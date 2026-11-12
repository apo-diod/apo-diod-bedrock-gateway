# apo-diod-bedrock-gateway

A lightweight, pure-Go (`CGO_ENABLED=0`), private-VPC-friendly HTTP gateway that exposes an **OpenAI-compatible REST API** backed by **AWS Bedrock**. Packaged as a minimal rootless distroless Docker image (`UID 65532`) consuming IAM Instance Profile / Task Role / IRSA credentials and authenticating API clients via Bearer tokens configured in `config.toml`.

## Features

- **OpenAI API Compatibility**:
  - `POST /v1/chat/completions` — Non-streaming and SSE streaming (`stream_options.include_usage`), function/tool calling, structured JSON outputs (`json_object`, `json_schema`), inline base64 images & PDF documents.
  - `POST /v1/responses`, `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` — Stateful multi-turn chaining via `previous_response_id` backed by embedded pure-Go SQLite (`WAL` mode) with configurable TTL expiry and typed Responses SSE events.
  - `POST /v1/embeddings` — Amazon Titan Text Embeddings v1/v2 (bounded concurrent worker pool) and Cohere Embed v3/v4 (native 96-item batching), supporting both `"float"` and `"base64"` (`float32` little-endian) encodings and `dimensions`.
  - `GET /v1/models` & `GET /v1/models/{model}` — Merges configured aliases with optional TTL-cached `bedrock:ListInferenceProfiles` and `bedrock:ListFoundationModels` discovery (gracefully falling back to static models when IAM only grants `bedrock-runtime` invoke permissions).
- **Bedrock Model Extensions**:
  - **Extended Thinking / Reasoning**: Maps `reasoning_effort` (`low`, `medium`, `high`) and `thinking.budget_tokens` to Anthropic Claude `thinking` and Amazon Nova `reasoningConfig`, preserving Claude cryptographic `signature` & `redactedContent` blocks across multi-turn `/v1/responses` conversations.
  - **Prompt Caching**: Explicit `cache_control: {"type": "ephemeral"}` and optional `enable_auto_prompt_cache` mapped to Bedrock `CachePoint` blocks, reporting `cached_tokens` in OpenAI usage payloads.
- **Security & Multi-Tenancy**:
  - Single-key shorthand (`access_key = "..."` or `GATEWAY_ACCESS_KEY`) AND/OR multi-tenant `[[tenants]]` definitions with zero-downtime key rotation (`access_keys`), per-tenant `allowed_models` wildcards, RPS token-bucket rate limiters, and concurrency semaphores.
  - Constant-time SHA-256 Bearer token comparison (`crypto/subtle.ConstantTimeCompare`).
  - Strict cross-tenant isolation across model catalogs and SQLite `/v1/responses` state.
  - Remote media URL fetching disabled by default (`allow_remote_media_urls = false`) to eliminate SSRF against EC2 IMDS (`169.254.169.254`), ECS (`169.254.170.2`), and private VPC hosts; connect-time IP blocklist enforced even when explicitly enabled.
  - Zero external runtime internet dependencies — runs cleanly inside air-gapped VPCs over AWS PrivateLink.
- **Observability & Operations**:
  - `GET /health` (`/healthz`), `GET /readyz` (verifies SQLite + AWS credential provider readiness), and `GET /metrics` (Prometheus metrics, Bearer-protected by default or bound to an isolated internal `metrics_listen_addr`).
  - Built-in `-healthcheck` binary flag for shell-free distroless container health probes.
  - Structured JSON `slog` request logs (never logging raw prompts or Bearer tokens).

---

## Minimal AWS IAM Policy

Attach the following IAM policy to your EC2 Instance Profile, ECS Task Role, or EKS Pod Role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockDataPlaneInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:*:inference-profile/*"
      ]
    },
    {
      "Sid": "OptionalControlPlaneDiscovery",
      "Effect": "Allow",
      "Action": [
        "bedrock:ListFoundationModels",
        "bedrock:ListInferenceProfiles"
      ],
      "Resource": "*"
    }
  ]
}
```

> **Note:** `OptionalControlPlaneDiscovery` is purely optional. If omitted (or set `discover_bedrock_models = false` in `config.toml`), the gateway logs a one-time warning per cache window and serves static configured models/aliases.

---

## Quick Start

### 1. Create `config.toml`

Copy `config.example.toml` to `config.toml` or create a minimal single-key config:

```toml
access_key = "sk-your-strong-random-gateway-token"

[aws]
region = "us-east-1"
credential_source = "imds" # "auto" | "imds" | "ecs" | "web_identity" | "env" | "profile"

[models]
default_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
default_embedding_model = "amazon.titan-embed-text-v2:0"

[models.aliases]
"gpt-4o" = "us.anthropic.claude-sonnet-4-20250514-v1:0"
"gpt-4o-mini" = "us.amazon.nova-lite-v1:0"
"text-embedding-3-small" = "amazon.titan-embed-text-v2:0"
```

### 2. Deploy with Docker / Docker Compose

#### Option A: EC2 Instance Profile with Docker Bridge Networking (Recommended)

EC2 IMDSv2 defaults to an IPv4 TTL hop limit of `1`, which drops token `PUT` responses before they cross the `docker0` bridge into a container. Set the instance hop limit to `2`:

```bash
aws ec2 modify-instance-metadata-options \
  --instance-id i-0123456789abcdef0 \
  --http-tokens required \
  --http-put-response-hop-limit 2
```

Then launch the container:

```bash
docker compose up -d --build
```

#### Option B: EC2 Instance Profile with Host Networking (Hop Limit = 1)

If you cannot modify EC2 instance metadata options, run the container in the host network namespace (`network_mode: host` in `docker-compose.yml` or `--network host`):

```bash
docker run -d --name bedrock-gateway \
  --network host \
  --read-only \
  -v "$(pwd)/config.toml:/etc/bedrock-gateway/config.toml:ro" \
  -v bedrock_gateway_data:/data \
  apo-diod-bedrock-gateway:latest
```

#### Option C: ECS Task Role / EKS IRSA / Pod Identity / `AWS_*` Environment Variables

Set `credential_source` in `config.toml` to `"ecs"`, `"web_identity"`, `"env"`, or `"auto"` to match your runtime environment:

- **`"ecs"`**: Uses `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` / `AWS_CONTAINER_CREDENTIALS_FULL_URI` injected by Amazon ECS or EKS Pod Identity.
- **`"web_identity"`**: Uses `AWS_ROLE_ARN` and `AWS_WEB_IDENTITY_TOKEN_FILE` injected by EKS IRSA.
- **`"env"`**: Uses `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`.
- **`"auto"`**: Evaluates the standard AWS SDK v2 credential chain automatically.

### 3. Private VPC / AWS PrivateLink Setup

For isolated subnets with no NAT Gateway or Internet Gateway, provision VPC Interface Endpoints for:
- `com.amazonaws.<region>.bedrock-runtime` (required)
- `com.amazonaws.<region>.bedrock` (only if `discover_bedrock_models = true`)
- `com.amazonaws.<region>.sts` (only if `credential_source = "web_identity"` or `assume_role_arn` is set)

If Private DNS is enabled on the VPC endpoints, standard AWS regional endpoints resolve privately automatically. Otherwise, set explicit endpoint URLs under `[aws]` in `config.toml`:

```toml
[aws]
region = "us-east-1"
credential_source = "imds"
bedrock_runtime_endpoint = "https://vpce-xxxx.bedrock-runtime.us-east-1.vpce.amazonaws.com"
bedrock_endpoint = "https://vpce-yyyy.bedrock.us-east-1.vpce.amazonaws.com"
```

---

## Client Usage Examples

### `curl`

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-strong-random-gateway-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain AWS PrivateLink in two sentences."}],
    "stream": false
  }'
```

### Python (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-strong-random-gateway-token",
)

# 1. Chat Completions (Streaming + Extended Thinking)
stream = client.chat.completions.create(
    model="gpt-4o",
    reasoning_effort="low",
    messages=[{"role": "user", "content": "Compare DynamoDB and Aurora."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# 2. Stateful Responses API (SQLite-backed previous_response_id)
r1 = client.responses.create(
    model="gpt-4o",
    input="Remember the codeword: ORBIT-42.",
)
r2 = client.responses.create(
    model="gpt-4o",
    previous_response_id=r1.id,
    input="What was the codeword?",
)
print("\nResponse:", r2.output_text)

# 3. Embeddings
emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Private VPC deployment", "AWS Bedrock Converse"],
)
print("Embedding dimensions:", len(emb.data[0].embedding))
```

### Node.js / TypeScript (`openai` SDK)

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-your-strong-random-gateway-token",
});

const completion = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello from Bedrock!" }],
});
console.log(completion.choices[0]?.message.content);
```

---

## Building & Testing from Source

```bash
# Run unit & integration tests with race detector
go test -race ./...

# Static analysis
go vet ./...

# Build static binary (zero C dependencies)
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o bedrock-gateway ./cmd/bedrock-gateway
```

## License

Released into the public domain under [The Unlicense](LICENSE).
