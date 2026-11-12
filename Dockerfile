# syntax=docker/dockerfile:1

FROM docker.io/library/golang:1.27-alpine AS builder

WORKDIR /src

# Download dependencies in a cacheable layer
COPY go.mod go.sum ./
RUN GOTOOLCHAIN=local go mod download

# Build static pure-Go binary (zero CGO dependencies)
COPY cmd/ ./cmd/
COPY internal/ ./internal/

ARG VERSION=dev
RUN CGO_ENABLED=0 GOTOOLCHAIN=local GOOS=linux go build \
    -trimpath \
    -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/bedrock-gateway ./cmd/bedrock-gateway && \
    mkdir -p /out/data && \
    chown -R 65532:65532 /out/data && \
    chmod 0750 /out/data

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /out/bedrock-gateway /usr/local/bin/bedrock-gateway
COPY --from=builder --chown=65532:65532 /out/data /data
COPY --chown=65532:65532 config.example.toml /etc/bedrock-gateway/config.example.toml

ENV GATEWAY_CONFIG=/etc/bedrock-gateway/config.toml

USER 65532:65532
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=4s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/bedrock-gateway", "-healthcheck"]

ENTRYPOINT ["/usr/local/bin/bedrock-gateway"]
CMD ["-config", "/etc/bedrock-gateway/config.toml"]
