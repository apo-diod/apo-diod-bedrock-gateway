package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLoadBytes_SingleAccessKeyShorthand(t *testing.T) {
	t.Setenv("GATEWAY_ACCESS_KEY", "")
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "")

	raw := `
access_key = "sk-simple-secret"

[aws]
region = "eu-central-1"
`
	cfg, err := LoadBytes([]byte(raw))
	if err != nil {
		t.Fatalf("LoadBytes returned error: %v", err)
	}
	if len(cfg.Tenants) != 1 {
		t.Fatalf("expected 1 default tenant, got %d", len(cfg.Tenants))
	}
	def := cfg.Tenants[0]
	if def.Name != "default" {
		t.Errorf("expected tenant name 'default', got %q", def.Name)
	}
	keys := def.AllKeys()
	if len(keys) != 1 || keys[0] != "sk-simple-secret" {
		t.Errorf("unexpected default tenant keys: %v", keys)
	}
	if len(def.AllowedModels) != 1 || def.AllowedModels[0] != "*" {
		t.Errorf("expected wildcard allowed_models, got %v", def.AllowedModels)
	}
	if cfg.AWS.Region != "eu-central-1" {
		t.Errorf("expected region eu-central-1, got %q", cfg.AWS.Region)
	}
	if cfg.Server.RequestTimeout.Duration != 300*time.Second {
		t.Errorf("expected default request_timeout 300s, got %v", cfg.Server.RequestTimeout.Duration)
	}
}

func TestLoadBytes_MultiTenantAndRotationKeys(t *testing.T) {
	t.Setenv("GATEWAY_ACCESS_KEY", "")

	raw := `
access_key = "sk-root-default"

[models.aliases]
"gpt-4o" = "us.anthropic.claude-sonnet-4-20250514-v1:0"

[[tenants]]
name = "team-alpha"
access_key = "sk-alpha-current"
access_keys = ["sk-alpha-old-1", "sk-alpha-old-2"]
allowed_models = ["gpt-4o", "us.anthropic.*"]
requests_per_second = 12.5
max_concurrent = 10

[[tenants]]
name = "team-beta"
access_keys = ["sk-beta-only"]
`
	cfg, err := LoadBytes([]byte(raw))
	if err != nil {
		t.Fatalf("LoadBytes failed: %v", err)
	}
	if len(cfg.Tenants) != 3 {
		t.Fatalf("expected 3 tenants (default + 2 named), got %d", len(cfg.Tenants))
	}
	alpha := cfg.Tenants[1]
	if alpha.Name != "team-alpha" {
		t.Fatalf("expected team-alpha at index 1, got %q", alpha.Name)
	}
	if got := len(alpha.AllKeys()); got != 3 {
		t.Errorf("expected 3 rotation keys for team-alpha, got %d", got)
	}
	if alpha.Burst != 13 {
		t.Errorf("expected default burst ceil(12.5)=13, got %d", alpha.Burst)
	}
}

func TestLoad_EnvOverridesAndPathResolution(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "gateway.toml")
	if err := os.WriteFile(cfgPath, []byte(`
[aws]
credential_source = "imds"
`), 0o600); err != nil {
		t.Fatalf("write temp config: %v", err)
	}

	t.Setenv("GATEWAY_CONFIG", cfgPath)
	t.Setenv("GATEWAY_ACCESS_KEY", "sk-from-env")
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "ap-northeast-1")

	if got := ResolvePath(""); got != cfgPath {
		t.Fatalf("ResolvePath empty flag expected %q, got %q", cfgPath, got)
	}
	if got := ResolvePath("/explicit.toml"); got != "/explicit.toml" {
		t.Fatalf("ResolvePath explicit flag expected /explicit.toml, got %q", got)
	}

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load via GATEWAY_CONFIG failed: %v", err)
	}
	if cfg.AWS.Region != "ap-northeast-1" {
		t.Errorf("expected AWS_DEFAULT_REGION fallback ap-northeast-1, got %q", cfg.AWS.Region)
	}
	if cfg.AWS.CredentialSource != CredentialSourceIMDS {
		t.Errorf("expected credential_source imds, got %q", cfg.AWS.CredentialSource)
	}
	if len(cfg.Tenants) != 1 || cfg.Tenants[0].AllKeys()[0] != "sk-from-env" {
		t.Errorf("expected GATEWAY_ACCESS_KEY tenant, got %+v", cfg.Tenants)
	}
}

func TestLoadBytes_InvalidConfigsRejected(t *testing.T) {
	t.Setenv("GATEWAY_ACCESS_KEY", "")

	cases := []struct {
		name    string
		toml    string
		wantSub string
	}{
		{
			name:    "no keys configured",
			toml:    `[aws]` + "\n" + `region = "us-east-1"`,
			wantSub: "at least one access key",
		},
		{
			name: "duplicate key across tenants",
			toml: `
[[tenants]]
name = "a"
access_key = "sk-shared"

[[tenants]]
name = "b"
access_key = "sk-shared"
`,
			wantSub: "duplicate access key",
		},
		{
			name: "duplicate key within tenant rotation list",
			toml: `
[[tenants]]
name = "a"
access_key = "sk-same"
access_keys = ["sk-same"]
`,
			wantSub: "duplicate access key",
		},
		{
			name: "invalid credential_source",
			toml: `
access_key = "sk-1"
[aws]
credential_source = "magic"
`,
			wantSub: "invalid aws.credential_source",
		},
		{
			name: "unknown toml key",
			toml: `
access_key = "sk-1"
typo_field = true
`,
			wantSub: "unknown configuration keys",
		},
		{
			name: "mismatched tls files",
			toml: `
access_key = "sk-1"
[server]
tls_cert_file = "/etc/ssl/cert.pem"
`,
			wantSub: "tls_cert_file and server.tls_key_file",
		},
		{
			name: "invalid duration",
			toml: `
access_key = "sk-1"
[server]
request_timeout = "forever"
`,
			wantSub: "invalid duration",
		},
		{
			name: "negative rate limit",
			toml: `
[[tenants]]
name = "neg"
access_key = "sk-neg"
requests_per_second = -1.0
`,
			wantSub: "requests_per_second",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadBytes([]byte(tc.toml))
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("expected error containing %q, got %q", tc.wantSub, err.Error())
			}
		})
	}
}

func TestConfigExampleTOML_IsValid(t *testing.T) {
	t.Setenv("GATEWAY_ACCESS_KEY", "")
	cfg, err := Load("../../config.example.toml")
	if err != nil {
		t.Fatalf("config.example.toml failed validation: %v", err)
	}
	if len(cfg.Tenants) != 1 || cfg.Tenants[0].Name != "team-alpha" {
		t.Fatalf("unexpected tenants in config.example.toml: %+v", cfg.Tenants)
	}
}

