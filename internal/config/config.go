package config

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// Credential source constants for [aws].credential_source.
const (
	CredentialSourceAuto        = "auto"
	CredentialSourceIMDS        = "imds"
	CredentialSourceECS         = "ecs"
	CredentialSourceWebIdentity = "web_identity"
	CredentialSourceEnv         = "env"
	CredentialSourceProfile     = "profile"
)

// Duration wraps time.Duration for TOML string decoding (e.g., "10s", "168h").
type Duration struct {
	time.Duration
}

// UnmarshalText parses a Go duration string.
func (d *Duration) UnmarshalText(text []byte) error {
	s := strings.TrimSpace(string(text))
	if s == "" || s == "0" {
		d.Duration = 0
		return nil
	}
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	d.Duration = parsed
	return nil
}

// MarshalText formats Duration as a standard Go duration string.
func (d Duration) MarshalText() ([]byte, error) {
	return []byte(d.Duration.String()), nil
}

// Config represents the top-level gateway configuration loaded from config.toml.
type Config struct {
	AccessKey string        `toml:"access_key"`
	Server    ServerConfig  `toml:"server"`
	AWS       AWSConfig     `toml:"aws"`
	Models    ModelsConfig  `toml:"models"`
	Storage   StorageConfig `toml:"storage"`
	Tenants   []TenantConfig `toml:"tenants"`
}

// ServerConfig configures the HTTP listeners, timeouts, and security guards.
type ServerConfig struct {
	ListenAddr           string   `toml:"listen_addr"`
	MetricsListenAddr    string   `toml:"metrics_listen_addr"`
	MetricsRequireAuth   bool     `toml:"metrics_require_auth"`
	ReadHeaderTimeout    Duration `toml:"read_header_timeout"`
	WriteTimeout         Duration `toml:"write_timeout"`
	RequestTimeout       Duration `toml:"request_timeout"`
	MaxRequestBodyBytes  int64    `toml:"max_request_body_bytes"`
	TLSCertFile          string   `toml:"tls_cert_file"`
	TLSKeyFile           string   `toml:"tls_key_file"`
	AllowRemoteMediaURLs bool     `toml:"allow_remote_media_urls"`
	LogLevel             string   `toml:"log_level"`
}

// AWSConfig configures AWS credential resolution, STS assume-role, and PrivateLink endpoints.
type AWSConfig struct {
	Region                     string `toml:"region"`
	CredentialSource           string `toml:"credential_source"`
	Profile                    string `toml:"profile"`
	AssumeRoleARN              string `toml:"assume_role_arn"`
	AssumeRoleExternalID       string `toml:"assume_role_external_id"`
	MaxAttempts                int    `toml:"max_attempts"`
	VerifyCredentialsOnStartup bool   `toml:"verify_credentials_on_startup"`
	BedrockRuntimeEndpoint     string `toml:"bedrock_runtime_endpoint"`
	BedrockEndpoint            string `toml:"bedrock_endpoint"`
	STSEndpoint                string `toml:"sts_endpoint"`
}

// ModelsConfig configures model defaults, aliases, passthrough, discovery, and prompt caching.
type ModelsConfig struct {
	DefaultModel          string            `toml:"default_model"`
	DefaultEmbeddingModel string            `toml:"default_embedding_model"`
	AllowPassthrough      bool              `toml:"allow_passthrough"`
	DiscoverBedrockModels bool              `toml:"discover_bedrock_models"`
	DiscoveryCacheTTL     Duration          `toml:"discovery_cache_ttl"`
	EnableAutoPromptCache bool              `toml:"enable_auto_prompt_cache"`
	Aliases               map[string]string `toml:"aliases"`
}

// StorageConfig configures the pure-Go SQLite store used by /v1/responses.
type StorageConfig struct {
	SQLitePath      string   `toml:"sqlite_path"`
	ResponsesTTL    Duration `toml:"responses_ttl"`
	CleanupInterval Duration `toml:"cleanup_interval"`
}

// TenantConfig defines an API consumer tenant and its rate/concurrency/model policies.
type TenantConfig struct {
	Name              string   `toml:"name"`
	AccessKey         string   `toml:"access_key"`
	AccessKeys        []string `toml:"access_keys"`
	AllowedModels     []string `toml:"allowed_models"`
	DefaultModel      string   `toml:"default_model"`
	RequestsPerSecond float64  `toml:"requests_per_second"`
	Burst             int      `toml:"burst"`
	MaxConcurrent     int      `toml:"max_concurrent"`
}

// AllKeys returns the normalized list of active Bearer tokens for this tenant.
func (t TenantConfig) AllKeys() []string {
	out := make([]string, 0, 1+len(t.AccessKeys))
	if t.AccessKey != "" {
		out = append(out, t.AccessKey)
	}
	out = append(out, t.AccessKeys...)
	return out
}

// Default returns a Config populated with production defaults prior to TOML decoding.
func Default() Config {
	return Config{
		Server: ServerConfig{
			ListenAddr:           "0.0.0.0:8080",
			MetricsListenAddr:    "",
			MetricsRequireAuth:   true,
			ReadHeaderTimeout:    Duration{10 * time.Second},
			WriteTimeout:         Duration{0},
			RequestTimeout:       Duration{300 * time.Second},
			MaxRequestBodyBytes:  33554432, // 32 MiB
			AllowRemoteMediaURLs: false,
			LogLevel:             "info",
		},
		AWS: AWSConfig{
			Region:                     "", // Populated by TOML, AWS_REGION, AWS_DEFAULT_REGION, or IMDS/SDK fallback
			CredentialSource:           CredentialSourceAuto,
			MaxAttempts:                3,
			VerifyCredentialsOnStartup: true,
		},
		Models: ModelsConfig{
			DefaultModel:          "us.anthropic.claude-sonnet-4-20250514-v1:0",
			DefaultEmbeddingModel: "amazon.titan-embed-text-v2:0",
			AllowPassthrough:      true,
			DiscoverBedrockModels: true,
			DiscoveryCacheTTL:     Duration{10 * time.Minute},
			EnableAutoPromptCache: false,
			Aliases:               map[string]string{},
		},
		Storage: StorageConfig{
			SQLitePath:      "/data/gateway.db",
			ResponsesTTL:    Duration{168 * time.Hour},
			CleanupInterval: Duration{1 * time.Hour},
		},
	}
}

// ResolvePath resolves the configuration file path from CLI flag, GATEWAY_CONFIG env, or default.
func ResolvePath(flagPath string) string {
	if p := strings.TrimSpace(flagPath); p != "" {
		return p
	}
	if p := strings.TrimSpace(os.Getenv("GATEWAY_CONFIG")); p != "" {
		return p
	}
	return "config.toml"
}

// Load reads, decodes, applies environment overrides, normalizes, and validates the TOML configuration.
func Load(path string) (*Config, error) {
	resolved := path
	if strings.TrimSpace(resolved) == "" {
		resolved = ResolvePath("")
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", resolved, err)
	}
	return LoadBytes(data)
}

// LoadBytes parses TOML bytes, applies environment overrides, normalizes tenants, and validates.
func LoadBytes(data []byte) (*Config, error) {
	cfg := Default()
	md, err := toml.NewDecoder(bytes.NewReader(data)).Decode(&cfg)
	if err != nil {
		return nil, fmt.Errorf("parse config toml: %w", err)
	}
	if undecoded := md.Undecoded(); len(undecoded) > 0 {
		keys := make([]string, len(undecoded))
		for i, k := range undecoded {
			keys[i] = k.String()
		}
		return nil, fmt.Errorf("unknown configuration keys: %s", strings.Join(keys, ", "))
	}

	applyEnvOverrides(&cfg)
	if err := cfg.NormalizeAndValidate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func applyEnvOverrides(cfg *Config) {
	if envKey := strings.TrimSpace(os.Getenv("GATEWAY_ACCESS_KEY")); envKey != "" {
		cfg.AccessKey = envKey
	}
	if strings.TrimSpace(cfg.AWS.Region) == "" {
		if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
			cfg.AWS.Region = r
		} else if r := strings.TrimSpace(os.Getenv("AWS_DEFAULT_REGION")); r != "" {
			cfg.AWS.Region = r
		}
	}
	if envListen := strings.TrimSpace(os.Getenv("GATEWAY_LISTEN_ADDR")); envListen != "" {
		cfg.Server.ListenAddr = envListen
	}
	if envDB := strings.TrimSpace(os.Getenv("GATEWAY_SQLITE_PATH")); envDB != "" {
		cfg.Storage.SQLitePath = envDB
	}
}

// NormalizeAndValidate normalizes single-key and multi-tenant definitions and checks all invariants.
func (c *Config) NormalizeAndValidate() error {
	// Server validation
	c.Server.ListenAddr = strings.TrimSpace(c.Server.ListenAddr)
	if c.Server.ListenAddr == "" {
		return errors.New("server.listen_addr must not be empty")
	}
	c.Server.MetricsListenAddr = strings.TrimSpace(c.Server.MetricsListenAddr)
	if c.Server.ReadHeaderTimeout.Duration < 0 {
		return errors.New("server.read_header_timeout must be >= 0")
	}
	if c.Server.WriteTimeout.Duration < 0 {
		return errors.New("server.write_timeout must be >= 0")
	}
	if c.Server.RequestTimeout.Duration <= 0 {
		return errors.New("server.request_timeout must be > 0")
	}
	if c.Server.MaxRequestBodyBytes <= 0 {
		return errors.New("server.max_request_body_bytes must be > 0")
	}
	c.Server.TLSCertFile = strings.TrimSpace(c.Server.TLSCertFile)
	c.Server.TLSKeyFile = strings.TrimSpace(c.Server.TLSKeyFile)
	if (c.Server.TLSCertFile == "") != (c.Server.TLSKeyFile == "") {
		return errors.New("server.tls_cert_file and server.tls_key_file must both be set or both be empty")
	}
	c.Server.LogLevel = strings.ToLower(strings.TrimSpace(c.Server.LogLevel))
	switch c.Server.LogLevel {
	case "debug", "info", "warn", "error":
	default:
		return fmt.Errorf("invalid server.log_level %q (expected debug, info, warn, or error)", c.Server.LogLevel)
	}

	// AWS validation
	c.AWS.Region = strings.TrimSpace(c.AWS.Region)
	c.AWS.CredentialSource = strings.ToLower(strings.TrimSpace(c.AWS.CredentialSource))
	if c.AWS.CredentialSource == "" {
		c.AWS.CredentialSource = CredentialSourceAuto
	}
	switch c.AWS.CredentialSource {
	case CredentialSourceAuto,
		CredentialSourceIMDS,
		CredentialSourceECS,
		CredentialSourceWebIdentity,
		CredentialSourceEnv,
		CredentialSourceProfile:
	default:
		return fmt.Errorf("invalid aws.credential_source %q (expected auto, imds, ecs, web_identity, env, or profile)", c.AWS.CredentialSource)
	}
	if c.AWS.MaxAttempts < 1 {
		return errors.New("aws.max_attempts must be >= 1")
	}
	c.AWS.Profile = strings.TrimSpace(c.AWS.Profile)
	c.AWS.AssumeRoleARN = strings.TrimSpace(c.AWS.AssumeRoleARN)
	c.AWS.AssumeRoleExternalID = strings.TrimSpace(c.AWS.AssumeRoleExternalID)
	c.AWS.BedrockRuntimeEndpoint = strings.TrimSpace(c.AWS.BedrockRuntimeEndpoint)
	c.AWS.BedrockEndpoint = strings.TrimSpace(c.AWS.BedrockEndpoint)
	c.AWS.STSEndpoint = strings.TrimSpace(c.AWS.STSEndpoint)

	// Models validation
	if c.Models.Aliases == nil {
		c.Models.Aliases = map[string]string{}
	}
	cleanedAliases := make(map[string]string, len(c.Models.Aliases))
	for alias, target := range c.Models.Aliases {
		a := strings.TrimSpace(alias)
		t := strings.TrimSpace(target)
		if a == "" || t == "" {
			return errors.New("models.aliases entries must have non-empty alias and target")
		}
		cleanedAliases[a] = t
	}
	c.Models.Aliases = cleanedAliases
	if c.Models.DiscoveryCacheTTL.Duration <= 0 {
		return errors.New("models.discovery_cache_ttl must be > 0")
	}

	// Storage validation
	c.Storage.SQLitePath = strings.TrimSpace(c.Storage.SQLitePath)
	if c.Storage.SQLitePath == "" {
		return errors.New("storage.sqlite_path must not be empty")
	}
	if c.Storage.ResponsesTTL.Duration < 0 {
		return errors.New("storage.responses_ttl must be >= 0")
	}
	if c.Storage.CleanupInterval.Duration <= 0 {
		return errors.New("storage.cleanup_interval must be > 0")
	}

	// Tenant normalization: map top-level access_key into "default" tenant if set.
	topKey := strings.TrimSpace(c.AccessKey)
	c.AccessKey = topKey
	if topKey != "" {
		merged := false
		for i := range c.Tenants {
			if strings.TrimSpace(c.Tenants[i].Name) == "default" {
				if strings.TrimSpace(c.Tenants[i].AccessKey) == "" {
					c.Tenants[i].AccessKey = topKey
				} else if strings.TrimSpace(c.Tenants[i].AccessKey) != topKey {
					c.Tenants[i].AccessKeys = append([]string{topKey}, c.Tenants[i].AccessKeys...)
				}
				merged = true
				break
			}
		}
		if !merged {
			defaultTenant := TenantConfig{
				Name:          "default",
				AccessKey:     topKey,
				AllowedModels: []string{"*"},
			}
			c.Tenants = append([]TenantConfig{defaultTenant}, c.Tenants...)
		}
	}

	if len(c.Tenants) == 0 {
		return errors.New("at least one access key must be configured via top-level access_key, GATEWAY_ACCESS_KEY, or [[tenants]]")
	}

	seenTenantNames := make(map[string]struct{}, len(c.Tenants))
	seenTokens := make(map[string]string)

	for i := range c.Tenants {
		t := &c.Tenants[i]
		t.Name = strings.TrimSpace(t.Name)
		if t.Name == "" {
			return fmt.Errorf("tenants[%d].name must not be empty", i)
		}
		if _, exists := seenTenantNames[t.Name]; exists {
			return fmt.Errorf("duplicate tenant name %q", t.Name)
		}
		seenTenantNames[t.Name] = struct{}{}

		t.AccessKey = strings.TrimSpace(t.AccessKey)
		cleanedExtra := make([]string, 0, len(t.AccessKeys))
		for j, raw := range t.AccessKeys {
			k := strings.TrimSpace(raw)
			if k == "" {
				return fmt.Errorf("tenant %q access_keys[%d] must not be empty", t.Name, j)
			}
			cleanedExtra = append(cleanedExtra, k)
		}
		t.AccessKeys = cleanedExtra

		keys := t.AllKeys()
		if len(keys) == 0 {
			return fmt.Errorf("tenant %q must configure access_key or at least one key in access_keys", t.Name)
		}
		for _, k := range keys {
			if prevTenant, dup := seenTokens[k]; dup {
				return fmt.Errorf("duplicate access key configured on tenant %q (already used by tenant %q)", t.Name, prevTenant)
			}
			seenTokens[k] = t.Name
		}

		if len(t.AllowedModels) == 0 {
			t.AllowedModels = []string{"*"}
		} else {
			models := make([]string, 0, len(t.AllowedModels))
			for j, m := range t.AllowedModels {
				trimmed := strings.TrimSpace(m)
				if trimmed == "" {
					return fmt.Errorf("tenant %q allowed_models[%d] must not be empty", t.Name, j)
				}
				models = append(models, trimmed)
			}
			t.AllowedModels = models
		}

		t.DefaultModel = strings.TrimSpace(t.DefaultModel)

		if t.RequestsPerSecond < 0 || math.IsNaN(t.RequestsPerSecond) || math.IsInf(t.RequestsPerSecond, 0) {
			return fmt.Errorf("tenant %q requests_per_second must be a finite number >= 0", t.Name)
		}
		if t.Burst < 0 {
			return fmt.Errorf("tenant %q burst must be >= 0", t.Name)
		}
		if t.RequestsPerSecond > 0 && t.Burst == 0 {
			t.Burst = max(1, int(math.Ceil(t.RequestsPerSecond)))
		}
		if t.MaxConcurrent < 0 {
			return fmt.Errorf("tenant %q max_concurrent must be >= 0", t.Name)
		}
	}

	return nil
}
