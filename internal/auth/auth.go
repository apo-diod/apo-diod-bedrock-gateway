package auth

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"math"
	"net/http"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
	"golang.org/x/time/rate"
)

type tenantContextKey struct{}

// Tenant represents an authenticated gateway tenant at runtime.
type Tenant struct {
	Name          string
	AllowedModels []string
	DefaultModel  string
	limiter       *rate.Limiter
	sem           chan struct{}
}

// IsModelAllowed reports whether either requestedModel or resolvedModel matches the tenant's allowlist.
func (t *Tenant) IsModelAllowed(requestedModel, resolvedModel string) bool {
	if t == nil {
		return false
	}
	if len(t.AllowedModels) == 0 {
		return true
	}
	for _, pattern := range t.AllowedModels {
		if matchModelPattern(pattern, requestedModel) || (resolvedModel != "" && matchModelPattern(pattern, resolvedModel)) {
			return true
		}
	}
	return false
}

func matchModelPattern(pattern, candidate string) bool {
	if pattern == "*" {
		return true
	}
	if candidate == "" {
		return false
	}
	if pattern == candidate {
		return true
	}
	// Support prefix wildcard patterns containing slashes (such as inference profile ARNs)
	// in addition to standard shell-style wildcards.
	if strings.HasSuffix(pattern, "*") && !strings.Contains(pattern[:len(pattern)-1], "*") && !strings.Contains(pattern, "?") {
		prefix := pattern[:len(pattern)-1]
		return strings.HasPrefix(candidate, prefix)
	}
	if ok, err := path.Match(pattern, candidate); err == nil && ok {
		return true
	}
	return false
}

// WithTenant attaches t to ctx.
func WithTenant(ctx context.Context, t *Tenant) context.Context {
	return context.WithValue(ctx, tenantContextKey{}, t)
}

// TenantFromContext retrieves the authenticated *Tenant from ctx, if present.
func TenantFromContext(ctx context.Context) (*Tenant, bool) {
	t, ok := ctx.Value(tenantContextKey{}).(*Tenant)
	return t, ok && t != nil
}

type keyDigestEntry struct {
	digest [32]byte
	tenant *Tenant
}

// Authenticator validates Bearer tokens in constant time and enforces per-tenant limits.
type Authenticator struct {
	entries []keyDigestEntry
}

// NewAuthenticator builds an Authenticator from normalized TenantConfig definitions.
func NewAuthenticator(tenants []config.TenantConfig) *Authenticator {
	var entries []keyDigestEntry
	for _, tc := range tenants {
		var lim *rate.Limiter
		if tc.RequestsPerSecond > 0 {
			burst := tc.Burst
			if burst <= 0 {
				burst = max(1, int(math.Ceil(tc.RequestsPerSecond)))
			}
			lim = rate.NewLimiter(rate.Limit(tc.RequestsPerSecond), burst)
		}
		var sem chan struct{}
		if tc.MaxConcurrent > 0 {
			sem = make(chan struct{}, tc.MaxConcurrent)
		}
		rt := &Tenant{
			Name:          tc.Name,
			AllowedModels: append([]string(nil), tc.AllowedModels...),
			DefaultModel:  tc.DefaultModel,
			limiter:       lim,
			sem:           sem,
		}
		for _, k := range tc.AllKeys() {
			entries = append(entries, keyDigestEntry{
				digest: sha256.Sum256([]byte(k)),
				tenant: rt,
			})
		}
	}
	return &Authenticator{entries: entries}
}

// AuthenticateToken matches a raw token against all configured tenant key SHA-256 digests in constant time.
func (a *Authenticator) AuthenticateToken(token string) (*Tenant, bool) {
	if a == nil || token == "" || len(a.entries) == 0 {
		return nil, false
	}
	candidate := sha256.Sum256([]byte(token))
	var matched *Tenant
	found := 0
	for i := range a.entries {
		eq := subtle.ConstantTimeCompare(candidate[:], a.entries[i].digest[:])
		if eq == 1 {
			matched = a.entries[i].tenant
			found = 1
		}
	}
	return matched, found == 1
}

// Middleware authenticates Authorization: Bearer <token>, attaches Tenant context,
// and enforces per-tenant RPS and concurrency limits when enforceLimits is true.
func (a *Authenticator) Middleware(enforceLimits bool, onTenantAuthenticated func(r *http.Request, tenantName string)) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			token, ok := extractBearerToken(r.Header.Get("Authorization"))
			if !ok {
				w.Header().Set("WWW-Authenticate", `Bearer realm="bedrock-openai-gateway"`)
				openai.WriteError(
					w,
					http.StatusUnauthorized,
					openai.TypeAuthentication,
					"invalid_api_key",
					"Missing or malformed Authorization header. Expected 'Authorization: Bearer <access_key>'.",
				)
				return
			}

			tenant, valid := a.AuthenticateToken(token)
			if !valid {
				w.Header().Set("WWW-Authenticate", `Bearer realm="bedrock-openai-gateway"`)
				openai.WriteError(
					w,
					http.StatusUnauthorized,
					openai.TypeAuthentication,
					"invalid_api_key",
					"Invalid API key provided.",
				)
				return
			}

			if onTenantAuthenticated != nil {
				onTenantAuthenticated(r, tenant.Name)
			}

			if enforceLimits {
				if tenant.limiter != nil {
					res := tenant.limiter.Reserve()
					if !res.OK() {
						writeRateLimitError(w, 1, "Tenant rate limit exceeded.")
						return
					}
					delay := res.Delay()
					if delay > 0 {
						res.Cancel()
						retrySec := max(1, int(math.Ceil(delay.Seconds())))
						writeRateLimitError(w, retrySec, "Tenant rate limit exceeded.")
						return
					}
				}

				if tenant.sem != nil {
					select {
					case tenant.sem <- struct{}{}:
						defer func() { <-tenant.sem }()
					default:
						writeRateLimitError(w, 1, "Tenant maximum concurrent requests limit exceeded.")
						return
					}
				}
			}

			ctx := WithTenant(r.Context(), tenant)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func writeRateLimitError(w http.ResponseWriter, retryAfterSec int, msg string) {
	if retryAfterSec < 1 {
		retryAfterSec = 1
	}
	apiErr := openai.NewError(http.StatusTooManyRequests, openai.TypeRateLimit, "rate_limit_exceeded", msg)
	apiErr.RetryAfter = strconv.Itoa(retryAfterSec)
	openai.WriteAPIError(w, apiErr)
}

func extractBearerToken(header string) (string, bool) {
	header = strings.TrimSpace(header)
	if len(header) <= 7 {
		return "", false
	}
	if !strings.EqualFold(header[:7], "Bearer ") {
		return "", false
	}
	tok := strings.TrimSpace(header[7:])
	if tok == "" {
		return "", false
	}
	return tok, true
}

// NowUTC is exported for deterministic timestamp helpers when needed.
func NowUTC() time.Time {
	return time.Now().UTC()
}
