package auth

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
)

func TestAuthenticator_BearerAndRotationAndAllowlist(t *testing.T) {
	authn := NewAuthenticator([]config.TenantConfig{
		{
			Name:          "alpha",
			AccessKey:     "sk-alpha-primary",
			AccessKeys:    []string{"sk-alpha-rotated"},
			AllowedModels: []string{"gpt-4o", "us.anthropic.*", "arn:aws:bedrock:us-east-1:123456789012:inference-profile/*"},
		},
		{
			Name:          "beta",
			AccessKey:     "sk-beta",
			AllowedModels: []string{"us.amazon.nova-lite-v1:0"},
		},
	})

	if _, ok := authn.AuthenticateToken(""); ok {
		t.Fatal("empty token must not authenticate")
	}
	if _, ok := authn.AuthenticateToken("sk-unknown"); ok {
		t.Fatal("unknown token must not authenticate")
	}

	alpha1, ok := authn.AuthenticateToken("sk-alpha-primary")
	if !ok || alpha1.Name != "alpha" {
		t.Fatalf("expected alpha for primary key, got %v %v", alpha1, ok)
	}
	alpha2, ok := authn.AuthenticateToken("sk-alpha-rotated")
	if !ok || alpha2 != alpha1 {
		t.Fatalf("expected same alpha Tenant pointer for rotation key")
	}

	if !alpha1.IsModelAllowed("gpt-4o", "us.amazon.nova-pro-v1:0") {
		t.Error("expected explicit alias gpt-4o to be allowed")
	}
	if !alpha1.IsModelAllowed("custom-alias", "us.anthropic.claude-sonnet-4-20250514-v1:0") {
		t.Error("expected resolved Bedrock ID matching us.anthropic.* to be allowed")
	}
	if !alpha1.IsModelAllowed("arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-prof", "") {
		t.Error("expected ARN prefix wildcard to match")
	}
	if alpha1.IsModelAllowed("us.amazon.nova-lite-v1:0", "us.amazon.nova-lite-v1:0") {
		t.Error("expected disallowed model to be rejected for alpha")
	}
}

func TestMiddleware_AuthErrorsRateLimitAndConcurrency(t *testing.T) {
	authn := NewAuthenticator([]config.TenantConfig{
		{
			Name:              "rps-limited",
			AccessKey:         "sk-rps",
			AllowedModels:     []string{"*"},
			RequestsPerSecond: 1,
			Burst:             1,
		},
		{
			Name:          "conc-limited",
			AccessKey:     "sk-conc",
			AllowedModels: []string{"*"},
			MaxConcurrent: 1,
		},
	})

	gate := make(chan struct{})
	entered := make(chan struct{})
	var enteredOnce sync.Once

	handler := authn.Middleware(true, nil)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ten, ok := TenantFromContext(r.Context())
		if !ok {
			t.Error("missing tenant in context")
		}
		if ten.Name == "conc-limited" {
			enteredOnce.Do(func() { close(entered) })
			<-gate
		}
		w.WriteHeader(http.StatusOK)
	}))

	// 1. Missing Authorization header -> 401 OpenAI JSON error
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rr.Code)
	}
	var env openai.ErrorEnvelope
	if err := json.Unmarshal(rr.Body.Bytes(), &env); err != nil {
		t.Fatalf("decode 401 json: %v", err)
	}
	if env.Error.Type != openai.TypeAuthentication {
		t.Errorf("expected authentication_error, got %q", env.Error.Type)
	}

	// 2. RPS limiter: first passes, immediate second gets 429 + Retry-After
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req1.Header.Set("Authorization", "bearer sk-rps")
	rr1 := httptest.NewRecorder()
	handler.ServeHTTP(rr1, req1)
	if rr1.Code != http.StatusOK {
		t.Fatalf("expected first request 200, got %d", rr1.Code)
	}

	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req2.Header.Set("Authorization", "Bearer sk-rps")
	rr2 := httptest.NewRecorder()
	handler.ServeHTTP(rr2, req2)
	if rr2.Code != http.StatusTooManyRequests {
		t.Fatalf("expected second burst request 429, got %d", rr2.Code)
	}
	if rr2.Header().Get("Retry-After") == "" {
		t.Error("expected Retry-After header on 429 response")
	}

	// 3. Max concurrency semaphore: 1st holds slot, 2nd gets 429, after release 3rd gets 200
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		rHold := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		rHold.Header.Set("Authorization", "Bearer sk-conc")
		wHold := httptest.NewRecorder()
		handler.ServeHTTP(wHold, rHold)
	}()

	<-entered

	rBlocked := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rBlocked.Header.Set("Authorization", "Bearer sk-conc")
	wBlocked := httptest.NewRecorder()
	handler.ServeHTTP(wBlocked, rBlocked)
	if wBlocked.Code != http.StatusTooManyRequests {
		t.Fatalf("expected concurrent overflow 429, got %d", wBlocked.Code)
	}

	close(gate)
	wg.Wait()

	// Re-open gate for a subsequent request now that semaphore was released
	gate = make(chan struct{})
	close(gate)
	rAfter := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rAfter.Header.Set("Authorization", "Bearer sk-conc")
	wAfter := httptest.NewRecorder()
	handler.ServeHTTP(wAfter, rAfter)
	if wAfter.Code != http.StatusOK {
		t.Fatalf("expected 200 after semaphore release, got %d", wAfter.Code)
	}
}
