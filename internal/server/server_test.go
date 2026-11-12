package server

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
)

func TestServer_HealthReadyMetricsLoggingAndGuards(t *testing.T) {
	cfg := config.Default()
	cfg.Server.MaxRequestBodyBytes = 64
	cfg.Tenants = []config.TenantConfig{
		{
			Name:          "team-sec",
			AccessKey:     "sk-super-secret-token-never-log",
			AllowedModels: []string{"*"},
		},
	}

	var logBuf bytes.Buffer
	logger := NewLogger("info", &logBuf)
	mc := metrics.New()
	authn := auth.NewAuthenticator(cfg.Tenants)

	dbHealthy := true
	gw := New(Dependencies{
		Config:        &cfg,
		Authenticator: authn,
		Metrics:       mc,
		Logger:        logger,
		ReadinessChecks: []ReadinessCheck{
			{
				Name: "sqlite",
				Check: func(context.Context) error {
					if !dbHealthy {
						return errors.New("sqlite locked")
					}
					return nil
				},
			},
		},
		ChatCompletionsHandler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("X-Trigger-Panic") == "1" {
				panic("simulated handler crash")
			}
			_, err := io.ReadAll(r.Body)
			if apiErr, ok := openai.CheckMaxBytesError(err); ok {
				openai.WriteAPIError(w, apiErr)
				return
			}
			meta := openai.RequestMetaFromContext(r.Context())
			meta.SetModel("gpt-4o", "us.anthropic.claude-sonnet-4-20250514-v1:0")
			meta.AddUsage(120, 45, 80, 0, 15)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `{"ok":true}`)
		}),
	})

	// 1. /healthz unauthenticated
	rr := httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("expected /healthz 200, got %d", rr.Code)
	}

	// 2. /readyz 200 -> 503 when check fails
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("expected /readyz 200, got %d", rr.Code)
	}
	dbHealthy = false
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected /readyz 503, got %d", rr.Code)
	}

	// 3. Authenticated chat request records usage & logs structured JSON without secret leakage
	chatReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"prompt":"SENSITIVE_USER_PROMPT"}`))
	chatReq.Header.Set("Authorization", "Bearer sk-super-secret-token-never-log")
	chatReq.Header.Set("X-Request-Id", "req-custom-123")
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, chatReq)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected chat 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if rr.Header().Get("X-Request-Id") != "req-custom-123" {
		t.Errorf("expected echoed X-Request-Id req-custom-123, got %q", rr.Header().Get("X-Request-Id"))
	}

	// 4. Oversized body -> 413
	bigReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(strings.Repeat("x", 256)))
	bigReq.Header.Set("Authorization", "Bearer sk-super-secret-token-never-log")
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, bigReq)
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413 for oversized body, got %d", rr.Code)
	}

	// 5. Panic recovery -> 500 OpenAI JSON error
	panicReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	panicReq.Header.Set("Authorization", "Bearer sk-super-secret-token-never-log")
	panicReq.Header.Set("X-Trigger-Panic", "1")
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, panicReq)
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500 from panic recovery, got %d", rr.Code)
	}

	// 6. /metrics requires Bearer auth by default, and exports Prometheus counters
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected unauthenticated /metrics 401, got %d", rr.Code)
	}

	mReq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	mReq.Header.Set("Authorization", "Bearer sk-super-secret-token-never-log")
	rr = httptest.NewRecorder()
	gw.MainHandler.ServeHTTP(rr, mReq)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected authenticated /metrics 200, got %d", rr.Code)
	}
	metricsOut := rr.Body.String()
	for _, metricName := range []string{
		"gateway_http_requests_total",
		"gateway_http_request_duration_seconds",
		"gateway_tokens_total",
	} {
		if !strings.Contains(metricsOut, metricName) {
			t.Errorf("expected metric %q in /metrics output", metricName)
		}
	}

	// 7. Verify logs never contain Bearer secret or sensitive prompt text
	logs := logBuf.String()
	if strings.Contains(logs, "sk-super-secret-token-never-log") {
		t.Fatal("SECURITY VIOLATION: Bearer token leaked into structured logs")
	}
	if strings.Contains(logs, "SENSITIVE_USER_PROMPT") {
		t.Fatal("SECURITY VIOLATION: Raw prompt leaked into structured logs")
	}
	if !strings.Contains(logs, `"tenant":"team-sec"`) || !strings.Contains(logs, `"resolved_bedrock_model":"us.anthropic.claude-sonnet-4-20250514-v1:0"`) {
		t.Errorf("expected tenant and resolved model in structured logs, got:\n%s", logs)
	}
}
