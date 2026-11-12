package server

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
	"runtime/debug"
	"strings"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
)

// ReadinessCheck evaluates a subsystem dependency for /readyz.
type ReadinessCheck struct {
	Name  string
	Check func(ctx context.Context) error
}

// Dependencies holds the components wired into the HTTP gateway router.
type Dependencies struct {
	Config          *config.Config
	Authenticator   *auth.Authenticator
	Metrics         *metrics.Collector
	Logger          *slog.Logger
	ReadinessChecks []ReadinessCheck

	ChatCompletionsHandler     http.Handler
	ResponsesCreateHandler     http.Handler
	ResponsesGetHandler        http.Handler
	ResponsesDeleteHandler     http.Handler
	ResponsesInputItemsHandler http.Handler
	EmbeddingsHandler          http.Handler
	ModelsListHandler          http.Handler
	ModelsGetHandler           http.Handler
}

// NewLogger creates a structured JSON slog.Logger at the configured level.
func NewLogger(level string, w io.Writer) *slog.Logger {
	if w == nil {
		w = os.Stdout
	}
	var lvl slog.Level
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(w, &slog.HandlerOptions{Level: lvl}))
}

// GatewayServer wraps the primary HTTP server and optional isolated metrics server.
type GatewayServer struct {
	MainHandler    http.Handler
	MetricsHandler http.Handler
	MainServer     *http.Server
	MetricsServer  *http.Server
}

// New builds the HTTP muxes, middleware stacks, and http.Server instances.
func New(deps Dependencies) *GatewayServer {
	cfg := deps.Config
	if cfg == nil {
		def := config.Default()
		cfg = &def
	}
	logger := deps.Logger
	if logger == nil {
		logger = NewLogger(cfg.Server.LogLevel, os.Stdout)
	}
	mc := deps.Metrics
	if mc == nil {
		mc = metrics.New()
	}

	mux := http.NewServeMux()

	// Health & readiness probes (unauthenticated for ALB/ECS/Kubernetes/Docker healthcheck).
	liveness := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"ok"}`+"\n")
	})
	mux.Handle("GET /health", liveness)
	mux.Handle("GET /healthz", liveness)
	mux.Handle("GET /readyz", buildReadyHandler(deps.ReadinessChecks))

	onTenantAuth := func(r *http.Request, tenantName string) {
		if meta := openai.RequestMetaFromContext(r.Context()); meta != nil {
			meta.SetTenant(tenantName)
		}
	}

	// Metrics endpoint on main listener when metrics_listen_addr is empty.
	metricsHandler := mc.Handler()
	if cfg.Server.MetricsListenAddr == "" {
		var mh http.Handler = metricsHandler
		if cfg.Server.MetricsRequireAuth && deps.Authenticator != nil {
			mh = deps.Authenticator.Middleware(false, onTenantAuth)(mh)
		}
		mux.Handle("GET /metrics", mh)
	}

	wrapAPI := func(h http.Handler) http.Handler {
		if h == nil {
			h = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				openai.WriteError(w, http.StatusNotImplemented, openai.TypeServer, "not_implemented", "endpoint handler not configured")
			})
		}
		wrapped := withRequestTimeout(cfg.Server.RequestTimeout.Duration, h)
		if deps.Authenticator != nil {
			wrapped = deps.Authenticator.Middleware(true, onTenantAuth)(wrapped)
		}
		return wrapped
	}

	mux.Handle("POST /v1/chat/completions", wrapAPI(deps.ChatCompletionsHandler))
	mux.Handle("POST /v1/responses", wrapAPI(deps.ResponsesCreateHandler))
	mux.Handle("GET /v1/responses/{id}", wrapAPI(deps.ResponsesGetHandler))
	mux.Handle("DELETE /v1/responses/{id}", wrapAPI(deps.ResponsesDeleteHandler))
	mux.Handle("GET /v1/responses/{id}/input_items", wrapAPI(deps.ResponsesInputItemsHandler))
	mux.Handle("POST /v1/embeddings", wrapAPI(deps.EmbeddingsHandler))
	mux.Handle("GET /v1/models", wrapAPI(deps.ModelsListHandler))
	mux.Handle("GET /v1/models/{model...}", wrapAPI(deps.ModelsGetHandler))

	// OpenAI-formatted fallback for unknown routes or unsupported methods.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		openai.WriteError(
			w,
			http.StatusNotFound,
			openai.TypeInvalidRequest,
			"unknown_url",
			"Invalid URL ("+r.Method+" "+r.URL.Path+")",
		)
	})

	var root http.Handler = mux
	root = withMaxBodyBytes(cfg.Server.MaxRequestBodyBytes, root)
	root = withObservability(logger, mc, root)
	root = withPanicRecovery(logger, root)

	mainSrv := &http.Server{
		Addr:              cfg.Server.ListenAddr,
		Handler:           root,
		ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout.Duration,
		WriteTimeout:      cfg.Server.WriteTimeout.Duration,
	}

	var metricsSrv *http.Server
	var isolatedMetricsHandler http.Handler
	if cfg.Server.MetricsListenAddr != "" {
		isolatedMux := http.NewServeMux()
		var mh http.Handler = metricsHandler
		if cfg.Server.MetricsRequireAuth && deps.Authenticator != nil {
			mh = deps.Authenticator.Middleware(false, onTenantAuth)(mh)
		}
		isolatedMux.Handle("GET /metrics", mh)
		isolatedMux.Handle("GET /healthz", liveness)
		isolatedMetricsHandler = withPanicRecovery(logger, isolatedMux)
		metricsSrv = &http.Server{
			Addr:              cfg.Server.MetricsListenAddr,
			Handler:           isolatedMetricsHandler,
			ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout.Duration,
		}
	}

	return &GatewayServer{
		MainHandler:    root,
		MetricsHandler: isolatedMetricsHandler,
		MainServer:     mainSrv,
		MetricsServer:  metricsSrv,
	}
}

func buildReadyHandler(checks []ReadinessCheck) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		details := make(map[string]string, len(checks))
		ready := true
		for _, chk := range checks {
			if chk.Check == nil {
				continue
			}
			if err := chk.Check(ctx); err != nil {
				ready = false
				details[chk.Name] = "unavailable"
			} else {
				details[chk.Name] = "ok"
			}
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		status := "ready"
		code := http.StatusOK
		if !ready {
			status = "not_ready"
			code = http.StatusServiceUnavailable
		}
		w.WriteHeader(code)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": status,
			"checks": details,
		})
	})
}

func withRequestTimeout(timeout time.Duration, next http.Handler) http.Handler {
	if timeout <= 0 {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func withMaxBodyBytes(limit int64, next http.Handler) http.Handler {
	if limit <= 0 {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ContentLength > limit {
			openai.WriteError(
				w,
				http.StatusRequestEntityTooLarge,
				openai.TypeInvalidRequest,
				"request_too_large",
				"request body exceeds configured max_request_body_bytes limit",
			)
			return
		}
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, limit)
		}
		next.ServeHTTP(w, r)
	})
}

func withPanicRecovery(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		defer func() {
			if p := recover(); p != nil {
				logger.Error("recovered from panic in HTTP handler",
					"panic", p,
					"path", r.URL.Path,
					"stack", string(debug.Stack()),
				)
				if !rec.wroteHeader {
					openai.WriteError(
						rec,
						http.StatusInternalServerError,
						openai.TypeServer,
						"internal_error",
						"Internal server error.",
					)
				}
			}
		}()
		next.ServeHTTP(rec, r)
	})
}

func withObservability(logger *slog.Logger, mc *metrics.Collector, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		reqID := sanitizeRequestID(r.Header.Get("X-Request-Id"))
		if reqID == "" {
			reqID = openai.NewID("req_")
		}
		w.Header().Set("X-Request-Id", reqID)

		endpoint := classifyEndpoint(r.URL.Path)
		meta := &openai.RequestMeta{
			RequestID: reqID,
			Endpoint:  endpoint,
		}
		ctx := openai.WithRequestMeta(r.Context(), meta)
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		next.ServeHTTP(rec, r.WithContext(ctx))

		dur := time.Since(start)
		snap := meta.Snapshot()

		modelLabel := snap.ResolvedBedrockModel
		if modelLabel == "" {
			modelLabel = "none"
		}
		mc.ObserveRequest(snap.Tenant, modelLabel, endpoint, rec.status, dur)
		if snap.InputTokens > 0 || snap.OutputTokens > 0 || snap.CachedReadTokens > 0 || snap.CachedWriteTokens > 0 || snap.ReasoningTokens > 0 {
			mc.RecordTokens(
				snap.Tenant,
				modelLabel,
				endpoint,
				snap.InputTokens,
				snap.OutputTokens,
				snap.CachedReadTokens,
				snap.CachedWriteTokens,
				snap.ReasoningTokens,
			)
		}
		if snap.BedrockErrorCode != "" {
			mc.RecordBedrockError(snap.Tenant, modelLabel, endpoint, snap.BedrockErrorCode)
		}

		attrs := []any{
			"request_id", snap.RequestID,
			"method", r.Method,
			"endpoint", endpoint,
			"status", rec.status,
			"duration_ms", dur.Milliseconds(),
		}
		if snap.Tenant != "" {
			attrs = append(attrs, "tenant", snap.Tenant)
		}
		if snap.RequestedModel != "" {
			attrs = append(attrs, "model", snap.RequestedModel)
		}
		if snap.ResolvedBedrockModel != "" {
			attrs = append(attrs, "resolved_bedrock_model", snap.ResolvedBedrockModel)
		}
		if strings.HasPrefix(endpoint, "/v1/") {
			attrs = append(attrs, "stream", snap.Stream)
		}
		if snap.InputTokens > 0 || snap.OutputTokens > 0 {
			attrs = append(attrs,
				"input_tokens", snap.InputTokens,
				"output_tokens", snap.OutputTokens,
			)
		}
		if snap.CachedReadTokens > 0 || snap.CachedWriteTokens > 0 {
			attrs = append(attrs,
				"cached_read_tokens", snap.CachedReadTokens,
				"cached_write_tokens", snap.CachedWriteTokens,
			)
		}
		if snap.ReasoningTokens > 0 {
			attrs = append(attrs, "reasoning_tokens", snap.ReasoningTokens)
		}
		if snap.BedrockErrorCode != "" {
			attrs = append(attrs, "bedrock_error", snap.BedrockErrorCode)
		}

		if rec.status >= 500 {
			logger.Error("http_request", attrs...)
		} else if rec.status >= 400 {
			logger.Warn("http_request", attrs...)
		} else {
			logger.Info("http_request", attrs...)
		}
	})
}

func classifyEndpoint(p string) string {
	switch {
	case p == "/health" || p == "/healthz":
		return "/healthz"
	case p == "/readyz":
		return "/readyz"
	case p == "/metrics":
		return "/metrics"
	case p == "/v1/chat/completions":
		return "/v1/chat/completions"
	case p == "/v1/embeddings":
		return "/v1/embeddings"
	case p == "/v1/models":
		return "/v1/models"
	case strings.HasPrefix(p, "/v1/models/"):
		return "/v1/models/{model}"
	case p == "/v1/responses":
		return "/v1/responses"
	case strings.HasPrefix(p, "/v1/responses/"):
		if strings.HasSuffix(p, "/input_items") {
			return "/v1/responses/{id}/input_items"
		}
		return "/v1/responses/{id}"
	default:
		return "other"
	}
}

func sanitizeRequestID(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" || len(raw) > 128 {
		return ""
	}
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' {
			continue
		}
		return ""
	}
	return raw
}

type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (r *statusRecorder) WriteHeader(code int) {
	if !r.wroteHeader {
		r.status = code
		r.wroteHeader = true
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.wroteHeader = true
	}
	return r.ResponseWriter.Write(b)
}

func (r *statusRecorder) Flush() {
	if fl, ok := r.ResponseWriter.(http.Flusher); ok {
		fl.Flush()
	}
}

func (r *statusRecorder) Unwrap() http.ResponseWriter {
	return r.ResponseWriter
}
