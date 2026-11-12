package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Token direction labels for gateway_tokens_total.
const (
	DirectionInput       = "input"
	DirectionOutput      = "output"
	DirectionCachedRead  = "cached_read"
	DirectionCachedWrite = "cached_write"
	DirectionReasoning   = "reasoning"
)

// Collector owns an isolated Prometheus registry and all gateway metric instruments.
type Collector struct {
	registry        *prometheus.Registry
	requestsTotal   *prometheus.CounterVec
	requestDuration *prometheus.HistogramVec
	activeStreams   *prometheus.GaugeVec
	tokensTotal     *prometheus.CounterVec
	bedrockErrors   *prometheus.CounterVec
}

// New creates and registers all gateway metrics in a dedicated prometheus.Registry.
func New() *Collector {
	reg := prometheus.NewRegistry()
	reg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)

	c := &Collector{
		registry: reg,
		requestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_http_requests_total",
				Help: "Total HTTP requests processed by the Bedrock OpenAI gateway.",
			},
			[]string{"tenant", "model", "endpoint", "status"},
		),
		requestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_http_request_duration_seconds",
				Help:    "HTTP request duration in seconds.",
				Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300},
			},
			[]string{"tenant", "model", "endpoint", "status"},
		),
		activeStreams: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gateway_active_streams",
				Help: "Number of active SSE streaming responses currently open.",
			},
			[]string{"tenant", "endpoint"},
		),
		tokensTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_tokens_total",
				Help: "Total Bedrock tokens processed by direction.",
			},
			[]string{"tenant", "model", "endpoint", "direction"},
		),
		bedrockErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_bedrock_errors_total",
				Help: "Total upstream AWS Bedrock errors encountered.",
			},
			[]string{"tenant", "model", "endpoint", "code"},
		),
	}

	reg.MustRegister(
		c.requestsTotal,
		c.requestDuration,
		c.activeStreams,
		c.tokensTotal,
		c.bedrockErrors,
	)
	return c
}

// Registry returns the underlying Prometheus registry (useful for assertions in tests).
func (c *Collector) Registry() *prometheus.Registry {
	return c.registry
}

// Handler returns an http.Handler serving the Prometheus exposition format.
func (c *Collector) Handler() http.Handler {
	return promhttp.HandlerFor(c.registry, promhttp.HandlerOpts{})
}

// ObserveRequest records completed HTTP request count and latency with bounded labels.
func (c *Collector) ObserveRequest(tenant, model, endpoint string, status int, dur time.Duration) {
	if c == nil {
		return
	}
	tenant = sanitizeLabel(tenant, "unauthenticated")
	model = sanitizeLabel(model, "none")
	endpoint = sanitizeLabel(endpoint, "unknown")
	st := strconv.Itoa(status)
	c.requestsTotal.WithLabelValues(tenant, model, endpoint, st).Inc()
	c.requestDuration.WithLabelValues(tenant, model, endpoint, st).Observe(dur.Seconds())
}

// IncActiveStream increments the active SSE gauge and returns an idempotent decrement func.
func (c *Collector) TrackStream(tenant, endpoint string) func() {
	if c == nil {
		return func() {}
	}
	tenant = sanitizeLabel(tenant, "unauthenticated")
	endpoint = sanitizeLabel(endpoint, "unknown")
	g := c.activeStreams.WithLabelValues(tenant, endpoint)
	g.Inc()
	done := false
	return func() {
		if !done {
			done = true
			g.Dec()
		}
	}
}

// RecordTokens adds token usage counters across all positive directions.
func (c *Collector) RecordTokens(tenant, model, endpoint string, input, output, cachedRead, cachedWrite, reasoning int64) {
	if c == nil {
		return
	}
	tenant = sanitizeLabel(tenant, "unauthenticated")
	model = sanitizeLabel(model, "none")
	endpoint = sanitizeLabel(endpoint, "unknown")

	if input > 0 {
		c.tokensTotal.WithLabelValues(tenant, model, endpoint, DirectionInput).Add(float64(input))
	}
	if output > 0 {
		c.tokensTotal.WithLabelValues(tenant, model, endpoint, DirectionOutput).Add(float64(output))
	}
	if cachedRead > 0 {
		c.tokensTotal.WithLabelValues(tenant, model, endpoint, DirectionCachedRead).Add(float64(cachedRead))
	}
	if cachedWrite > 0 {
		c.tokensTotal.WithLabelValues(tenant, model, endpoint, DirectionCachedWrite).Add(float64(cachedWrite))
	}
	if reasoning > 0 {
		c.tokensTotal.WithLabelValues(tenant, model, endpoint, DirectionReasoning).Add(float64(reasoning))
	}
}

// RecordBedrockError increments gateway_bedrock_errors_total.
func (c *Collector) RecordBedrockError(tenant, model, endpoint, code string) {
	if c == nil {
		return
	}
	tenant = sanitizeLabel(tenant, "unauthenticated")
	model = sanitizeLabel(model, "none")
	endpoint = sanitizeLabel(endpoint, "unknown")
	code = sanitizeLabel(code, "unknown")
	c.bedrockErrors.WithLabelValues(tenant, model, endpoint, code).Inc()
}

func sanitizeLabel(v, fallback string) string {
	if v == "" || len(v) > 128 {
		return fallback
	}
	return v
}
