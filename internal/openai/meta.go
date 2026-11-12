package openai

import (
	"context"
	"sync"
)

type metaContextKey struct{}

// RequestMeta accumulates request-scoped metadata for structured logging and Prometheus metrics.
// Raw prompt bodies and Bearer tokens are intentionally never stored here.
type RequestMeta struct {
	mu                   sync.Mutex
	RequestID            string
	Tenant               string
	Endpoint             string
	RequestedModel       string
	ResolvedBedrockModel string
	Stream               bool
	InputTokens          int64
	OutputTokens         int64
	CachedReadTokens     int64
	CachedWriteTokens    int64
	ReasoningTokens      int64
	BedrockErrorCode     string
}

// WithRequestMeta attaches meta to ctx.
func WithRequestMeta(ctx context.Context, meta *RequestMeta) context.Context {
	return context.WithValue(ctx, metaContextKey{}, meta)
}

// RequestMetaFromContext retrieves RequestMeta from ctx, or returns nil.
func RequestMetaFromContext(ctx context.Context) *RequestMeta {
	m, _ := ctx.Value(metaContextKey{}).(*RequestMeta)
	return m
}

// SetTenant records the authenticated tenant name.
func (m *RequestMeta) SetTenant(tenant string) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.Tenant = tenant
	m.mu.Unlock()
}

// SetModel records the requested model alias/ID and resolved Bedrock model ID.
func (m *RequestMeta) SetModel(requested, resolved string) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.RequestedModel = requested
	m.ResolvedBedrockModel = resolved
	m.mu.Unlock()
}

// SetStream marks whether the request is using SSE streaming.
func (m *RequestMeta) SetStream(stream bool) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.Stream = stream
	m.mu.Unlock()
}

// AddUsage accumulates Bedrock token counts for metrics and structured logging.
func (m *RequestMeta) AddUsage(input, output, cachedRead, cachedWrite, reasoning int64) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.InputTokens += input
	m.OutputTokens += output
	m.CachedReadTokens += cachedRead
	m.CachedWriteTokens += cachedWrite
	m.ReasoningTokens += reasoning
	m.mu.Unlock()
}

// SetBedrockError records a bounded-cardinality upstream Bedrock error code.
func (m *RequestMeta) SetBedrockError(code string) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.BedrockErrorCode = code
	m.mu.Unlock()
}

// Snapshot returns a thread-safe value copy of RequestMeta.
func (m *RequestMeta) Snapshot() RequestMeta {
	if m == nil {
		return RequestMeta{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return RequestMeta{
		RequestID:            m.RequestID,
		Tenant:               m.Tenant,
		Endpoint:             m.Endpoint,
		RequestedModel:       m.RequestedModel,
		ResolvedBedrockModel: m.ResolvedBedrockModel,
		Stream:               m.Stream,
		InputTokens:          m.InputTokens,
		OutputTokens:         m.OutputTokens,
		CachedReadTokens:     m.CachedReadTokens,
		CachedWriteTokens:    m.CachedWriteTokens,
		ReasoningTokens:      m.ReasoningTokens,
		BedrockErrorCode:     m.BedrockErrorCode,
	}
}
