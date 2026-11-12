package bedrockconv

import (
	"context"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/awsclient"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Canonical roles.
const (
	RoleSystem    = "system"
	RoleDeveloper = "developer"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Canonical content part types.
const (
	PartText       = "text"
	PartImage      = "image"
	PartDocument   = "document"
	PartToolUse    = "tool_use"
	PartToolResult = "tool_result"
	PartReasoning  = "reasoning"
)

// ContentPart is a JSON-serializable canonical content element shared by
// /v1/chat/completions, /v1/responses, and SQLite conversation history storage.
type ContentPart struct {
	Type string `json:"type"`

	// Text content
	Text string `json:"text,omitempty"`

	// Binary or remote media (image / document)
	MediaType string `json:"media_type,omitempty"`
	Data      []byte `json:"data,omitempty"`
	RemoteURL string `json:"remote_url,omitempty"`
	Name      string `json:"name,omitempty"`

	// Assistant tool invocation
	ToolCallID   string `json:"tool_call_id,omitempty"`
	ToolName     string `json:"tool_name,omitempty"`
	ToolArgsJSON string `json:"tool_args_json,omitempty"`

	// User/tool output
	ToolResultText string `json:"tool_result_text,omitempty"`
	IsError        bool   `json:"is_error,omitempty"`

	// Extended thinking / reasoning block (lossless signature + redacted bytes)
	ReasoningText   string `json:"reasoning_text,omitempty"`
	Signature       string `json:"signature,omitempty"`
	RedactedContent []byte `json:"redacted_content,omitempty"`

	// Prompt caching marker
	CacheControl bool `json:"cache_control,omitempty"`
}

// Message is a canonical conversation turn.
type Message struct {
	Role         string        `json:"role"`
	Parts        []ContentPart `json:"parts"`
	CacheControl bool          `json:"cache_control,omitempty"`
}

// ToolDef represents a function tool definition.
type ToolDef struct {
	Name         string         `json:"name"`
	Description  string         `json:"description,omitempty"`
	Parameters   map[string]any `json:"parameters,omitempty"`
	CacheControl bool           `json:"cache_control,omitempty"`
}

// ToolChoice specifies tool selection behavior ("", "auto", "none", "required", "tool").
type ToolChoice struct {
	Mode     string `json:"mode"`
	ToolName string `json:"tool_name,omitempty"`
}

// ResponseFormat specifies text, json_object, or json_schema output formatting.
type ResponseFormat struct {
	Type        string         `json:"type"` // "", "text", "json_object", "json_schema"
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
}

// ReasoningConfig specifies extended thinking / reasoning effort and optional token budget.
type ReasoningConfig struct {
	Effort       string `json:"effort,omitempty"` // "low", "medium", "high"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// Request is the canonical model invocation request translated into Bedrock Converse / ConverseStream.
type Request struct {
	BedrockModelID  string
	Messages        []Message
	Tools           []ToolDef
	ToolChoice      ToolChoice
	ResponseFormat  ResponseFormat
	Reasoning       *ReasoningConfig
	MaxTokens       *int
	Temperature     *float32
	TopP            *float32
	Stop            []string
	AutoPromptCache bool
}

// ToolCallOutput represents a completed assistant tool call.
type ToolCallOutput struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Usage holds normalized token accounting across Bedrock Converse responses.
type Usage struct {
	InputTokens            int64 `json:"input_tokens"`
	OutputTokens           int64 `json:"output_tokens"`
	TotalTokens            int64 `json:"total_tokens"`
	CachedReadInputTokens  int64 `json:"cached_read_input_tokens"`
	CachedWriteInputTokens int64 `json:"cached_write_input_tokens"`
	ReasoningTokens        int64 `json:"reasoning_tokens"`
}

// Response is the canonical non-streaming result from Bedrock Converse.
type Response struct {
	AssistantMessage  Message
	ContentText       string
	ReasoningText     string
	ToolCalls         []ToolCallOutput
	FinishReason      string // "stop", "length", "tool_calls", "content_filter"
	BedrockStopReason string
	Usage             Usage
}

// StreamEventKind identifies normalized streaming events from ConverseStream.
type StreamEventKind string

const (
	EventMessageStart   StreamEventKind = "message_start"
	EventTextDelta      StreamEventKind = "text_delta"
	EventReasoningDelta StreamEventKind = "reasoning_delta"
	EventToolCallStart  StreamEventKind = "tool_call_start"
	EventToolCallDelta  StreamEventKind = "tool_call_delta"
	EventToolCallDone   StreamEventKind = "tool_call_done"
	EventMessageStop    StreamEventKind = "message_stop"
	EventUsage          StreamEventKind = "usage"
)

// StreamEvent represents one normalized event emitted during streaming inference.
type StreamEvent struct {
	Kind            StreamEventKind
	Text            string
	ReasoningText   string
	SignatureDelta  string
	RedactedContent []byte
	ToolIndex       int
	ToolCallID      string
	ToolName        string
	ArgumentsDelta  string
	FinishReason    string
	Usage           Usage
}

// StreamSummary captures the full reconstructed assistant turn once a stream completes.
type StreamSummary struct {
	AssistantMessage Message
	ContentText      string
	ReasoningText    string
	ToolCalls        []ToolCallOutput
	FinishReason     string
	Usage            Usage
}

// RuntimeClient abstracts Bedrock Converse and ConverseStream so unit tests can mock event streams cleanly.
type RuntimeClient interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(ctx context.Context, params *bedrockruntime.ConverseStreamInput) (bedrockruntime.ConverseStreamOutputReader, error)
}

type sdkRuntimeAdapter struct {
	api awsclient.BedrockRuntimeAPI
}

// NewSDKRuntimeClient wraps an awsclient.BedrockRuntimeAPI into a RuntimeClient.
func NewSDKRuntimeClient(api awsclient.BedrockRuntimeAPI) RuntimeClient {
	return &sdkRuntimeAdapter{api: api}
}

func (a *sdkRuntimeAdapter) Converse(ctx context.Context, params *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	return a.api.Converse(ctx, params)
}

func (a *sdkRuntimeAdapter) ConverseStream(ctx context.Context, params *bedrockruntime.ConverseStreamInput) (bedrockruntime.ConverseStreamOutputReader, error) {
	out, err := a.api.ConverseStream(ctx, params)
	if err != nil {
		return nil, err
	}
	return out.GetStream(), nil
}
