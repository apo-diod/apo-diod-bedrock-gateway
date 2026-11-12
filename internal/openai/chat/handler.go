package chat

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/bedrockconv"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
)

// CompletionRequest models OpenAI POST /v1/chat/completions requests.
type CompletionRequest struct {
	Model               string              `json:"model"`
	Messages            []RequestMessage    `json:"messages"`
	Tools               []RequestTool       `json:"tools,omitempty"`
	ToolChoice          json.RawMessage     `json:"tool_choice,omitempty"`
	ResponseFormat      *RequestRespFormat  `json:"response_format,omitempty"`
	MaxTokens           *int                `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                `json:"max_completion_tokens,omitempty"`
	Temperature         *float32            `json:"temperature,omitempty"`
	TopP                *float32            `json:"top_p,omitempty"`
	Stop                json.RawMessage     `json:"stop,omitempty"`
	N                   *int                `json:"n,omitempty"`
	Stream              bool                `json:"stream,omitempty"`
	StreamOptions       *StreamOptions      `json:"stream_options,omitempty"`
	ReasoningEffort     string              `json:"reasoning_effort,omitempty"`
	Thinking            *ThinkingParam      `json:"thinking,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ThinkingParam struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

type CacheControl struct {
	Type string `json:"type"`
}

type RequestMessage struct {
	Role             string            `json:"role"`
	Content          json.RawMessage   `json:"content,omitempty"`
	ToolCalls        []MessageToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	Signature        string            `json:"signature,omitempty"`
	CacheControl     *CacheControl     `json:"cache_control,omitempty"`
}

type MessageToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function MessageToolFunction `json:"function"`
}

type MessageToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type RequestTool struct {
	Type         string          `json:"type"`
	Function     ToolFunctionDef `json:"function"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

type ToolFunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type RequestRespFormat struct {
	Type       string             `json:"type"`
	JSONSchema *RequestJSONSchema `json:"json_schema,omitempty"`
}

type RequestJSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
}

// CompletionResponse is the non-streaming OpenAI chat.completion object.
type CompletionResponse struct {
	ID                string             `json:"id"`
	Object            string             `json:"object"`
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	SystemFingerprint string             `json:"system_fingerprint"`
	Choices           []CompletionChoice `json:"choices"`
	Usage             CompletionUsage    `json:"usage"`
}

type CompletionChoice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type ResponseMessage struct {
	Role             string            `json:"role"`
	Content          *string           `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []MessageToolCall `json:"tool_calls,omitempty"`
}

type CompletionUsage struct {
	PromptTokens        int64               `json:"prompt_tokens"`
	CompletionTokens    int64               `json:"completion_tokens"`
	TotalTokens         int64               `json:"total_tokens"`
	PromptTokensDetails PromptTokensDetails `json:"prompt_tokens_details"`
}

type PromptTokensDetails struct {
	CachedTokens int64 `json:"cached_tokens"`
}

// CompletionChunk is an OpenAI SSE streaming chat.completion.chunk frame.
type CompletionChunk struct {
	ID                string           `json:"id"`
	Object            string           `json:"object"`
	Created           int64            `json:"created"`
	Model             string           `json:"model"`
	SystemFingerprint string           `json:"system_fingerprint"`
	Choices           []ChunkChoice    `json:"choices"`
	Usage             *CompletionUsage `json:"usage,omitempty"`
}

type ChunkChoice struct {
	Index        int        `json:"index"`
	Delta        ChunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

type ChunkDelta struct {
	Role             string          `json:"role,omitempty"`
	Content          *string         `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	ToolCalls        []ChunkToolCall `json:"tool_calls,omitempty"`
}

type ChunkToolCall struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function ChunkToolFunction `json:"function"`
}

type ChunkToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// Handler serves POST /v1/chat/completions.
type Handler struct {
	registry        *models.Registry
	translator      *bedrockconv.Translator
	metrics         *metrics.Collector
	autoPromptCache bool
}

// NewHandler creates a new Chat Completions HTTP handler.
func NewHandler(registry *models.Registry, translator *bedrockconv.Translator, mc *metrics.Collector, autoPromptCache bool) *Handler {
	return &Handler{
		registry:        registry,
		translator:      translator,
		metrics:         mc,
		autoPromptCache: autoPromptCache,
	}
}

// ServeHTTP handles POST /v1/chat/completions.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		if apiErr, ok := openai.CheckMaxBytesError(err); ok {
			openai.WriteAPIError(w, apiErr)
			return
		}
		openai.WriteError(w, http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_body", "Failed to read request body.")
		return
	}

	var req CompletionRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		openai.WriteError(w, http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_json", fmt.Sprintf("Invalid JSON payload: %v", err))
		return
	}

	if req.N != nil && *req.N > 1 {
		openai.WriteAPIError(w, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"unsupported_parameter",
			"n",
			"Only n=1 is supported by AWS Bedrock Converse.",
		))
		return
	}

	tenant, _ := auth.TenantFromContext(r.Context())
	requestedModel, bedrockModel, apiErr := h.registry.ResolveChatModel(tenant, req.Model)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	meta := openai.RequestMetaFromContext(r.Context())
	if meta != nil {
		meta.SetModel(requestedModel, bedrockModel)
		meta.SetStream(req.Stream)
	}

	canonReq, apiErr := h.toCanonicalRequest(req, bedrockModel)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	if req.Stream {
		h.serveStream(w, r, tenant, meta, requestedModel, req, canonReq)
		return
	}

	resp, apiErr := h.translator.Invoke(r.Context(), canonReq)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	if meta != nil {
		meta.AddUsage(
			resp.Usage.InputTokens,
			resp.Usage.OutputTokens,
			resp.Usage.CachedReadInputTokens,
			resp.Usage.CachedWriteInputTokens,
			resp.Usage.ReasoningTokens,
		)
	}

	var toolCalls []MessageToolCall
	for _, tc := range resp.ToolCalls {
		toolCalls = append(toolCalls, MessageToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: MessageToolFunction{
				Name:      tc.Name,
				Arguments: tc.Arguments,
			},
		})
	}

	var contentPtr *string
	if resp.ContentText != "" || len(toolCalls) == 0 {
		txt := resp.ContentText
		contentPtr = &txt
	}

	out := CompletionResponse{
		ID:                openai.NewID("chatcmpl-"),
		Object:            "chat.completion",
		Created:           time.Now().Unix(),
		Model:             requestedModel,
		SystemFingerprint: "fp_bedrock",
		Choices: []CompletionChoice{
			{
				Index: 0,
				Message: ResponseMessage{
					Role:             "assistant",
					Content:          contentPtr,
					ReasoningContent: resp.ReasoningText,
					ToolCalls:        toolCalls,
				},
				FinishReason: resp.FinishReason,
			},
		},
		Usage: formatChatUsage(resp.Usage),
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(out)
}

func (h *Handler) serveStream(
	w http.ResponseWriter,
	r *http.Request,
	tenant *auth.Tenant,
	meta *openai.RequestMeta,
	requestedModel string,
	req CompletionRequest,
	canonReq bedrockconv.Request,
) {
	flusher, canFlush := w.(http.Flusher)
	if !canFlush {
		openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "streaming_unsupported", "ResponseWriter does not support streaming flush.")
		return
	}

	completionID := openai.NewID("chatcmpl-")
	created := time.Now().Unix()
	headersSent := false
	roleSent := false
	var streamUsage *bedrockconv.Usage

	tenantName := ""
	if tenant != nil {
		tenantName = tenant.Name
	}
	var releaseStreamGauge func()
	defer func() {
		if releaseStreamGauge != nil {
			releaseStreamGauge()
		}
	}()

	ensureHeaders := func() {
		if headersSent {
			return
		}
		headersSent = true
		releaseStreamGauge = h.metrics.TrackStream(tenantName, "/v1/chat/completions")
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-transform")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
	}

	writeChunk := func(chunk CompletionChunk) error {
		ensureHeaders()
		b, err := json.Marshal(chunk)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	ensureRoleChunk := func() error {
		if roleSent {
			return nil
		}
		roleSent = true
		empty := ""
		return writeChunk(CompletionChunk{
			ID:                completionID,
			Object:            "chat.completion.chunk",
			Created:           created,
			Model:             requestedModel,
			SystemFingerprint: "fp_bedrock",
			Choices: []ChunkChoice{
				{
					Index: 0,
					Delta: ChunkDelta{
						Role:    "assistant",
						Content: &empty,
					},
				},
			},
		})
	}

	summary, apiErr := h.translator.InvokeStream(r.Context(), canonReq, func(ev bedrockconv.StreamEvent) error {
		switch ev.Kind {
		case bedrockconv.EventMessageStart:
			return ensureRoleChunk()

		case bedrockconv.EventReasoningDelta:
			if ev.ReasoningText == "" {
				return nil
			}
			if err := ensureRoleChunk(); err != nil {
				return err
			}
			return writeChunk(CompletionChunk{
				ID:                completionID,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             requestedModel,
				SystemFingerprint: "fp_bedrock",
				Choices: []ChunkChoice{
					{
						Index: 0,
						Delta: ChunkDelta{
							ReasoningContent: ev.ReasoningText,
						},
					},
				},
			})

		case bedrockconv.EventTextDelta:
			if err := ensureRoleChunk(); err != nil {
				return err
			}
			txt := ev.Text
			return writeChunk(CompletionChunk{
				ID:                completionID,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             requestedModel,
				SystemFingerprint: "fp_bedrock",
				Choices: []ChunkChoice{
					{
						Index: 0,
						Delta: ChunkDelta{
							Content: &txt,
						},
					},
				},
			})

		case bedrockconv.EventToolCallStart:
			if err := ensureRoleChunk(); err != nil {
				return err
			}
			return writeChunk(CompletionChunk{
				ID:                completionID,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             requestedModel,
				SystemFingerprint: "fp_bedrock",
				Choices: []ChunkChoice{
					{
						Index: 0,
						Delta: ChunkDelta{
							ToolCalls: []ChunkToolCall{
								{
									Index: ev.ToolIndex,
									ID:    ev.ToolCallID,
									Type:  "function",
									Function: ChunkToolFunction{
										Name:      ev.ToolName,
										Arguments: "",
									},
								},
							},
						},
					},
				},
			})

		case bedrockconv.EventToolCallDelta:
			if err := ensureRoleChunk(); err != nil {
				return err
			}
			return writeChunk(CompletionChunk{
				ID:                completionID,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             requestedModel,
				SystemFingerprint: "fp_bedrock",
				Choices: []ChunkChoice{
					{
						Index: 0,
						Delta: ChunkDelta{
							ToolCalls: []ChunkToolCall{
								{
									Index: ev.ToolIndex,
									Function: ChunkToolFunction{
										Arguments: ev.ArgumentsDelta,
									},
								},
							},
						},
					},
				},
			})

		case bedrockconv.EventMessageStop:
			if err := ensureRoleChunk(); err != nil {
				return err
			}
			fr := ev.FinishReason
			return writeChunk(CompletionChunk{
				ID:                completionID,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             requestedModel,
				SystemFingerprint: "fp_bedrock",
				Choices: []ChunkChoice{
					{
						Index:        0,
						Delta:        ChunkDelta{},
						FinishReason: &fr,
					},
				},
			})

		case bedrockconv.EventUsage:
			u := ev.Usage
			streamUsage = &u
			return nil
		}
		return nil
	})

	if apiErr != nil {
		if !headersSent {
			openai.WriteAPIError(w, apiErr)
			return
		}
		// Mid-stream failure after HTTP 200 headers were already sent: emit OpenAI SSE error frame.
		var codePtr *string
		if apiErr.Code != "" {
			c := apiErr.Code
			codePtr = &c
		}
		errFrame, _ := json.Marshal(openai.ErrorEnvelope{
			Error: openai.ErrorBody{
				Message: apiErr.Message,
				Type:    apiErr.Type,
				Param:   apiErr.Param,
				Code:    codePtr,
			},
		})
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", errFrame)
		flusher.Flush()
		return
	}

	if summary != nil && meta != nil {
		meta.AddUsage(
			summary.Usage.InputTokens,
			summary.Usage.OutputTokens,
			summary.Usage.CachedReadInputTokens,
			summary.Usage.CachedWriteInputTokens,
			summary.Usage.ReasoningTokens,
		)
	}

	// Ensure headers are sent even if upstream returned an empty stream without errors.
	if err := ensureRoleChunk(); err != nil {
		return
	}

	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		u := bedrockconv.Usage{}
		if streamUsage != nil {
			u = *streamUsage
		} else if summary != nil {
			u = summary.Usage
		}
		cu := formatChatUsage(u)
		_ = writeChunk(CompletionChunk{
			ID:                completionID,
			Object:            "chat.completion.chunk",
			Created:           created,
			Model:             requestedModel,
			SystemFingerprint: "fp_bedrock",
			Choices:           []ChunkChoice{},
			Usage:             &cu,
		})
	}

	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (h *Handler) toCanonicalRequest(req CompletionRequest, bedrockModel string) (bedrockconv.Request, *openai.APIError) {
	if len(req.Messages) == 0 {
		return bedrockconv.Request{}, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"missing_messages",
			"messages",
			"messages must contain at least one message.",
		)
	}

	msgs := make([]bedrockconv.Message, 0, len(req.Messages))
	for i, rm := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(rm.Role))
		msgCache := rm.CacheControl != nil && rm.CacheControl.Type != ""

		switch role {
		case bedrockconv.RoleSystem, bedrockconv.RoleDeveloper, bedrockconv.RoleUser, bedrockconv.RoleAssistant:
			parts, apiErr := parseMessageContentParts(rm.Content)
			if apiErr != nil {
				return bedrockconv.Request{}, apiErr
			}
			if role == bedrockconv.RoleAssistant {
				if rm.ReasoningContent != "" {
					parts = append([]bedrockconv.ContentPart{{
						Type:          bedrockconv.PartReasoning,
						ReasoningText: rm.ReasoningContent,
						Signature:     rm.Signature,
					}}, parts...)
				}
				for _, tc := range rm.ToolCalls {
					parts = append(parts, bedrockconv.ContentPart{
						Type:         bedrockconv.PartToolUse,
						ToolCallID:   tc.ID,
						ToolName:     tc.Function.Name,
						ToolArgsJSON: tc.Function.Arguments,
					})
				}
			}
			msgs = append(msgs, bedrockconv.Message{
				Role:         role,
				Parts:        parts,
				CacheControl: msgCache,
			})

		case bedrockconv.RoleTool:
			parts, apiErr := parseMessageContentParts(rm.Content)
			if apiErr != nil {
				return bedrockconv.Request{}, apiErr
			}
			var sb strings.Builder
			for _, p := range parts {
				if p.Type == bedrockconv.PartText {
					sb.WriteString(p.Text)
				}
			}
			msgs = append(msgs, bedrockconv.Message{
				Role: bedrockconv.RoleTool,
				Parts: []bedrockconv.ContentPart{
					{
						Type:           bedrockconv.PartToolResult,
						ToolCallID:     rm.ToolCallID,
						ToolResultText: sb.String(),
						CacheControl:   msgCache,
					},
				},
			})

		default:
			return bedrockconv.Request{}, openai.NewParamError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"invalid_role",
				fmt.Sprintf("messages[%d].role", i),
				fmt.Sprintf("Unsupported role %q.", rm.Role),
			)
		}
	}

	tools := make([]bedrockconv.ToolDef, 0, len(req.Tools))
	for _, rt := range req.Tools {
		if rt.Type != "" && rt.Type != "function" {
			return bedrockconv.Request{}, openai.NewParamError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"unsupported_tool_type",
				"tools",
				fmt.Sprintf("Unsupported tool type %q; only 'function' tools are supported.", rt.Type),
			)
		}
		tools = append(tools, bedrockconv.ToolDef{
			Name:         rt.Function.Name,
			Description:  rt.Function.Description,
			Parameters:   rt.Function.Parameters,
			CacheControl: rt.CacheControl != nil && rt.CacheControl.Type != "",
		})
	}

	tc, apiErr := parseToolChoice(req.ToolChoice)
	if apiErr != nil {
		return bedrockconv.Request{}, apiErr
	}

	stops, apiErr := parseStopSequences(req.Stop)
	if apiErr != nil {
		return bedrockconv.Request{}, apiErr
	}

	var rf bedrockconv.ResponseFormat
	if req.ResponseFormat != nil {
		rf.Type = req.ResponseFormat.Type
		if req.ResponseFormat.JSONSchema != nil {
			rf.Name = req.ResponseFormat.JSONSchema.Name
			rf.Description = req.ResponseFormat.JSONSchema.Description
			rf.Schema = req.ResponseFormat.JSONSchema.Schema
		}
	}

	maxTok := req.MaxTokens
	if req.MaxCompletionTokens != nil {
		maxTok = req.MaxCompletionTokens
	}

	var reasoning *bedrockconv.ReasoningConfig
	if req.ReasoningEffort != "" && !strings.EqualFold(req.ReasoningEffort, "none") {
		reasoning = &bedrockconv.ReasoningConfig{Effort: req.ReasoningEffort}
	}
	if req.Thinking != nil && strings.EqualFold(req.Thinking.Type, "enabled") {
		if reasoning == nil {
			reasoning = &bedrockconv.ReasoningConfig{}
		}
		reasoning.BudgetTokens = req.Thinking.BudgetTokens
	}

	return bedrockconv.Request{
		BedrockModelID:  bedrockModel,
		Messages:        msgs,
		Tools:           tools,
		ToolChoice:      tc,
		ResponseFormat:  rf,
		Reasoning:       reasoning,
		MaxTokens:       maxTok,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stop:            stops,
		AutoPromptCache: h.autoPromptCache,
	}, nil
}

type rawContentItem struct {
	Type         string          `json:"type"`
	Text         string          `json:"text"`
	ImageURL     json.RawMessage `json:"image_url"`
	File         *rawFileInput   `json:"file"`
	CacheControl *CacheControl   `json:"cache_control"`
}

type rawFileInput struct {
	Filename string `json:"filename"`
	FileData string `json:"file_data"`
}

func parseMessageContentParts(raw json.RawMessage) ([]bedrockconv.ContentPart, *openai.APIError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_content", err.Error())
		}
		if s == "" {
			return nil, nil
		}
		return []bedrockconv.ContentPart{{Type: bedrockconv.PartText, Text: s}}, nil
	}
	if trimmed[0] != '[' {
		return nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_content", "message content must be a string, array, or null")
	}

	var items []rawContentItem
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_content", fmt.Sprintf("invalid content array: %v", err))
	}

	parts := make([]bedrockconv.ContentPart, 0, len(items))
	for _, it := range items {
		cc := it.CacheControl != nil && it.CacheControl.Type != ""
		switch it.Type {
		case "text", "input_text", "output_text":
			parts = append(parts, bedrockconv.ContentPart{
				Type:         bedrockconv.PartText,
				Text:         it.Text,
				CacheControl: cc,
			})
		case "image_url", "input_image":
			imgURL, err := extractURLField(it.ImageURL)
			if err != nil {
				return nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_image_url", err.Error())
			}
			parts = append(parts, bedrockconv.ContentPart{
				Type:         bedrockconv.PartImage,
				RemoteURL:    imgURL,
				CacheControl: cc,
			})
		case "file", "input_file", "document":
			if it.File == nil || strings.TrimSpace(it.File.FileData) == "" {
				return nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_file", "file content block requires file.file_data")
			}
			part, apiErr := parseFileInput(it.File.Filename, it.File.FileData, cc)
			if apiErr != nil {
				return nil, apiErr
			}
			parts = append(parts, part)
		default:
			return nil, openai.NewError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"unsupported_content_type",
				fmt.Sprintf("Unsupported message content part type %q.", it.Type),
			)
		}
	}
	return parts, nil
}

func parseFileInput(filename, fileData string, cacheControl bool) (bedrockconv.ContentPart, *openai.APIError) {
	fd := strings.TrimSpace(fileData)
	if strings.HasPrefix(fd, "data:") || strings.HasPrefix(fd, "http://") || strings.HasPrefix(fd, "https://") {
		return bedrockconv.ContentPart{
			Type:         bedrockconv.PartDocument,
			Name:         filename,
			RemoteURL:    fd,
			CacheControl: cacheControl,
		}, nil
	}
	// Raw base64 string without data: prefix (used by some OpenAI SDK helpers)
	decoded, err := base64.StdEncoding.DecodeString(fd)
	if err != nil {
		return bedrockconv.ContentPart{}, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_file_data", "file.file_data must be a data: URI or valid base64")
	}
	mt := ""
	if strings.HasPrefix(string(decoded), "%PDF-") {
		mt = "application/pdf"
	}
	return bedrockconv.ContentPart{
		Type:         bedrockconv.PartDocument,
		Name:         filename,
		MediaType:    mt,
		Data:         decoded,
		CacheControl: cacheControl,
	}, nil
}

func extractURLField(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "", fmt.Errorf("image_url is empty")
	}
	if trimmed[0] == '"' {
		var u string
		if err := json.Unmarshal(trimmed, &u); err != nil {
			return "", err
		}
		return u, nil
	}
	var obj struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal(trimmed, &obj); err != nil {
		return "", err
	}
	if strings.TrimSpace(obj.URL) == "" {
		return "", fmt.Errorf("image_url.url must not be empty")
	}
	return obj.URL, nil
}

func parseToolChoice(raw json.RawMessage) (bedrockconv.ToolChoice, *openai.APIError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return bedrockconv.ToolChoice{}, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return bedrockconv.ToolChoice{}, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_tool_choice", "tool_choice", err.Error())
		}
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "auto", "none", "required":
			return bedrockconv.ToolChoice{Mode: strings.ToLower(strings.TrimSpace(s))}, nil
		default:
			return bedrockconv.ToolChoice{}, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_tool_choice", "tool_choice", fmt.Sprintf("Unsupported tool_choice %q.", s))
		}
	}
	var obj struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(trimmed, &obj); err != nil {
		return bedrockconv.ToolChoice{}, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_tool_choice", "tool_choice", err.Error())
	}
	fnName := strings.TrimSpace(obj.Function.Name)
	if fnName == "" {
		fnName = strings.TrimSpace(obj.Name)
	}
	if fnName == "" {
		return bedrockconv.ToolChoice{}, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_tool_choice", "tool_choice", "named tool_choice requires function.name")
	}
	return bedrockconv.ToolChoice{Mode: "tool", ToolName: fnName}, nil
}

func parseStopSequences(raw json.RawMessage) ([]string, *openai.APIError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_stop", "stop", err.Error())
		}
		if s == "" {
			return nil, nil
		}
		return []string{s}, nil
	}
	var list []string
	if err := json.Unmarshal(trimmed, &list); err != nil {
		return nil, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_stop", "stop", "stop must be a string or array of strings")
	}
	return list, nil
}

func formatChatUsage(u bedrockconv.Usage) CompletionUsage {
	prompt := u.InputTokens + u.CachedReadInputTokens + u.CachedWriteInputTokens
	total := prompt + u.OutputTokens
	return CompletionUsage{
		PromptTokens:     prompt,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      total,
		PromptTokensDetails: PromptTokensDetails{
			CachedTokens: u.CachedReadInputTokens,
		},
	}
}
