package responses

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
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
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/store"
)

// CreateRequest models POST /v1/responses request payloads.
type CreateRequest struct {
	Model              string             `json:"model"`
	Input              json.RawMessage    `json:"input"`
	Instructions       string             `json:"instructions,omitempty"`
	PreviousResponseID string             `json:"previous_response_id,omitempty"`
	Tools              []ResponseTool     `json:"tools,omitempty"`
	ToolChoice         json.RawMessage    `json:"tool_choice,omitempty"`
	Text               *ResponseTextParam `json:"text,omitempty"`
	Reasoning          *ReasoningParam    `json:"reasoning,omitempty"`
	MaxOutputTokens    *int               `json:"max_output_tokens,omitempty"`
	Temperature        *float32           `json:"temperature,omitempty"`
	TopP               *float32           `json:"top_p,omitempty"`
	Store              *bool              `json:"store,omitempty"`
	Stream             bool               `json:"stream,omitempty"`
	Metadata           map[string]string  `json:"metadata,omitempty"`
}

type ResponseTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	// Fallback nested function definition for callers migrating from Chat Completions
	Function *struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	} `json:"function,omitempty"`
}

type ResponseTextParam struct {
	Format *ResponseTextFormat `json:"format,omitempty"`
}

type ResponseTextFormat struct {
	Type        string         `json:"type"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
}

type ReasoningParam struct {
	Effort       string `json:"effort,omitempty"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// ResponseObject is the OpenAI Responses API resource returned by POST/GET /v1/responses.
type ResponseObject struct {
	ID                 string             `json:"id"`
	Object             string             `json:"object"`
	CreatedAt          int64              `json:"created_at"`
	Status             string             `json:"status"`
	Error              *ResponseErrorObj  `json:"error"`
	IncompleteDetails  *IncompleteDetails `json:"incomplete_details"`
	Instructions       *string            `json:"instructions"`
	MaxOutputTokens    *int               `json:"max_output_tokens"`
	Model              string             `json:"model"`
	PreviousResponseID *string            `json:"previous_response_id"`
	Output             []OutputItem       `json:"output"`
	ParallelToolCalls  bool               `json:"parallel_tool_calls"`
	Store              bool               `json:"store"`
	Temperature        *float32           `json:"temperature,omitempty"`
	TopP               *float32           `json:"top_p,omitempty"`
	Usage              *ResponseUsage     `json:"usage,omitempty"`
	Metadata           map[string]string  `json:"metadata"`
}

type ResponseErrorObj struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type IncompleteDetails struct {
	Reason string `json:"reason"`
}

type OutputItem struct {
	ID        string           `json:"id"`
	Type      string           `json:"type"` // "message", "function_call", "reasoning"
	Status    string           `json:"status,omitempty"`
	Role      string           `json:"role,omitempty"`
	Content   []OutputPart     `json:"content,omitempty"`
	CallID    string           `json:"call_id,omitempty"`
	Name      string           `json:"name,omitempty"`
	Arguments string           `json:"arguments,omitempty"`
	Summary   []SummaryTextObj `json:"summary,omitempty"`
}

type OutputPart struct {
	Type        string `json:"type"` // "output_text"
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

type SummaryTextObj struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

// InputItemStored is the normalized OpenAI input item returned by GET /v1/responses/{id}/input_items.
type InputItemStored struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`
	CallID  string `json:"call_id,omitempty"`
	Name    string `json:"name,omitempty"`
	Output  string `json:"output,omitempty"`
}

type ResponseUsage struct {
	InputTokens         int64               `json:"input_tokens"`
	InputTokensDetails  InputTokensDetails  `json:"input_tokens_details"`
	OutputTokens        int64               `json:"output_tokens"`
	OutputTokensDetails OutputTokensDetails `json:"output_tokens_details"`
	TotalTokens         int64               `json:"total_tokens"`
}

type InputTokensDetails struct {
	CachedTokens int64 `json:"cached_tokens"`
}

type OutputTokensDetails struct {
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

// Handler serves /v1/responses CRUD and input_items endpoints.
type Handler struct {
	registry        *models.Registry
	translator      *bedrockconv.Translator
	store           *store.Store
	metrics         *metrics.Collector
	autoPromptCache bool
}

// NewHandler creates a new Responses API handler.
func NewHandler(registry *models.Registry, translator *bedrockconv.Translator, st *store.Store, mc *metrics.Collector, autoPromptCache bool) *Handler {
	return &Handler{
		registry:        registry,
		translator:      translator,
		store:           st,
		metrics:         mc,
		autoPromptCache: autoPromptCache,
	}
}

// Create handles POST /v1/responses.
func (h *Handler) Create(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		if apiErr, ok := openai.CheckMaxBytesError(err); ok {
			openai.WriteAPIError(w, apiErr)
			return
		}
		openai.WriteError(w, http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_body", "Failed to read request body.")
		return
	}

	var req CreateRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		openai.WriteError(w, http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_json", fmt.Sprintf("Invalid JSON payload: %v", err))
		return
	}

	tenant, _ := auth.TenantFromContext(r.Context())
	tenantID := "default"
	if tenant != nil && tenant.Name != "" {
		tenantID = tenant.Name
	}

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

	// Load prior conversation chain when previous_response_id is supplied (strictly tenant-scoped).
	var priorHistory []bedrockconv.Message
	var priorInputItems []InputItemStored
	prevID := strings.TrimSpace(req.PreviousResponseID)
	if prevID != "" {
		if h.store == nil {
			openai.WriteError(w, http.StatusBadRequest, openai.TypeInvalidRequest, "storage_unavailable", "Response state storage is not configured.")
			return
		}
		prevRec, err := h.store.GetResponse(r.Context(), tenantID, prevID)
		if errors.Is(err, store.ErrNotFound) {
			openai.WriteAPIError(w, openai.NewParamError(
				http.StatusNotFound,
				openai.TypeNotFound,
				"previous_response_not_found",
				"previous_response_id",
				fmt.Sprintf("Response %q was not found.", prevID),
			))
			return
		}
		if err != nil {
			openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "storage_error", "Failed to load previous response state.")
			return
		}
		if len(prevRec.CanonicalHistoryJSON) > 0 {
			_ = json.Unmarshal(prevRec.CanonicalHistoryJSON, &priorHistory)
		}
		if len(prevRec.InputItemsJSON) > 0 {
			_ = json.Unmarshal(prevRec.InputItemsJSON, &priorInputItems)
		}
	}

	turnMsgs, turnInputItems, apiErr := parseResponsesInput(req.Input)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	// Assemble canonical messages: optional instructions -> prior history -> new turn messages.
	var invokeMsgs []bedrockconv.Message
	if strings.TrimSpace(req.Instructions) != "" {
		invokeMsgs = append(invokeMsgs, bedrockconv.Message{
			Role: bedrockconv.RoleDeveloper,
			Parts: []bedrockconv.ContentPart{
				{Type: bedrockconv.PartText, Text: req.Instructions},
			},
		})
	}
	historyWithoutEphemeralInstructions := make([]bedrockconv.Message, 0, len(priorHistory)+len(turnMsgs))
	historyWithoutEphemeralInstructions = append(historyWithoutEphemeralInstructions, priorHistory...)
	historyWithoutEphemeralInstructions = append(historyWithoutEphemeralInstructions, turnMsgs...)
	invokeMsgs = append(invokeMsgs, historyWithoutEphemeralInstructions...)

	canonReq, apiErr := h.buildCanonicalRequest(req, bedrockModel, invokeMsgs)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	shouldStore := true
	if req.Store != nil {
		shouldStore = *req.Store
	}
	allInputItems := make([]InputItemStored, 0, len(priorInputItems)+len(turnInputItems))
	allInputItems = append(allInputItems, priorInputItems...)
	allInputItems = append(allInputItems, turnInputItems...)

	if req.Stream {
		h.serveStream(w, r, tenantID, meta, requestedModel, req, canonReq, historyWithoutEphemeralInstructions, allInputItems, shouldStore)
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

	respID := openai.NewID("resp_")
	createdAt := time.Now().Unix()
	outputItems := buildOutputItems(resp.ReasoningText, resp.ContentText, resp.ToolCalls)
	respObj := buildResponseObject(respID, createdAt, requestedModel, req, shouldStore, resp.FinishReason, outputItems, &resp.Usage)

	if shouldStore && h.store != nil {
		fullHistory := append(historyWithoutEphemeralInstructions, resp.AssistantMessage)
		if err := h.persistResponse(r.Context(), tenantID, respObj, allInputItems, outputItems, fullHistory); err != nil {
			openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "storage_error", "Failed to persist response state.")
			return
		}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(respObj)
}

// Get handles GET /v1/responses/{id}.
func (h *Handler) Get(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PathValue("id"))
	tenantID := tenantIDFromRequest(r)
	if h.store == nil {
		openai.WriteError(w, http.StatusNotFound, openai.TypeNotFound, "not_found", fmt.Sprintf("Response %q not found.", id))
		return
	}
	rec, err := h.store.GetResponse(r.Context(), tenantID, id)
	if errors.Is(err, store.ErrNotFound) {
		openai.WriteError(w, http.StatusNotFound, openai.TypeNotFound, "not_found", fmt.Sprintf("Response %q not found.", id))
		return
	}
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "storage_error", "Failed to read response.")
		return
	}
	if meta := openai.RequestMetaFromContext(r.Context()); meta != nil {
		meta.SetModel(rec.Model, rec.Model)
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(rec.ResponseJSON)
	_, _ = io.WriteString(w, "\n")
}

// Delete handles DELETE /v1/responses/{id}.
func (h *Handler) Delete(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PathValue("id"))
	tenantID := tenantIDFromRequest(r)
	if h.store == nil {
		openai.WriteError(w, http.StatusNotFound, openai.TypeNotFound, "not_found", fmt.Sprintf("Response %q not found.", id))
		return
	}
	deleted, err := h.store.DeleteResponse(r.Context(), tenantID, id)
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "storage_error", "Failed to delete response.")
		return
	}
	if !deleted {
		openai.WriteError(w, http.StatusNotFound, openai.TypeNotFound, "not_found", fmt.Sprintf("Response %q not found.", id))
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      id,
		"object":  "response.deleted",
		"deleted": true,
	})
}

// ListInputItems handles GET /v1/responses/{id}/input_items.
func (h *Handler) ListInputItems(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PathValue("id"))
	tenantID := tenantIDFromRequest(r)
	if h.store == nil {
		openai.WriteError(w, http.StatusNotFound, openai.TypeNotFound, "not_found", fmt.Sprintf("Response %q not found.", id))
		return
	}
	rec, err := h.store.GetResponse(r.Context(), tenantID, id)
	if errors.Is(err, store.ErrNotFound) {
		openai.WriteError(w, http.StatusNotFound, openai.TypeNotFound, "not_found", fmt.Sprintf("Response %q not found.", id))
		return
	}
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "storage_error", "Failed to load input items.")
		return
	}

	var items []InputItemStored
	if len(rec.InputItemsJSON) > 0 {
		_ = json.Unmarshal(rec.InputItemsJSON, &items)
	}
	if items == nil {
		items = []InputItemStored{}
	}
	var firstID, lastID *string
	if len(items) > 0 {
		f := items[0].ID
		l := items[len(items)-1].ID
		firstID = &f
		lastID = &l
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object":   "list",
		"data":     items,
		"first_id": firstID,
		"last_id":  lastID,
		"has_more": false,
	})
}

func (h *Handler) serveStream(
	w http.ResponseWriter,
	r *http.Request,
	tenantID string,
	meta *openai.RequestMeta,
	requestedModel string,
	req CreateRequest,
	canonReq bedrockconv.Request,
	baseHistory []bedrockconv.Message,
	allInputItems []InputItemStored,
	shouldStore bool,
) {
	flusher, canFlush := w.(http.Flusher)
	if !canFlush {
		openai.WriteError(w, http.StatusInternalServerError, openai.TypeServer, "streaming_unsupported", "ResponseWriter does not support streaming flush.")
		return
	}

	respID := openai.NewID("resp_")
	createdAt := time.Now().Unix()
	headersSent := false
	seqNum := 0

	var releaseStreamGauge func()
	defer func() {
		if releaseStreamGauge != nil {
			releaseStreamGauge()
		}
	}()

	sendSSE := func(eventType string, payload map[string]any) error {
		if !headersSent {
			headersSent = true
			releaseStreamGauge = h.metrics.TrackStream(tenantID, "/v1/responses")
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.Header().Set("Cache-Control", "no-cache, no-transform")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
		}
		payload["type"] = eventType
		payload["sequence_number"] = seqNum
		seqNum++
		b, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, b); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	startedLifecycle := false
	ensureLifecycleStarted := func() error {
		if startedLifecycle {
			return nil
		}
		startedLifecycle = true
		inProg := buildResponseObject(respID, createdAt, requestedModel, req, shouldStore, "", []OutputItem{}, nil)
		inProg.Status = "in_progress"
		if err := sendSSE("response.created", map[string]any{"response": inProg}); err != nil {
			return err
		}
		return sendSSE("response.in_progress", map[string]any{"response": inProg})
	}

	nextOutputIndex := 0

	// Reasoning output item state
	reasoningOpen := false
	reasoningDone := false
	reasoningItemID := ""
	reasoningOutIdx := 0
	var reasoningBuf strings.Builder

	closeReasoningIfOpen := func() error {
		if !reasoningOpen || reasoningDone {
			return nil
		}
		reasoningDone = true
		doneItem := OutputItem{
			ID:     reasoningItemID,
			Type:   "reasoning",
			Status: "completed",
			Summary: []SummaryTextObj{
				{Type: "summary_text", Text: reasoningBuf.String()},
			},
		}
		return sendSSE("response.output_item.done", map[string]any{
			"output_index": reasoningOutIdx,
			"item":         doneItem,
		})
	}

	// Message text output item state
	msgOpen := false
	msgDone := false
	msgItemID := ""
	msgOutIdx := 0
	var msgTextBuf strings.Builder

	closeMsgIfOpen := func() error {
		if !msgOpen || msgDone {
			return nil
		}
		msgDone = true
		fullTxt := msgTextBuf.String()
		if err := sendSSE("response.output_text.done", map[string]any{
			"item_id":       msgItemID,
			"output_index":  msgOutIdx,
			"content_index": 0,
			"text":          fullTxt,
		}); err != nil {
			return err
		}
		doneItem := OutputItem{
			ID:     msgItemID,
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []OutputPart{
				{Type: "output_text", Text: fullTxt, Annotations: []any{}},
			},
		}
		return sendSSE("response.output_item.done", map[string]any{
			"output_index": msgOutIdx,
			"item":         doneItem,
		})
	}

	type streamToolState struct {
		itemID    string
		outIdx    int
		callID    string
		name      string
		argsBuf   strings.Builder
		completed bool
	}
	toolStates := make(map[int]*streamToolState)
	var finalOutputItems []OutputItem

	summary, apiErr := h.translator.InvokeStream(r.Context(), canonReq, func(ev bedrockconv.StreamEvent) error {
		switch ev.Kind {
		case bedrockconv.EventMessageStart:
			return ensureLifecycleStarted()

		case bedrockconv.EventReasoningDelta:
			if ev.ReasoningText == "" {
				return nil
			}
			if err := ensureLifecycleStarted(); err != nil {
				return err
			}
			if !reasoningOpen {
				reasoningOpen = true
				reasoningItemID = openai.NewID("rs_")
				reasoningOutIdx = nextOutputIndex
				nextOutputIndex++
				if err := sendSSE("response.output_item.added", map[string]any{
					"output_index": reasoningOutIdx,
					"item": OutputItem{
						ID:      reasoningItemID,
						Type:    "reasoning",
						Status:  "in_progress",
						Summary: []SummaryTextObj{},
					},
				}); err != nil {
					return err
				}
			}
			reasoningBuf.WriteString(ev.ReasoningText)
			return sendSSE("response.reasoning_summary_text.delta", map[string]any{
				"item_id":       reasoningItemID,
				"output_index":  reasoningOutIdx,
				"summary_index": 0,
				"delta":         ev.ReasoningText,
			})

		case bedrockconv.EventTextDelta:
			if err := ensureLifecycleStarted(); err != nil {
				return err
			}
			if err := closeReasoningIfOpen(); err != nil {
				return err
			}
			if !msgOpen {
				msgOpen = true
				msgItemID = openai.NewID("msg_")
				msgOutIdx = nextOutputIndex
				nextOutputIndex++
				if err := sendSSE("response.output_item.added", map[string]any{
					"output_index": msgOutIdx,
					"item": OutputItem{
						ID:      msgItemID,
						Type:    "message",
						Status:  "in_progress",
						Role:    "assistant",
						Content: []OutputPart{},
					},
				}); err != nil {
					return err
				}
				if err := sendSSE("response.content_part.added", map[string]any{
					"item_id":       msgItemID,
					"output_index":  msgOutIdx,
					"content_index": 0,
					"part":          OutputPart{Type: "output_text", Text: "", Annotations: []any{}},
				}); err != nil {
					return err
				}
			}
			msgTextBuf.WriteString(ev.Text)
			return sendSSE("response.output_text.delta", map[string]any{
				"item_id":       msgItemID,
				"output_index":  msgOutIdx,
				"content_index": 0,
				"delta":         ev.Text,
			})

		case bedrockconv.EventToolCallStart:
			if err := ensureLifecycleStarted(); err != nil {
				return err
			}
			if err := closeReasoningIfOpen(); err != nil {
				return err
			}
			if err := closeMsgIfOpen(); err != nil {
				return err
			}
			st := &streamToolState{
				itemID: openai.NewID("fc_"),
				outIdx: nextOutputIndex,
				callID: ev.ToolCallID,
				name:   ev.ToolName,
			}
			nextOutputIndex++
			toolStates[ev.ToolIndex] = st
			return sendSSE("response.output_item.added", map[string]any{
				"output_index": st.outIdx,
				"item": OutputItem{
					ID:        st.itemID,
					Type:      "function_call",
					Status:    "in_progress",
					CallID:    st.callID,
					Name:      st.name,
					Arguments: "",
				},
			})

		case bedrockconv.EventToolCallDelta:
			st, ok := toolStates[ev.ToolIndex]
			if !ok {
				return nil
			}
			st.argsBuf.WriteString(ev.ArgumentsDelta)
			return sendSSE("response.function_call_arguments.delta", map[string]any{
				"item_id":      st.itemID,
				"output_index": st.outIdx,
				"delta":        ev.ArgumentsDelta,
			})

		case bedrockconv.EventToolCallDone:
			st, ok := toolStates[ev.ToolIndex]
			if !ok || st.completed {
				return nil
			}
			st.completed = true
			args := strings.TrimSpace(st.argsBuf.String())
			if args == "" {
				args = "{}"
			}
			if err := sendSSE("response.function_call_arguments.done", map[string]any{
				"item_id":      st.itemID,
				"output_index": st.outIdx,
				"arguments":    args,
			}); err != nil {
				return err
			}
			doneItem := OutputItem{
				ID:        st.itemID,
				Type:      "function_call",
				Status:    "completed",
				CallID:    st.callID,
				Name:      st.name,
				Arguments: args,
			}
			return sendSSE("response.output_item.done", map[string]any{
				"output_index": st.outIdx,
				"item":         doneItem,
			})

		case bedrockconv.EventMessageStop:
			if err := closeReasoningIfOpen(); err != nil {
				return err
			}
			return closeMsgIfOpen()
		}
		return nil
	})

	if apiErr != nil {
		if !headersSent {
			openai.WriteAPIError(w, apiErr)
			return
		}
		failedObj := buildResponseObject(respID, createdAt, requestedModel, req, shouldStore, "stop", []OutputItem{}, nil)
		failedObj.Status = "failed"
		failedObj.Error = &ResponseErrorObj{
			Code:    apiErr.Code,
			Message: apiErr.Message,
		}
		_ = sendSSE("response.failed", map[string]any{"response": failedObj})
		return
	}

	if err := ensureLifecycleStarted(); err != nil {
		return
	}
	_ = closeReasoningIfOpen()
	_ = closeMsgIfOpen()

	// Assemble ordered final output items matching emitted item IDs.
	ordered := make([]OutputItem, nextOutputIndex)
	if reasoningOpen {
		ordered[reasoningOutIdx] = OutputItem{
			ID:     reasoningItemID,
			Type:   "reasoning",
			Status: "completed",
			Summary: []SummaryTextObj{
				{Type: "summary_text", Text: reasoningBuf.String()},
			},
		}
	}
	if msgOpen {
		ordered[msgOutIdx] = OutputItem{
			ID:     msgItemID,
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []OutputPart{
				{Type: "output_text", Text: msgTextBuf.String(), Annotations: []any{}},
			},
		}
	}
	for _, st := range toolStates {
		args := strings.TrimSpace(st.argsBuf.String())
		if args == "" {
			args = "{}"
		}
		ordered[st.outIdx] = OutputItem{
			ID:        st.itemID,
			Type:      "function_call",
			Status:    "completed",
			CallID:    st.callID,
			Name:      st.name,
			Arguments: args,
		}
	}
	finalOutputItems = ordered

	var usageVal bedrockconv.Usage
	finishReason := "stop"
	var assistantMsg bedrockconv.Message
	if summary != nil {
		usageVal = summary.Usage
		finishReason = summary.FinishReason
		assistantMsg = summary.AssistantMessage
	}

	if meta != nil {
		meta.AddUsage(
			usageVal.InputTokens,
			usageVal.OutputTokens,
			usageVal.CachedReadInputTokens,
			usageVal.CachedWriteInputTokens,
			usageVal.ReasoningTokens,
		)
	}

	respObj := buildResponseObject(respID, createdAt, requestedModel, req, shouldStore, finishReason, finalOutputItems, &usageVal)

	// Persist BEFORE emitting response.completed so immediate follow-up previous_response_id lookups succeed.
	if shouldStore && h.store != nil {
		fullHistory := append(baseHistory, assistantMsg)
		_ = h.persistResponse(r.Context(), tenantID, respObj, allInputItems, finalOutputItems, fullHistory)
	}

	_ = sendSSE("response.completed", map[string]any{"response": respObj})
}

func (h *Handler) persistResponse(
	ctx context.Context,
	tenantID string,
	respObj ResponseObject,
	inputItems []InputItemStored,
	outputItems []OutputItem,
	canonicalHistory []bedrockconv.Message,
) error {
	inBytes, err := json.Marshal(inputItems)
	if err != nil {
		return err
	}
	outBytes, err := json.Marshal(outputItems)
	if err != nil {
		return err
	}
	histBytes, err := json.Marshal(canonicalHistory)
	if err != nil {
		return err
	}
	respBytes, err := json.Marshal(respObj)
	if err != nil {
		return err
	}

	var expPtr *int64
	if ttl := h.store.TTL(); ttl > 0 {
		exp := time.Unix(respObj.CreatedAt, 0).Add(ttl).Unix()
		expPtr = &exp
	}

	prevID := ""
	if respObj.PreviousResponseID != nil {
		prevID = *respObj.PreviousResponseID
	}

	return h.store.SaveResponse(ctx, store.StoredResponse{
		ID:                   respObj.ID,
		TenantID:             tenantID,
		PreviousResponseID:   prevID,
		Model:                respObj.Model,
		Status:               respObj.Status,
		InputItemsJSON:       inBytes,
		OutputItemsJSON:      outBytes,
		CanonicalHistoryJSON: histBytes,
		ResponseJSON:         respBytes,
		CreatedAt:            respObj.CreatedAt,
		ExpiresAt:            expPtr,
	})
}

func (h *Handler) buildCanonicalRequest(req CreateRequest, bedrockModel string, msgs []bedrockconv.Message) (bedrockconv.Request, *openai.APIError) {
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
		name := rt.Name
		desc := rt.Description
		params := rt.Parameters
		if name == "" && rt.Function != nil {
			name = rt.Function.Name
			desc = rt.Function.Description
			params = rt.Function.Parameters
		}
		tools = append(tools, bedrockconv.ToolDef{
			Name:        name,
			Description: desc,
			Parameters:  params,
		})
	}

	tc, apiErr := parseResponsesToolChoice(req.ToolChoice)
	if apiErr != nil {
		return bedrockconv.Request{}, apiErr
	}

	var rf bedrockconv.ResponseFormat
	if req.Text != nil && req.Text.Format != nil {
		rf = bedrockconv.ResponseFormat{
			Type:        req.Text.Format.Type,
			Name:        req.Text.Format.Name,
			Description: req.Text.Format.Description,
			Schema:      req.Text.Format.Schema,
		}
	}

	var reasoning *bedrockconv.ReasoningConfig
	if req.Reasoning != nil && ((req.Reasoning.Effort != "" && !strings.EqualFold(req.Reasoning.Effort, "none")) || req.Reasoning.BudgetTokens > 0) {
		reasoning = &bedrockconv.ReasoningConfig{
			Effort:       req.Reasoning.Effort,
			BudgetTokens: req.Reasoning.BudgetTokens,
		}
	}

	return bedrockconv.Request{
		BedrockModelID:  bedrockModel,
		Messages:        msgs,
		Tools:           tools,
		ToolChoice:      tc,
		ResponseFormat:  rf,
		Reasoning:       reasoning,
		MaxTokens:       req.MaxOutputTokens,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		AutoPromptCache: h.autoPromptCache,
	}, nil
}

type rawResponsesInputItem struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Output    json.RawMessage `json:"output"`
}

func parseResponsesInput(raw json.RawMessage) ([]bedrockconv.Message, []InputItemStored, *openai.APIError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"missing_input",
			"input",
			"You must provide a non-empty input string or array of input items.",
		)
	}

	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, nil, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_input", "input", err.Error())
		}
		msgs := []bedrockconv.Message{
			{
				Role:  bedrockconv.RoleUser,
				Parts: []bedrockconv.ContentPart{{Type: bedrockconv.PartText, Text: s}},
			},
		}
		stored := []InputItemStored{
			{
				ID:   openai.NewID("msg_"),
				Type: "message",
				Role: "user",
				Content: []map[string]any{
					{"type": "input_text", "text": s},
				},
			},
		}
		return msgs, stored, nil
	}

	if trimmed[0] != '[' {
		return nil, nil, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"invalid_input",
			"input",
			"input must be a string or an array of input items.",
		)
	}

	var items []rawResponsesInputItem
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return nil, nil, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_input", "input", fmt.Sprintf("invalid input items array: %v", err))
	}

	var msgs []bedrockconv.Message
	var stored []InputItemStored

	for _, it := range items {
		itemType := strings.ToLower(strings.TrimSpace(it.Type))
		if itemType == "" && it.Role != "" {
			itemType = "message"
		}
		itemID := strings.TrimSpace(it.ID)

		switch itemType {
		case "message":
			role := strings.ToLower(strings.TrimSpace(it.Role))
			if role == "" {
				role = bedrockconv.RoleUser
			}
			parts, normalizedContent, apiErr := parseResponsesContentParts(it.Content)
			if apiErr != nil {
				return nil, nil, apiErr
			}
			msgs = append(msgs, bedrockconv.Message{
				Role:  role,
				Parts: parts,
			})
			if itemID == "" {
				itemID = openai.NewID("msg_")
			}
			stored = append(stored, InputItemStored{
				ID:      itemID,
				Type:    "message",
				Role:    role,
				Content: normalizedContent,
			})

		case "function_call":
			callID := strings.TrimSpace(it.CallID)
			if callID == "" {
				callID = itemID
			}
			msgs = append(msgs, bedrockconv.Message{
				Role: bedrockconv.RoleAssistant,
				Parts: []bedrockconv.ContentPart{
					{
						Type:         bedrockconv.PartToolUse,
						ToolCallID:   callID,
						ToolName:     it.Name,
						ToolArgsJSON: it.Arguments,
					},
				},
			})
			if itemID == "" {
				itemID = openai.NewID("fc_")
			}
			stored = append(stored, InputItemStored{
				ID:     itemID,
				Type:   "function_call",
				CallID: callID,
				Name:   it.Name,
			})

		case "function_call_output":
			outStr := decodeRawStringOrJSON(it.Output)
			msgs = append(msgs, bedrockconv.Message{
				Role: bedrockconv.RoleTool,
				Parts: []bedrockconv.ContentPart{
					{
						Type:           bedrockconv.PartToolResult,
						ToolCallID:     it.CallID,
						ToolResultText: outStr,
					},
				},
			})
			if itemID == "" {
				itemID = openai.NewID("fco_")
			}
			stored = append(stored, InputItemStored{
				ID:     itemID,
				Type:   "function_call_output",
				CallID: it.CallID,
				Output: outStr,
			})

		case "reasoning":
			// Stateless reasoning item echoed back by client: ignore if lacking Bedrock signature
			continue

		default:
			return nil, nil, openai.NewParamError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"unsupported_input_item",
				"input",
				fmt.Sprintf("Unsupported input item type %q.", it.Type),
			)
		}
	}

	return msgs, stored, nil
}

func parseResponsesContentParts(raw json.RawMessage) ([]bedrockconv.ContentPart, any, *openai.APIError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, []any{}, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_content", err.Error())
		}
		return []bedrockconv.ContentPart{{Type: bedrockconv.PartText, Text: s}},
			[]map[string]any{{"type": "input_text", "text": s}},
			nil
	}

	var list []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		ImageURL json.RawMessage `json:"image_url"`
		Filename string          `json:"filename"`
		FileData string          `json:"file_data"`
	}
	if err := json.Unmarshal(trimmed, &list); err != nil {
		return nil, nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_content", err.Error())
	}

	var parts []bedrockconv.ContentPart
	var normalized []map[string]any
	for _, elem := range list {
		switch elem.Type {
		case "input_text", "text", "output_text":
			parts = append(parts, bedrockconv.ContentPart{Type: bedrockconv.PartText, Text: elem.Text})
			normalized = append(normalized, map[string]any{"type": elem.Type, "text": elem.Text})
		case "input_image", "image_url":
			u, err := extractResponsesImageURL(elem.ImageURL)
			if err != nil {
				return nil, nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_image_url", err.Error())
			}
			parts = append(parts, bedrockconv.ContentPart{Type: bedrockconv.PartImage, RemoteURL: u})
			normalized = append(normalized, map[string]any{"type": "input_image"})
		case "input_file", "file":
			fd := strings.TrimSpace(elem.FileData)
			if fd == "" {
				return nil, nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_file", "input_file requires file_data")
			}
			if strings.HasPrefix(fd, "data:") || strings.HasPrefix(fd, "http://") || strings.HasPrefix(fd, "https://") {
				parts = append(parts, bedrockconv.ContentPart{
					Type:      bedrockconv.PartDocument,
					Name:      elem.Filename,
					RemoteURL: fd,
				})
			} else {
				decoded, err := base64.StdEncoding.DecodeString(fd)
				if err != nil {
					return nil, nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_file_data", "file_data must be a data: URI or base64")
				}
				mt := ""
				if strings.HasPrefix(string(decoded), "%PDF-") {
					mt = "application/pdf"
				}
				parts = append(parts, bedrockconv.ContentPart{
					Type:      bedrockconv.PartDocument,
					Name:      elem.Filename,
					MediaType: mt,
					Data:      decoded,
				})
			}
			normalized = append(normalized, map[string]any{"type": "input_file", "filename": elem.Filename})
		default:
			return nil, nil, openai.NewError(http.StatusBadRequest, openai.TypeInvalidRequest, "unsupported_content_type", fmt.Sprintf("Unsupported content part type %q.", elem.Type))
		}
	}
	return parts, normalized, nil
}

func extractResponsesImageURL(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "", errors.New("image_url is empty")
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
		return "", errors.New("image_url.url is empty")
	}
	return obj.URL, nil
}

func parseResponsesToolChoice(raw json.RawMessage) (bedrockconv.ToolChoice, *openai.APIError) {
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
	name := strings.TrimSpace(obj.Name)
	if name == "" {
		name = strings.TrimSpace(obj.Function.Name)
	}
	if name == "" {
		return bedrockconv.ToolChoice{}, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_tool_choice", "tool_choice", "function tool_choice requires name")
	}
	return bedrockconv.ToolChoice{Mode: "tool", ToolName: name}, nil
}

func decodeRawStringOrJSON(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err == nil {
			return s
		}
	}
	return string(trimmed)
}

func buildOutputItems(reasoningText, contentText string, toolCalls []bedrockconv.ToolCallOutput) []OutputItem {
	var out []OutputItem
	if reasoningText != "" {
		out = append(out, OutputItem{
			ID:     openai.NewID("rs_"),
			Type:   "reasoning",
			Status: "completed",
			Summary: []SummaryTextObj{
				{Type: "summary_text", Text: reasoningText},
			},
		})
	}
	if contentText != "" || len(toolCalls) == 0 {
		out = append(out, OutputItem{
			ID:     openai.NewID("msg_"),
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []OutputPart{
				{Type: "output_text", Text: contentText, Annotations: []any{}},
			},
		})
	}
	for _, tc := range toolCalls {
		out = append(out, OutputItem{
			ID:        openai.NewID("fc_"),
			Type:      "function_call",
			Status:    "completed",
			CallID:    tc.ID,
			Name:      tc.Name,
			Arguments: tc.Arguments,
		})
	}
	return out
}

func buildResponseObject(
	id string,
	createdAt int64,
	model string,
	req CreateRequest,
	shouldStore bool,
	finishReason string,
	outputItems []OutputItem,
	usage *bedrockconv.Usage,
) ResponseObject {
	status := "completed"
	var inc *IncompleteDetails
	if finishReason == "length" {
		status = "incomplete"
		inc = &IncompleteDetails{Reason: "max_output_tokens"}
	} else if finishReason == "content_filter" {
		status = "incomplete"
		inc = &IncompleteDetails{Reason: "content_filter"}
	}

	var instPtr *string
	if req.Instructions != "" {
		inst := req.Instructions
		instPtr = &inst
	}
	var prevPtr *string
	if strings.TrimSpace(req.PreviousResponseID) != "" {
		p := strings.TrimSpace(req.PreviousResponseID)
		prevPtr = &p
	}
	metaMap := req.Metadata
	if metaMap == nil {
		metaMap = map[string]string{}
	}
	if outputItems == nil {
		outputItems = []OutputItem{}
	}

	var respUsage *ResponseUsage
	if usage != nil {
		inTotal := usage.InputTokens + usage.CachedReadInputTokens + usage.CachedWriteInputTokens
		respUsage = &ResponseUsage{
			InputTokens: inTotal,
			InputTokensDetails: InputTokensDetails{
				CachedTokens: usage.CachedReadInputTokens,
			},
			OutputTokens: usage.OutputTokens,
			OutputTokensDetails: OutputTokensDetails{
				ReasoningTokens: usage.ReasoningTokens,
			},
			TotalTokens: inTotal + usage.OutputTokens,
		}
	}

	return ResponseObject{
		ID:                 id,
		Object:             "response",
		CreatedAt:          createdAt,
		Status:             status,
		IncompleteDetails:  inc,
		Instructions:       instPtr,
		MaxOutputTokens:    req.MaxOutputTokens,
		Model:              model,
		PreviousResponseID: prevPtr,
		Output:             outputItems,
		ParallelToolCalls:  true,
		Store:              shouldStore,
		Temperature:        req.Temperature,
		TopP:               req.TopP,
		Usage:              respUsage,
		Metadata:           metaMap,
	}
}

func tenantIDFromRequest(r *http.Request) string {
	if t, ok := auth.TenantFromContext(r.Context()); ok && t.Name != "" {
		return t.Name
	}
	return "default"
}
