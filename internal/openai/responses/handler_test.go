package responses

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/bedrockconv"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/store"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type fakeRespStreamReader struct {
	ch  chan types.ConverseStreamOutput
	err error
}

func (f *fakeRespStreamReader) Events() <-chan types.ConverseStreamOutput { return f.ch }
func (f *fakeRespStreamReader) Close() error                              { return nil }
func (f *fakeRespStreamReader) Err() error                                { return f.err }

type fakeRespRuntime struct {
	converseFn func(in *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error)
	lastInput  *bedrockruntime.ConverseInput
	streamEvs  []types.ConverseStreamOutput
}

func (f *fakeRespRuntime) Converse(_ context.Context, params *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	f.lastInput = params
	return f.converseFn(params)
}

func (f *fakeRespRuntime) ConverseStream(_ context.Context, _ *bedrockruntime.ConverseStreamInput) (bedrockruntime.ConverseStreamOutputReader, error) {
	ch := make(chan types.ConverseStreamOutput, len(f.streamEvs))
	for _, ev := range f.streamEvs {
		ch <- ev
	}
	close(ch)
	return &fakeRespStreamReader{ch: ch}, nil
}

func TestResponsesAPI_MultiTurnChainingCrossTenantIsolationStoreFalseAndSSE(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "nested", "gateway.db")
	st, err := store.Open(context.Background(), dbPath, 24*time.Hour)
	if err != nil {
		t.Fatalf("store.Open failed: %v", err)
	}
	defer st.Close()
	if err := st.Ping(context.Background()); err != nil {
		t.Fatalf("store.Ping failed: %v", err)
	}

	cleanupCtx, cancelCleanup := context.WithCancel(context.Background())
	st.StartCleanupLoop(cleanupCtx, 50*time.Millisecond, nil)
	defer cancelCleanup()

	turn := 0
	rt := &fakeRespRuntime{
		converseFn: func(in *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
			turn++
			if turn == 1 {
				return &bedrockruntime.ConverseOutput{
					StopReason: types.StopReasonToolUse,
					Output: &types.ConverseOutputMemberMessage{
						Value: types.Message{
							Role: types.ConversationRoleAssistant,
							Content: []types.ContentBlock{
								&types.ContentBlockMemberReasoningContent{
									Value: &types.ReasoningContentBlockMemberReasoningText{
										Value: types.ReasoningTextBlock{
											Text:      aws.String("Check DB inventory first."),
											Signature: aws.String("sig_roundtrip_99"),
										},
									},
								},
								&types.ContentBlockMemberToolUse{
									Value: types.ToolUseBlock{
										ToolUseId: aws.String("call_inv_1"),
										Name:      aws.String("get_inventory"),
										Input:     document.NewLazyDocument(map[string]any{"sku": "ABC"}),
									},
								},
							},
						},
					},
					Usage: &types.TokenUsage{InputTokens: aws.Int32(30), OutputTokens: aws.Int32(20)},
				}, nil
			}
			return &bedrockruntime.ConverseOutput{
				StopReason: types.StopReasonEndTurn,
				Output: &types.ConverseOutputMemberMessage{
					Value: types.Message{
						Role: types.ConversationRoleAssistant,
						Content: []types.ContentBlock{
							&types.ContentBlockMemberText{Value: "SKU ABC has 17 units in stock."},
						},
					},
				},
				Usage: &types.TokenUsage{InputTokens: aws.Int32(60), OutputTokens: aws.Int32(12)},
			}, nil
		},
		streamEvs: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberMessageStart{
				Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(0),
					Delta: &types.ContentBlockDeltaMemberReasoningContent{
						Value: &types.ReasoningContentBlockDeltaMemberText{Value: "Quick thought."},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(1),
					Delta:             &types.ContentBlockDeltaMemberText{Value: "Streamed hello!"},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockStart{
				Value: types.ContentBlockStartEvent{
					ContentBlockIndex: aws.Int32(2),
					Start: &types.ContentBlockStartMemberToolUse{
						Value: types.ToolUseBlockStart{
							ToolUseId: aws.String("call_s1"),
							Name:      aws.String("ping_svc"),
						},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(2),
					Delta: &types.ContentBlockDeltaMemberToolUse{
						Value: types.ToolUseBlockDelta{Input: aws.String(`{"host":"vpc.local"}`)},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockStop{
				Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(2)},
			},
			&types.ConverseStreamOutputMemberMessageStop{
				Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse},
			},
			&types.ConverseStreamOutputMemberMetadata{
				Value: types.ConverseStreamMetadataEvent{
					Usage: &types.TokenUsage{InputTokens: aws.Int32(15), OutputTokens: aws.Int32(10)},
				},
			},
		},
	}

	reg := models.NewRegistry(config.ModelsConfig{
		DefaultModel:      "us.anthropic.claude-sonnet-4-20250514-v1:0",
		AllowPassthrough:  true,
		DiscoveryCacheTTL: config.Duration{Duration: time.Minute},
		Aliases:           map[string]string{"gpt-4o": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
	}, nil, nil)
	tr := bedrockconv.NewTranslator(rt, bedrockconv.Options{})
	h := NewHandler(reg, tr, st, metrics.New(), false)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", h.Create)
	mux.HandleFunc("GET /v1/responses/{id}", h.Get)
	mux.HandleFunc("DELETE /v1/responses/{id}", h.Delete)
	mux.HandleFunc("GET /v1/responses/{id}/input_items", h.ListInputItems)

	tenantA := &auth.Tenant{Name: "tenant-a", AllowedModels: []string{"*"}}
	tenantB := &auth.Tenant{Name: "tenant-b", AllowedModels: []string{"*"}}

	// 1. Turn 1 (Tenant A): Create response -> returns reasoning + function_call
	req1Body := `{
		"model": "gpt-4o",
		"instructions": "Be concise.",
		"reasoning": {"effort": "low"},
		"input": "How many SKU ABC do we have?",
		"tools": [{"type": "function", "name": "get_inventory", "parameters": {"type": "object"}}]
	}`
	req1 := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(req1Body))
	req1 = req1.WithContext(auth.WithTenant(req1.Context(), tenantA))
	rr1 := httptest.NewRecorder()
	mux.ServeHTTP(rr1, req1)
	if rr1.Code != http.StatusOK {
		t.Fatalf("Turn 1 failed (%d): %s", rr1.Code, rr1.Body.String())
	}
	var resp1 ResponseObject
	if err := json.Unmarshal(rr1.Body.Bytes(), &resp1); err != nil {
		t.Fatalf("decode resp1: %v", err)
	}
	if !strings.HasPrefix(resp1.ID, "resp_") || len(resp1.Output) != 2 {
		t.Fatalf("expected reasoning + function_call output items, got %+v", resp1)
	}

	// 2. Cross-tenant isolation: Tenant B CANNOT read, list input_items, delete, or chain resp1.ID
	for _, tc := range []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/v1/responses/" + resp1.ID, ""},
		{http.MethodGet, "/v1/responses/" + resp1.ID + "/input_items", ""},
		{http.MethodDelete, "/v1/responses/" + resp1.ID, ""},
		{http.MethodPost, "/v1/responses", `{"model":"gpt-4o","previous_response_id":"` + resp1.ID + `","input":"hijack"}`},
	} {
		rCross := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
		rCross = rCross.WithContext(auth.WithTenant(rCross.Context(), tenantB))
		wCross := httptest.NewRecorder()
		mux.ServeHTTP(wCross, rCross)
		if wCross.Code != http.StatusNotFound {
			t.Fatalf("SECURITY: Tenant B %s %s expected 404, got %d: %s", tc.method, tc.path, wCross.Code, wCross.Body.String())
		}
	}

	// 3. Turn 2 (Tenant A): Chain previous_response_id with function_call_output
	req2Body := `{
		"model": "gpt-4o",
		"previous_response_id": "` + resp1.ID + `",
		"input": [{"type": "function_call_output", "call_id": "call_inv_1", "output": "{\"stock\": 17}"}]
	}`
	req2 := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(req2Body))
	req2 = req2.WithContext(auth.WithTenant(req2.Context(), tenantA))
	rr2 := httptest.NewRecorder()
	mux.ServeHTTP(rr2, req2)
	if rr2.Code != http.StatusOK {
		t.Fatalf("Turn 2 failed (%d): %s", rr2.Code, rr2.Body.String())
	}
	var resp2 ResponseObject
	_ = json.Unmarshal(rr2.Body.Bytes(), &resp2)
	if len(resp2.Output) != 1 || resp2.Output[0].Content[0].Text != "SKU ABC has 17 units in stock." {
		t.Fatalf("unexpected Turn 2 output: %+v", resp2.Output)
	}

	// Verify Bedrock received 3 alternating messages including round-tripped Claude reasoning signature!
	if len(rt.lastInput.Messages) != 3 {
		t.Fatalf("expected 3 historical + new messages sent to Bedrock, got %d", len(rt.lastInput.Messages))
	}
	rcBlock, ok := rt.lastInput.Messages[1].Content[0].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("expected persisted ReasoningContentBlock in Turn 2, got %T", rt.lastInput.Messages[1].Content[0])
	}
	rtVal := rcBlock.Value.(*types.ReasoningContentBlockMemberReasoningText)
	if aws.ToString(rtVal.Value.Signature) != "sig_roundtrip_99" {
		t.Errorf("expected persisted signature sig_roundtrip_99, got %q", aws.ToString(rtVal.Value.Signature))
	}

	// 4. GET /v1/responses/{id} and GET /v1/responses/{id}/input_items for Tenant A
	rGet := httptest.NewRequest(http.MethodGet, "/v1/responses/"+resp2.ID, nil)
	rGet = rGet.WithContext(auth.WithTenant(rGet.Context(), tenantA))
	wGet := httptest.NewRecorder()
	mux.ServeHTTP(wGet, rGet)
	if wGet.Code != http.StatusOK {
		t.Fatalf("GET response failed: %d", wGet.Code)
	}

	rItems := httptest.NewRequest(http.MethodGet, "/v1/responses/"+resp2.ID+"/input_items", nil)
	rItems = rItems.WithContext(auth.WithTenant(rItems.Context(), tenantA))
	wItems := httptest.NewRecorder()
	mux.ServeHTTP(wItems, rItems)
	if wItems.Code != http.StatusOK || !strings.Contains(wItems.Body.String(), "call_inv_1") {
		t.Fatalf("GET input_items unexpected (%d): %s", wItems.Code, wItems.Body.String())
	}

	// 5. store: false must NOT persist in SQLite
	reqNoStore := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-4o","store":false,"input":"ephemeral"}`))
	reqNoStore = reqNoStore.WithContext(auth.WithTenant(reqNoStore.Context(), tenantA))
	wNoStore := httptest.NewRecorder()
	mux.ServeHTTP(wNoStore, reqNoStore)
	var noStoreObj ResponseObject
	_ = json.Unmarshal(wNoStore.Body.Bytes(), &noStoreObj)

	rCheckGone := httptest.NewRequest(http.MethodGet, "/v1/responses/"+noStoreObj.ID, nil)
	rCheckGone = rCheckGone.WithContext(auth.WithTenant(rCheckGone.Context(), tenantA))
	wCheckGone := httptest.NewRecorder()
	mux.ServeHTTP(wCheckGone, rCheckGone)
	if wCheckGone.Code != http.StatusNotFound {
		t.Fatalf("expected store:false response to return 404 on GET, got %d", wCheckGone.Code)
	}

	// 6. DELETE /v1/responses/{id} removes stored response
	rDel := httptest.NewRequest(http.MethodDelete, "/v1/responses/"+resp1.ID, nil)
	rDel = rDel.WithContext(auth.WithTenant(rDel.Context(), tenantA))
	wDel := httptest.NewRecorder()
	mux.ServeHTTP(wDel, rDel)
	if wDel.Code != http.StatusOK || !strings.Contains(wDel.Body.String(), `"deleted":true`) {
		t.Fatalf("DELETE failed (%d): %s", wDel.Code, wDel.Body.String())
	}

	// 7. Typed SSE streaming sequence
	reqStream := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-4o","stream":true,"input":"stream test"}`))
	reqStream = reqStream.WithContext(auth.WithTenant(reqStream.Context(), tenantA))
	wStream := httptest.NewRecorder()
	mux.ServeHTTP(wStream, reqStream)
	if wStream.Code != http.StatusOK {
		t.Fatalf("SSE stream failed (%d): %s", wStream.Code, wStream.Body.String())
	}
	sseBody := wStream.Body.String()
	for _, evName := range []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		"event: response.reasoning_summary_text.delta",
		"event: response.content_part.added",
		"event: response.output_text.delta",
		"event: response.output_text.done",
		"event: response.function_call_arguments.delta",
		"event: response.function_call_arguments.done",
		"event: response.output_item.done",
		"event: response.completed",
	} {
		if !strings.Contains(sseBody, evName) {
			t.Errorf("missing expected Responses SSE event %q in:\n%s", evName, sseBody)
		}
	}

	// 8. Verify TTL expiry reaper removes expired rows
	past := time.Now().Add(-1 * time.Hour).Unix()
	if err := st.SaveResponse(context.Background(), store.StoredResponse{
		ID:                   "resp_expired_1",
		TenantID:             "tenant-a",
		Model:                "gpt-4o",
		Status:               "completed",
		InputItemsJSON:       []byte("[]"),
		OutputItemsJSON:      []byte("[]"),
		CanonicalHistoryJSON: []byte("[]"),
		ResponseJSON:         []byte("{}"),
		CreatedAt:            past - 3600,
		ExpiresAt:            &past,
	}); err != nil {
		t.Fatalf("save expired row: %v", err)
	}
	reaped, err := st.CleanupExpired(context.Background(), time.Now().Unix())
	if err != nil || reaped != 1 {
		t.Fatalf("expected 1 expired row reaped, got %d (err=%v)", reaped, err)
	}
}
