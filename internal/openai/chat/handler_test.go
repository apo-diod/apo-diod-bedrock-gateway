package chat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/bedrockconv"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
)

type fakeStreamReader struct {
	ch  chan types.ConverseStreamOutput
	err error
}

func (f *fakeStreamReader) Events() <-chan types.ConverseStreamOutput { return f.ch }
func (f *fakeStreamReader) Close() error                              { return nil }
func (f *fakeStreamReader) Err() error                                { return f.err }

type fakeRuntime struct {
	converseOut  *bedrockruntime.ConverseOutput
	streamEvents []types.ConverseStreamOutput
	streamErr    error
}

func (f *fakeRuntime) Converse(_ context.Context, _ *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	return f.converseOut, nil
}

func (f *fakeRuntime) ConverseStream(_ context.Context, _ *bedrockruntime.ConverseStreamInput) (bedrockruntime.ConverseStreamOutputReader, error) {
	ch := make(chan types.ConverseStreamOutput, len(f.streamEvents))
	for _, ev := range f.streamEvents {
		ch <- ev
	}
	close(ch)
	return &fakeStreamReader{ch: ch, err: f.streamErr}, nil
}

func TestChatCompletions_NonStreamingAndSSEStreaming(t *testing.T) {
	reg := models.NewRegistry(config.ModelsConfig{
		DefaultModel:      "us.anthropic.claude-sonnet-4-20250514-v1:0",
		AllowPassthrough:  true,
		DiscoveryCacheTTL: config.Duration{Duration: time.Minute},
		Aliases: map[string]string{
			"gpt-4o": "us.anthropic.claude-sonnet-4-20250514-v1:0",
		},
	}, nil, nil)

	rt := &fakeRuntime{
		converseOut: &bedrockruntime.ConverseOutput{
			StopReason: types.StopReasonToolUse,
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Role: types.ConversationRoleAssistant,
					Content: []types.ContentBlock{
						&types.ContentBlockMemberReasoningContent{
							Value: &types.ReasoningContentBlockMemberReasoningText{
								Value: types.ReasoningTextBlock{
									Text:      aws.String("Need weather tool."),
									Signature: aws.String("sig_1"),
								},
							},
						},
						&types.ContentBlockMemberToolUse{
							Value: types.ToolUseBlock{
								ToolUseId: aws.String("call_weather_1"),
								Name:      aws.String("get_weather"),
								Input:     document.NewLazyDocument(map[string]any{"city": "Paris"}),
							},
						},
					},
				},
			},
			Usage: &types.TokenUsage{
				InputTokens:          aws.Int32(40),
				OutputTokens:         aws.Int32(25),
				CacheReadInputTokens: aws.Int32(100),
			},
		},
		streamEvents: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberMessageStart{
				Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(0),
					Delta: &types.ContentBlockDeltaMemberReasoningContent{
						Value: &types.ReasoningContentBlockDeltaMemberText{Value: "Thinking..."},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(1),
					Delta:             &types.ContentBlockDeltaMemberText{Value: "Hello "},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockStart{
				Value: types.ContentBlockStartEvent{
					ContentBlockIndex: aws.Int32(2),
					Start: &types.ContentBlockStartMemberToolUse{
						Value: types.ToolUseBlockStart{
							ToolUseId: aws.String("call_abc"),
							Name:      aws.String("lookup"),
						},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(2),
					Delta: &types.ContentBlockDeltaMemberToolUse{
						Value: types.ToolUseBlockDelta{Input: aws.String(`{"q":"bedrock"}`)},
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
					Usage: &types.TokenUsage{
						InputTokens:          aws.Int32(20),
						OutputTokens:         aws.Int32(15),
						CacheReadInputTokens: aws.Int32(30),
					},
				},
			},
		},
	}

	mc := metrics.New()
	tr := bedrockconv.NewTranslator(rt, bedrockconv.Options{})
	h := NewHandler(reg, tr, mc, true)
	tenant := &auth.Tenant{Name: "alpha", AllowedModels: []string{"*"}}

	// 1. Non-streaming tool call + reasoning + cached_tokens
	bodyNonStream := `{
		"model": "gpt-4o",
		"reasoning_effort": "low",
		"messages": [{"role": "user", "content": "Weather in Paris?"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
	}`
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(bodyNonStream))
	req1 = req1.WithContext(auth.WithTenant(req1.Context(), tenant))
	rr1 := httptest.NewRecorder()
	h.ServeHTTP(rr1, req1)

	if rr1.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr1.Code, rr1.Body.String())
	}
	var cResp CompletionResponse
	if err := json.Unmarshal(rr1.Body.Bytes(), &cResp); err != nil {
		t.Fatalf("decode CompletionResponse: %v", err)
	}
	if cResp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason tool_calls, got %q", cResp.Choices[0].FinishReason)
	}
	if cResp.Choices[0].Message.Content != nil {
		t.Errorf("expected null message.content when only tool_calls returned, got %v", *cResp.Choices[0].Message.Content)
	}
	if cResp.Choices[0].Message.ReasoningContent != "Need weather tool." {
		t.Errorf("unexpected reasoning_content: %q", cResp.Choices[0].Message.ReasoningContent)
	}
	if len(cResp.Choices[0].Message.ToolCalls) != 1 || cResp.Choices[0].Message.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("unexpected tool_calls: %+v", cResp.Choices[0].Message.ToolCalls)
	}
	if cResp.Usage.PromptTokens != 140 || cResp.Usage.PromptTokensDetails.CachedTokens != 100 {
		t.Errorf("unexpected usage: %+v", cResp.Usage)
	}

	// 2. SSE streaming with stream_options.include_usage = true
	bodyStream := `{
		"model": "gpt-4o",
		"stream": true,
		"stream_options": {"include_usage": true},
		"messages": [{"role": "user", "content": "Search bedrock"}]
	}`
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(bodyStream))
	req2 = req2.WithContext(auth.WithTenant(req2.Context(), tenant))
	rr2 := httptest.NewRecorder()
	h.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusOK {
		t.Fatalf("expected SSE 200, got %d: %s", rr2.Code, rr2.Body.String())
	}
	if ct := rr2.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected text/event-stream, got %q", ct)
	}
	if rr2.Header().Get("X-Accel-Buffering") != "no" {
		t.Errorf("expected X-Accel-Buffering: no")
	}

	sseOut := rr2.Body.String()
	if !strings.HasSuffix(sseOut, "data: [DONE]\n\n") {
		t.Fatalf("expected SSE stream to end with 'data: [DONE]\\n\\n', got:\n%s", sseOut)
	}
	for _, expectedSub := range []string{
		`"reasoning_content":"Thinking..."`,
		`"content":"Hello "`,
		`"name":"lookup"`,
		`"arguments":"{\"q\":\"bedrock\"}"`,
		`"finish_reason":"tool_calls"`,
		`"cached_tokens":30`,
	} {
		if !strings.Contains(sseOut, expectedSub) {
			t.Errorf("expected %s in SSE output:\n%s", expectedSub, sseOut)
		}
	}

	// 3. Mid-stream Bedrock ThrottlingException emits SSE error frame after headers sent
	rt.streamErr = &smithy.GenericAPIError{Code: "ThrottlingException", Message: "Rate exceeded mid-stream"}
	req3 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(bodyStream))
	req3 = req3.WithContext(auth.WithTenant(req3.Context(), tenant))
	rr3 := httptest.NewRecorder()
	h.ServeHTTP(rr3, req3)
	if !strings.Contains(rr3.Body.String(), `"rate_limit_exceeded"`) {
		t.Errorf("expected mid-stream SSE error frame with rate_limit_exceeded, got:\n%s", rr3.Body.String())
	}
}
