package bedrockconv

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type mockStreamReader struct {
	ch  chan types.ConverseStreamOutput
	err error
}

func (m *mockStreamReader) Events() <-chan types.ConverseStreamOutput { return m.ch }
func (m *mockStreamReader) Close() error                              { return nil }
func (m *mockStreamReader) Err() error                                { return m.err }

type mockRuntime struct {
	lastInput       *bedrockruntime.ConverseInput
	lastStreamInput *bedrockruntime.ConverseStreamInput
	converseOut     *bedrockruntime.ConverseOutput
	converseErr     error
	streamEvents    []types.ConverseStreamOutput
	streamErr       error
}

func (m *mockRuntime) Converse(_ context.Context, params *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	m.lastInput = params
	return m.converseOut, m.converseErr
}

func (m *mockRuntime) ConverseStream(_ context.Context, params *bedrockruntime.ConverseStreamInput) (bedrockruntime.ConverseStreamOutputReader, error) {
	m.lastStreamInput = params
	ch := make(chan types.ConverseStreamOutput, len(m.streamEvents))
	for _, ev := range m.streamEvents {
		ch <- ev
	}
	close(ch)
	return &mockStreamReader{ch: ch, err: m.streamErr}, nil
}

func TestBuildConverseInput_AlternationToolCoalescingMediaAndCaching(t *testing.T) {
	rt := &mockRuntime{}
	tr := NewTranslator(rt, Options{AllowRemoteMediaURLs: false})

	pngB64 := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nfake"))
	pdfB64 := base64.StdEncoding.EncodeToString([]byte("%PDF-1.7 fake pdf"))

	req := Request{
		BedrockModelID:  "us.anthropic.claude-sonnet-4-20250514-v1:0",
		AutoPromptCache: true,
		Messages: []Message{
			{Role: RoleSystem, Parts: []ContentPart{{Type: PartText, Text: "System rule 1."}}},
			{Role: RoleDeveloper, Parts: []ContentPart{{Type: PartText, Text: "Developer rule 2."}}},
			{
				Role: RoleUser,
				Parts: []ContentPart{
					{Type: PartText, Text: "Inspect this image and PDF."},
					{Type: PartImage, RemoteURL: "data:image/png;base64," + pngB64},
					{Type: PartDocument, Name: "Q4_Report_2026.pdf", RemoteURL: "data:application/pdf;base64," + pdfB64, CacheControl: true},
				},
			},
			// Adjacent user message must merge into previous user message
			{Role: RoleUser, Parts: []ContentPart{{Type: PartText, Text: "And call both tools."}}},
			{
				Role: RoleAssistant,
				Parts: []ContentPart{
					{Type: PartToolUse, ToolCallID: "call_1", ToolName: "get_weather", ToolArgsJSON: `{"city":"Berlin"}`},
					{Type: PartToolUse, ToolCallID: "call_2", ToolName: "get_stock", ToolArgsJSON: `{"ticker":"AMZN"}`},
				},
			},
			// Two consecutive OpenAI tool messages must coalesce into ONE Bedrock user message with 2 ToolResultBlocks
			{Role: RoleTool, Parts: []ContentPart{{Type: PartToolResult, ToolCallID: "call_1", ToolResultText: `{"temp":18}`}}},
			{Role: RoleTool, Parts: []ContentPart{{Type: PartToolResult, ToolCallID: "call_2", ToolResultText: `{"price":210}`}}},
		},
		Tools: []ToolDef{
			{Name: "get_weather", Parameters: map[string]any{"type": "object"}},
			{Name: "get_stock", Parameters: map[string]any{"type": "object"}},
		},
	}

	in, _, apiErr := tr.BuildConverseInput(context.Background(), req)
	if apiErr != nil {
		t.Fatalf("BuildConverseInput failed: %v", apiErr)
	}

	// System blocks: 2 text + 1 auto CachePoint = 3
	if len(in.System) != 3 {
		t.Fatalf("expected 3 system blocks (2 text + 1 cachePoint), got %d", len(in.System))
	}
	if _, ok := in.System[2].(*types.SystemContentBlockMemberCachePoint); !ok {
		t.Errorf("expected trailing system CachePointBlock, got %T", in.System[2])
	}

	// Messages must strictly alternate: [0]=user, [1]=assistant, [2]=user (with 2 coalesced ToolResultBlocks)
	if len(in.Messages) != 3 {
		t.Fatalf("expected 3 strictly alternating messages, got %d", len(in.Messages))
	}
	if in.Messages[0].Role != types.ConversationRoleUser ||
		in.Messages[1].Role != types.ConversationRoleAssistant ||
		in.Messages[2].Role != types.ConversationRoleUser {
		t.Fatalf("unexpected role sequence: %v, %v, %v", in.Messages[0].Role, in.Messages[1].Role, in.Messages[2].Role)
	}
	if len(in.Messages[2].Content) != 2 {
		t.Fatalf("expected 2 coalesced ToolResultBlocks in user turn, got %d", len(in.Messages[2].Content))
	}

	// Verify sanitized PDF document block name (no .pdf extension)
	foundDoc := false
	for _, cb := range in.Messages[0].Content {
		if docMember, ok := cb.(*types.ContentBlockMemberDocument); ok {
			foundDoc = true
			if got := aws.ToString(docMember.Value.Name); got != "Q4 Report 2026" {
				t.Errorf("expected sanitized doc name 'Q4 Report 2026', got %q", got)
			}
		}
	}
	if !foundDoc {
		t.Error("expected DocumentBlock in first user message")
	}

	// Tools: 2 toolSpecs + 1 auto CachePoint = 3
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) != 3 {
		t.Fatalf("expected 3 ToolConfig entries (2 specs + 1 cachePoint), got %+v", in.ToolConfig)
	}
}

func TestSSRFProtection_MediaBlocking(t *testing.T) {
	blockedIPs := []string{
		"169.254.169.254",        // EC2 IMDSv2
		"169.254.170.2",          // ECS task credentials
		"::ffff:169.254.169.254", // IPv4-mapped IPv6 IMDS bypass attempt
		"127.0.0.1",
		"::1",
		"10.0.0.1",
		"172.16.5.4",
		"192.168.1.1",
		"100.64.10.10", // CGNAT
		"fd00:ec2::23", // EKS Pod Identity ULA
		"fe80::1",
	}
	for _, raw := range blockedIPs {
		ip, err := netip.ParseAddr(raw)
		if err != nil {
			t.Fatalf("parse %q: %v", raw, err)
		}
		if !IsBlockedMediaIP(ip) {
			t.Errorf("SECURITY: expected IP %s to be blocked by SSRF filter", raw)
		}
	}

	publicIP := netip.MustParseAddr("93.184.216.34")
	if IsBlockedMediaIP(publicIP) {
		t.Errorf("expected public IP %s to be allowed", publicIP)
	}

	// 1. Default policy rejects remote URLs outright
	trDisabled := NewTranslator(&mockRuntime{}, Options{AllowRemoteMediaURLs: false})
	_, _, apiErr := trDisabled.BuildConverseInput(context.Background(), Request{
		BedrockModelID: "us.anthropic.claude-sonnet-4-20250514-v1:0",
		Messages: []Message{
			{Role: RoleUser, Parts: []ContentPart{{Type: PartImage, RemoteURL: "https://93.184.216.34/pic.png"}}},
		},
	})
	if apiErr == nil || apiErr.Code != "remote_media_disabled" {
		t.Fatalf("expected remote_media_disabled error, got %+v", apiErr)
	}

	// 2. Even when AllowRemoteMediaURLs=true, dialing a local/private server is blocked at connect time
	localSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("SSRF dialer should have blocked connection before HTTP handler ran")
	}))
	defer localSrv.Close()

	trEnabled := NewTranslator(&mockRuntime{}, Options{AllowRemoteMediaURLs: true})
	_, _, apiErr = trEnabled.BuildConverseInput(context.Background(), Request{
		BedrockModelID: "us.anthropic.claude-sonnet-4-20250514-v1:0",
		Messages: []Message{
			{Role: RoleUser, Parts: []ContentPart{{Type: PartImage, RemoteURL: localSrv.URL + "/imds.png"}}},
		},
	})
	if apiErr == nil || !strings.Contains(apiErr.Message, "SSRF protection") {
		t.Fatalf("expected connect-time SSRF protection error, got %+v", apiErr)
	}
}

func TestClaudeThinkingSignatureRoundTripAndNovaReasoning(t *testing.T) {
	rt := &mockRuntime{
		streamEvents: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberMessageStart{
				Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(0),
					Delta: &types.ContentBlockDeltaMemberReasoningContent{
						Value: &types.ReasoningContentBlockDeltaMemberText{Value: "Step 1: analyze. "},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(0),
					Delta: &types.ContentBlockDeltaMemberReasoningContent{
						Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig_claude_abc123"},
					},
				},
			},
			&types.ConverseStreamOutputMemberContentBlockDelta{
				Value: types.ContentBlockDeltaEvent{
					ContentBlockIndex: aws.Int32(1),
					Delta:             &types.ContentBlockDeltaMemberText{Value: "Final answer."},
				},
			},
			&types.ConverseStreamOutputMemberMessageStop{
				Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn},
			},
			&types.ConverseStreamOutputMemberMetadata{
				Value: types.ConverseStreamMetadataEvent{
					Usage: &types.TokenUsage{
						InputTokens:          aws.Int32(100),
						OutputTokens:         aws.Int32(50),
						TotalTokens:          aws.Int32(150),
						CacheReadInputTokens: aws.Int32(60),
					},
				},
			},
		},
	}

	tr := NewTranslator(rt, Options{})
	temp := float32(0.7)
	smallMax := 500
	summary, apiErr := tr.InvokeStream(context.Background(), Request{
		BedrockModelID: "us.anthropic.claude-sonnet-4-20250514-v1:0",
		Messages:       []Message{{Role: RoleUser, Parts: []ContentPart{{Type: PartText, Text: "Solve this."}}}},
		Reasoning:      &ReasoningConfig{Effort: "medium"},
		Temperature:    &temp,
		MaxTokens:      &smallMax,
	}, func(StreamEvent) error { return nil })
	if apiErr != nil {
		t.Fatalf("InvokeStream failed: %v", apiErr)
	}

	// Verify Claude thinking stripped Temperature and raised MaxTokens >= 4096+1024
	if rt.lastStreamInput.InferenceConfig.Temperature != nil {
		t.Error("expected Temperature to be stripped when Claude thinking is enabled")
	}
	if gotMax := aws.ToInt32(rt.lastStreamInput.InferenceConfig.MaxTokens); gotMax < 5120 {
		t.Errorf("expected MaxTokens >= 5120 (budget 4096 + 1024), got %d", gotMax)
	}
	addlBytes, err := rt.lastStreamInput.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatalf("MarshalSmithyDocument(claude): %v", err)
	}
	if !strings.Contains(string(addlBytes), `"budget_tokens":4096`) || !strings.Contains(string(addlBytes), `"type":"enabled"`) {
		t.Errorf("expected thinking budget_tokens=4096, got %s", string(addlBytes))
	}

	// Verify reconstructed AssistantMessage preserves signature for multi-turn /v1/responses chaining
	if summary.ReasoningText != "Step 1: analyze. " || summary.ContentText != "Final answer." {
		t.Fatalf("unexpected stream summary text: %+v", summary)
	}
	if summary.Usage.CachedReadInputTokens != 60 {
		t.Errorf("expected CachedReadInputTokens=60, got %d", summary.Usage.CachedReadInputTokens)
	}
	if len(summary.AssistantMessage.Parts) != 2 || summary.AssistantMessage.Parts[0].Signature != "sig_claude_abc123" {
		t.Fatalf("expected signature preserved in AssistantMessage, got %+v", summary.AssistantMessage.Parts)
	}

	// Round-trip AssistantMessage into a follow-up turn and verify ReasoningTextBlock + Signature is emitted to Bedrock
	inFollow, _, apiErr := tr.BuildConverseInput(context.Background(), Request{
		BedrockModelID: "us.anthropic.claude-sonnet-4-20250514-v1:0",
		Messages: []Message{
			{Role: RoleUser, Parts: []ContentPart{{Type: PartText, Text: "Solve this."}}},
			summary.AssistantMessage,
			{Role: RoleUser, Parts: []ContentPart{{Type: PartText, Text: "Follow up."}}},
		},
	})
	if apiErr != nil {
		t.Fatalf("follow-up BuildConverseInput failed: %v", apiErr)
	}
	rcBlock, ok := inFollow.Messages[1].Content[0].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("expected ReasoningContentBlock on round-trip, got %T", inFollow.Messages[1].Content[0])
	}
	rtMember, ok := rcBlock.Value.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok || aws.ToString(rtMember.Value.Signature) != "sig_claude_abc123" {
		t.Fatalf("expected round-tripped signature sig_claude_abc123, got %+v", rcBlock.Value)
	}

	// Verify Nova reasoningConfig translation
	inNova, _, apiErr := tr.BuildConverseInput(context.Background(), Request{
		BedrockModelID: "us.amazon.nova-pro-v1:0",
		Messages:       []Message{{Role: RoleUser, Parts: []ContentPart{{Type: PartText, Text: "Hi"}}}},
		Reasoning:      &ReasoningConfig{Effort: "high"},
	})
	if apiErr != nil {
		t.Fatalf("Nova BuildConverseInput failed: %v", apiErr)
	}
	novaBytes, err := inNova.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatalf("MarshalSmithyDocument(nova): %v", err)
	}
	if !strings.Contains(string(novaBytes), `"maxReasoningEffort":"high"`) {
		t.Errorf("expected Nova maxReasoningEffort=high, got %s", string(novaBytes))
	}
}

func TestStructuredOutputJSONSchemaSyntheticToolUnwrapping(t *testing.T) {
	rt := &mockRuntime{
		converseOut: &bedrockruntime.ConverseOutput{
			StopReason: types.StopReasonToolUse,
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Role: types.ConversationRoleAssistant,
					Content: []types.ContentBlock{
						&types.ContentBlockMemberToolUse{
							Value: types.ToolUseBlock{
								ToolUseId: aws.String("tooluse_1"),
								Name:      aws.String(syntheticJSONSchemaToolName),
								Input:     document.NewLazyDocument(map[string]any{"answer": 42}),
							},
						},
					},
				},
			},
			Usage: &types.TokenUsage{InputTokens: aws.Int32(10), OutputTokens: aws.Int32(5)},
		},
	}

	tr := NewTranslator(rt, Options{})
	resp, apiErr := tr.Invoke(context.Background(), Request{
		BedrockModelID: "us.anthropic.claude-sonnet-4-20250514-v1:0",
		Messages:       []Message{{Role: RoleUser, Parts: []ContentPart{{Type: PartText, Text: "Give 42"}}}},
		ResponseFormat: ResponseFormat{
			Type:   "json_schema",
			Name:   "math_result",
			Schema: map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "integer"}}},
		},
	})
	if apiErr != nil {
		t.Fatalf("Invoke(json_schema) failed: %v", apiErr)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop' after unwrapping synthetic schema tool, got %q", resp.FinishReason)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("expected 0 user-visible tool calls, got %+v", resp.ToolCalls)
	}
	if resp.ContentText != `{"answer":42}` {
		t.Errorf("expected extracted JSON content '{\"answer\":42}', got %q", resp.ContentText)
	}
}
