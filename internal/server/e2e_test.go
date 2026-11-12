package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/store"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type e2eStreamReader struct {
	ch chan types.ConverseStreamOutput
}

func (r *e2eStreamReader) Events() <-chan types.ConverseStreamOutput { return r.ch }
func (r *e2eStreamReader) Close() error                              { return nil }
func (r *e2eStreamReader) Err() error                                { return nil }

type e2eMockBedrock struct{}

func (m *e2eMockBedrock) Converse(_ context.Context, in *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	reply := "Hello from " + aws.ToString(in.ModelId)
	return &bedrockruntime.ConverseOutput{
		StopReason: types.StopReasonEndTurn,
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: reply},
				},
			},
		},
		Usage: &types.TokenUsage{
			InputTokens:  aws.Int32(12),
			OutputTokens: aws.Int32(6),
			TotalTokens:  aws.Int32(18),
		},
	}, nil
}

func (m *e2eMockBedrock) ConverseStream(_ context.Context, _ *bedrockruntime.ConverseStreamInput) (bedrockruntime.ConverseStreamOutputReader, error) {
	ch := make(chan types.ConverseStreamOutput, 4)
	ch <- &types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &types.ContentBlockDeltaMemberText{Value: "E2E stream ok"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}}
	ch <- &types.ConverseStreamOutputMemberMetadata{
		Value: types.ConverseStreamMetadataEvent{
			Usage: &types.TokenUsage{InputTokens: aws.Int32(10), OutputTokens: aws.Int32(4)},
		},
	}
	close(ch)
	return &e2eStreamReader{ch: ch}, nil
}

func (m *e2eMockBedrock) InvokeModel(_ context.Context, _ *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	return &bedrockruntime.InvokeModelOutput{
		Body: []byte(`{"embedding":[0.25,0.75],"inputTextTokenCount":4}`),
	}, nil
}

type e2eRawRuntimeAdapter struct {
	*e2eMockBedrock
}

func (a e2eRawRuntimeAdapter) Converse(ctx context.Context, params *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return a.e2eMockBedrock.Converse(ctx, params)
}

func (a e2eRawRuntimeAdapter) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return nil, nil
}

func TestEndToEndGatewayStack(t *testing.T) {
	t.Setenv("GATEWAY_ACCESS_KEY", "")
	dbFile := filepath.Join(t.TempDir(), "e2e.db")
	rawTOML := `
access_key = "sk-default-top-key"

[models]
default_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
default_embedding_model = "amazon.titan-embed-text-v2:0"
discover_bedrock_models = false

[models.aliases]
"gpt-4o" = "us.anthropic.claude-sonnet-4-20250514-v1:0"
"text-embedding-3-small" = "amazon.titan-embed-text-v2:0"

[storage]
sqlite_path = "` + dbFile + `"

[[tenants]]
name = "alpha"
access_key = "sk-alpha-1"
access_keys = ["sk-alpha-2"]
allowed_models = ["gpt-4o", "text-embedding-3-small"]
`
	cfg, err := config.LoadBytes([]byte(rawTOML))
	if err != nil {
		t.Fatalf("LoadBytes: %v", err)
	}

	st, err := store.Open(context.Background(), cfg.Storage.SQLitePath, cfg.Storage.ResponsesTTL.Duration)
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	defer st.Close()

	mockBR := &e2eMockBedrock{}
	var logBuf bytes.Buffer
	gw := BuildGateway(WireInput{
		Config:        cfg,
		RuntimeClient: mockBR,
		RawRuntime:    e2eRawRuntimeAdapter{mockBR},
		AWSReadyCheck: func(context.Context) error { return nil },
		Store:         st,
		Logger:        NewLogger("info", &logBuf),
	})

	srv := httptest.NewServer(gw.MainHandler)
	defer srv.Close()

	doReq := func(method, path, key, body string) (*http.Response, string) {
		t.Helper()
		var rdr io.Reader
		if body != "" {
			rdr = strings.NewReader(body)
		}
		req, err := http.NewRequest(method, srv.URL+path, rdr)
		if err != nil {
			t.Fatalf("NewRequest: %v", err)
		}
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatalf("Do(%s %s): %v", method, path, err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp, string(b)
	}

	// 1. /readyz passes both SQLite + AWS checks
	resp, body := doReq(http.MethodGet, "/readyz", "", "")
	if resp.StatusCode != http.StatusOK || !strings.Contains(body, `"ready"`) {
		t.Fatalf("expected /readyz 200, got %d: %s", resp.StatusCode, body)
	}

	// 2. Top-level access_key works on /v1/chat/completions SSE streaming
	resp, body = doReq(http.MethodPost, "/v1/chat/completions", "sk-default-top-key", `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK || !strings.Contains(body, "E2E stream ok") || !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("SSE chat failed (%d): %s", resp.StatusCode, body)
	}

	// 3. Rotation key sk-alpha-2 creates a /v1/responses state; top-level default tenant cannot read it
	resp, body = doReq(http.MethodPost, "/v1/responses", "sk-alpha-2", `{"model":"gpt-4o","input":"First turn"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/responses failed (%d): %s", resp.StatusCode, body)
	}
	var created struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal([]byte(body), &created)

	// Same tenant using primary key sk-alpha-1 CAN read it
	resp, _ = doReq(http.MethodGet, "/v1/responses/"+created.ID, "sk-alpha-1", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected rotated key sibling sk-alpha-1 to read response, got %d", resp.StatusCode)
	}
	// Different tenant ("default") gets 404
	resp, _ = doReq(http.MethodGet, "/v1/responses/"+created.ID, "sk-default-top-key", "")
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected cross-tenant GET to return 404, got %d", resp.StatusCode)
	}

	// 4. /v1/embeddings works with base64 encoding
	resp, body = doReq(http.MethodPost, "/v1/embeddings", "sk-alpha-1", `{"model":"text-embedding-3-small","encoding_format":"base64","input":"hello"}`)
	if resp.StatusCode != http.StatusOK || !strings.Contains(body, `"object":"embedding"`) {
		t.Fatalf("POST /v1/embeddings failed (%d): %s", resp.StatusCode, body)
	}
}
