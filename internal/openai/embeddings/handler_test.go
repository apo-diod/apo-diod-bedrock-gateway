package embeddings

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/embeddings"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type fakeInvokeRuntime struct {
	titanCalls  atomic.Int32
	cohereCalls atomic.Int32
}

func (f *fakeInvokeRuntime) Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return nil, nil
}

func (f *fakeInvokeRuntime) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return nil, nil
}

func (f *fakeInvokeRuntime) InvokeModel(_ context.Context, params *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	modelID := aws.ToString(params.ModelId)
	if strings.Contains(modelID, "titan") {
		f.titanCalls.Add(1)
		var payload struct {
			InputText  string `json:"inputText"`
			Dimensions int    `json:"dimensions"`
			Normalize  bool   `json:"normalize"`
		}
		_ = json.Unmarshal(params.Body, &payload)
		var idx int
		_, _ = fmt.Sscanf(payload.InputText, "doc-%d", &idx)
		// Stagger even/odd workers slightly to verify concurrent order preservation
		if idx%2 == 0 {
			time.Sleep(2 * time.Millisecond)
		}
		respBytes, _ := json.Marshal(map[string]any{
			"embedding":           []float32{float32(idx), 0.5, -1.25},
			"inputTextTokenCount": 5,
		})
		return &bedrockruntime.InvokeModelOutput{Body: respBytes}, nil
	}

	f.cohereCalls.Add(1)
	var cPayload struct {
		Texts     []string `json:"texts"`
		InputType string   `json:"input_type"`
	}
	_ = json.Unmarshal(params.Body, &cPayload)
	vecs := make([][]float32, len(cPayload.Texts))
	for i := range cPayload.Texts {
		vecs[i] = []float32{float32(i), 1.0}
	}
	respBytes, _ := json.Marshal(map[string]any{
		"embeddings": map[string]any{"float": vecs},
		"meta": map[string]any{
			"billed_units": map[string]any{"input_tokens": len(cPayload.Texts) * 3},
		},
	})
	return &bedrockruntime.InvokeModelOutput{Body: respBytes}, nil
}

func TestEmbeddingsHandler_TitanConcurrentCohereBatchingFloatBase64AndTokenError(t *testing.T) {
	rt := &fakeInvokeRuntime{}
	reg := models.NewRegistry(config.ModelsConfig{
		DefaultEmbeddingModel: "amazon.titan-embed-text-v2:0",
		AllowPassthrough:      true,
		DiscoveryCacheTTL:     config.Duration{Duration: time.Minute},
		Aliases: map[string]string{
			"text-embedding-3-small": "amazon.titan-embed-text-v2:0",
		},
	}, nil, nil)

	svc := embeddings.NewService(rt, 4)
	h := NewHandler(reg, svc)
	tenant := &auth.Tenant{Name: "alpha", AllowedModels: []string{"*"}}

	// 1. Titan concurrent batch of 12 strings with float encoding
	inputs := make([]string, 12)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("doc-%d", i)
	}
	inJSON, _ := json.Marshal(inputs)
	reqFloat := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(fmt.Sprintf(`{"model":"text-embedding-3-small","dimensions":256,"input":%s}`, inJSON)))
	reqFloat = reqFloat.WithContext(auth.WithTenant(reqFloat.Context(), tenant))
	rrFloat := httptest.NewRecorder()
	h.ServeHTTP(rrFloat, reqFloat)
	if rrFloat.Code != http.StatusOK {
		t.Fatalf("Titan float failed (%d): %s", rrFloat.Code, rrFloat.Body.String())
	}
	var outFloat struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage EmbeddingUsage `json:"usage"`
	}
	if err := json.Unmarshal(rrFloat.Body.Bytes(), &outFloat); err != nil {
		t.Fatalf("decode float response: %v", err)
	}
	if len(outFloat.Data) != 12 || outFloat.Usage.PromptTokens != 60 {
		t.Fatalf("unexpected Titan float response: len=%d usage=%+v", len(outFloat.Data), outFloat.Usage)
	}
	for i, d := range outFloat.Data {
		if d.Index != i || d.Embedding[0] != float32(i) {
			t.Fatalf("order mismatch at index %d: got %+v", i, d)
		}
	}

	// 2. Base64 IEEE-754 little-endian float32 encoding
	reqB64 := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","encoding_format":"base64","input":"doc-7"}`))
	reqB64 = reqB64.WithContext(auth.WithTenant(reqB64.Context(), tenant))
	rrB64 := httptest.NewRecorder()
	h.ServeHTTP(rrB64, reqB64)
	if rrB64.Code != http.StatusOK {
		t.Fatalf("Titan base64 failed (%d): %s", rrB64.Code, rrB64.Body.String())
	}
	var outB64 struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	_ = json.Unmarshal(rrB64.Body.Bytes(), &outB64)
	rawBytes, err := base64.StdEncoding.DecodeString(outB64.Data[0].Embedding)
	if err != nil || len(rawBytes) != 12 {
		t.Fatalf("expected 12 decoded bytes (3 float32s), got len=%d err=%v", len(rawBytes), err)
	}
	v0 := math.Float32frombits(binary.LittleEndian.Uint32(rawBytes[0:4]))
	v2 := math.Float32frombits(binary.LittleEndian.Uint32(rawBytes[8:12]))
	if v0 != 7.0 || v2 != -1.25 {
		t.Errorf("unexpected decoded float32 values: v0=%v v2=%v", v0, v2)
	}

	// 3. Cohere batching > 96 texts (120 texts -> 2 InvokeModel calls: 96 + 24)
	cohereInputs := make([]string, 120)
	for i := range cohereInputs {
		cohereInputs[i] = fmt.Sprintf("chunk-%d", i)
	}
	cohereJSON, _ := json.Marshal(cohereInputs)
	reqCohere := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(fmt.Sprintf(`{"model":"cohere.embed-english-v3","input":%s}`, cohereJSON)))
	reqCohere.Header.Set("X-Cohere-Input-Type", "search_query")
	reqCohere = reqCohere.WithContext(auth.WithTenant(reqCohere.Context(), tenant))
	rrCohere := httptest.NewRecorder()
	h.ServeHTTP(rrCohere, reqCohere)
	if rrCohere.Code != http.StatusOK {
		t.Fatalf("Cohere batch failed (%d): %s", rrCohere.Code, rrCohere.Body.String())
	}
	if rt.cohereCalls.Load() != 2 {
		t.Errorf("expected 2 chunked Cohere InvokeModel calls (96 + 24), got %d", rt.cohereCalls.Load())
	}

	// 4. Token-ID integer array rejected with 400 token_array_unsupported
	reqTok := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","input":[15496,11,995]}`))
	reqTok = reqTok.WithContext(auth.WithTenant(reqTok.Context(), tenant))
	rrTok := httptest.NewRecorder()
	h.ServeHTTP(rrTok, reqTok)
	if rrTok.Code != http.StatusBadRequest || !strings.Contains(rrTok.Body.String(), "token_array_unsupported") {
		t.Fatalf("expected 400 token_array_unsupported, got %d: %s", rrTok.Code, rrTok.Body.String())
	}
}
