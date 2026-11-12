package embeddings

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/embeddings"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
)

// CreateRequest models OpenAI POST /v1/embeddings requests.
type CreateRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	Dimensions     *int            `json:"dimensions,omitempty"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	User           string          `json:"user,omitempty"`
	InputType      string          `json:"input_type,omitempty"` // Optional Cohere input_type extension
}

// CreateResponse models OpenAI POST /v1/embeddings responses.
type CreateResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingData holds either a []float32 or a base64-encoded little-endian float32 string.
type EmbeddingData struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}

type EmbeddingUsage struct {
	PromptTokens int64 `json:"prompt_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

// Handler serves POST /v1/embeddings.
type Handler struct {
	registry *models.Registry
	service  *embeddings.Service
}

// NewHandler creates a new OpenAI /v1/embeddings HTTP handler.
func NewHandler(registry *models.Registry, service *embeddings.Service) *Handler {
	return &Handler{
		registry: registry,
		service:  service,
	}
}

// ServeHTTP handles POST /v1/embeddings.
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

	var req CreateRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		openai.WriteError(w, http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_json", fmt.Sprintf("Invalid JSON payload: %v", err))
		return
	}

	encFmt := strings.ToLower(strings.TrimSpace(req.EncodingFormat))
	if encFmt == "" {
		encFmt = "float"
	}
	if encFmt != "float" && encFmt != "base64" {
		openai.WriteAPIError(w, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"invalid_encoding_format",
			"encoding_format",
			fmt.Sprintf("Unsupported encoding_format %q (expected 'float' or 'base64').", req.EncodingFormat),
		))
		return
	}

	if req.Dimensions != nil && *req.Dimensions <= 0 {
		openai.WriteAPIError(w, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"invalid_dimensions",
			"dimensions",
			"dimensions must be a positive integer.",
		))
		return
	}

	texts, apiErr := parseEmbeddingInput(req.Input)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	tenant, _ := auth.TenantFromContext(r.Context())
	requestedModel, bedrockModel, apiErr := h.registry.ResolveEmbeddingModel(tenant, req.Model)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	meta := openai.RequestMetaFromContext(r.Context())
	if meta != nil {
		meta.SetModel(requestedModel, bedrockModel)
	}

	inputType := strings.TrimSpace(req.InputType)
	if inputType == "" {
		inputType = strings.TrimSpace(r.Header.Get("X-Cohere-Input-Type"))
	}

	res, apiErr := h.service.Embed(r.Context(), embeddings.EmbedRequest{
		BedrockModelID: bedrockModel,
		Texts:          texts,
		Dimensions:     req.Dimensions,
		InputType:      inputType,
	})
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}

	if meta != nil {
		meta.AddUsage(res.InputTokens, 0, 0, 0, 0)
	}

	data := make([]EmbeddingData, len(res.Vectors))
	for i, vec := range res.Vectors {
		var encoded any = vec
		if encFmt == "base64" {
			encoded = EncodeFloat32Base64(vec)
		}
		data[i] = EmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: encoded,
		}
	}

	out := CreateResponse{
		Object: "list",
		Data:   data,
		Model:  requestedModel,
		Usage: EmbeddingUsage{
			PromptTokens: res.InputTokens,
			TotalTokens:  res.InputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(out)
}

// EncodeFloat32Base64 packs float32 values as little-endian IEEE-754 bytes encoded in standard base64,
// matching the official Python openai SDK's numpy float32 decoder.
func EncodeFloat32Base64(vec []float32) string {
	buf := make([]byte, len(vec)*4)
	for i, f := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func parseEmbeddingInput(raw json.RawMessage) ([]string, *openai.APIError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"missing_input",
			"input",
			"input must be a string or non-empty array of strings.",
		)
	}

	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "invalid_input", "input", err.Error())
		}
		return []string{s}, nil
	}

	if trimmed[0] == '[' {
		var strs []string
		if err := json.Unmarshal(trimmed, &strs); err == nil {
			if len(strs) == 0 {
				return nil, openai.NewParamError(
					http.StatusBadRequest,
					openai.TypeInvalidRequest,
					"empty_input",
					"input",
					"input array must not be empty.",
				)
			}
			return strs, nil
		}

		// Detect token-ID integer arrays ([15496, ...] or [[15496, ...]]) sent by tiktoken clients.
		var ints []int
		if err := json.Unmarshal(trimmed, &ints); err == nil {
			return nil, openai.NewParamError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"token_array_unsupported",
				"input",
				"Token-ID integer arrays are not supported because AWS Bedrock Titan/Cohere models do not use OpenAI tiktoken vocabularies; pass raw text strings instead.",
			)
		}
		var nestedInts [][]int
		if err := json.Unmarshal(trimmed, &nestedInts); err == nil {
			return nil, openai.NewParamError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"token_array_unsupported",
				"input",
				"Token-ID integer arrays are not supported because AWS Bedrock Titan/Cohere models do not use OpenAI tiktoken vocabularies; pass raw text strings instead.",
			)
		}
	}

	return nil, openai.NewParamError(
		http.StatusBadRequest,
		openai.TypeInvalidRequest,
		"invalid_input",
		"input",
		"input must be a string or an array of strings.",
	)
}
