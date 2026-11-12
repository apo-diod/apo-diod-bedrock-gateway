package embeddings

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/awsclient"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"golang.org/x/sync/errgroup"
)

const (
	defaultTitanConcurrency = 8
	cohereMaxBatchSize      = 96
)

// EmbedRequest represents a normalized batch embeddings invocation.
type EmbedRequest struct {
	BedrockModelID string
	Texts          []string
	Dimensions     *int
	InputType      string // Optional Cohere input_type override (defaults to "search_document")
}

// EmbedResult contains ordered float32 vectors and total prompt token count.
type EmbedResult struct {
	Vectors     [][]float32
	InputTokens int64
}

// Service executes embeddings requests over BedrockRuntime InvokeModel.
type Service struct {
	runtime          awsclient.BedrockRuntimeAPI
	titanConcurrency int
}

// NewService creates a new Bedrock embeddings Service.
func NewService(runtime awsclient.BedrockRuntimeAPI, titanConcurrency int) *Service {
	if titanConcurrency <= 0 {
		titanConcurrency = defaultTitanConcurrency
	}
	return &Service{
		runtime:          runtime,
		titanConcurrency: titanConcurrency,
	}
}

// Embed routes the embeddings request to the appropriate Bedrock model family adapter.
func (s *Service) Embed(ctx context.Context, req EmbedRequest) (*EmbedResult, *openai.APIError) {
	lowerID := strings.ToLower(strings.TrimSpace(req.BedrockModelID))
	switch {
	case strings.Contains(lowerID, "titan-embed-text"):
		isV2 := strings.Contains(lowerID, "titan-embed-text-v2")
		return s.embedTitan(ctx, req, isV2)
	case strings.Contains(lowerID, "cohere.embed"):
		isV4 := strings.Contains(lowerID, "embed-v4")
		return s.embedCohere(ctx, req, isV4)
	default:
		return nil, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"unsupported_embedding_model",
			"model",
			fmt.Sprintf("Model %q is not a supported Bedrock text embedding model (supported families: amazon.titan-embed-text-*, cohere.embed-*).", req.BedrockModelID),
		)
	}
}

type titanRequest struct {
	InputText  string `json:"inputText"`
	Dimensions *int   `json:"dimensions,omitempty"`
	Normalize  *bool  `json:"normalize,omitempty"`
}

type titanResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int64     `json:"inputTextTokenCount"`
}

func (s *Service) embedTitan(ctx context.Context, req EmbedRequest, isV2 bool) (*EmbedResult, *openai.APIError) {
	vectors := make([][]float32, len(req.Texts))
	var totalTokens atomic.Int64

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.titanConcurrency)

	for i, text := range req.Texts {
		idx := i
		txt := text
		g.Go(func() error {
			payload := titanRequest{InputText: txt}
			if isV2 {
				norm := true
				payload.Normalize = &norm
				if req.Dimensions != nil && *req.Dimensions > 0 {
					payload.Dimensions = req.Dimensions
				}
			}
			bodyBytes, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			out, err := s.runtime.InvokeModel(gCtx, &bedrockruntime.InvokeModelInput{
				ModelId:     aws.String(req.BedrockModelID),
				ContentType: aws.String("application/json"),
				Accept:      aws.String("application/json"),
				Body:        bodyBytes,
			})
			if err != nil {
				return err
			}
			var parsed titanResponse
			if err := json.Unmarshal(out.Body, &parsed); err != nil {
				return fmt.Errorf("decode Titan embedding response: %w", err)
			}
			vectors[idx] = parsed.Embedding
			totalTokens.Add(parsed.InputTextTokenCount)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		if meta := openai.RequestMetaFromContext(ctx); meta != nil {
			meta.SetBedrockError(openai.BedrockErrorCode(err))
		}
		return nil, openai.FromBedrockError(err)
	}

	return &EmbedResult{
		Vectors:     vectors,
		InputTokens: totalTokens.Load(),
	}, nil
}

type cohereRequest struct {
	Texts           []string `json:"texts"`
	InputType       string   `json:"input_type"`
	EmbeddingTypes  []string `json:"embedding_types,omitempty"`
	OutputDimension *int     `json:"output_dimension,omitempty"`
}

type cohereResponse struct {
	Embeddings json.RawMessage `json:"embeddings"`
	Meta       *struct {
		BilledUnits *struct {
			InputTokens int64 `json:"input_tokens"`
		} `json:"billed_units"`
	} `json:"meta"`
}

func (s *Service) embedCohere(ctx context.Context, req EmbedRequest, isV4 bool) (*EmbedResult, *openai.APIError) {
	inputType := strings.TrimSpace(req.InputType)
	if inputType == "" {
		inputType = "search_document"
	}

	vectors := make([][]float32, 0, len(req.Texts))
	var totalTokens int64

	for start := 0; start < len(req.Texts); start += cohereMaxBatchSize {
		end := min(start+cohereMaxBatchSize, len(req.Texts))
		chunk := req.Texts[start:end]

		payload := cohereRequest{
			Texts:          chunk,
			InputType:      inputType,
			EmbeddingTypes: []string{"float"},
		}
		if isV4 && req.Dimensions != nil && *req.Dimensions > 0 {
			payload.OutputDimension = req.Dimensions
		}
		bodyBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, openai.FromBedrockError(err)
		}

		out, err := s.runtime.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String(req.BedrockModelID),
			ContentType: aws.String("application/json"),
			Accept:      aws.String("application/json"),
			Body:        bodyBytes,
		})
		if err != nil {
			if meta := openai.RequestMetaFromContext(ctx); meta != nil {
				meta.SetBedrockError(openai.BedrockErrorCode(err))
			}
			return nil, openai.FromBedrockError(err)
		}

		var parsed cohereResponse
		if err := json.Unmarshal(out.Body, &parsed); err != nil {
			return nil, openai.NewError(http.StatusBadGateway, openai.TypeServer, "upstream_decode_error", fmt.Sprintf("decode Cohere response: %v", err))
		}

		batchVecs, err := decodeCohereEmbeddingsField(parsed.Embeddings)
		if err != nil {
			return nil, openai.NewError(http.StatusBadGateway, openai.TypeServer, "upstream_decode_error", err.Error())
		}
		vectors = append(vectors, batchVecs...)

		if parsed.Meta != nil && parsed.Meta.BilledUnits != nil {
			totalTokens += parsed.Meta.BilledUnits.InputTokens
		}
	}

	return &EmbedResult{
		Vectors:     vectors,
		InputTokens: totalTokens,
	}, nil
}

func decodeCohereEmbeddingsField(raw json.RawMessage) ([][]float32, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return nil, fmt.Errorf("missing embeddings field in Cohere response")
	}
	if strings.HasPrefix(trimmed, "{") {
		var byType struct {
			Float [][]float32 `json:"float"`
		}
		if err := json.Unmarshal(raw, &byType); err != nil {
			return nil, fmt.Errorf("decode Cohere embeddings.float: %w", err)
		}
		return byType.Float, nil
	}
	var flat [][]float32
	if err := json.Unmarshal(raw, &flat); err != nil {
		return nil, fmt.Errorf("decode Cohere embeddings array: %w", err)
	}
	return flat, nil
}
