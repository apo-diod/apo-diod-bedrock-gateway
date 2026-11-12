package models

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	"github.com/aws/smithy-go"
)

type fakeBedrockControl struct {
	calls     atomic.Int32
	denyError bool
}

func (f *fakeBedrockControl) ListFoundationModels(_ context.Context, _ *bedrock.ListFoundationModelsInput, _ ...func(*bedrock.Options)) (*bedrock.ListFoundationModelsOutput, error) {
	f.calls.Add(1)
	if f.denyError {
		return nil, &smithy.GenericAPIError{
			Code:    "AccessDeniedException",
			Message: "User is not authorized to perform bedrock:ListFoundationModels",
		}
	}
	return &bedrock.ListFoundationModelsOutput{
		ModelSummaries: []bedrocktypes.FoundationModelSummary{
			{
				ModelId:      aws.String("amazon.titan-embed-text-v2:0"),
				ProviderName: aws.String("Amazon"),
			},
			{
				ModelId:      aws.String("cohere.embed-english-v3"),
				ProviderName: aws.String("Cohere"),
			},
		},
	}, nil
}

func (f *fakeBedrockControl) ListInferenceProfiles(_ context.Context, _ *bedrock.ListInferenceProfilesInput, _ ...func(*bedrock.Options)) (*bedrock.ListInferenceProfilesOutput, error) {
	f.calls.Add(1)
	if f.denyError {
		return nil, &smithy.GenericAPIError{
			Code:    "AccessDeniedException",
			Message: "User is not authorized to perform bedrock:ListInferenceProfiles",
		}
	}
	return &bedrock.ListInferenceProfilesOutput{
		InferenceProfileSummaries: []bedrocktypes.InferenceProfileSummary{
			{
				InferenceProfileId: aws.String("us.anthropic.claude-sonnet-4-20250514-v1:0"),
				Status:             bedrocktypes.InferenceProfileStatusActive,
			},
		},
	}, nil
}

func TestModelsRegistryAndHandlers_DiscoveryCachingTenantFilteringAnd403Fallback(t *testing.T) {
	ctrl := &fakeBedrockControl{}
	var logBuf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&logBuf, nil))

	mcfg := config.ModelsConfig{
		DefaultModel:          "us.anthropic.claude-sonnet-4-20250514-v1:0",
		DefaultEmbeddingModel: "amazon.titan-embed-text-v2:0",
		AllowPassthrough:      true,
		DiscoverBedrockModels: true,
		DiscoveryCacheTTL:     config.Duration{Duration: 5 * time.Minute},
		Aliases: map[string]string{
			"gpt-4o":      "us.anthropic.claude-sonnet-4-20250514-v1:0",
			"gpt-4o-mini": "us.amazon.nova-lite-v1:0",
		},
	}

	reg := models.NewRegistry(mcfg, ctrl, logger)
	h := NewHandler(reg)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", h.List)
	mux.HandleFunc("GET /v1/models/{model...}", h.Get)

	alpha := &auth.Tenant{Name: "alpha", AllowedModels: []string{"*"}}
	restricted := &auth.Tenant{Name: "restricted", AllowedModels: []string{"us.anthropic.*"}}

	// 1. Alpha sees aliases + discovered Cohere model, cached across calls
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req = req.WithContext(auth.WithTenant(req.Context(), alpha))
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rr.Code)
		}
		var list ListResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &list); err != nil {
			t.Fatalf("decode list: %v", err)
		}
		if len(list.Data) < 4 {
			t.Fatalf("expected merged static + discovered models, got %+v", list.Data)
		}
	}
	if ctrl.calls.Load() != 2 { // 1 ListInferenceProfiles + 1 ListFoundationModels, second HTTP request served from TTL cache
		t.Errorf("expected 2 cached AWS control-plane calls, got %d", ctrl.calls.Load())
	}

	// 2. Restricted tenant only sees gpt-4o (whose target matches us.anthropic.*) and us.anthropic.claude-sonnet-4-20250514-v1:0
	reqRest := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	reqRest = reqRest.WithContext(auth.WithTenant(reqRest.Context(), restricted))
	rrRest := httptest.NewRecorder()
	mux.ServeHTTP(rrRest, reqRest)
	var restList ListResponse
	_ = json.Unmarshal(rrRest.Body.Bytes(), &restList)
	if len(restList.Data) != 2 {
		t.Fatalf("expected exactly 2 models for restricted tenant, got %+v", restList.Data)
	}

	// Restricted tenant requesting disallowed model via GET /v1/models/{model} gets 404
	reqDeniedGet := httptest.NewRequest(http.MethodGet, "/v1/models/cohere.embed-english-v3", nil)
	reqDeniedGet = reqDeniedGet.WithContext(auth.WithTenant(reqDeniedGet.Context(), restricted))
	rrDeniedGet := httptest.NewRecorder()
	mux.ServeHTTP(rrDeniedGet, reqDeniedGet)
	if rrDeniedGet.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for disallowed model lookup, got %d", rrDeniedGet.Code)
	}

	// Restricted tenant requesting allowed Bedrock model ID with colon gets 200
	reqAllowedGet := httptest.NewRequest(http.MethodGet, "/v1/models/us.anthropic.claude-sonnet-4-20250514-v1:0", nil)
	reqAllowedGet = reqAllowedGet.WithContext(auth.WithTenant(reqAllowedGet.Context(), restricted))
	rrAllowedGet := httptest.NewRecorder()
	mux.ServeHTTP(rrAllowedGet, reqAllowedGet)
	if rrAllowedGet.Code != http.StatusOK {
		t.Fatalf("expected 200 for allowed model lookup, got %d", rrAllowedGet.Code)
	}

	// ResolveChatModel rejects disallowed model with 403
	if _, _, apiErr := reg.ResolveChatModel(restricted, "gpt-4o-mini"); apiErr == nil || apiErr.Status != http.StatusForbidden {
		t.Fatalf("expected 403 on disallowed chat model, got %+v", apiErr)
	}

	// 3. AccessDeniedException graceful fallback on a fresh registry
	denyCtrl := &fakeBedrockControl{denyError: true}
	regDeny := models.NewRegistry(mcfg, denyCtrl, logger)
	fallbackModels := regDeny.ListForTenant(context.Background(), alpha)
	if len(fallbackModels) == 0 {
		t.Fatal("expected static configured models when Bedrock List* returns AccessDeniedException")
	}
	if !bytes.Contains(logBuf.Bytes(), []byte("AccessDeniedException")) {
		t.Errorf("expected structured warning mentioning AccessDeniedException, got: %s", logBuf.String())
	}
}
