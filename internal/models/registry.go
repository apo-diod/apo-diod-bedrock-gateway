package models

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/awsclient"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	"github.com/aws/smithy-go"
	"golang.org/x/sync/singleflight"
)

// ModelInfo represents an OpenAI-compatible model entry enriched with internal target mapping.
type ModelInfo struct {
	ID              string `json:"id"`
	Object          string `json:"object"`
	Created         int64  `json:"created"`
	OwnedBy         string `json:"owned_by"`
	TargetBedrockID string `json:"-"`
}

// Registry resolves model aliases, enforces passthrough and tenant allowlists,
// and caches optional AWS Bedrock control-plane model discovery.
type Registry struct {
	cfg       config.ModelsConfig
	control   awsclient.BedrockControlAPI
	logger    *slog.Logger
	createdAt int64

	sf         singleflight.Group
	mu         sync.RWMutex
	discovered []ModelInfo
	cacheExp   time.Time
}

// NewRegistry creates a new model Registry.
func NewRegistry(cfg config.ModelsConfig, control awsclient.BedrockControlAPI, logger *slog.Logger) *Registry {
	if logger == nil {
		logger = slog.Default()
	}
	return &Registry{
		cfg:       cfg,
		control:   control,
		logger:    logger,
		createdAt: 1700000000, // Deterministic epoch baseline when AWS omits CreatedAt
	}
}

// ResolveChatModel resolves and authorizes a requested model for chat or responses inference.
func (r *Registry) ResolveChatModel(tenant *auth.Tenant, requested string) (string, string, *openai.APIError) {
	fallback := r.cfg.DefaultModel
	if tenant != nil && tenant.DefaultModel != "" {
		fallback = tenant.DefaultModel
	}
	return r.resolveModel(tenant, requested, fallback)
}

// ResolveEmbeddingModel resolves and authorizes a requested model for embeddings inference.
func (r *Registry) ResolveEmbeddingModel(tenant *auth.Tenant, requested string) (string, string, *openai.APIError) {
	return r.resolveModel(tenant, requested, r.cfg.DefaultEmbeddingModel)
}

func (r *Registry) resolveModel(tenant *auth.Tenant, requested, fallback string) (string, string, *openai.APIError) {
	reqModel := strings.TrimSpace(requested)
	if reqModel == "" {
		reqModel = strings.TrimSpace(fallback)
	}
	if reqModel == "" {
		return "", "", openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"missing_model",
			"model",
			"You must provide a model parameter.",
		)
	}

	var resolved string
	if target, isAlias := r.cfg.Aliases[reqModel]; isAlias {
		resolved = target
	} else if r.cfg.AllowPassthrough {
		resolved = reqModel
	} else {
		return reqModel, "", openai.NewParamError(
			http.StatusNotFound,
			openai.TypeNotFound,
			"model_not_found",
			"model",
			fmt.Sprintf("The model %q does not exist in configured aliases and passthrough is disabled.", reqModel),
		)
	}

	if tenant != nil && !tenant.IsModelAllowed(reqModel, resolved) {
		return reqModel, "", openai.NewParamError(
			http.StatusForbidden,
			openai.TypePermission,
			"model_not_allowed",
			"model",
			fmt.Sprintf("Tenant %q does not have access to model %q.", tenant.Name, reqModel),
		)
	}

	return reqModel, resolved, nil
}

// ListForTenant returns the deduplicated, sorted list of models accessible to tenant.
func (r *Registry) ListForTenant(ctx context.Context, tenant *auth.Tenant) []ModelInfo {
	byID := make(map[string]ModelInfo)

	add := func(info ModelInfo) {
		if info.ID == "" {
			return
		}
		if tenant != nil && !tenant.IsModelAllowed(info.ID, info.TargetBedrockID) {
			return
		}
		if _, exists := byID[info.ID]; !exists {
			byID[info.ID] = info
		}
	}

	// 1. Configured aliases
	for alias, target := range r.cfg.Aliases {
		add(ModelInfo{
			ID:              alias,
			Object:          "model",
			Created:         r.createdAt,
			OwnedBy:         inferOwner(target),
			TargetBedrockID: target,
		})
		if r.cfg.AllowPassthrough {
			add(ModelInfo{
				ID:              target,
				Object:          "model",
				Created:         r.createdAt,
				OwnedBy:         inferOwner(target),
				TargetBedrockID: target,
			})
		}
	}

	// 2. Configured defaults (when passthrough is allowed)
	if r.cfg.AllowPassthrough {
		for _, def := range []string{r.cfg.DefaultModel, r.cfg.DefaultEmbeddingModel} {
			def = strings.TrimSpace(def)
			if def != "" {
				add(ModelInfo{
					ID:              def,
					Object:          "model",
					Created:         r.createdAt,
					OwnedBy:         inferOwner(def),
					TargetBedrockID: def,
				})
			}
		}
	}

	// 3. Discovered Bedrock foundation models & inference profiles (when enabled and passthrough is allowed)
	if r.cfg.DiscoverBedrockModels && r.cfg.AllowPassthrough && r.control != nil {
		for _, dm := range r.getDiscoveredModels(ctx) {
			add(dm)
		}
	}

	out := make([]ModelInfo, 0, len(byID))
	for _, m := range byID {
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ID < out[j].ID
	})
	return out
}

// GetForTenant retrieves a single model visible to tenant or returns a 404 OpenAI error.
func (r *Registry) GetForTenant(ctx context.Context, tenant *auth.Tenant, modelID string) (*ModelInfo, *openai.APIError) {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		return nil, openai.NewError(http.StatusNotFound, openai.TypeNotFound, "model_not_found", "Model not found.")
	}
	for _, m := range r.ListForTenant(ctx, tenant) {
		if m.ID == modelID {
			found := m
			return &found, nil
		}
	}
	return nil, openai.NewParamError(
		http.StatusNotFound,
		openai.TypeNotFound,
		"model_not_found",
		"model",
		fmt.Sprintf("The model %q does not exist or you do not have access to it.", modelID),
	)
}

func (r *Registry) getDiscoveredModels(ctx context.Context) []ModelInfo {
	now := time.Now()
	r.mu.RLock()
	if now.Before(r.cacheExp) {
		cached := append([]ModelInfo(nil), r.discovered...)
		r.mu.RUnlock()
		return cached
	}
	r.mu.RUnlock()

	val, _, _ := r.sf.Do("discover", func() (any, error) {
		// Double-check cache inside singleflight
		r.mu.RLock()
		if time.Now().Before(r.cacheExp) {
			cached := append([]ModelInfo(nil), r.discovered...)
			r.mu.RUnlock()
			return cached, nil
		}
		prevCached := append([]ModelInfo(nil), r.discovered...)
		r.mu.RUnlock()

		discCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		fetched, ok := r.fetchBedrockCatalog(discCtx)
		ttl := r.cfg.DiscoveryCacheTTL.Duration
		if ttl <= 0 {
			ttl = 10 * time.Minute
		}
		if !ok && len(fetched) == 0 && len(prevCached) > 0 {
			// Retain last good discovery snapshot on transient control-plane failure
			fetched = prevCached
		}

		r.mu.Lock()
		r.discovered = fetched
		r.cacheExp = time.Now().Add(ttl)
		r.mu.Unlock()
		return fetched, nil
	})

	if list, ok := val.([]ModelInfo); ok {
		return list
	}
	return nil
}

func (r *Registry) fetchBedrockCatalog(ctx context.Context) ([]ModelInfo, bool) {
	var results []ModelInfo
	anySuccess := false

	// 1. Inference Profiles (cross-region & application profiles)
	var nextToken *string
	for page := 0; page < 5; page++ {
		out, err := r.control.ListInferenceProfiles(ctx, &bedrock.ListInferenceProfilesInput{
			NextToken: nextToken,
		})
		if err != nil {
			r.logDiscoveryWarning("ListInferenceProfiles", err)
			break
		}
		anySuccess = true
		for _, prof := range out.InferenceProfileSummaries {
			if prof.Status != "" && prof.Status != bedrocktypes.InferenceProfileStatusActive {
				continue
			}
			id := aws.ToString(prof.InferenceProfileId)
			if id == "" {
				continue
			}
			created := r.createdAt
			if prof.CreatedAt != nil {
				created = prof.CreatedAt.Unix()
			}
			results = append(results, ModelInfo{
				ID:              id,
				Object:          "model",
				Created:         created,
				OwnedBy:         inferOwner(id),
				TargetBedrockID: id,
			})
		}
		if out.NextToken == nil || *out.NextToken == "" {
			break
		}
		nextToken = out.NextToken
	}

	// 2. Foundation Models
	fmOut, err := r.control.ListFoundationModels(ctx, &bedrock.ListFoundationModelsInput{})
	if err != nil {
		r.logDiscoveryWarning("ListFoundationModels", err)
	} else {
		anySuccess = true
		for _, fm := range fmOut.ModelSummaries {
			if fm.ModelLifecycle != nil && fm.ModelLifecycle.Status != "" &&
				fm.ModelLifecycle.Status != bedrocktypes.FoundationModelLifecycleStatusActive {
				continue
			}
			id := aws.ToString(fm.ModelId)
			if id == "" {
				continue
			}
			owner := strings.ToLower(strings.TrimSpace(aws.ToString(fm.ProviderName)))
			if owner == "" {
				owner = inferOwner(id)
			}
			results = append(results, ModelInfo{
				ID:              id,
				Object:          "model",
				Created:         r.createdAt,
				OwnedBy:         owner,
				TargetBedrockID: id,
			})
		}
	}

	return results, anySuccess
}

func (r *Registry) logDiscoveryWarning(op string, err error) {
	var smithyErr smithy.APIError
	code := "unknown"
	if errors.As(err, &smithyErr) {
		code = smithyErr.ErrorCode()
	}
	r.logger.Warn("bedrock control-plane model discovery unavailable; falling back to static configured models",
		"operation", op,
		"aws_error_code", code,
		"error", err.Error(),
	)
}

func inferOwner(bedrockID string) string {
	lower := strings.ToLower(bedrockID)
	// Strip cross-region prefix if present (us., eu., apac., global.)
	for _, pfx := range []string{"us.", "eu.", "apac.", "global."} {
		if strings.HasPrefix(lower, pfx) {
			lower = strings.TrimPrefix(lower, pfx)
			break
		}
	}
	switch {
	case strings.HasPrefix(lower, "anthropic.") || strings.Contains(lower, ":foundation-model/anthropic."):
		return "anthropic"
	case strings.HasPrefix(lower, "amazon.") || strings.Contains(lower, ":foundation-model/amazon."):
		return "amazon"
	case strings.HasPrefix(lower, "cohere.") || strings.Contains(lower, ":foundation-model/cohere."):
		return "cohere"
	case strings.HasPrefix(lower, "meta.") || strings.Contains(lower, ":foundation-model/meta."):
		return "meta"
	case strings.HasPrefix(lower, "mistral.") || strings.Contains(lower, ":foundation-model/mistral."):
		return "mistral"
	case strings.HasPrefix(lower, "deepseek.") || strings.Contains(lower, ":foundation-model/deepseek."):
		return "deepseek"
	default:
		return "aws-bedrock"
	}
}
