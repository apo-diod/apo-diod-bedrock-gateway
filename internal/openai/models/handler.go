package models

import (
	"encoding/json"
	"net/http"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
)

// ListResponse is the OpenAI GET /v1/models JSON envelope.
type ListResponse struct {
	Object string             `json:"object"`
	Data   []models.ModelInfo `json:"data"`
}

// Handler serves GET /v1/models and GET /v1/models/{model}.
type Handler struct {
	registry *models.Registry
}

// NewHandler creates a new OpenAI models HTTP handler.
func NewHandler(registry *models.Registry) *Handler {
	return &Handler{registry: registry}
}

// List handles GET /v1/models.
func (h *Handler) List(w http.ResponseWriter, r *http.Request) {
	tenant, _ := auth.TenantFromContext(r.Context())
	items := h.registry.ListForTenant(r.Context(), tenant)
	if items == nil {
		items = []models.ModelInfo{}
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(ListResponse{
		Object: "list",
		Data:   items,
	})
}

// Get handles GET /v1/models/{model...}.
func (h *Handler) Get(w http.ResponseWriter, r *http.Request) {
	tenant, _ := auth.TenantFromContext(r.Context())
	modelID := r.PathValue("model")
	info, apiErr := h.registry.GetForTenant(r.Context(), tenant, modelID)
	if apiErr != nil {
		openai.WriteAPIError(w, apiErr)
		return
	}
	if meta := openai.RequestMetaFromContext(r.Context()); meta != nil {
		meta.SetModel(info.ID, info.TargetBedrockID)
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(info)
}
