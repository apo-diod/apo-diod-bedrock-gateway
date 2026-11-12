package server

import (
	"context"
	"log/slog"
	"net/http"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/auth"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/awsclient"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/bedrockconv"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/config"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/embeddings"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/metrics"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/models"
	openaichat "github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai/chat"
	openaiembed "github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai/embeddings"
	openaimodels "github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai/models"
	openairesponses "github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai/responses"
	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/store"
)

// WireInput bundles all runtime subsystems needed to assemble the complete OpenAI gateway.
type WireInput struct {
	Config          *config.Config
	RuntimeClient   bedrockconv.RuntimeClient
	RawRuntime      awsclient.BedrockRuntimeAPI
	ControlClient   awsclient.BedrockControlAPI
	AWSReadyCheck   func(ctx context.Context) error
	Store           *store.Store
	Logger          *slog.Logger
	Metrics         *metrics.Collector
}

// BuildGateway wires authentication, model registry, canonical translator, embeddings,
// SQLite responses store, Prometheus metrics, and all OpenAI HTTP endpoints.
func BuildGateway(in WireInput) *GatewayServer {
	cfg := in.Config
	logger := in.Logger
	if logger == nil {
		logger = NewLogger(cfg.Server.LogLevel, nil)
	}
	mc := in.Metrics
	if mc == nil {
		mc = metrics.New()
	}

	authn := auth.NewAuthenticator(cfg.Tenants)
	reg := models.NewRegistry(cfg.Models, in.ControlClient, logger)

	rtClient := in.RuntimeClient
	if rtClient == nil && in.RawRuntime != nil {
		rtClient = bedrockconv.NewSDKRuntimeClient(in.RawRuntime)
	}

	translator := bedrockconv.NewTranslator(rtClient, bedrockconv.Options{
		AllowRemoteMediaURLs: cfg.Server.AllowRemoteMediaURLs,
	})
	embedSvc := embeddings.NewService(in.RawRuntime, 0)

	chatH := openaichat.NewHandler(reg, translator, mc, cfg.Models.EnableAutoPromptCache)
	respH := openairesponses.NewHandler(reg, translator, in.Store, mc, cfg.Models.EnableAutoPromptCache)
	embedH := openaiembed.NewHandler(reg, embedSvc)
	modelsH := openaimodels.NewHandler(reg)

	var checks []ReadinessCheck
	if in.Store != nil {
		checks = append(checks, ReadinessCheck{
			Name:  "sqlite",
			Check: in.Store.Ping,
		})
	}
	if in.AWSReadyCheck != nil {
		checks = append(checks, ReadinessCheck{
			Name:  "aws_credentials",
			Check: in.AWSReadyCheck,
		})
	}

	return New(Dependencies{
		Config:                     cfg,
		Authenticator:              authn,
		Metrics:                    mc,
		Logger:                     logger,
		ReadinessChecks:            checks,
		ChatCompletionsHandler:     chatH,
		ResponsesCreateHandler:     http.HandlerFunc(respH.Create),
		ResponsesGetHandler:        http.HandlerFunc(respH.Get),
		ResponsesDeleteHandler:     http.HandlerFunc(respH.Delete),
		ResponsesInputItemsHandler: http.HandlerFunc(respH.ListInputItems),
		EmbeddingsHandler:          embedH,
		ModelsListHandler:          http.HandlerFunc(modelsH.List),
		ModelsGetHandler:           http.HandlerFunc(modelsH.Get),
	})
}
