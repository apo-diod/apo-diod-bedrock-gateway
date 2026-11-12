package bedrockconv

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/apo-diod/apo-diod-bedrock-gateway/internal/openai"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithydoc "github.com/aws/smithy-go/document"
)

const syntheticJSONSchemaToolName = "__structured_json_output"

// Options configures Translator media policy and HTTP client injection.
type Options struct {
	AllowRemoteMediaURLs bool
	MediaHTTPClient      *http.Client
}

// Translator converts between canonical gateway requests/responses and AWS Bedrock Converse / ConverseStream.
type Translator struct {
	runtime RuntimeClient
	opts    Options
}

// NewTranslator creates a new Bedrock Converse Translator.
func NewTranslator(runtime RuntimeClient, opts Options) *Translator {
	if opts.MediaHTTPClient == nil && opts.AllowRemoteMediaURLs {
		opts.MediaHTTPClient = NewSSRFSafeHTTPClient(0)
	}
	return &Translator{
		runtime: runtime,
		opts:    opts,
	}
}

type buildPlan struct {
	syntheticSchemaTool bool
}

// BuildConverseInput translates a canonical Request into a Bedrock ConverseInput.
func (t *Translator) BuildConverseInput(ctx context.Context, req Request) (*bedrockruntime.ConverseInput, *buildPlan, *openai.APIError) {
	if strings.TrimSpace(req.BedrockModelID) == "" {
		return nil, nil, openai.NewParamError(http.StatusBadRequest, openai.TypeInvalidRequest, "missing_model", "model", "model must not be empty")
	}

	plan := &buildPlan{}
	canCache := supportsPromptCache(req.BedrockModelID)
	isClaude := isAnthropicClaude(req.BedrockModelID)
	isNova := isAmazonNova(req.BedrockModelID)
	claudeThinkingActive := isClaude && req.Reasoning != nil && (req.Reasoning.Effort != "" || req.Reasoning.BudgetTokens > 0)
	disableTools := strings.EqualFold(req.ToolChoice.Mode, "none")

	var sysBlocks []types.SystemContentBlock
	hasExplicitSysCache := false

	// 1. Extract system & developer instructions into SystemContentBlock slice.
	for _, msg := range req.Messages {
		if msg.Role != RoleSystem && msg.Role != RoleDeveloper {
			continue
		}
		for _, part := range msg.Parts {
			if part.Type == PartText && strings.TrimSpace(part.Text) != "" {
				sysBlocks = append(sysBlocks, &types.SystemContentBlockMemberText{Value: part.Text})
				if canCache && part.CacheControl {
					sysBlocks = append(sysBlocks, newSysCachePoint())
					hasExplicitSysCache = true
				}
			}
		}
		if canCache && msg.CacheControl && len(sysBlocks) > 0 {
			if _, already := sysBlocks[len(sysBlocks)-1].(*types.SystemContentBlockMemberCachePoint); !already {
				sysBlocks = append(sysBlocks, newSysCachePoint())
				hasExplicitSysCache = true
			}
		}
	}

	// 2. Structured output (response_format) handling.
	effectiveTools := append([]ToolDef(nil), req.Tools...)
	effectiveChoice := req.ToolChoice

	switch req.ResponseFormat.Type {
	case "", "text":
	case "json_object":
		sysBlocks = append(sysBlocks, &types.SystemContentBlockMemberText{
			Value: "You must respond strictly with a valid JSON object. Do not wrap the JSON in markdown code fences or add explanatory text.",
		})
	case "json_schema":
		schema := req.ResponseFormat.Schema
		if len(schema) == 0 {
			schema = map[string]any{"type": "object"}
		}
		// Use synthetic forced tool extraction when no user tools are present, tools are not disabled,
		// and Claude extended thinking is not active (Bedrock forbids forced tool_choice with Claude thinking).
		if len(effectiveTools) == 0 && !disableTools && !claudeThinkingActive {
			plan.syntheticSchemaTool = true
			desc := req.ResponseFormat.Description
			if desc == "" {
				desc = "Emit the final assistant response strictly adhering to the required JSON schema."
			}
			effectiveTools = []ToolDef{
				{
					Name:        syntheticJSONSchemaToolName,
					Description: desc,
					Parameters:  normalizeToolSchema(schema),
				},
			}
			effectiveChoice = ToolChoice{
				Mode:     "tool",
				ToolName: syntheticJSONSchemaToolName,
			}
		} else {
			schemaBytes, _ := json.Marshal(schema)
			sysBlocks = append(sysBlocks, &types.SystemContentBlockMemberText{
				Value: fmt.Sprintf("Your final text response must be a single valid JSON document conforming strictly to this JSON Schema: %s", string(schemaBytes)),
			})
		}
	default:
		return nil, nil, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"invalid_response_format",
			"response_format",
			fmt.Sprintf("Unsupported response_format.type %q.", req.ResponseFormat.Type),
		)
	}

	if canCache && req.AutoPromptCache && !hasExplicitSysCache && len(sysBlocks) > 0 {
		sysBlocks = append(sysBlocks, newSysCachePoint())
	}

	// 3. Normalize conversation messages: tool coalescing + strict user/assistant alternation.
	historicalToolNames := make(map[string]struct{})
	var convMsgs []types.Message
	docIndex := 0

	appendBlocks := func(role types.ConversationRole, blocks []types.ContentBlock) {
		if len(blocks) == 0 {
			return
		}
		if n := len(convMsgs); n > 0 && convMsgs[n-1].Role == role {
			convMsgs[n-1].Content = append(convMsgs[n-1].Content, blocks...)
			return
		}
		convMsgs = append(convMsgs, types.Message{
			Role:    role,
			Content: blocks,
		})
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case RoleSystem, RoleDeveloper:
			continue

		case RoleTool:
			// Coalesce tool outputs into a ConversationRoleUser message containing ToolResultBlocks.
			var blocks []types.ContentBlock
			for _, part := range msg.Parts {
				if part.Type != PartToolResult {
					continue
				}
				if disableTools {
					blocks = append(blocks, &types.ContentBlockMemberText{
						Value: fmt.Sprintf("[Tool result %s]: %s", part.ToolCallID, part.ToolResultText),
					})
					continue
				}
				status := types.ToolResultStatusSuccess
				if part.IsError {
					status = types.ToolResultStatusError
				}
				callID := part.ToolCallID
				if callID == "" {
					callID = "call_unknown"
				}
				blocks = append(blocks, &types.ContentBlockMemberToolResult{
					Value: types.ToolResultBlock{
						ToolUseId: aws.String(callID),
						Status:    status,
						Content: []types.ToolResultContentBlock{
							&types.ToolResultContentBlockMemberText{Value: part.ToolResultText},
						},
					},
				})
				if canCache && part.CacheControl {
					blocks = append(blocks, newMsgCachePoint())
				}
			}
			if canCache && msg.CacheControl && len(blocks) > 0 {
				blocks = ensureTrailingMsgCachePoint(blocks)
			}
			appendBlocks(types.ConversationRoleUser, blocks)

		case RoleUser, RoleAssistant:
			targetRole := types.ConversationRoleUser
			if msg.Role == RoleAssistant {
				targetRole = types.ConversationRoleAssistant
			}
			var blocks []types.ContentBlock
			for _, part := range msg.Parts {
				switch part.Type {
				case PartText:
					if part.Text != "" {
						blocks = append(blocks, &types.ContentBlockMemberText{Value: part.Text})
					}
				case PartImage:
					mt, raw, apiErr := resolveMediaBytes(ctx, part, t.opts.AllowRemoteMediaURLs, t.opts.MediaHTTPClient)
					if apiErr != nil {
						return nil, nil, apiErr
					}
					imgFmt, apiErr := mapImageFormat(mt, raw)
					if apiErr != nil {
						return nil, nil, apiErr
					}
					blocks = append(blocks, &types.ContentBlockMemberImage{
						Value: types.ImageBlock{
							Format: imgFmt,
							Source: &types.ImageSourceMemberBytes{Value: raw},
						},
					})
				case PartDocument:
					mt, raw, apiErr := resolveMediaBytes(ctx, part, t.opts.AllowRemoteMediaURLs, t.opts.MediaHTTPClient)
					if apiErr != nil {
						return nil, nil, apiErr
					}
					docFmt, apiErr := mapDocumentFormat(mt, part.Name)
					if apiErr != nil {
						return nil, nil, apiErr
					}
					docName := sanitizeBedrockDocName(part.Name, docIndex)
					docIndex++
					blocks = append(blocks, &types.ContentBlockMemberDocument{
						Value: types.DocumentBlock{
							Format: docFmt,
							Name:   aws.String(docName),
							Source: &types.DocumentSourceMemberBytes{Value: raw},
						},
					})
				case PartReasoning:
					if targetRole != types.ConversationRoleAssistant {
						continue
					}
					if len(part.RedactedContent) > 0 {
						blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
							Value: &types.ReasoningContentBlockMemberRedactedContent{
								Value: part.RedactedContent,
							},
						})
					} else if part.ReasoningText != "" {
						// Claude requires a valid cryptographic signature when passing historical ReasoningContent back.
						if isClaude && part.Signature == "" {
							continue
						}
						rtb := types.ReasoningTextBlock{
							Text: aws.String(part.ReasoningText),
						}
						if part.Signature != "" {
							rtb.Signature = aws.String(part.Signature)
						}
						blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
							Value: &types.ReasoningContentBlockMemberReasoningText{Value: rtb},
						})
					}
				case PartToolUse:
					if targetRole != types.ConversationRoleAssistant {
						continue
					}
					if disableTools {
						blocks = append(blocks, &types.ContentBlockMemberText{
							Value: fmt.Sprintf("[Called tool %s(%s) id=%s]", part.ToolName, part.ToolArgsJSON, part.ToolCallID),
						})
						continue
					}
					callID := part.ToolCallID
					if callID == "" {
						callID = openai.NewID("call_")
					}
					toolName := part.ToolName
					if toolName == "" {
						toolName = "unknown_tool"
					}
					historicalToolNames[toolName] = struct{}{}
					blocks = append(blocks, &types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String(callID),
							Name:      aws.String(toolName),
							Input:     parseJSONArgsToDocument(part.ToolArgsJSON),
						},
					})
				case PartToolResult:
					if disableTools {
						blocks = append(blocks, &types.ContentBlockMemberText{
							Value: fmt.Sprintf("[Tool result %s]: %s", part.ToolCallID, part.ToolResultText),
						})
						continue
					}
					status := types.ToolResultStatusSuccess
					if part.IsError {
						status = types.ToolResultStatusError
					}
					blocks = append(blocks, &types.ContentBlockMemberToolResult{
						Value: types.ToolResultBlock{
							ToolUseId: aws.String(part.ToolCallID),
							Status:    status,
							Content: []types.ToolResultContentBlock{
								&types.ToolResultContentBlockMemberText{Value: part.ToolResultText},
							},
						},
					})
				}

				if canCache && part.CacheControl && len(blocks) > 0 {
					blocks = ensureTrailingMsgCachePoint(blocks)
				}
			}
			if canCache && msg.CacheControl && len(blocks) > 0 {
				blocks = ensureTrailingMsgCachePoint(blocks)
			}
			appendBlocks(targetRole, blocks)

		default:
			return nil, nil, openai.NewParamError(
				http.StatusBadRequest,
				openai.TypeInvalidRequest,
				"invalid_role",
				"messages",
				fmt.Sprintf("Unsupported message role %q.", msg.Role),
			)
		}
	}

	if len(convMsgs) == 0 {
		return nil, nil, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"empty_messages",
			"messages",
			"Request must include at least one non-empty user message.",
		)
	}
	if convMsgs[0].Role != types.ConversationRoleUser {
		// Prepend a minimal user turn if caller began conversation history with an assistant prefill.
		convMsgs = append([]types.Message{
			{
				Role: types.ConversationRoleUser,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "Continue."},
				},
			},
		}, convMsgs...)
	}

	// 4. Build ToolConfig (including stubs for historical ToolUse blocks if caller omitted tool specs on follow-up turns).
	var toolCfg *types.ToolConfiguration
	if !disableTools && (len(effectiveTools) > 0 || len(historicalToolNames) > 0) {
		defined := make(map[string]struct{}, len(effectiveTools))
		var brTools []types.Tool
		hasExplicitToolCache := false

		for _, td := range effectiveTools {
			name := strings.TrimSpace(td.Name)
			if name == "" {
				continue
			}
			defined[name] = struct{}{}
			spec := types.ToolSpecification{
				Name: aws.String(name),
				InputSchema: &types.ToolInputSchemaMemberJson{
					Value: document.NewLazyDocument(normalizeToolSchema(td.Parameters)),
				},
			}
			if strings.TrimSpace(td.Description) != "" {
				spec.Description = aws.String(td.Description)
			}
			brTools = append(brTools, &types.ToolMemberToolSpec{Value: spec})
			if canCache && td.CacheControl {
				brTools = append(brTools, newToolCachePoint())
				hasExplicitToolCache = true
			}
		}

		for histName := range historicalToolNames {
			if _, ok := defined[histName]; !ok {
				brTools = append(brTools, &types.ToolMemberToolSpec{
					Value: types.ToolSpecification{
						Name: aws.String(histName),
						InputSchema: &types.ToolInputSchemaMemberJson{
							Value: document.NewLazyDocument(map[string]any{"type": "object"}),
						},
					},
				})
			}
		}

		if canCache && req.AutoPromptCache && !hasExplicitToolCache && len(brTools) > 0 {
			brTools = append(brTools, newToolCachePoint())
		}

		if len(brTools) > 0 {
			toolCfg = &types.ToolConfiguration{Tools: brTools}
			switch strings.ToLower(strings.TrimSpace(effectiveChoice.Mode)) {
			case "", "auto":
				toolCfg.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
			case "required":
				if !claudeThinkingActive {
					toolCfg.ToolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
				}
			case "tool":
				if !claudeThinkingActive && effectiveChoice.ToolName != "" {
					toolCfg.ToolChoice = &types.ToolChoiceMemberTool{
						Value: types.SpecificToolChoice{Name: aws.String(effectiveChoice.ToolName)},
					}
				}
			}
		}
	}

	// 5. InferenceConfig + Claude & Nova extended reasoning parameters.
	infCfg := &types.InferenceConfiguration{}
	hasInf := false

	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		mt := int32(*req.MaxTokens)
		infCfg.MaxTokens = &mt
		hasInf = true
	}
	if req.Temperature != nil {
		temp := *req.Temperature
		infCfg.Temperature = &temp
		hasInf = true
	}
	if req.TopP != nil {
		tp := *req.TopP
		infCfg.TopP = &tp
		hasInf = true
	}
	if len(req.Stop) > 0 {
		infCfg.StopSequences = append([]string(nil), req.Stop...)
		hasInf = true
	}

	var additionalFields map[string]any
	if req.Reasoning != nil && (req.Reasoning.Effort != "" || req.Reasoning.BudgetTokens > 0) {
		effort, budget, apiErr := resolveReasoningBudget(req.Reasoning)
		if apiErr != nil {
			return nil, nil, apiErr
		}
		if isClaude {
			additionalFields = map[string]any{
				"thinking": map[string]any{
					"type":          "enabled",
					"budget_tokens": budget,
				},
			}
			minMaxTokens := int32(budget + 1024)
			if infCfg.MaxTokens == nil || *infCfg.MaxTokens < minMaxTokens {
				defaultMax := int32(budget + 4096)
				if infCfg.MaxTokens != nil && *infCfg.MaxTokens > 0 {
					defaultMax = minMaxTokens
				}
				infCfg.MaxTokens = &defaultMax
			}
			// Anthropic Claude rejects requests that combine thinking with temperature or topP.
			infCfg.Temperature = nil
			infCfg.TopP = nil
			hasInf = true
		} else if isNova {
			additionalFields = map[string]any{
				"reasoningConfig": map[string]any{
					"type":               "enabled",
					"maxReasoningEffort": effort,
				},
			}
			if effort == "high" {
				infCfg.Temperature = nil
				infCfg.TopP = nil
			}
		}
	}

	input := &bedrockruntime.ConverseInput{
		ModelId:    aws.String(req.BedrockModelID),
		Messages:   convMsgs,
		System:     sysBlocks,
		ToolConfig: toolCfg,
	}
	if hasInf {
		input.InferenceConfig = infCfg
	}
	if len(additionalFields) > 0 {
		input.AdditionalModelRequestFields = document.NewLazyDocument(additionalFields)
	}

	return input, plan, nil
}

// Invoke executes a non-streaming Bedrock Converse call and normalizes the response.
func (t *Translator) Invoke(ctx context.Context, req Request) (*Response, *openai.APIError) {
	input, plan, apiErr := t.BuildConverseInput(ctx, req)
	if apiErr != nil {
		return nil, apiErr
	}

	out, err := t.runtime.Converse(ctx, input)
	if err != nil {
		if meta := openai.RequestMetaFromContext(ctx); meta != nil {
			meta.SetBedrockError(openai.BedrockErrorCode(err))
		}
		return nil, openai.FromBedrockError(err)
	}

	var (
		assistantParts []ContentPart
		textBuilder    strings.Builder
		reasonBuilder  strings.Builder
		toolCalls      []ToolCallOutput
	)

	if msgOut, ok := out.Output.(*types.ConverseOutputMemberMessage); ok {
		for _, block := range msgOut.Value.Content {
			switch b := block.(type) {
			case *types.ContentBlockMemberText:
				if b.Value != "" {
					textBuilder.WriteString(b.Value)
					assistantParts = append(assistantParts, ContentPart{
						Type: PartText,
						Text: b.Value,
					})
				}
			case *types.ContentBlockMemberReasoningContent:
				switch rb := b.Value.(type) {
				case *types.ReasoningContentBlockMemberReasoningText:
					txt := aws.ToString(rb.Value.Text)
					sig := aws.ToString(rb.Value.Signature)
					reasonBuilder.WriteString(txt)
					assistantParts = append(assistantParts, ContentPart{
						Type:          PartReasoning,
						ReasoningText: txt,
						Signature:     sig,
					})
				case *types.ReasoningContentBlockMemberRedactedContent:
					assistantParts = append(assistantParts, ContentPart{
						Type:            PartReasoning,
						RedactedContent: append([]byte(nil), rb.Value...),
					})
				}
			case *types.ContentBlockMemberToolUse:
				name := aws.ToString(b.Value.Name)
				callID := aws.ToString(b.Value.ToolUseId)
				argsJSON := marshalDocumentToJSONString(b.Value.Input)
				if plan.syntheticSchemaTool && name == syntheticJSONSchemaToolName {
					textBuilder.WriteString(argsJSON)
					assistantParts = append(assistantParts, ContentPart{
						Type: PartText,
						Text: argsJSON,
					})
					continue
				}
				toolCalls = append(toolCalls, ToolCallOutput{
					ID:        callID,
					Name:      name,
					Arguments: argsJSON,
				})
				assistantParts = append(assistantParts, ContentPart{
					Type:         PartToolUse,
					ToolCallID:   callID,
					ToolName:     name,
					ToolArgsJSON: argsJSON,
				})
			}
		}
	}

	finishReason := mapStopReason(out.StopReason)
	if plan.syntheticSchemaTool && out.StopReason == types.StopReasonToolUse && len(toolCalls) == 0 {
		finishReason = "stop"
	}

	usage := extractUsage(out.Usage)

	return &Response{
		AssistantMessage: Message{
			Role:  RoleAssistant,
			Parts: assistantParts,
		},
		ContentText:       textBuilder.String(),
		ReasoningText:     reasonBuilder.String(),
		ToolCalls:         toolCalls,
		FinishReason:      finishReason,
		BedrockStopReason: string(out.StopReason),
		Usage:             usage,
	}, nil
}

type activeBlockState struct {
	kind           string // "text", "reasoning", "tool", "synthetic_schema"
	toolIndex      int
	toolCallID     string
	toolName       string
	textBuf        strings.Builder
	reasoningBuf   strings.Builder
	signatureBuf   strings.Builder
	redactedBytes  []byte
	toolArgsBuf    strings.Builder
}

// InvokeStream executes Bedrock ConverseStream, invoking onEvent sequentially and returning a reconstructed StreamSummary.
func (t *Translator) InvokeStream(ctx context.Context, req Request, onEvent func(StreamEvent) error) (*StreamSummary, *openai.APIError) {
	cInput, plan, apiErr := t.BuildConverseInput(ctx, req)
	if apiErr != nil {
		return nil, apiErr
	}

	sInput := &bedrockruntime.ConverseStreamInput{
		ModelId:                           cInput.ModelId,
		Messages:                          cInput.Messages,
		System:                            cInput.System,
		ToolConfig:                        cInput.ToolConfig,
		InferenceConfig:                   cInput.InferenceConfig,
		AdditionalModelRequestFields:      cInput.AdditionalModelRequestFields,
		AdditionalModelResponseFieldPaths: cInput.AdditionalModelResponseFieldPaths,
		OutputConfig:                      cInput.OutputConfig,
		PerformanceConfig:                 cInput.PerformanceConfig,
		PromptVariables:                   cInput.PromptVariables,
		RequestMetadata:                   cInput.RequestMetadata,
		ServiceTier:                       cInput.ServiceTier,
	}

	reader, err := t.runtime.ConverseStream(ctx, sInput)
	if err != nil {
		if meta := openai.RequestMetaFromContext(ctx); meta != nil {
			meta.SetBedrockError(openai.BedrockErrorCode(err))
		}
		return nil, openai.FromBedrockError(err)
	}
	defer func() { _ = reader.Close() }()

	blocks := make(map[int32]*activeBlockState)
	var orderedBlocks []int32
	nextToolIdx := 0
	var usage Usage
	finishReason := "stop"

	getOrInitBlock := func(idx int32, defaultKind string) *activeBlockState {
		if st, ok := blocks[idx]; ok {
			return st
		}
		st := &activeBlockState{kind: defaultKind, toolIndex: -1}
		blocks[idx] = st
		orderedBlocks = append(orderedBlocks, idx)
		return st
	}

	for ev := range reader.Events() {
		switch e := ev.(type) {
		case *types.ConverseStreamOutputMemberMessageStart:
			if err := onEvent(StreamEvent{Kind: EventMessageStart}); err != nil {
				return nil, openai.FromBedrockError(err)
			}

		case *types.ConverseStreamOutputMemberContentBlockStart:
			idx := aws.ToInt32(e.Value.ContentBlockIndex)
			if tuStart, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
				name := aws.ToString(tuStart.Value.Name)
				callID := aws.ToString(tuStart.Value.ToolUseId)
				if plan.syntheticSchemaTool && name == syntheticJSONSchemaToolName {
					st := getOrInitBlock(idx, "synthetic_schema")
					st.kind = "synthetic_schema"
					continue
				}
				st := getOrInitBlock(idx, "tool")
				st.kind = "tool"
				st.toolIndex = nextToolIdx
				nextToolIdx++
				st.toolCallID = callID
				st.toolName = name
				if err := onEvent(StreamEvent{
					Kind:       EventToolCallStart,
					ToolIndex:  st.toolIndex,
					ToolCallID: callID,
					ToolName:   name,
				}); err != nil {
					return nil, openai.FromBedrockError(err)
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			idx := aws.ToInt32(e.Value.ContentBlockIndex)
			switch d := e.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				st := getOrInitBlock(idx, "text")
				st.textBuf.WriteString(d.Value)
				if d.Value != "" {
					if err := onEvent(StreamEvent{
						Kind: EventTextDelta,
						Text: d.Value,
					}); err != nil {
						return nil, openai.FromBedrockError(err)
					}
				}
			case *types.ContentBlockDeltaMemberToolUse:
				chunk := aws.ToString(d.Value.Input)
				st := getOrInitBlock(idx, "tool")
				if st.kind == "synthetic_schema" {
					st.textBuf.WriteString(chunk)
					if chunk != "" {
						if err := onEvent(StreamEvent{
							Kind: EventTextDelta,
							Text: chunk,
						}); err != nil {
							return nil, openai.FromBedrockError(err)
						}
					}
					continue
				}
				st.toolArgsBuf.WriteString(chunk)
				if chunk != "" {
					if err := onEvent(StreamEvent{
						Kind:           EventToolCallDelta,
						ToolIndex:      st.toolIndex,
						ToolCallID:     st.toolCallID,
						ToolName:       st.toolName,
						ArgumentsDelta: chunk,
					}); err != nil {
						return nil, openai.FromBedrockError(err)
					}
				}
			case *types.ContentBlockDeltaMemberReasoningContent:
				st := getOrInitBlock(idx, "reasoning")
				st.kind = "reasoning"
				switch rd := d.Value.(type) {
				case *types.ReasoningContentBlockDeltaMemberText:
					st.reasoningBuf.WriteString(rd.Value)
					if rd.Value != "" {
						if err := onEvent(StreamEvent{
							Kind:          EventReasoningDelta,
							ReasoningText: rd.Value,
						}); err != nil {
							return nil, openai.FromBedrockError(err)
						}
					}
				case *types.ReasoningContentBlockDeltaMemberSignature:
					st.signatureBuf.WriteString(rd.Value)
					if rd.Value != "" {
						if err := onEvent(StreamEvent{
							Kind:           EventReasoningDelta,
							SignatureDelta: rd.Value,
						}); err != nil {
							return nil, openai.FromBedrockError(err)
						}
					}
				case *types.ReasoningContentBlockDeltaMemberRedactedContent:
					st.redactedBytes = append(st.redactedBytes, rd.Value...)
					if len(rd.Value) > 0 {
						if err := onEvent(StreamEvent{
							Kind:            EventReasoningDelta,
							RedactedContent: append([]byte(nil), rd.Value...),
						}); err != nil {
							return nil, openai.FromBedrockError(err)
						}
					}
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockStop:
			idx := aws.ToInt32(e.Value.ContentBlockIndex)
			if st, ok := blocks[idx]; ok && st.kind == "tool" {
				if err := onEvent(StreamEvent{
					Kind:       EventToolCallDone,
					ToolIndex:  st.toolIndex,
					ToolCallID: st.toolCallID,
					ToolName:   st.toolName,
				}); err != nil {
					return nil, openai.FromBedrockError(err)
				}
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			finishReason = mapStopReason(e.Value.StopReason)
			if plan.syntheticSchemaTool && e.Value.StopReason == types.StopReasonToolUse && nextToolIdx == 0 {
				finishReason = "stop"
			}
			if err := onEvent(StreamEvent{
				Kind:         EventMessageStop,
				FinishReason: finishReason,
			}); err != nil {
				return nil, openai.FromBedrockError(err)
			}

		case *types.ConverseStreamOutputMemberMetadata:
			usage = extractUsage(e.Value.Usage)
			if err := onEvent(StreamEvent{
				Kind:  EventUsage,
				Usage: usage,
			}); err != nil {
				return nil, openai.FromBedrockError(err)
			}
		}
	}

	if streamErr := reader.Err(); streamErr != nil {
		if meta := openai.RequestMetaFromContext(ctx); meta != nil {
			meta.SetBedrockError(openai.BedrockErrorCode(streamErr))
		}
		return nil, openai.FromBedrockError(streamErr)
	}

	var (
		parts       []ContentPart
		fullText    strings.Builder
		fullReason  strings.Builder
		toolOutputs []ToolCallOutput
	)
	for _, idx := range orderedBlocks {
		st := blocks[idx]
		switch st.kind {
		case "text", "synthetic_schema":
			txt := st.textBuf.String()
			if st.kind == "synthetic_schema" && txt == "" {
				txt = "{}"
			}
			if txt != "" {
				fullText.WriteString(txt)
				parts = append(parts, ContentPart{Type: PartText, Text: txt})
			}
		case "reasoning":
			if len(st.redactedBytes) > 0 {
				parts = append(parts, ContentPart{
					Type:            PartReasoning,
					RedactedContent: st.redactedBytes,
				})
			} else if st.reasoningBuf.Len() > 0 || st.signatureBuf.Len() > 0 {
				rt := st.reasoningBuf.String()
				fullReason.WriteString(rt)
				parts = append(parts, ContentPart{
					Type:          PartReasoning,
					ReasoningText: rt,
					Signature:     st.signatureBuf.String(),
				})
			}
		case "tool":
			args := strings.TrimSpace(st.toolArgsBuf.String())
			if args == "" {
				args = "{}"
			}
			toolOutputs = append(toolOutputs, ToolCallOutput{
				ID:        st.toolCallID,
				Name:      st.toolName,
				Arguments: args,
			})
			parts = append(parts, ContentPart{
				Type:         PartToolUse,
				ToolCallID:   st.toolCallID,
				ToolName:     st.toolName,
				ToolArgsJSON: args,
			})
		}
	}

	return &StreamSummary{
		AssistantMessage: Message{
			Role:  RoleAssistant,
			Parts: parts,
		},
		ContentText:   fullText.String(),
		ReasoningText: fullReason.String(),
		ToolCalls:     toolOutputs,
		FinishReason:  finishReason,
		Usage:         usage,
	}, nil
}

func resolveReasoningBudget(rc *ReasoningConfig) (string, int, *openai.APIError) {
	effort := strings.ToLower(strings.TrimSpace(rc.Effort))
	var defaultBudget int
	switch effort {
	case "", "medium":
		if effort == "" {
			effort = "medium"
		}
		defaultBudget = 4096
	case "low", "minimal":
		effort = "low"
		defaultBudget = 1024
	case "high", "xhigh", "max":
		effort = "high"
		defaultBudget = 16384
	default:
		return "", 0, openai.NewParamError(
			http.StatusBadRequest,
			openai.TypeInvalidRequest,
			"invalid_reasoning_effort",
			"reasoning_effort",
			fmt.Sprintf("Unsupported reasoning_effort %q (expected low, medium, or high).", rc.Effort),
		)
	}
	budget := defaultBudget
	if rc.BudgetTokens > 0 {
		budget = max(1024, rc.BudgetTokens)
	}
	return effort, budget, nil
}

func isAnthropicClaude(modelID string) bool {
	m := strings.ToLower(modelID)
	return strings.Contains(m, "anthropic.claude")
}

func isAmazonNova(modelID string) bool {
	m := strings.ToLower(modelID)
	return strings.Contains(m, "amazon.nova")
}

func supportsPromptCache(modelID string) bool {
	return isAnthropicClaude(modelID) || isAmazonNova(modelID)
}

func newSysCachePoint() *types.SystemContentBlockMemberCachePoint {
	return &types.SystemContentBlockMemberCachePoint{
		Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
	}
}

func newMsgCachePoint() *types.ContentBlockMemberCachePoint {
	return &types.ContentBlockMemberCachePoint{
		Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
	}
}

func newToolCachePoint() *types.ToolMemberCachePoint {
	return &types.ToolMemberCachePoint{
		Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
	}
}

func ensureTrailingMsgCachePoint(blocks []types.ContentBlock) []types.ContentBlock {
	if len(blocks) == 0 {
		return blocks
	}
	if _, already := blocks[len(blocks)-1].(*types.ContentBlockMemberCachePoint); already {
		return blocks
	}
	return append(blocks, newMsgCachePoint())
}

func normalizeToolSchema(schema map[string]any) map[string]any {
	if len(schema) == 0 {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	cloned := make(map[string]any, len(schema)+1)
	for k, v := range schema {
		cloned[k] = v
	}
	if _, hasType := cloned["type"]; !hasType {
		cloned["type"] = "object"
	}
	return cloned
}

func parseJSONArgsToDocument(raw string) document.Interface {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return document.NewLazyDocument(map[string]any{})
	}
	var obj any
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return document.NewLazyDocument(map[string]any{"_raw": raw})
	}
	if obj == nil {
		return document.NewLazyDocument(map[string]any{})
	}
	return document.NewLazyDocument(obj)
}

func marshalDocumentToJSONString(doc document.Interface) string {
	if doc == nil {
		return "{}"
	}
	if rawBytes, err := doc.MarshalSmithyDocument(); err == nil && len(rawBytes) > 0 {
		return string(rawBytes)
	}
	var val any
	if err := doc.UnmarshalSmithyDocument(&val); err != nil || val == nil {
		return "{}"
	}
	b, err := json.Marshal(normalizeSmithyNumbers(val))
	if err != nil {
		return "{}"
	}
	return string(b)
}

func normalizeSmithyNumbers(v any) any {
	switch x := v.(type) {
	case smithydoc.Number:
		return json.Number(string(x))
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			out[k] = normalizeSmithyNumbers(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = normalizeSmithyNumbers(val)
		}
		return out
	default:
		return v
	}
}

func mapStopReason(sr types.StopReason) string {
	switch sr {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return "stop"
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return "length"
	case types.StopReasonToolUse:
		return "tool_calls"
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return "content_filter"
	default:
		return "stop"
	}
}

func extractUsage(u *types.TokenUsage) Usage {
	if u == nil {
		return Usage{}
	}
	in := int64(aws.ToInt32(u.InputTokens))
	out := int64(aws.ToInt32(u.OutputTokens))
	cachedRead := int64(aws.ToInt32(u.CacheReadInputTokens))
	cachedWrite := int64(aws.ToInt32(u.CacheWriteInputTokens))
	total := int64(aws.ToInt32(u.TotalTokens))
	if total == 0 {
		total = in + out
	}
	return Usage{
		InputTokens:            in,
		OutputTokens:           out,
		TotalTokens:            total,
		CachedReadInputTokens:  cachedRead,
		CachedWriteInputTokens: cachedWrite,
	}
}
