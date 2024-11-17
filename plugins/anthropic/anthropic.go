package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/firebase/genkit/go/ai"
)

const (
	provider    = "anthropic"
	labelPrefix = "Anthropic"
	apiKeyEnv   = "ANTHROPIC_API_KEY"
)

var state struct {
	mu      sync.Mutex
	initted bool
	client  *anthropic.Client
}

type GenerationAnthropicConfig struct {
	ai.GenerationCommonConfig
	ToolChoiceParam anthropic.ToolChoiceUnionParam
	Metadata        anthropic.MetadataParam
}

var (
	knownCaps = map[string]ai.ModelCapabilities{
		anthropic.ModelClaude3_5Haiku20241022:   Multimodal,
		anthropic.ModelClaude3_5Sonnet20241022:  Multimodal,
		anthropic.ModelClaude_3_Haiku_20240307:  Multimodal,
		anthropic.ModelClaude_3_Sonnet_20240229: Multimodal,
		anthropic.ModelClaude_3_Opus_20240229:   Multimodal,
	}
)

type Config struct {
	APIKey string
}

func Init(ctx context.Context, cfg *Config) (err error) {
	if cfg == nil {
		cfg = &Config{}
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.initted {
		panic(provider + ".Init not called")
	}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(apiKeyEnv)
		if apiKey == "" {
			return fmt.Errorf("Anthropic requires setting %s in the environment. You can get an API key at https://console.anthropic.com/settings/keys", apiKeyEnv)
		}
	}

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)
	state.client = client
	state.initted = true

	for model, caps := range knownCaps {
		defineModel(model, caps)
	}

	return nil
}

func DefineModel(name string, caps *ai.ModelCapabilities) (ai.Model, error) {
	state.mu.Lock()
	defer state.mu.Unlock()
	if !state.initted {
		panic(provider + ".Init not called")
	}
	var mc ai.ModelCapabilities
	if caps == nil {
		var ok bool
		mc, ok = knownCaps[name]
		if !ok {
			return nil, fmt.Errorf("%s.DefineModel: called with unknown model %q and nil ModelCapabilities", provider, name)
		}
	} else {
		mc = *caps
	}
	return defineModel(name, mc), nil
}

func IsDefinedModel(name string) bool {
	return ai.IsDefinedModel(provider, name)
}

func Model(name string) ai.Model {
	return ai.LookupModel(provider, name)
}

func defineModel(name string, caps ai.ModelCapabilities) ai.Model {
	meta := &ai.ModelMetadata{
		Label:    labelPrefix + " - " + name,
		Supports: caps,
	}
	return ai.DefineModel(provider, name, meta, func(
		ctx context.Context,
		input *ai.GenerateRequest,
		cb func(context.Context, *ai.GenerateResponseChunk) error,
	) (*ai.GenerateResponse, error) {
		return generate(ctx, state.client, name, input, cb)
	})
}

func generate(
	ctx context.Context,
	client *anthropic.Client,
	model string,
	input *ai.GenerateRequest,
	cb func(context.Context, *ai.GenerateResponseChunk) error,
) (*ai.GenerateResponse, error) {
	req, err := convertRequest(model, input)
	if err != nil {
		return nil, err
	}

	res, err := client.Messages.New(ctx, req)
	if err != nil {
		return nil, err
	}

	r := translateResponse(res)
	r.Request = input
	return r, nil
}

func convertRequest(model string, input *ai.GenerateRequest) (anthropic.MessageNewParams, error) {
	if input.Output != nil && input.Output.Format != ai.OutputFormatText {
		err := fmt.Errorf("anthropic does not support %q output format, only %q is supported", input.Output.Format, ai.OutputFormatText)
		return anthropic.MessageNewParams{}, err
	}

	system, messages, err := convertMessages(input.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}
	tools, err := convertTools(input.Tools)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	params := anthropic.MessageNewParams{
		System:   anthropic.F(system),
		Model:    anthropic.F(model),
		Messages: anthropic.F(messages),
		Tools:    anthropic.F(tools),
	}

	if c, ok := input.Config.(*ai.GenerationCommonConfig); ok && c != nil {
		params.MaxTokens = anthropic.F(int64(c.MaxOutputTokens))
		params.TopK = anthropic.F(int64(c.TopK))
		params.TopP = anthropic.F(c.TopP)
		params.Temperature = anthropic.F(c.Temperature)
		params.StopSequences = anthropic.F(c.StopSequences)
	}

	if c, ok := input.Config.(*GenerationAnthropicConfig); ok && c != nil {
		params.MaxTokens = anthropic.F(int64(c.MaxOutputTokens))
		params.TopK = anthropic.F(int64(c.TopK))
		params.TopP = anthropic.F(c.TopP)
		params.Temperature = anthropic.F(c.Temperature)
		params.StopSequences = anthropic.F(c.StopSequences)
		params.ToolChoice = anthropic.F(c.ToolChoiceParam)
		params.Metadata = anthropic.F(c.Metadata)
	}

	return params, nil
}

func convertMessages(messages []*ai.Message) ([]anthropic.TextBlockParam, []anthropic.MessageParam, error) {
	var system = []anthropic.TextBlockParam{}
	var user = []anthropic.MessageParam{}

	for _, msg := range messages {
		var content = []anthropic.MessageParamContentUnion{}
		for _, part := range msg.Content {
			switch part.Kind {
			case ai.PartText:
				content = append(content, anthropic.NewTextBlock(part.Text))
			case ai.PartMedia:
				content = append(content, anthropic.NewImageBlockBase64(part.ContentType, part.Text))
			case ai.PartToolRequest:
				content = append(content, anthropic.NewToolUseBlockParam(
					part.ToolRequest.Name,
					part.ToolRequest.Name,
					part.ToolRequest.Input,
				))
			case ai.PartToolResponse:
				content = append(content, anthropic.NewToolResultBlock(
					part.ToolResponse.Name,
					convertToolResponse(part.ToolResponse),
					false,
				))
			default:
				return nil, nil, fmt.Errorf("unsupported message part kind %q", part.Kind)
			}
		}
	}
	return system, user, nil
}

func convertToolResponse(response *ai.ToolResponse) string {
	if url, ok := response.Output["url"].(string); ok {
		if base64Data, contentType, err := extractDataFromBase64URL(url); err == nil {
			if mediaContentType, ok := response.Output["contentType"].(string); ok {
				contentType = mediaContentType
			}
			return fmt.Sprintf(`{"type":"image","source":{"type":"base64","data":"%s","media_type":"%s"}}`,
				base64Data, contentType)
		}
	}

	// Handle other output types by JSON stringifying
	if bytes, err := json.Marshal(response.Output); err == nil {
		return fmt.Sprintf(`{"type":"text","text":%s}`, string(bytes))
	}

	return ""
}

func extractDataFromBase64URL(url string) (string, string, error) {
	re := regexp.MustCompile(`^data:([^;]+);base64,(.+)$`)
	matches := re.FindStringSubmatch(url)
	if matches == nil {
		return "", "", fmt.Errorf("invalid base64 data URL format")
	}
	return matches[2], matches[1], nil
}

func convertTools(tools []*ai.ToolDefinition) ([]anthropic.ToolParam, error) {
	res := []anthropic.ToolParam{}
	for _, t := range tools {
		res = append(res, anthropic.ToolParam{
			Name:        anthropic.F(t.Name),
			InputSchema: anthropic.Raw[interface{}](t.InputSchema),
			Description: anthropic.F(t.Description),
		})
	}
	return res, nil
}

func convertRole(role ai.Role) (anthropic.MessageParamRole, error) {
	switch role {
	case ai.RoleUser:
		return anthropic.MessageParamRoleUser, nil
	case ai.RoleSystem:
		return anthropic.MessageParamRoleAssistant, nil
	case ai.RoleModel:
		return anthropic.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unsupported role %q", role)
	}
}

func translateResponse(res *anthropic.Message) *ai.GenerateResponse {
	r := &ai.GenerateResponse{}

	for _, c := range res.Content {
		r.Candidates = append(r.Candidates, translateCandidate(&c, res.StopReason))
	}

	r.Usage = &ai.GenerationUsage{
		InputTokens:  int(res.Usage.InputTokens),
		OutputTokens: int(res.Usage.OutputTokens),
		TotalTokens:  int(res.Usage.InputTokens + res.Usage.OutputTokens),
	}

	return r
}

func translateCandidate(c *anthropic.ContentBlock, stopReason anthropic.MessageStopReason) *ai.Candidate {
	res := &ai.Candidate{}
	m := &ai.Message{
		Role: ai.RoleModel,
	}
	switch stopReason {
	case anthropic.MessageStopReasonEndTurn:
		res.FinishReason = ai.FinishReasonStop
	case anthropic.MessageStopReasonMaxTokens:
		res.FinishReason = ai.FinishReasonLength
	case anthropic.MessageStopReasonStopSequence:
		res.FinishReason = ai.FinishReasonOther
	case anthropic.MessageStopReasonToolUse:
		res.FinishReason = ai.FinishReasonOther
	default:
		res.FinishReason = ai.FinishReasonUnknown
	}

	m.Content = append(m.Content, ai.NewTextPart(c.Text))

	res.Message = m
	return res
}
