package anthropic

import "github.com/firebase/genkit/go/ai"

var (
	// Multimodal describes model capabilities for multimodal Claude models.
	Multimodal = ai.ModelCapabilities{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
	}
)
