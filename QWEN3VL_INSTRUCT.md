# Qwen3VL Prompt Instructions for `ZImageTurboSinglePromptConditioning`

## Goal
Generate one high-quality prompt string for Tongyi Z-Image-Turbo that can be pasted directly into the `prompt` field.

## Node target
- Node: `ZImageTurboSinglePromptConditioning`
- Field: `prompt`
- Guidance: Keep output as one paragraph, no bullets, no metadata wrappers.

## Qwen3VL System Prompt
You are a prompt writer for Tongyi Z-Image-Turbo. Produce one single-paragraph positive prompt only. Use precise visual language, strong composition, realistic material/lighting details, and clean constraints. Avoid negative prompt syntax. Do not output JSON, markdown, lists, or explanations.

## Qwen3VL User Prompt Template
Create one prompt for Tongyi Z-Image-Turbo.

Subject: <describe subject>
Style: <photo / cinematic / illustration / anime / etc>
Lighting: <describe light direction, quality, mood>
Camera: <lens/focal/depth cues if relevant>
Scene/Background: <environment and context>
Critical details: <must-have details>

Hard requirements:
- single paragraph
- 45 to 120 words
- include composition cues
- include texture/material realism
- include color palette cues
- include one short quality ending: "masterpiece, high quality"

## Example Qwen3VL Input
Create one prompt for Tongyi Z-Image-Turbo.

Subject: high-end product photo of a black mechanical keyboard
Style: commercial studio photography
Lighting: soft key from top-left, subtle rim light
Camera: 50mm, shallow depth of field
Scene/Background: dark matte desk with minimal props
Critical details: crisp key legends, brushed aluminum texture, clean reflections

Hard requirements:
- single paragraph
- 45 to 120 words
- include composition cues
- include texture/material realism
- include color palette cues
- include one short quality ending: "masterpiece, high quality"

## Example Expected Output
Three-quarter hero shot of a premium black mechanical keyboard centered on a dark matte desk, framed with negative space and a clean diagonal leading line from the front edge, commercial studio photography style with a controlled 50mm look and shallow depth of field. Soft top-left key light defines the brushed aluminum chassis while a subtle rim light separates the silhouette, revealing crisp key legends, fine PBT grain, precise chamfers, and restrained reflections in a charcoal and graphite palette, masterpiece, high quality.

## ComfyUI Use
1. Paste Qwen3VL output into `ZImageTurboSinglePromptConditioning.prompt`.
2. Keep `enhancement_profile` at `capitan_daily` for most prompts.
3. Connect node output to `ZImageTurboSampler` as usual.
