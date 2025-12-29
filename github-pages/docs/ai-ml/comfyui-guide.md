---
layout: docs
title: ComfyUI Comprehensive Guide
parent: AI/ML Documentation
nav_order: 2
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


{: .no_toc }

<div class="code-example" markdown="1">
Build visual AI workflows by connecting nodes, from simple image generation to complex multi-model pipelines.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why Use ComfyUI?

ComfyUI takes a different approach from other AI interfaces. Instead of hiding complexity behind menus, it shows you exactly how each part of the generation process connects to the next. You build workflows by linking nodes together, like connecting pipes in a plumbing diagram.

This approach offers several benefits:

- **See how generation actually works** - The visual layout teaches you what each component does
- **Customize everything** - Change any step, add new processing, or reroute the pipeline
- **Reuse and share workflows** - Save successful setups and load workflows from others
- **Efficient iteration** - Only recalculate what changes, not the entire pipeline

**Consider the following before starting:**

ComfyUI has a learning curve. The node interface feels unfamiliar at first. But once you understand the basics, you gain capabilities that simpler interfaces cannot provide. If you just want quick results, start with a simpler tool. If you want control and understanding, ComfyUI rewards the investment.

### When ComfyUI Makes Sense

| Use Case | Why ComfyUI | Alternative |
|----------|-------------|-------------|
| Learning how generation works | Visual pipeline shows connections | Read documentation |
| Complex multi-step workflows | Nodes make complexity manageable | Script-based automation |
| Batch processing with variations | Queue system handles it | Manual repeated generation |
| Sharing reproducible workflows | JSON export captures everything | Write setup instructions |
| Experimenting with new techniques | Modify workflow visually | Edit code |

## Getting Started

### Quick Start with Docker

The fastest way to get ComfyUI running:

```bash
docker-compose up -d comfyui-server
# Open http://localhost:8188 in your browser
```

### Manual Installation

If you prefer a local installation:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

For GPU acceleration, install PyTorch with CUDA support before running the requirements installation.

### Where Models Go

ComfyUI looks for models in specific folders:

| Model Type | Folder | Examples |
|------------|--------|----------|
| Checkpoints | `models/checkpoints/` | SDXL Base, Juggernaut |
| LoRAs | `models/loras/` | Style LoRAs, character LoRAs |
| VAEs | `models/vae/` | sdxl_vae, anime VAE |
| ControlNets | `models/controlnet/` | OpenPose, Canny |
| Embeddings | `models/embeddings/` | EasyNegative |

Generated images save to `output/` by default.

## Understanding the Interface

### How Nodes Work

Everything in ComfyUI is a node. Nodes are boxes that do one specific thing. They have:

- **Inputs** (left side) - Data coming in
- **Outputs** (right side) - Data going out
- **Settings** (inside the box) - Parameters you can adjust

You build workflows by connecting outputs to inputs. Data flows left to right through your connections.

### The Essential Node Types

| Category | What They Do | Examples |
|----------|--------------|----------|
| Loaders | Load models and images | CheckpointLoader, LoraLoader |
| Conditioning | Process text prompts | CLIPTextEncode |
| Sampling | Run the generation | KSampler |
| Latent | Work with compressed data | EmptyLatentImage, LatentUpscale |
| Image | Handle final images | VAEDecode, SaveImage |

### How Execution Works

When you click "Queue Prompt":
1. ComfyUI traces backward from output nodes
2. It only runs nodes whose inputs changed
3. Cached results are reused when possible

This means if you only change your prompt, the model does not reload. If you only change LoRA strength, previously computed steps are reused. This makes iteration fast.

## Essential Nodes Reference

These are the nodes you will use in almost every workflow.

### Loading Models

**CheckpointLoaderSimple** - Loads your base model and outputs three things:
- MODEL (the core generation model)
- CLIP (the text encoder)
- VAE (the image encoder/decoder)

**LoraLoader** - Adds a LoRA to your model. Key settings:
- `strength_model`: How much the LoRA affects generation (start at 0.7)
- `strength_clip`: How much it affects text understanding (usually match model strength)

### Processing Text

**CLIPTextEncode** - Converts your text prompt into numbers the model understands. Connect your CLIP output here and type your prompt in the text field.

### Generating Images

**KSampler** - The heart of generation. Key settings:

| Setting | What It Controls | Typical Values |
|---------|------------------|----------------|
| steps | Number of refinement passes | 20-35 |
| cfg | Prompt adherence strength | 5-9 |
| sampler_name | Denoising algorithm | euler, dpmpp_2m |
| scheduler | Noise reduction curve | karras, normal |
| seed | Randomness control | -1 for random |

### Output

**VAEDecode** - Converts the latent result to a viewable image.

**SaveImage** - Saves the image to disk. Set `filename_prefix` to organize your outputs.

## Building Your First Workflow

### The Minimal Text-to-Image Workflow

This is the simplest working workflow. Every other workflow builds from this foundation:

1. **CheckpointLoaderSimple** - Outputs: MODEL, CLIP, VAE
2. **EmptyLatentImage** - Creates blank canvas to generate on
3. **CLIPTextEncode (positive)** - Your main prompt
4. **CLIPTextEncode (negative)** - What to avoid
5. **KSampler** - Does the actual generation
6. **VAEDecode** - Converts result to image
7. **SaveImage** - Saves to disk

Connect them: Checkpoint outputs go to relevant inputs. Text encoders feed positive/negative conditioning to KSampler. EmptyLatentImage feeds latent_image. KSampler output goes to VAEDecode, which feeds SaveImage.

### Adding a LoRA

Insert a LoraLoader between CheckpointLoaderSimple and KSampler:

1. Connect CheckpointLoader MODEL and CLIP to LoraLoader
2. Connect LoraLoader outputs to where Checkpoint outputs originally went

You can chain multiple LoraLoaders for stacking.

### Image-to-Image Modification

Start from an existing image instead of empty latent:

1. Add LoadImage node and load your source image
2. Add VAEEncode node
3. Connect LoadImage to VAEEncode, VAEEncode to KSampler's latent_image
4. Set KSampler's denoise to 0.5-0.8 (lower = closer to original)

## Common Workflow Patterns

### Using FLUX Models

FLUX requires different settings than SD/SDXL:

| Setting | FLUX Value | SDXL Value |
|---------|------------|------------|
| cfg | 1.0 (always) | 5-9 |
| guidance | 3.5 (via FluxGuidance node) | N/A |
| sampler | euler | dpmpp_2m |
| scheduler | simple | karras |
| steps | 20-25 | 25-35 |

FLUX also needs FluxGuidance node for guidance control instead of using cfg directly.

### Stacking Multiple LoRAs

Chain LoraLoader nodes, reducing strength as you add more:

- First LoRA: 0.7-0.8 strength
- Second LoRA: 0.5-0.6 strength
- Third LoRA: 0.3-0.4 strength

Total combined effect should stay reasonable. Too much LoRA influence causes artifacts.

### Upscaling Generated Images

Two-step process for high-quality upscaling:

1. **Generate at native resolution** (1024x1024 for SDXL)
2. **Upscale and refine:**
   - Load result with LoadImage or keep in workflow
   - Use UpscaleModelLoader + ImageUpscaleWithModel for 2x-4x
   - VAEEncode the upscaled image
   - KSampler with low denoise (0.3-0.5) to add detail
   - VAEDecode to final image

This adds genuine detail rather than just enlarging pixels.

## Extending ComfyUI with Custom Nodes

The base ComfyUI installation handles core generation. Custom nodes add specialized capabilities.

### Essential Custom Node Packs

| Pack | What It Adds | When You Need It |
|------|--------------|------------------|
| ComfyUI Manager | Node installation, model downloads | Always - install this first |
| ControlNet Aux | Preprocessors for poses, edges, depth | Using ControlNet |
| Impact Pack | Face detection, regional processing | Face work, inpainting |
| IP-Adapter Plus | Image-as-prompt functionality | Style transfer, consistency |
| Efficiency Nodes | Batch processing, optimization | Large-scale generation |

### Installing Custom Nodes

**With ComfyUI Manager (recommended):**
1. Click "Manager" button in UI
2. Select "Install Custom Nodes"
3. Search for the pack you need
4. Install and restart ComfyUI

**Manually:**
```bash
cd ComfyUI/custom_nodes
git clone [repository-url]
# Restart ComfyUI
```

## Automation with the API

ComfyUI exposes an API for programmatic control, useful for batch processing or integration with other tools.

### Basic API Usage

```python
import requests

# Submit a workflow
response = requests.post("http://localhost:8188/prompt", json={
    "prompt": workflow_json
})
```

### Practical Automation Example

Export your workflow from ComfyUI (Save as API format), then submit it programmatically with modified parameters:

```python
import json

# Load your exported workflow
with open("my_workflow_api.json") as f:
    workflow = json.load(f)

# Modify prompt text in the workflow
workflow["6"]["inputs"]["text"] = "New prompt here"

# Submit
requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
```

This enables batch generation, integration with other systems, or scheduled generation tasks.

## Optimization and Troubleshooting

### Making Workflows Faster

| Strategy | How To Do It | Benefit |
|----------|--------------|---------|
| Use PreviewImage during iteration | Only SaveImage for finals | Faster feedback loop |
| Cache text encoding | Keep CLIPTextEncode results | Skip re-encoding |
| Reduce steps for previews | 15-20 steps while iterating | 2x faster iteration |
| Use fp16/fp8 models | Download quantized versions | Fits in less VRAM |

### Handling Low VRAM

If you hit memory errors:

1. **Start ComfyUI with low VRAM mode:**
   ```bash
   python main.py --lowvram
   ```

2. **Use tiled VAE** for large images - enable in VAEDecode settings

3. **Load fewer models simultaneously** - use separate workflows instead of one complex one

4. **Use quantized checkpoints** - fp8 FLUX uses ~12GB instead of ~24GB

### Common Issues and Solutions

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| "Failed to validate prompt" | Missing connection | Check all required inputs have connections |
| Out of memory | Model too large | Use quantized model, enable low VRAM mode |
| Slow generation | Unoptimized settings | Reduce steps, use faster sampler |
| Black image output | VAE mismatch | Use correct VAE for your checkpoint |
| Workflow won't load | Missing custom nodes | Install required nodes via Manager |

### Debugging Strategies

1. **Add PreviewImage nodes** after each major step to see intermediate results
2. **Hover over connections** to verify data types match
3. **Check the console** for error messages (terminal where ComfyUI runs)
4. **Simplify the workflow** - disconnect parts to isolate the problem

## Organizing Your Workflows

### Best Practices

- **Name your nodes** - Right-click and set descriptive titles
- **Use groups** - Box related nodes together with colors
- **Add notes** - Use Note nodes to document settings and requirements
- **Save often** - Keep versioned copies of working workflows

### Sharing Workflows

To share a workflow with others:

1. Save the workflow (Ctrl+S)
2. Note which custom nodes are required
3. List which models the workflow needs
4. Share the JSON file

Recipients need the same custom nodes and models (or compatible alternatives) to run your workflow.

## Conclusion

ComfyUI provides unmatched control over AI image generation through its visual node system. The initial learning investment pays off in:

- **Understanding** - You see exactly how generation works
- **Flexibility** - Build workflows no other tool can match
- **Efficiency** - Intelligent caching speeds iteration
- **Sharing** - JSON workflows capture complete setups

Start with the basic text-to-image workflow, then add complexity as you need it. Each new technique builds on what you have already learned.

---

## See Also
- [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) - Core concepts behind the generation process
- [ControlNet](controlnet.html) - Add precision control to ComfyUI workflows
- [LoRA Training](lora-training.html) - Train custom LoRAs for use in ComfyUI
- [Model Types](model-types.html) - Understanding models, LoRAs, and VAEs
- [Base Models Comparison](base-models-comparison.html) - Choosing the right base model
- [Advanced Techniques](advanced-techniques.html) - Expert workflow patterns
- [Output Formats](output-formats.html) - Image formats and optimization
- [AI/ML Documentation Hub](./) - Complete AI/ML documentation index