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
Master ComfyUI's node-based workflow system for advanced AI image generation, from basic concepts to complex workflows.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What is ComfyUI?

ComfyUI is a powerful node-based interface for Stable Diffusion and other diffusion models that enables users to create complex image generation workflows through visual programming. Unlike traditional UIs, ComfyUI exposes the entire generation pipeline as modular nodes that can be connected in countless ways.

As of 2024, ComfyUI has become the de facto standard for advanced workflows, supporting all major models including SDXL, SD3, FLUX, and various video/audio models. Its modular architecture makes it ideal for experimenting with cutting-edge techniques.

### Key Advantages

- **Visual Workflow Design**: See and control every step of the generation process
- **Reusable Workflows**: Save and share complex setups
- **Maximum Flexibility**: Create workflows impossible in other UIs
- **Efficient Processing**: Only recalculate changed nodes
- **Extensible**: Huge ecosystem of custom nodes

## Installation and Setup

### Using Docker (Recommended)

```bash
# Clone the ComfyUI MCP setup
git clone https://github.com/andrewaltimit/Documentation comfyui-mcp
cd comfyui-mcp

# Build and start services
docker-compose build
docker-compose up -d

# Access ComfyUI at http://localhost:8188
```

### Manual Installation

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install PyTorch with CUDA support first (choose your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for production):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Install xformers for 2x+ speedup (highly recommended)
# Match your PyTorch/CUDA version
pip install xformers

# Optional: Install Flash Attention 2 for even better performance
pip install flash-attn --no-build-isolation

# Run ComfyUI
python main.py

# Or with specific arguments
python main.py --listen --port 8188 --preview-method auto
```

### Directory Structure

```
ComfyUI/
├── models/
│   ├── checkpoints/     # Base models (SD, SDXL, FLUX)
│   ├── loras/          # LoRA models
│   ├── vae/            # VAE models
│   ├── clip/           # CLIP models
│   ├── controlnet/     # ControlNet models
│   └── embeddings/     # Textual inversions
├── input/              # Input images
├── output/             # Generated images
├── custom_nodes/       # Extensions
├── web/                # Frontend files
└── user/               # User data
    └── default/
        └── workflows/   # Saved workflows
```

## Core Concepts

### Nodes

Nodes are the building blocks of ComfyUI workflows. Each node:
- Has input ports (left side)
- Has output ports (right side)
- Performs a specific operation
- Can be connected to other nodes

### Node Categories

1. **Loaders**: Load models, images, or data
2. **Conditioning**: Process text prompts and embeddings
3. **Sampling**: Core generation nodes
4. **Image**: Image processing and manipulation
5. **Latent**: Operations in latent space
6. **Control**: ControlNet and guidance nodes
7. **Advanced**: Custom sampling, model merging
8. **Utils**: Helper nodes for workflow control

### Workflow Execution

ComfyUI uses intelligent execution:
1. **Lazy Evaluation**: Only executes necessary nodes
2. **Caching**: Reuses results from unchanged nodes
3. **Queue System**: Manages multiple generation requests
4. **Progress Tracking**: Shows execution status

## Essential Nodes

### Model Loading

**CheckpointLoaderSimple**
```
Outputs:
- MODEL: The main model
- CLIP: Text encoder
- VAE: Image encoder/decoder
```

**LoraLoader**
```
Inputs:
- model: Base model
- clip: CLIP model
- lora_name: LoRA file
- strength_model: Model strength (0-2)
- strength_clip: CLIP strength (0-2)
```

### Text Encoding

**CLIPTextEncode**
```
Inputs:
- clip: CLIP model
- text: Your prompt
Outputs:
- CONDITIONING: Encoded prompt
```

### Sampling

**KSampler**
```
Inputs:
- model: Generation model
- positive: Positive conditioning
- negative: Negative conditioning
- latent_image: Starting latent
- seed: Random seed
- steps: Sampling steps
- cfg: CFG scale
- sampler_name: Algorithm
- scheduler: Noise schedule
```

**KSampler Advanced**
```
Additional inputs:
- add_noise: Enable/disable noise addition
- start_at_step: Begin at specific step
- end_at_step: Stop at specific step
- return_with_leftover_noise: Keep residual noise
```

### Image Handling

**VAEDecode**
```
Inputs:
- samples: Latent image
- vae: VAE model
Outputs:
- IMAGE: Decoded image
```

**SaveImage**
```
Inputs:
- images: Images to save
- filename_prefix: File naming
```

## Basic Workflows

### Simple Text-to-Image

```
[CheckpointLoaderSimple] → MODEL → [KSampler]
                      ↓            ↑
                     CLIP → [CLIPTextEncode] → CONDITIONING
                      ↓
                     VAE → [VAEDecode] → [SaveImage]
```

### Image-to-Image

```
[LoadImage] → [VAEEncode] → LATENT → [KSampler] → [VAEDecode] → [SaveImage]
```

### LoRA Integration

```
[CheckpointLoaderSimple] → [LoraLoader] → [LoraLoader] → [KSampler]
                                ↑              ↑
                          lora1.safetensors  lora2.safetensors
```

## Advanced Workflows

### FLUX Workflow

FLUX requires specific node configurations:

```python
# Key FLUX settings
cfg = 1.0  # Must be 1.0, not 7-8
guidance = 3.5  # Via FluxGuidance node
sampler = "euler"
scheduler = "simple"
steps = 20-25
```

Complete FLUX workflow:
```
[CheckpointLoaderSimple] 
    ↓
[UNETLoader] → [LoraLoader] → [FluxGuidance]
    ↓                              ↑
[CLIPLoader] → [CLIPTextEncode] → CONDITIONING
    ↓
[VAELoader] → [VAEDecode]
```

### Multi-LoRA Stacking

```python
# Chain multiple LoRAs
[CheckpointLoaderSimple]
    ↓
[LoraLoader: style_lora @ 0.8]
    ↓
[LoraLoader: character_lora @ 0.6]
    ↓
[LoraLoader: detail_lora @ 0.4]
    ↓
[KSampler]
```

### Regional Prompting

Use attention masking for different prompts in different areas:

```
[CLIPTextEncode: "forest background"] → [Attention Mask]
                                              ↓
[CLIPTextEncode: "knight in armor"] → [Attention Mask] → [Combine] → [KSampler]
```

### Upscaling Workflow

```
[Generated Image] → [Upscale Model Loader] → [Image Upscale]
                                                    ↓
                            [VAEEncode] → [KSampler @ 0.4 denoise] → [VAEDecode]
```

## Custom Nodes

### Essential Custom Node Packs

1. **ComfyUI Manager**
   - In-UI installation of custom nodes
   - Model downloading
   - Workflow management

2. **Efficiency Nodes**
   - Optimized samplers
   - Batch processing
   - Memory management

3. **Impact Pack**
   - Advanced detectors
   - Regional processing
   - Face enhancement

4. **ControlNet Aux**
   - Preprocessors for ControlNet
   - Edge detection
   - Pose estimation
   - Depth estimation

5. **AnimateDiff Evolved**
   - Video generation
   - Motion LoRAs
   - Frame interpolation

6. **IP-Adapter Plus**
   - Advanced image prompting
   - Style transfer
   - Face consistency

7. **GGUF Support**
   - Load quantized models
   - Reduced VRAM usage
   - Mobile deployment

### Installing Custom Nodes

Via ComfyUI Manager:
```
1. Click "Manager" in UI
2. Select "Install Custom Nodes"
3. Search and install
4. Restart ComfyUI
```

Manual installation:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/user/custom-node-pack
# Restart ComfyUI
```

## MCP Integration

### Using ComfyUI with MCP

The ComfyUI MCP server enables programmatic control:

```python
# Generate image via MCP
import requests
import json

response = requests.post("http://localhost:8005/mcp/tool", json={
    "tool": "generate-image",
    "arguments": {
        "prompt": "cyberpunk city, neon lights, rain",
        "checkpoint": "sdxl_base.safetensors",
        "lora": "cyberpunk_style.safetensors",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras"
    }
})
```

### Workflow Submission

```python
# Submit custom workflow
with open("my_workflow.json", "r") as f:
    workflow = json.load(f)

response = requests.post("http://localhost:8189/mcp/tool", json={
    "tool": "submit-workflow",
    "arguments": {
        "workflow": workflow
    }
})
```

### Model Management

```python
# Upload LoRA with metadata
response = requests.post("http://localhost:8005/mcp/tool", json={
    "tool": "upload-lora",
    "arguments": {
        "filename": "my_style.safetensors",
        "content": base64_content,
        "metadata": {
            "trigger_words": ["mystyle"],
            "recommended_strength": 0.8,
            "base_model": "SDXL",
            "training_details": {
                "steps": 2000,
                "dataset_size": 50
            }
        }
    }
})
```

## Workflow Optimization

### Performance Tips

1. **Node Arrangement**
   - Group related nodes
   - Minimize connection crossings
   - Use reroute nodes for clarity

2. **Caching Strategy**
   - Place variable inputs last
   - Reuse encoded prompts
   - Cache VAE outputs when iterating

3. **Batch Processing**
   ```
   [Batch Images] → [VAEEncode Batch] → [KSampler] → [VAEDecode Batch]
   ```

### Memory Management

**Low VRAM Techniques**:
- Use `--lowvram` or `--cpu` flags
- Enable sequential CPU offloading
- Use tiled VAE for large images
- Reduce batch size
- Use fp8/int8 quantized models
- Enable smart memory management:
  ```bash
  python main.py --lowvram --use-split-cross-attention
  ```

**Efficient Node Usage**:
```python
# Instead of multiple saves
[KSampler] → [VAEDecode] → [SaveImage]

# Use preview for iteration
[KSampler] → [VAEDecode] → [PreviewImage]
                         ↓
                    [SaveImage] (only final)
```

## Common Workflows

### Style Transfer

```
[LoadImage: style] → [VAEEncode] → [StyleModel]
                                        ↓
[LoadImage: content] → [VAEEncode] → [Combine] → [VAEDecode]
```

### Character Consistency

```
[Reference Image] → [CLIPVisionEncode] → [IPAdapter]
                                            ↓
[Text Prompt] → [CLIPTextEncode] → [Combine] → [KSampler]
```

### Animation Workflow

```
[Frame 1] → [VAEEncode] → [Latent Interpolate] → [Batch] → [KSampler]
[Frame 2] → [VAEEncode] ↗
```

## Troubleshooting

### Common Issues

**"Failed to validate prompt"**
- Check all required inputs are connected
- Verify node compatibility
- Ensure models are loaded

**Out of Memory**
```python
# Add these nodes
[VAETileDecode]  # Instead of VAEDecode
[TiledKSampler]  # Instead of KSampler
```

**Slow Generation**
- Reduce image size initially
- Use faster samplers (DPM++ 2M)
- Disable preview during batch

### Debugging Workflows

1. **Use Preview Nodes**: Add PreviewImage after each major step
2. **Check Data Types**: Hover over connections to see types
3. **Isolate Issues**: Disconnect nodes systematically
4. **Monitor Console**: Check for error messages

## Advanced Techniques

### Latent Space Operations

**Latent Composite**
```python
# Combine multiple latents
[Latent1] → [LatentComposite: x=0, y=0]
[Latent2] → [LatentComposite: x=512, y=0] → [Combined Latent]
```

**Latent Upscale**
```python
[Latent] → [LatentUpscale: 2x] → [KSampler: denoise=0.5]
```

### Prompt Scheduling

```python
# Animate prompt weights over time
[CLIPTextEncode: "(cat:1.0)"] → frame 0
[CLIPTextEncode: "(cat:0.5) (dog:0.5)"] → frame 15  
[CLIPTextEncode: "(dog:1.0)"] → frame 30
```

### Multi-Model Ensemble

```python
# Blend predictions from multiple models
[Model1] → [KSampler] → [Latent]
                           ↓
                    [LatentBlend: 0.5]
                           ↑
[Model2] → [KSampler] → [Latent]
```

## Recent ComfyUI Developments (2024)

### New Features

1. **Execution Caching**: Smart caching prevents re-execution
2. **Node Templates**: Save and reuse node configurations
3. **Workflow Components**: Encapsulate sub-workflows
4. **Better Preview**: Real-time latent preview options
5. **Auto Queue Management**: Intelligent batch processing

### Performance Improvements

- **Torch Compile**: Up to 30% speed improvement
- **Better Memory Management**: Automatic VRAM optimization
- **Parallel Execution**: Multiple independent paths
- **Smart Model Loading**: Reduced loading times

## Best Practices

### Workflow Organization

1. **Naming Convention**
   - Use descriptive node titles
   - Group nodes by function  
   - Color code by purpose
   - Use reroute nodes for clarity

2. **Documentation**
   - Add Note nodes for complex sections
   - Include parameter explanations
   - Document model requirements
   - Version your workflows

3. **Version Control**
   - Save incremental versions
   - Export workflows as JSON
   - Track model dependencies
   - Use git for workflow files

### Sharing Workflows

**Workflow Export**:
1. Clear cache to reduce file size
2. Use relative model paths
3. Include custom node list
4. Document required models

**Workflow Format**:
```json
{
  "last_node_id": 50,
  "last_link_id": 100,
  "nodes": [...],
  "links": [...],
  "groups": [...],
  "version": 0.4
}
```

## Advanced Workflows Examples

### LCM Speed Workflow
```python
# 4-step generation with LCM
[SDXL Checkpoint] → [LCM-LoRA Loader @ 1.0] → [KSampler]
                                                    │
                                              cfg: 1.5-2.5
                                              steps: 4-8
                                              sampler: lcm
```

### SD3 Workflow
```python
# SD3 with triple text encoding
[SD3 Checkpoint] → [Triple Text Encode]
                          │
                    [CLIP L] [CLIP G] [T5]
                          ↓
                    [KSampler: shift=3.0]
```

## Example: Complete FLUX LoRA Workflow

```python
# Complete FLUX workflow with LoRA
{
  "nodes": [
    {
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "outputs": ["MODEL", "CLIP", "VAE"]
    },
    {
      "type": "LoraLoader",
      "inputs": {
        "lora_name": "my_style.safetensors",
        "strength_model": 1.0,
        "strength_clip": 1.0
      }
    },
    {
      "type": "CLIPTextEncode",
      "inputs": {
        "text": "my_style digital art of a dragon"
      }
    },
    {
      "type": "FluxGuidance",
      "inputs": {
        "guidance": 3.5
      }
    },
    {
      "type": "KSamplerAdvanced",
      "inputs": {
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "steps": 25
      }
    },
    {
      "type": "VAEDecode"
    },
    {
      "type": "SaveImage",
      "inputs": {
        "filename_prefix": "flux_lora"
      }
    }
  ]
}
```

## Conclusion

ComfyUI's node-based approach offers unparalleled control over the image generation process. While the learning curve is steeper than traditional UIs, the flexibility and power it provides make it the tool of choice for advanced users and those seeking to push the boundaries of AI image generation.

Start with simple workflows and gradually incorporate more complex nodes and techniques. The visual nature of ComfyUI makes it excellent for understanding how different components interact, leading to better results and deeper knowledge of the generation process.