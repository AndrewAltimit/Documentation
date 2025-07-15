---
layout: default
title: Base Models Comparison
parent: AI/ML Documentation
nav_order: 4
---

# Base Models Comparison Guide
{: .no_toc }

<div class="code-example" markdown="1">
Comprehensive comparison of popular diffusion models: SD 1.5, SD 2.x, SDXL, Pony, and FLUX, with their strengths, requirements, and use cases.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The landscape of diffusion models has evolved rapidly, with each generation bringing improvements in quality, capabilities, and efficiency. This guide compares the major base models to help you choose the right one for your needs.

## Quick Comparison Table

| Model | Resolution | VRAM (Min) | Quality | Speed | Flexibility | Release |
|-------|------------|------------|---------|-------|-------------|---------|
| SD 1.5 | 512×512 | 4GB | Good | Fast | Excellent | 2022 |
| SD 2.1 | 768×768 | 6GB | Better | Medium | Good | 2022 |
| SDXL | 1024×1024 | 8GB | Excellent | Slow | Very Good | 2023 |
| Pony | 1024×1024 | 8GB | Excellent* | Medium | Specialized | 2024 |
| FLUX | 1024×1024+ | 12GB | State-of-art | Slow | Excellent | 2024 |

*Excellent for anime/stylized content

## Stable Diffusion 1.5

### Overview

SD 1.5 remains the most popular and widely supported model due to its balance of quality, speed, and compatibility. Despite newer models, it maintains relevance through extensive community support and optimization.

### Technical Specifications

```yaml
Architecture: U-Net with attention
Parameters: 859M
Training Resolution: 512×512
VAE: KL-f8 autoencoder
Text Encoder: CLIP ViT-L/14
Max Tokens: 77
Precision: FP16/FP32
File Size: ~2GB (pruned)
```

### Strengths

- **Massive ecosystem**: Thousands of LoRAs, embeddings, and tools
- **Low requirements**: Runs on 4GB VRAM GPUs
- **Fast generation**: 20-50 steps typical
- **Highly optimized**: Years of community optimization
- **Versatile**: Works well for most content types

### Weaknesses

- **Lower resolution**: Native 512×512
- **Text rendering**: Poor text generation
- **Anatomy**: Struggles with hands/complex poses
- **Modern concepts**: Limited understanding of recent topics

### Optimal Settings

```python
{
    "resolution": "512x512",
    "steps": 20-30,
    "cfg_scale": 7-9,
    "sampler": "euler_a",
    "clip_skip": 1-2,
    "vae": "vae-ft-mse-840000"
}
```

### Use Cases

Best for:
- Quick prototyping
- Low-resource environments
- Artistic/stylized content
- When extensive LoRA support needed
- Web applications

## Stable Diffusion 2.x

### Overview

SD 2.x improved upon 1.5 with better training data and higher resolution, but faced adoption challenges due to changed aesthetic preferences and compatibility issues.

### Technical Specifications

```yaml
Architecture: Improved U-Net
Parameters: 865M (2.0), 865M (2.1)
Training Resolution: 768×768
VAE: Improved KL-f8
Text Encoder: OpenCLIP ViT-H/14
Max Tokens: 77
File Size: ~2.5GB
```

### Key Differences from 1.5

- **Different CLIP model**: OpenCLIP vs CLIP
- **Cleaner dataset**: NSFW content filtered
- **Higher resolution**: 768×768 native
- **Better architecture**: Improved attention

### Strengths

- **Better quality**: Improved detail and coherence
- **Higher resolution**: Native 768×768
- **Cleaner outputs**: Less prone to artifacts
- **Better concepts**: Improved understanding

### Weaknesses

- **Limited adoption**: Fewer LoRAs and tools
- **Different aesthetic**: Less "artistic" by default
- **Compatibility**: Not backward compatible with SD 1.5
- **Prompt differences**: Requires different prompting style

### Optimal Settings

```python
{
    "resolution": "768x768",
    "steps": 25-35,
    "cfg_scale": 6-8,
    "sampler": "dpm++_2m",
    "negative_prompt": "Essential for good results"
}
```

## SDXL (Stable Diffusion XL)

### Overview

SDXL represents a major leap in quality and resolution, introducing a two-stage pipeline with separate base and refiner models for unprecedented detail.

### Technical Specifications

```yaml
Architecture: Enlarged U-Net + Refiner
Parameters: 3.5B (base) + 3.5B (refiner)
Training Resolution: 1024×1024
VAE: SDXL VAE
Text Encoders: CLIP ViT-L + OpenCLIP ViT-G
Max Tokens: 77 × 2
Conditioning: Size + Crop conditioning
File Size: ~6.5GB (base only)
```

### Unique Features

1. **Two-stage pipeline**:
   ```
   Base Model (0.8 denoising) → Refiner Model (0.2 denoising)
   ```

2. **Dual text encoders**: Combines two CLIP models
3. **Conditioning augmentation**: Resolution and crop parameters
4. **Improved VAE**: Better color accuracy

### Strengths

- **Superior quality**: Photorealistic capabilities
- **High resolution**: Native 1024×1024+
- **Better text**: Improved text rendering
- **Fine details**: Excellent micro-details
- **Versatility**: Works across all styles

### Weaknesses

- **Resource intensive**: 8GB+ VRAM minimum
- **Slower generation**: 40-50% slower than SD 1.5
- **Complex pipeline**: Two models for best results
- **Large size**: ~13GB for full pipeline

### Optimal Settings

```python
{
    "resolution": "1024x1024",
    "base_steps": 25-35,
    "refiner_steps": 10-15,
    "cfg_scale": 5-7,
    "sampler": "dpm++_2m_sde",
    "refiner_switch": 0.8,
    "negative_prompt": "Critical for quality"
}
```

### SDXL Workflow

```python
# ComfyUI workflow
Base Model → KSampler (end_at_step: 20)
                ↓
         Latent Output
                ↓
Refiner Model → KSampler (start_at_step: 20)
```

## Pony Diffusion

### Overview

Pony Diffusion is a specialized SDXL fine-tune focused on anime, furry, and cartoon content, becoming the go-to model for stylized artwork generation.

### Technical Specifications

```yaml
Base: SDXL architecture
Specialization: Anime/Furry/Cartoon
Training Data: Curated booru datasets
Special Tokens: Quality tags system
Resolution: 1024×1024
File Size: ~6.5GB
```

### Unique Features

1. **Score-based prompting**:
   ```
   "score_9, score_8_up, score_7_up, [your prompt]"
   ```

2. **Style tags**: Extensive style control
3. **Character knowledge**: Recognizes many characters
4. **Booru tags**: Uses danbooru-style tagging

### Strengths

- **Anime excellence**: Best-in-class for anime
- **Style consistency**: Maintains style well
- **Character accuracy**: Great for fan art
- **Tag system**: Intuitive for booru users
- **Active community**: Regular updates

### Weaknesses

- **Specialized**: Not ideal for photorealism
- **Learning curve**: Unique prompting style
- **Content bias**: Optimized for specific content
- **NSFW tendency**: Requires careful prompting

### Optimal Settings

```python
{
    "resolution": "1024x1024",
    "steps": 25-30,
    "cfg_scale": 6-8,
    "sampler": "euler_a",
    "clip_skip": 2,  # Important for anime
    "prompt_prefix": "score_9, score_8_up, score_7_up, masterpiece"
}
```

### Prompting Guide

```
# Good Pony prompt structure
score_9, score_8_up, masterpiece, best quality,
1girl, [character_name], [outfit], [pose], [expression],
[background], [lighting], [style_tags]

# Negative prompt
score_6, score_5, score_4, worst quality, low quality,
bad anatomy, bad hands
```

## FLUX

### Overview

FLUX represents the current state-of-the-art in open diffusion models, with revolutionary architecture changes and significantly improved capabilities.

### Technical Specifications

```yaml
Architecture: Transformer-based (DiT)
Parameters: 12B
Training: Flow matching
Resolution: 1024×1024 - 2048×2048
Text Encoder: T5 XXL + CLIP
Max Tokens: 256
Guidance: Separate guidance model
File Size: ~24GB (FP16)
```

### Model Variants

1. **FLUX.1-dev**: Full quality, slower
2. **FLUX.1-schnell**: Distilled, 4-step generation
3. **FLUX-fp8**: Quantized for lower VRAM

### Revolutionary Features

1. **No CFG needed**: Uses guidance model instead
   ```python
   cfg = 1.0  # Always 1.0 for FLUX
   guidance = 3.5  # Separate parameter
   ```

2. **Better text understanding**: T5 encoder
3. **Improved composition**: Better spatial awareness
4. **Consistent anatomy**: Rarely fails on hands

### Strengths

- **Unmatched quality**: Best available quality
- **Text rendering**: Can generate readable text
- **Prompt adherence**: Follows complex prompts
- **Consistency**: Reliable anatomy/physics
- **Modern knowledge**: Trained on recent data

### Weaknesses

- **Resource heavy**: 12GB+ VRAM minimum
- **Slower**: 2-3x slower than SDXL
- **Large size**: 24GB+ for full precision
- **New ecosystem**: Fewer LoRAs initially
- **Different workflow**: Requires adjustment

### Optimal Settings

```python
{
    "resolution": "1024x1024",
    "steps": 20-25,
    "cfg": 1.0,  # CRITICAL: Must be 1.0
    "guidance": 3.5,
    "sampler": "euler",
    "scheduler": "simple",
    "model": "flux-fp8.safetensors"  # For lower VRAM
}
```

### FLUX Workflow Example

```python
# ComfyUI FLUX workflow
[FLUX Checkpoint] → [CLIP Encode] → [FLUX Guidance]
                                           ↓
                        [KSampler: cfg=1.0] → [VAE Decode]
```

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Alternative |
|----------|------------------|-------------|
| Quick prototypes | SD 1.5 | FLUX-schnell |
| Photorealism | FLUX | SDXL |
| Anime/Manga | Pony | SD 1.5 + LoRA |
| Game assets | SDXL | SD 1.5 |
| Product renders | FLUX | SDXL |
| Artistic styles | SD 1.5 | SDXL |
| Text in images | FLUX | SDXL (limited) |
| Low VRAM (4-6GB) | SD 1.5 | SD 2.1 |
| Best quality | FLUX | SDXL + Refiner |

### By Hardware

| VRAM | Optimal Model | Settings |
|------|--------------|----------|
| 4GB | SD 1.5 | 512×512, FP16 |
| 6GB | SD 2.1 | 768×768, FP16 |
| 8GB | SDXL | 1024×1024, FP16, no refiner |
| 12GB | FLUX-fp8 | 1024×1024, optimized |
| 16GB+ | Any model | Full quality |

## Performance Comparison

### Generation Speed (RTX 4090)

| Model | Resolution | Steps | Time | It/s |
|-------|------------|-------|------|------|
| SD 1.5 | 512×512 | 25 | 3s | 8.3 |
| SD 2.1 | 768×768 | 30 | 6s | 5.0 |
| SDXL | 1024×1024 | 30 | 15s | 2.0 |
| Pony | 1024×1024 | 25 | 12s | 2.1 |
| FLUX | 1024×1024 | 25 | 40s | 0.6 |

### Quality Metrics

| Model | FID Score | CLIP Score | User Preference |
|-------|-----------|------------|-----------------|
| SD 1.5 | 12.6 | 31.7 | 72% |
| SD 2.1 | 10.2 | 32.5 | 78% |
| SDXL | 8.1 | 33.8 | 86% |
| Pony | 9.2* | 32.1* | 91%** |
| FLUX | 6.3 | 35.2 | 94% |

*On anime dataset **Among target audience

## Migration Guide

### From SD 1.5 to SDXL

```python
# SD 1.5 prompt
"masterpiece, best quality, 1girl, sitting, park bench"

# SDXL prompt (more natural)
"A young woman sitting on a park bench in a sunny day, 
professional photography, shallow depth of field"
```

### From SDXL to FLUX

```python
# SDXL settings
cfg_scale = 7.5
steps = 30

# FLUX settings
cfg = 1.0  # MUST be 1.0
guidance = 3.5
steps = 25
```

### Prompting Differences

| Model | Prompt Style | Example |
|-------|--------------|---------|
| SD 1.5 | Tag-based | "1girl, red hair, blue eyes, smile, outdoors" |
| SDXL | Natural + tags | "A girl with red hair and blue eyes smiling outdoors, masterpiece" |
| Pony | Score + tags | "score_9, 1girl, red hair, blue eyes, smile, outdoors" |
| FLUX | Natural language | "A cheerful young woman with vibrant red hair and striking blue eyes" |

## Future Considerations

### Emerging Trends

1. **Smaller, faster models**: Distillation techniques
2. **Better architectures**: Transformer-based models
3. **Multi-modal**: Combined image/video/3D
4. **Real-time generation**: Sub-second inference
5. **Mobile deployment**: Edge computing models

### Choosing Future-Proof Models

- **FLUX**: Current best for quality and capabilities
- **SDXL**: Stable choice with growing ecosystem
- **SD 1.5**: Will remain relevant for specialized uses

## Conclusion

Each model serves different needs:

- **SD 1.5**: Speed, compatibility, and low requirements
- **SD 2.x**: Middle ground (mostly superseded)
- **SDXL**: Quality and resolution balance
- **Pony**: Specialized excellence for anime/stylized
- **FLUX**: Cutting-edge quality and capabilities

Choose based on your specific requirements for quality, speed, hardware, and content type. The ecosystem continues to evolve rapidly, but understanding these foundational models will help you adapt to future developments.