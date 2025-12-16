---
layout: docs
title: LoRA Training Guide
parent: AI/ML Documentation
nav_order: 3
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


{: .no_toc }

<div class="code-example" markdown="1">
Learn how to train custom LoRA (Low-Rank Adaptation) models to add new styles, concepts, or characters to Stable Diffusion models.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a training technique that allows you to fine-tune large models like Stable Diffusion by training only a small number of parameters. Instead of modifying the entire model, LoRA adds trainable rank decomposition matrices to existing weights.

In 2024, LoRA training has become significantly more accessible with tools like AI Toolkit, Kohya SS, and cloud-based solutions. New variants like LCM-LoRA and DoRA offer specialized capabilities beyond traditional style/character training.

### Key Benefits

- **Efficient**: Requires 100-1000x less storage than full fine-tuning
- **Flexible**: Multiple LoRAs can be combined and weighted
- **Fast**: Training takes hours instead of days
- **Preserves Base Model**: Original model remains unchanged

### How LoRA Works

LoRA decomposes weight updates into low-rank matrices:

```
W' = W + ΔW = W + BA
```

Where:
- `W` is the original weight matrix (frozen)
- `B` and `A` are low-rank matrices (trainable)
- Rank `r << d` (typically 4-128)

## Setup with AI Toolkit

### Using Docker (Recommended)

```bash
# Clone AI Toolkit MCP setup
git clone https://github.com/andrewaltimit/Documentation ai-toolkit-trainer
cd ai-toolkit-trainer

# Build and start services
docker-compose build
docker-compose up -d

# Access services
# Web UI: http://localhost:8675
# MCP API: http://localhost:8190
```

### Required Hardware

| Model Type | Minimum VRAM | Recommended VRAM | Training Time (1k steps) |
|------------|--------------|------------------|-------------------------|
| SD 1.5     | 6GB          | 8GB              | 15-30 min              |
| SD 2.x     | 8GB          | 12GB             | 20-40 min              |
| SDXL       | 12GB         | 16GB             | 30-60 min              |
| SD3 Medium | 14GB         | 20GB             | 45-90 min              |
| FLUX Dev   | 16GB         | 24GB             | 60-120 min             |
| FLUX Schnell| N/A         | N/A              | (Distilled, no training)|

## Dataset Preparation

### Image Requirements

1. **Quality Standards**:
   - High resolution (512x512 minimum, 1024x1024 for FLUX/SDXL)
   - Clear, well-lit subjects
   - Diverse angles and contexts
   - Consistent quality across dataset

2. **Dataset Size Guidelines**:
   - **Style LoRA**: 10-50 images (quality > quantity)
   - **Character/Person**: 20-100 images (varied poses/expressions)
   - **Object/Concept**: 15-50 images (multiple angles)
   - **Complex Style**: 50-200 images (diverse examples)
   - **LCM-LoRA**: 100-500 images (for speed optimization)

### Captioning Best Practices

Each image needs a corresponding `.txt` file with the same name:

```
dataset/
├── image001.png
├── image001.txt
├── image002.png
├── image002.txt
└── ...
```

#### Caption Structure

```
[trigger_word] [subject] [details], [style], [quality], [composition]
```

Examples:
```
# Character LoRA
xyz_person portrait of a woman, wearing a blue dress, soft lighting, professional photograph

# Style LoRA
xyz_style digital painting of a landscape, vibrant colors, fantasy art style, highly detailed

# Object LoRA
xyz_object a red sports car, studio lighting, product photography, white background
```

#### Model-Specific Caption Guidelines

**FLUX Captions**:
```
xyz_style portrait of a knight, three-quarter view, in a medieval castle, torch lighting
```
- Natural language preferred
- Include camera angles and lighting
- Detailed scene descriptions

**SDXL Captions**:
```
xyz_style, digital artwork, fantasy knight, detailed armor, castle interior, dramatic lighting, masterpiece
```
- Mix of tags and natural language
- Quality tags important
- Booru-style tags work well

**SD3 Captions**:
```
A knight in shining armor standing in a medieval castle, xyz_style artwork
```
- Very natural language
- Trigger word can be mid-sentence
- Detailed descriptions excel

### Automated Captioning

Using AI Toolkit's caption generation:

```python
# Modern captioning options
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "generate-captions",
    "arguments": {
        "dataset_path": "/ai-toolkit/datasets/my-dataset",
        "caption_model": "blip2",  # or "wd14", "cogvlm", "florence2"
        "trigger_word": "xyz_style",
        "caption_style": "natural",  # or "booru", "mixed"
        "add_quality_tags": true
    }
})
```

**Caption Models**:
- **BLIP2**: Best for natural descriptions
- **WD14**: Excellent for anime/booru tags
- **CogVLM**: Advanced understanding
- **Florence2**: Detailed region descriptions

## Training Configuration

### Basic Configuration via MCP

```python
import requests
import json

# Create training configuration
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "create-training-config",
    "arguments": {
        "name": "my-style-lora",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "dataset_path": "/ai-toolkit/datasets/my-style",
        "resolution": 512,
        "batch_size": 1,
        "learning_rate": 0.0002,
        "steps": 2000,
        "rank": 32,
        "alpha": 32,
        "trigger_word": "xyz_style",
        "test_prompts": [
            "xyz_style portrait of a woman",
            "xyz_style landscape with mountains",
            "xyz_style still life painting",
            "xyz_style in a cyberpunk city at night"
        ]
    }
})
```

### Key Parameters Explained

#### Learning Rate
- **Default**: 2e-4 (0.0002) - Good for most cases
- **Conservative**: 1e-4 (0.0001) - Slower but safer
- **Aggressive**: 3e-4 (0.0003) - For stubborn concepts  
- **Fine-tuning**: 5e-5 (0.00005) - For existing styles
- **Prodigy Optimizer**: 1.0 - Self-adjusting (recommended)
- **AdamW8bit**: 3e-4 - Memory efficient option

#### Steps Calculation
```
Optimal Steps = 100 × number_of_images
```

Examples:
- 20 images → 2000 steps
- 50 images → 5000 steps
- Single image → 100-500 steps

#### Rank Selection

| Rank  | Use Case                          | File Size |
|-------|-----------------------------------|-----------|
| 4-8   | Simple style adjustments          | ~10MB     |
| 16-32 | Standard character/style LoRAs    | ~50MB     |
| 64-96 | Complex concepts, multiple subjects| ~150MB    |
| 128   | Maximum detail/flexibility        | ~300MB    |

### Advanced Configuration

```python
{
    "name": "advanced-lora",
    "model_name": "black-forest-labs/FLUX.1-dev",  # FLUX model
    "dataset_path": "/ai-toolkit/datasets/my-dataset",
    
    # Training parameters
    "resolution": 1024,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0002,
    "lr_scheduler": "cosine_with_restarts",
    "lr_warmup_steps": 100,
    
    # LoRA parameters
    "rank": 32,
    "alpha": 32,
    "dropout": 0.1,
    
    # Optimization
    "optimizer": "adamw",
    "mixed_precision": "fp16",
    "gradient_checkpointing": true,
    "low_vram": true,
    
    # Regularization
    "prior_preservation": true,
    "prior_loss_weight": 1.0,
    
    # Sampling
    "sample_every": 100,
    "sample_prompts": [...],
    
    # Advanced
    "network_dim": 32,
    "network_alpha": 16,
    "clip_skip": 2,
    "max_token_length": 225,
    
    # 2024 Features
    "use_prodigy_optimizer": false,
    "masked_loss": false,
    "debiased_estimation": true,
    "ip_noise_gamma": 0.1,
    "min_snr_gamma": 5
}
```

## Training Process

### Starting Training

```python
# Start training job
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "start-training",
    "arguments": {
        "config_name": "my-style-lora"
    }
})

job_id = response.json()["result"]["job_id"]
```

### Monitoring Progress

```python
# Check training status
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "get-training-status",
    "arguments": {
        "job_id": job_id
    }
})

# Get training logs
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "get-training-logs",
    "arguments": {
        "job_id": job_id,
        "lines": 50
    }
})
```

### Understanding Training Metrics

Key metrics to monitor:
- **Loss**: Should decrease over time (target: 0.05-0.15)
- **Learning Rate**: Follows scheduler (cosine, linear, etc.)
- **Gradient Norm**: Indicates training stability

## Dataset Upload

### Direct Upload (Small Datasets)

```python
import base64

images = []
for img_path, caption_path in dataset_files:
    with open(img_path, "rb") as f:
        img_content = base64.b64encode(f.read()).decode()
    with open(caption_path, "r") as f:
        caption = f.read().strip()
    
    images.append({
        "filename": os.path.basename(img_path),
        "content": img_content,
        "caption": caption
    })

response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "upload-dataset",
    "arguments": {
        "dataset_name": "my-style",
        "images": images
    }
})
```

### Chunked Upload (Large Files)

```python
# For LoRAs > 100MB
CHUNK_SIZE = 256 * 1024  # 256KB chunks

# Start upload
upload_id = str(uuid.uuid4())
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "upload-lora-chunked-start",
    "arguments": {
        "upload_id": upload_id,
        "filename": "large_lora.safetensors",
        "total_size": file_size,
        "metadata": {
            "trigger_words": ["xyz_style"],
            "base_model": "FLUX"
        }
    }
})

# Upload chunks
for i in range(0, file_size, CHUNK_SIZE):
    chunk = file_data[i:i+CHUNK_SIZE]
    response = requests.post("http://localhost:8190/mcp/tool", json={
        "tool": "upload-lora-chunked-append",
        "arguments": {
            "upload_id": upload_id,
            "chunk": base64.b64encode(chunk).decode(),
            "chunk_index": i // CHUNK_SIZE
        }
    })

# Finalize
response = requests.post("http://localhost:8190/mcp/tool", json={
    "tool": "upload-lora-chunked-finish",
    "arguments": {
        "upload_id": upload_id
    }
})
```

## Specialized LoRA Types

### LCM-LoRA Training

Train for fast generation:
```python
{
    "name": "lcm-lora",
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "teacher_model": "lcm-sdxl",
    "distillation_mode": true,
    "learning_rate": 1e-4,
    "steps": 5000,
    "guidance_scale_range": [1.0, 2.0]
}
```

### DoRA (Weight-Decomposed LoRA)

```python
{
    "name": "dora-style",
    "use_dora": true,
    "magnitude_learning_rate": 1e-4,
    "direction_learning_rate": 2e-4,
    "rank": 64  # Can use higher ranks effectively
}
```

### Control-LoRA

Combine LoRA with ControlNet:
```python
{
    "name": "control-lora",
    "control_type": "openpose",
    "control_weight": 0.5,
    "train_control_adapter": true
}
```

## Common Training Scenarios

### Style LoRA

Training an artistic style:

```python
config = {
    "name": "watercolor-style",
    "dataset_path": "/ai-toolkit/datasets/watercolor",
    "trigger_word": "wc_style",
    "steps": 1500,  # 15 images × 100
    "rank": 16,
    "test_prompts": [
        "wc_style painting of a sunset",
        "wc_style portrait of an elderly man",
        "wc_style still life with flowers",
        "wc_style abstract composition"
    ]
}
```

### Character LoRA

Training a specific character:

```python
config = {
    "name": "game-character",
    "dataset_path": "/ai-toolkit/datasets/character",
    "trigger_word": "xyz_character",
    "steps": 3000,  # 30 images × 100
    "rank": 32,
    "clip_skip": 2,  # For anime styles
    "test_prompts": [
        "xyz_character standing pose, full body",
        "xyz_character portrait, smiling",
        "xyz_character in battle armor",
        "xyz_character in casual clothes"
    ]
}
```

### Photographic Subject

Training a person or object:

```python
config = {
    "name": "product-photos",
    "model_name": "stabilityai/stable-diffusion-2-1",
    "dataset_path": "/ai-toolkit/datasets/product",
    "trigger_word": "xyz_product",
    "resolution": 768,
    "steps": 2000,
    "rank": 24,
    "test_prompts": [
        "xyz_product on white background, product photography",
        "xyz_product in use, lifestyle photography",
        "xyz_product close-up detail shot",
        "xyz_product with natural lighting"
    ]
}
```

## Optimization Techniques

### Memory Optimization

For limited VRAM:

```python
{
    "low_vram": true,
    "gradient_checkpointing": true,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",
    "optimizer": "adafactor"  # Uses less memory than AdamW
}
```

### Speed Optimization

For faster training:

```python
{
    "disable_sampling": true,  # Skip sample generation
    "save_every": 500,  # Less frequent saves
    "optimizer": "lion",  # Faster convergence
    "lr_scheduler": "constant",  # Simple scheduler
    "cache_latents": true  # Pre-compute VAE encodings
}
```

### Quality Optimization

For best results:

```python
{
    "rank": 64,
    "alpha": 32,  # Lower than rank for regularization
    "learning_rate": 0.0001,
    "lr_scheduler": "cosine_with_restarts",
    "steps": 5000,
    "sample_every": 100,
    "save_every": 250,
    "prior_preservation": true
}
```

## Troubleshooting

### Common Issues

#### Overfitting

**Symptoms**: LoRA only generates training images
**Solutions**:
- Reduce learning rate
- Decrease training steps
- Add dropout (0.1-0.3)
- Use more diverse captions
- Enable prior preservation

#### Underfitting

**Symptoms**: LoRA has no effect
**Solutions**:
- Increase learning rate
- Train for more steps
- Increase rank
- Check trigger word usage
- Verify dataset quality

#### Style Bleeding

**Symptoms**: LoRA affects unintended aspects
**Solutions**:
- Improve caption specificity
- Use regularization images
- Reduce LoRA strength when using
- Train with narrower focus

### Training Diagnostics

Monitor these indicators:

```python
# Good training
- Loss: Steadily decreasing
- Sample quality: Improving each checkpoint
- Gradient norm: Stable (not exploding)

# Problems
- Loss: Plateaued early → Increase learning rate
- Loss: Oscillating → Decrease learning rate
- Loss: Sudden spike → Check for corrupt data
```

## Advanced Techniques

### Multi-Concept Training

Train multiple concepts in one LoRA:

```python
# Dataset structure
dataset/
├── concept1/
│   ├── xyz_cat_001.jpg
│   └── xyz_cat_001.txt: "xyz_cat photo of a cat..."
├── concept2/
│   ├── xyz_dog_001.jpg
│   └── xyz_dog_001.txt: "xyz_dog photo of a dog..."
```

### DreamBooth-style Training

For maximum fidelity:

```python
{
    "model_name": "runwayml/stable-diffusion-v1-5",
    "instance_prompt": "xyz_person person",
    "class_prompt": "person",
    "prior_preservation": true,
    "prior_preservation_weight": 1.0,
    "num_class_images": 200,
    "steps": 3000
}
```

### Progressive Training

Train in stages for complex concepts:

```python
# Stage 1: Basic form
train_config(rank=16, steps=1000, lr=0.0002)

# Stage 2: Details
train_config(rank=32, steps=2000, lr=0.0001, resume_from="stage1")

# Stage 3: Polish
train_config(rank=32, steps=1000, lr=0.00005, resume_from="stage2")
```

## Using Trained LoRAs

### In ComfyUI

```python
# Load in workflow
{
    "type": "LoraLoader",
    "inputs": {
        "lora_name": "my-style-lora.safetensors",
        "strength_model": 0.8,
        "strength_clip": 0.8
    }
}
```

### Combining Multiple LoRAs

```python
# Stack LoRAs
[Base Model] → [LoRA 1 @ 0.6] → [LoRA 2 @ 0.4] → [LoRA 3 @ 0.3]
```

### Optimal Strength Settings

| LoRA Type | Model Strength | CLIP Strength |
|-----------|----------------|---------------|
| Style     | 0.6-1.0        | 0.6-1.0       |
| Character | 0.7-0.9        | 0.7-0.9       |
| Pose      | 0.4-0.7        | 0.3-0.6       |
| Details   | 0.3-0.6        | 0.3-0.6       |

## Best Practices Summary

### Do's
✓ Use unique trigger words (xyz_style, not "style")  
✓ Include trigger word in all captions
✓ Vary descriptions while maintaining consistency
✓ Test with prompts during training
✓ Save checkpoints regularly
✓ Start with conservative settings
✓ Use regularization images for better generalization
✓ Monitor validation loss

### Don'ts
✗ Don't overtrain (watch for overfitting)
✗ Don't use copyrighted content without permission
✗ Don't skip dataset quality control
✗ Don't use generic trigger words
✗ Don't ignore sampling results
✗ Don't train at full resolution if not needed
✗ Don't mix incompatible model versions
✗ Don't forget to backup successful LoRAs

## Conclusion

LoRA training opens up endless possibilities for customizing AI image generation. With proper dataset preparation, configuration tuning, and monitoring, you can create LoRAs that seamlessly integrate new concepts while maintaining the flexibility of the base model. Start with simple projects and gradually tackle more complex training scenarios as you gain experience.

---

## See Also
- [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) - Understanding the base models you'll train on
- [ComfyUI Guide](comfyui-guide.html) - Use trained LoRAs in advanced workflows
- [ControlNet](controlnet.html) - Combine LoRAs with ControlNet for precision control
- [Model Types](model-types.html) - Understanding LoRAs, checkpoints, and embeddings
- [Base Models Comparison](base-models-comparison.html) - Choosing the right base for training
- [Advanced Techniques](advanced-techniques.html) - Expert LoRA usage patterns
- [AI Fundamentals](../technology/ai.html) - Neural network training concepts
- [AI/ML Documentation Hub](index.html) - Complete AI/ML documentation index