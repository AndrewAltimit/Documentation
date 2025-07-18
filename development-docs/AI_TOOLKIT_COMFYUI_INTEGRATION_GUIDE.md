# AI Toolkit & ComfyUI Integration Guide

This guide documents the complete workflow for training LoRA models with AI Toolkit and using them in ComfyUI, including all the gotchas and solutions discovered during implementation. Updated for 2024 with the latest API changes and best practices.

## Table of Contents
1. [Training LoRAs with AI Toolkit](#training-loras-with-ai-toolkit)
2. [Transferring LoRAs to ComfyUI](#transferring-loras-to-comfyui)
3. [FLUX Workflow Requirements](#flux-workflow-requirements)
4. [Common Issues and Solutions](#common-issues-and-solutions)

## Training LoRAs with AI Toolkit

### Critical Dataset Path Requirement

**Issue**: Dataset path must be absolute, not relative.

❌ **Wrong**:
```json
"dataset_path": "pixel_cat_dataset"
```

✅ **Correct**:
```json
"dataset_path": "/ai-toolkit/datasets/pixel_cat_dataset"
```

**Error if wrong**: `[Errno 2] No such file or directory: 'pixel_cat_dataset'`

### Dataset Upload Structure

When uploading datasets via MCP:
1. Images are saved to `/ai-toolkit/datasets/[dataset_name]/`
2. Caption files (.txt) are created automatically
3. A `.aitk_size.json` metadata file is generated for AI Toolkit compatibility

### Training Configuration Best Practices

```python
{
    "name": "pixel_cat_birthday_lora_v2",
    "model_name": "ostris/Flex.1-alpha",  # Use publicly accessible models
    "dataset_path": "/ai-toolkit/datasets/pixel_cat_dataset",  # FULL PATH!
    "resolution": 512,
    "steps": 1000,  # ~30 minutes for FLUX on RTX 4090/A6000
    "rank": 16,
    "alpha": 16,
    "low_vram": true,  # Essential for <24GB GPUs
    "gradient_checkpointing": true,  # Further memory optimization
    "mixed_precision": "fp16",  # Faster training with minimal quality loss
    "trigger_word": "pixel_cat",
    "test_prompts": [
        # Include trigger word in all prompts
        "pixel_cat, a snowshoe cat sitting elegantly...",
        # 3 similar + 1 creative prompt recommended
    ]
}
```

### Training Time Estimates

Based on actual training runs (2024 hardware):
- **FLUX/Flex LoRA**: ~30 minutes for 1000 steps (RTX 4090)
- **Training speed**: ~1.15 sec/iteration (with gradient checkpointing)
- **Model loading**: 5-10 minutes initial setup
- **SDXL LoRA**: ~15-20 minutes for 1000 steps
- **Memory usage**: 16-20GB VRAM with optimizations

## Transferring LoRAs to ComfyUI

### The Size Limit Challenge

Standard HTTP upload fails for files >50-100MB. Our 112MB LoRA required chunked upload.

### Chunked Upload Discovery

**Important**: The MCP tools list endpoint may not show all available tools!

```bash
# This might not show chunked upload tools
curl http://192.168.0.152:8189/mcp/tools

# But they exist in the gist implementation!
# Always check: https://gist.github.com/AndrewAltimit/f2a21b1a075cc8c9a151483f89e0f11e

# Alternative: Use the /mcp/tools/list endpoint (newer API)
curl http://192.168.0.152:8189/mcp/tools/list
```

### Correct Chunked Upload Parameters

The documentation and actual implementation had differences:

**Initial attempt** (wrong parameters - common mistake):
```json
{
    "tool": "upload-lora-chunked-start",
    "arguments": {
        "filename": "model.safetensors",
        "total_chunks": 450  // ❌ Wrong parameter name
    }
}
```

**Correct parameters**:
```json
{
    "tool": "upload-lora-chunked-start",
    "arguments": {
        "upload_id": "uuid-here",      // ✅ Must provide this
        "filename": "model.safetensors",
        "total_size": 117861152,       // ✅ Size in bytes, not chunks
        "metadata": {}
    }
}
```

**Append chunks with**:
```json
{
    "tool": "upload-lora-chunked-append",
    "arguments": {
        "upload_id": "same-uuid",
        "chunk": "base64-data",        // ✅ "chunk" not "chunk_data"
        "chunk_index": 0
    }
}
```

### Complete Transfer Script

See `transfer_lora_between_services.py` for working implementation:
- Downloads from AI Toolkit with metadata
- Generates UUID for upload session
- Splits into 256KB chunks (450 chunks for 112MB file)
- Uploads with progress tracking (~21 seconds for 112MB on gigabit LAN)
- Verifies model availability
- Handles network interruptions with retry logic

### Useful Commands

```bash
# Check training status (AI Toolkit API v2)
curl -s http://192.168.0.152:8675/api/jobs | jq '.jobs[0]'

# Alternative status endpoint (newer)
curl -s http://192.168.0.152:8675/api/v2/training/status

# Transfer LoRA between services
python3 transfer_lora_between_services.py

# List available LoRAs in ComfyUI
curl -X POST http://192.168.0.152:8189/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "list-loras", "arguments": {}}'

# Verify LoRA metadata
curl -X POST http://192.168.0.152:8189/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "get-lora-info", "arguments": {"lora_name": "model.safetensors"}}'
```

## FLUX Workflow Requirements

### Critical Differences from SD Workflows

FLUX workflows are NOT the same as Stable Diffusion workflows!

**❌ Wrong (SD-style)**:
```json
{
    "class_type": "KSampler",
    "inputs": {
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": ["3", 0],
        "negative": ["4", 0]  // Direct conditioning
    }
}
```

**✅ Correct (FLUX-style - 2024 update)**:
```json
{
    "class_type": "FluxGuidance",
    "inputs": {
        "guidance": 3.5,  // NOT cfg! Range: 1.0-10.0
        "conditioning": ["positive_clip", 0]
    }
},
{
    "class_type": "KSampler",
    "inputs": {
        "cfg": 1.0,  // Always 1.0 for FLUX
        "sampler_name": "heunpp2",  // NOT euler (also try: dpmpp_2m, euler_ancestral)
        "scheduler": "simple",  // NOT normal (also: beta, sgm_uniform)
        "positive": ["flux_guidance", 0],  // From FluxGuidance
        "negative": ["empty_clip", 0]  // Cannot be null!
    }
}
```

### FLUX Workflow Node Structure (2024 Best Practices)

1. **CheckpointLoaderSimple** → Loads FLUX model (dev or schnell variant)
2. **LoraLoader** → Loads custom LoRA (strength 0.8-1.0 recommended)
3. **CLIPTextEncode** (positive) → Your prompt with trigger word
4. **CLIPTextEncode** (negative) → Empty string (NOT null!)
5. **FluxGuidance** → Applies guidance to positive conditioning (3.5-5.0 range)
6. **EmptyLatentImage** → 1024x1024 for FLUX (or 512x512 for faster generation)
7. **KSampler** → cfg=1.0, heunpp2, simple scheduler, 20-30 steps
8. **VAEDecode** → Decodes latent (use tiled=True for large images)
9. **SaveImage** → Saves result (supports PNG/JPEG/WebP)

### LoRA Connection Points

The LoRA loader MUST be connected properly:
- Model output → Next model input
- CLIP output → Next CLIP input

```json
"LoraLoader": {
    "inputs": {
        "model": ["CheckpointLoader", 0],  // From checkpoint
        "clip": ["CheckpointLoader", 1],   // From checkpoint
        "lora_name": "your_lora.safetensors",
        "strength_model": 1.0,
        "strength_clip": 1.0
    }
}
```

## Common Issues and Solutions

### Issue 1: "Value not in list" for LoRA

**Cause**: LoRA file not in ComfyUI's models directory
**Solution**: Use chunked upload or verify transfer completed
**Alternative**: Check file permissions (should be readable by ComfyUI process)

### Issue 2: "'NoneType' object is not iterable"

**Cause**: Passing null to KSampler negative input
**Solution**: Use empty CLIPTextEncode node, not null

### Issue 3: Size mismatch in chunked upload

**Cause**: Using wrong parameter name ("chunk_data" vs "chunk")
**Solution**: Check actual API implementation, not just docs
**Note**: API parameter names may vary between versions - always test first

### Issue 4: Training fails immediately

**Cause**: Relative dataset path
**Solution**: Always use absolute paths starting with /ai-toolkit/datasets/

### Issue 5: No LoRA effect in generated images

**Cause**: LoRA not properly connected in workflow or strength too low
**Solution**: Ensure LoRA loader is between checkpoint and sampler, use strength ≥0.8
**Debug**: Check LoRA weights are actually being applied with get-lora-info tool

## Quick Checklist

Before training:
- [ ] Dataset uploaded to `/ai-toolkit/datasets/[name]/`
- [ ] Using absolute dataset path in config
- [ ] Model name is publicly accessible (not gated)
- [ ] Trigger word defined and used in test prompts
- [ ] Sufficient disk space (>50GB for checkpoints)
- [ ] VRAM requirements met (16GB+ recommended)

Before transferring:
- [ ] Check file size (<100MB use direct, >100MB use chunked)
- [ ] Generate UUID for chunked upload session
- [ ] Use correct parameter names (check implementation)

Before generating:
- [ ] LoRA appears in list-loras output
- [ ] Using FLUX workflow structure (not SD)
- [ ] FluxGuidance node present (guidance 3.5-5.0)
- [ ] CFG = 1.0, sampler = heunpp2 or dpmpp_2m
- [ ] Empty negative prompt (not null)
- [ ] Trigger word in prompt
- [ ] Appropriate step count (20-30 for quality)
- [ ] VAE tile mode enabled for large images

## Example Files

Working examples from successful implementation:
- **Transfer Script**: `transfer_lora_between_services.py` - Complete chunked upload implementation
- **FLUX Workflow**: `pixel_birthday_flux_workflow_fixed.json` - Properly structured FLUX workflow with LoRA
- **SD Workflow (incorrect)**: `pixel_birthday_workflow.json` - Shows what NOT to do for FLUX

## Resources

- [AI Toolkit MCP Gist](https://gist.github.com/AndrewAltimit/fc5ba068b73e7002cbe4e9721cebb0f5)
- [ComfyUI MCP Gist](https://gist.github.com/AndrewAltimit/f2a21b1a075cc8c9a151483f89e0f11e)
- [FLUX Model Documentation](https://github.com/black-forest-labs/flux)
- [ComfyUI Custom Nodes](https://github.com/comfyanonymous/ComfyUI_examples)

## Related Documentation

- [LORA_TRANSFER_DOCUMENTATION.md](LORA_TRANSFER_DOCUMENTATION.md) - Detailed transfer methods
- [MCP_TOOLS.md](MCP_TOOLS.md) - Complete MCP tool reference
- [CONTAINERIZED_CI.md](CONTAINERIZED_CI.md) - Docker setup for services
