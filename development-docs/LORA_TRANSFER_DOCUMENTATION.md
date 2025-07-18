# LoRA Model Transfer Documentation

This document describes the process of transferring trained LoRA models from AI Toolkit to ComfyUI. Updated for 2024 with the latest API changes and best practices.

## Overview

When training a LoRA model with AI Toolkit and using it in ComfyUI for image generation, the model file needs to be transferred between services. Both services run on the same server (192.168.0.152) but have separate model directories.

## Transfer Methods

### Method 1: API Transfer (Size Limited)

**Limitation**: HTTP APIs have request size limits that prevent uploading files larger than ~50-100MB (varies by server configuration).

For small LoRA files (<50MB):
```python
# Download from AI Toolkit (API v2)
response = requests.post('http://192.168.0.152:8190/mcp/tool', 
    headers={'Content-Type': 'application/json'},
    json={
        "tool": "download-model",
        "arguments": {
            "model_path": "model_name/model.safetensors",
            "include_metadata": True,
            "format": "base64"  # New in 2024
        }
    })

# Upload to ComfyUI (will fail for large files)
upload_response = requests.post('http://192.168.0.152:8189/mcp/tool', 
    headers={'Content-Type': 'application/json'},
    timeout=30,  # Add timeout for large requests
    json={
        "tool": "upload-lora",
        "arguments": {
            "filename": "model.safetensors",
            "content": base64_content,  # Too large!
            "overwrite": True  # New option in 2024
        }
    })
```

### Method 2: Server-Side File Copy (Recommended for Local Transfer)

Since both services run on the same server, the most efficient method is direct file copying:

```bash
# On the server (192.168.0.152)
# Copy from AI Toolkit outputs to ComfyUI models
cp /ai-toolkit/outputs/[model_name]/[model_file].safetensors \
   /comfyui/models/loras/[model_file].safetensors
```

For Docker deployments:
```bash
# Copy between Docker containers
docker cp ai-toolkit-container:/ai-toolkit/outputs/model.safetensors \
          comfyui-container:/comfyui/models/loras/model.safetensors

# Alternative: Using volumes
docker run --rm \
  -v ai-toolkit-outputs:/source:ro \
  -v comfyui-models:/dest \
  alpine cp /source/model.safetensors /dest/loras/
```

### Method 3: Chunked Upload (Recommended for API Transfer)

The ComfyUI MCP server supports chunked uploads for large files:
- Split file into 256KB chunks (configurable up to 1MB in 2024)
- Use upload-lora-chunked-start/append/finish sequence
- Successfully tested with files up to 500MB
- Supports resume on failure (new in 2024)

Example implementation:
```python
# See transfer_lora_between_services.py for full implementation
import uuid
import base64

upload_id = str(uuid.uuid4())
chunk_size = 256 * 1024  # 256KB chunks

# Start upload
start_response = requests.post(url, json={
    "tool": "upload-lora-chunked-start",
    "arguments": {
        "upload_id": upload_id,
        "filename": "model.safetensors",
        "total_size": file_size,
        "chunk_size": chunk_size,  # New in 2024
        "metadata": {"trained_on": "2024-01-01"}
    }
})

# Append chunks with progress tracking
for i, chunk in enumerate(chunks):
    append_response = requests.post(url, json={
        "tool": "upload-lora-chunked-append",
        "arguments": {
            "upload_id": upload_id,
            "chunk": base64.b64encode(chunk).decode(),
            "chunk_index": i
        }
    })

# Finish upload
finish_response = requests.post(url, json={
    "tool": "upload-lora-chunked-finish",
    "arguments": {
        "upload_id": upload_id,
        "verify_checksum": True  # New in 2024
    }
})
```

## Example Transfer Case

For a typical FLUX LoRA model:
- **Model**: custom_lora_v2.safetensors
- **Size**: 100-150 MB (requires chunked upload)
- **Source**: `/ai-toolkit/outputs/[model_name]/[model_name].safetensors`
- **Destination**: `/comfyui/models/loras/[model_name].safetensors`
- **Transfer time**: ~20-30 seconds on gigabit LAN

## Workflow Integration

Once the LoRA is available in ComfyUI's models directory, it can be used in workflows:

```json
{
  "2": {
    "class_type": "LoraLoader",
    "inputs": {
      "lora_name": "pixel_cat_birthday_lora.safetensors",
      "strength_model": 1.0,
      "strength_clip": 1.0,
      "model": ["1", 0],  // From CheckpointLoaderSimple
      "clip": ["1", 1]    // From CheckpointLoaderSimple
    }
  }
}
```

## Important Notes

1. **LoRA Weight**: Always connect the LoRA loader between the checkpoint loader and the sampler
2. **Strength**: Use 0.8-1.0 for full effect, 0.3-0.7 for subtle influence
3. **Trigger Words**: Include the trigger word in prompts for activation
4. **Verification**: Use list-loras tool to verify the model is available after transfer
5. **FLUX Workflows**: Require different structure - see AI_TOOLKIT_COMFYUI_INTEGRATION_GUIDE.md
6. **Model Compatibility**: Ensure LoRA matches base model architecture (SDXL/FLUX)
7. **Memory Usage**: Large LoRAs may require additional VRAM allocation

## Error Prevention

Common issues and solutions:
- **"Value not in list" error**: LoRA file not found in ComfyUI models directory
  - Solution: Check file permissions and path
- **"Request Entity Too Large"**: File too big for HTTP upload
  - Solution: Use chunked transfer or adjust server limits
- **No LoRA effect**: LoRA not properly connected or strength too low
  - Solution: Check workflow connections and increase strength
- **"NoneType not iterable"**: Null value in workflow
  - Solution: Use empty string for negative prompt
- **Wrong parameters**: API parameter mismatch
  - Solution: Check actual implementation and API version
- **Size mismatch**: Chunk upload fails
  - Solution: Verify chunk size and total size match
- **Timeout errors**: Large file transfer timeout
  - Solution: Increase timeout or use smaller chunks

## Key Discoveries

1. **Hidden Tools**: The MCP tools list may not show all available tools!
   - Always check the source implementation
   - [ComfyUI MCP gist](https://gist.github.com/AndrewAltimit/f2a21b1a075cc8c9a151483f89e0f11e)
   - Chunked upload tools exist even if not listed in /mcp/tools

2. **API Versioning**: Different MCP servers may have different API versions
   - Check /mcp/version endpoint if available
   - Parameter names may vary between versions

3. **Network Optimization**: For local transfers
   - Use server-side copy when possible
   - Increase chunk size for faster transfers
   - Consider compression for text-heavy models

## Related Documentation

- [AI_TOOLKIT_COMFYUI_INTEGRATION_GUIDE.md](AI_TOOLKIT_COMFYUI_INTEGRATION_GUIDE.md) - Complete integration guide
- [MCP_TOOLS.md](MCP_TOOLS.md) - MCP tool reference
- [CONTAINERIZED_CI.md](CONTAINERIZED_CI.md) - Docker setup for services
