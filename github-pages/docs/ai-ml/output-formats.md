---
layout: default
title: Output Formats Guide
parent: AI/ML Documentation
nav_order: 6
---

# Output Formats Guide
{: .no_toc }

<div class="code-example" markdown="1">
Comprehensive guide to different output formats in AI generation: images, videos, audio, text, and multimodal outputs.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Modern AI models can generate various types of content beyond static images. This guide covers the different output formats, their generation methods, and best practices for each medium.

## Image Generation

### Standard Image Formats

#### PNG (Recommended)
```yaml
Format: Lossless compression
Bit Depth: 8-bit (24-bit color) or 16-bit
Alpha Channel: Supported
File Size: Larger
Best For: Final outputs, transparency needs
```

#### JPEG
```yaml
Format: Lossy compression  
Quality: Adjustable (70-95 recommended)
Alpha Channel: Not supported
File Size: Smaller
Best For: Sharing, web use
```

#### WebP
```yaml
Format: Modern compression
Features: Lossy/lossless, animation
File Size: 25-35% smaller than JPEG
Best For: Web optimization
```

### Image Parameters

```python
{
    "width": 1024,
    "height": 1024,
    "batch_size": 4,
    "format": "png",
    "metadata": {
        "prompt": "...",
        "seed": 12345,
        "model": "..."
    }
}
```

### Resolution Standards

| Aspect Ratio | SD 1.5 | SDXL/FLUX | Use Case |
|--------------|--------|-----------|----------|
| 1:1 | 512×512 | 1024×1024 | Social media |
| 4:3 | 576×432 | 1152×864 | Photography |
| 16:9 | 768×432 | 1344×768 | Wallpapers |
| 9:16 | 432×768 | 768×1344 | Mobile/Stories |
| 2:3 | 512×768 | 1024×1536 | Portraits |

### High-Resolution Generation

#### Direct Generation
```python
# FLUX/SDXL can generate directly at high res
{
    "width": 2048,
    "height": 2048,
    "steps": 30,
    "memory_optimization": true
}
```

#### Upscaling Pipeline
```
Base Generation (1024×1024) → AI Upscaler → High Res (4096×4096)
                                   ↓
                            Img2Img Refinement
```

### HDR and Extended Formats

#### EXR Format
```python
# For HDR/professional workflows
{
    "format": "exr",
    "bit_depth": 32,
    "color_space": "linear",
    "channels": ["R", "G", "B", "A", "Z"]  # Includes depth
}
```

## Video Generation

### Video Diffusion Models

#### AnimateDiff
```yaml
Type: Temporal consistency module
Base: Any SD 1.5/SDXL model
Length: 16-32 frames typical
FPS: 8-24 fps
Resolution: Same as base model
```

#### Stable Video Diffusion (SVD)
```yaml
Type: Dedicated video model
Input: Single image
Output: 14-25 frames
Resolution: 1024×576
FPS: 6-30 fps configurable
```

#### ModelScope/ZeroScope
```yaml
Type: Text-to-video
Length: 16-24 frames
Resolution: 256×256 to 1024×576
Quality: Lower than image models
```

### Video Generation Workflow

```python
# ComfyUI AnimateDiff workflow
[Checkpoint] → [AnimateDiff Loader] → [Sampler] → [VAE Decode]
                                                        ↓
                                                [Video Combine] → MP4
```

### Video Parameters

```python
{
    "frames": 24,
    "fps": 12,
    "motion_module": "mm_sd_v15_v2.ckpt",
    "context_schedule": "uniform",
    "context_length": 16,
    "format": "mp4",
    "codec": "h264",
    "bitrate": "10M"
}
```

### Frame Interpolation

Increase smoothness with interpolation:
```
Original (8 fps) → RIFE/FILM → Smooth (24 fps)
```

Tools:
- **RIFE**: Real-time interpolation
- **FILM**: Google's frame interpolation
- **DAIN**: Depth-aware interpolation

### Video Formats

| Format | Container | Codec | Use Case |
|--------|-----------|-------|----------|
| MP4 | .mp4 | H.264/H.265 | Universal compatibility |
| WebM | .webm | VP9 | Web streaming |
| GIF | .gif | GIF89a | Short loops |
| APNG | .png | APNG | Animated PNG |
| MOV | .mov | ProRes | Professional editing |

## Audio Generation

### Audio Diffusion Models

#### AudioLDM
```yaml
Type: Text-to-audio
Duration: Up to 10 seconds
Sample Rate: 16kHz-48kHz
Quality: Good for effects
```

#### MusicGen
```yaml
Type: Text-to-music
Duration: Up to 30 seconds
Genres: Multiple styles
Control: Melody conditioning
```

#### Bark
```yaml
Type: Text-to-speech
Languages: Multiple
Features: Emotion, singing
Non-speech: Laughs, sighs
```

### Audio Generation Pipeline

```python
# Basic audio generation
text_prompt = "thunderstorm with heavy rain"
audio = audioldm_model.generate(
    prompt=text_prompt,
    duration=5.0,
    sample_rate=44100,
    guidance_scale=3.5
)
```

### Audio Parameters

```python
{
    "duration": 10.0,        # seconds
    "sample_rate": 44100,    # Hz
    "channels": 2,           # stereo
    "bit_depth": 16,         # bits
    "format": "wav",         # or mp3, flac
    "guidance_scale": 3.5,
    "steps": 50
}
```

### Audio Formats

| Format | Compression | Quality | File Size | Use Case |
|--------|-------------|---------|-----------|----------|
| WAV | None | Highest | Large | Professional |
| FLAC | Lossless | High | Medium | Archival |
| MP3 | Lossy | Good | Small | Distribution |
| OGG | Lossy | Good | Small | Web/Games |
| M4A | Lossy/Lossless | Variable | Variable | Apple ecosystem |

## Text Generation Integration

### Text in Images

#### Native Text Rendering
FLUX excels at text rendering:
```python
prompt = 'A sign that says "HELLO WORLD" in bold letters'
# FLUX can render this accurately
```

#### ControlNet Text
```python
# Use ControlNet with text image
[Text Image] → [ControlNet] → [Generation with text]
```

### Text Effects Workflow

```
Text Prompt → [Generate Base] → [Depth Map] → [3D Effect]
                                     ↓
                              [Style Transfer] → Final
```

### Font Styles via Prompting

```python
text_styles = {
    "handwritten": "handwritten text that says",
    "neon": "neon sign displaying",
    "carved": "stone carved inscription reading",
    "digital": "LED display showing",
    "graffiti": "street art graffiti spelling"
}
```

## Multimodal Outputs

### Image + Depth

Generate image with depth information:
```python
# ComfyUI workflow
[Generate] → [Image Output]
    ↓
[MiDaS Depth] → [Depth Map] → [Save EXR with Depth]
```

Uses:
- 3D reconstruction
- AR applications
- Post-processing effects

### Image + Segmentation

```python
# Generate with semantic maps
[Generate] → [Image]
    ↓
[SAM Segmentation] → [Masks] → [Layered PSD]
```

### Image + Motion Vectors

For video game integration:
```python
outputs = {
    "albedo": "color_information.png",
    "normal": "surface_normals.exr",
    "motion": "motion_vectors.exr",
    "depth": "depth_map.exr"
}
```

## 3D Output Formats

### 3D from 2D

#### DreamGaussian
```yaml
Input: Single image
Output: 3D Gaussian Splatting
Format: .ply, .splat
Time: 1-2 minutes
```

#### One-2-3-45
```yaml
Input: Single image
Output: Multi-view + 3D
Format: .obj with textures
Quality: Good geometry
```

### 3D Formats

| Format | Features | Use Case |
|--------|----------|----------|
| OBJ | Geometry + Materials | Universal |
| FBX | Animation + Rigging | Game engines |
| GLTF | Web optimized | Web 3D |
| USD | Scene description | VFX/Film |
| PLY | Point clouds | Scanning |

### Neural Radiance Fields (NeRF)

```python
# NeRF output structure
nerf_output/
├── checkpoints/     # Trained model
├── renders/         # Video outputs
├── depth_maps/      # Depth information
└── mesh/           # Extracted mesh
```

## Optimization Strategies

### Batch Processing

```python
# Efficient batch generation
batch_config = {
    "batch_size": 4,
    "variations": [
        {"seed": 1, "prompt_suffix": "morning light"},
        {"seed": 2, "prompt_suffix": "sunset"},
        {"seed": 3, "prompt_suffix": "night time"},
        {"seed": 4, "prompt_suffix": "foggy weather"}
    ]
}
```

### Format Selection

| Priority | Image | Video | Audio | 3D |
|----------|-------|-------|-------|-----|
| Quality | PNG/EXR | ProRes | WAV/FLAC | FBX |
| Size | JPEG/WebP | H.265 | MP3/AAC | GLTF |
| Speed | JPEG | H.264 | MP3 | OBJ |
| Compatibility | JPEG | H.264 | MP3 | OBJ |

### Compression Guidelines

```python
# Image compression
if file_size_critical:
    format = "jpeg"
    quality = 85
elif transparency_needed:
    format = "png"
    optimize = True
else:
    format = "webp"
    quality = 90
```

## Post-Processing Pipelines

### Image Pipeline
```
Raw Output → Color Grading → Sharpening → 
Format Conversion → Metadata Injection → Final Output
```

### Video Pipeline
```
Frame Sequence → Interpolation → Color Correction → 
Stabilization → Encoding → Audio Sync → Final Video
```

### Audio Pipeline
```
Raw Audio → Noise Reduction → EQ → 
Compression → Normalization → Format Export
```

## Metadata Standards

### Image Metadata
```python
metadata = {
    "prompt": "full prompt text",
    "negative_prompt": "negative prompt",
    "model": "model_name",
    "sampler": "euler_a",
    "steps": 25,
    "cfg_scale": 7.5,
    "seed": 12345,
    "size": "1024x1024",
    "created": "2024-01-01T00:00:00Z"
}
```

### Preservation Tools
- **PNG**: tEXt chunks
- **JPEG**: EXIF data
- **WebP**: XMP metadata
- **Videos**: MP4 metadata atoms

## Real-time Output

### Streaming Generation

```python
# Progressive image streaming
async def stream_generation():
    for step in range(total_steps):
        if step % 5 == 0:  # Every 5 steps
            preview = vae.decode(latents)
            yield encode_preview(preview)
```

### Live Preview Systems

```python
{
    "preview_method": "fast",  # fast/accurate
    "preview_interval": 5,     # steps
    "preview_size": 512,       # pixels
    "stream_format": "jpeg",   # lightweight
}
```

## Export Workflows

### Professional Pipeline

```python
# ComfyUI professional export
[Generate] → [16-bit PNG] → [Color Space Convert] → 
[EXR with passes] → [Archive]

passes = {
    "beauty": "final_render.exr",
    "diffuse": "diffuse_pass.exr", 
    "specular": "specular_pass.exr",
    "normal": "normal_pass.exr",
    "z_depth": "depth_pass.exr"
}
```

### Web Optimization

```python
# Automated web export
def optimize_for_web(image):
    # Resize for breakpoints
    sizes = [320, 768, 1024, 1920]
    outputs = {}
    
    for size in sizes:
        resized = image.resize(size)
        outputs[f"image-{size}w.webp"] = save_webp(resized, 85)
        outputs[f"image-{size}w.jpg"] = save_jpeg(resized, 85)
    
    return outputs
```

## Future Formats

### Emerging Standards

1. **AVIF**: Next-gen image format
2. **JXL**: JPEG XL for better compression
3. **Gaussian Splatting**: .splat files
4. **Neural Representations**: Model-based formats

### Integration Formats

```python
# Game engine integration
unity_export = {
    "textures": {
        "albedo": "DDS/BC7",
        "normal": "DDS/BC5",
        "masks": "DDS/BC4"
    },
    "models": "FBX",
    "animations": "FBX"
}
```

## Best Practices

### Format Selection Checklist

1. **Purpose**: Web, print, archive, real-time?
2. **Quality**: Lossless required?
3. **Size**: Storage/bandwidth constraints?
4. **Compatibility**: Target platform support?
5. **Features**: Alpha, HDR, metadata needed?

### Quality Settings

| Use Case | Image | Video | Audio |
|----------|-------|-------|-------|
| Archive | PNG/EXR | ProRes 422 | FLAC |
| Share | JPEG 90 | H.264 CRF 23 | MP3 320k |
| Web | WebP 85 | H.265 CRF 28 | MP3 192k |
| Preview | JPEG 70 | H.264 CRF 30 | MP3 128k |

## Conclusion

Understanding output formats enables you to:
- Choose optimal formats for your use case
- Balance quality and file size
- Ensure compatibility across platforms
- Preserve generation metadata
- Build efficient pipelines

The key is matching format capabilities to your specific requirements while considering the entire workflow from generation to final delivery.