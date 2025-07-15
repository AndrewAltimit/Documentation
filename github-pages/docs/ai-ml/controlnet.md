---
layout: docs
title: ControlNet Guide
parent: AI/ML Documentation
nav_order: 7
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


{: .no_toc }

<div class="code-example" markdown="1">
Master ControlNet for precise control over AI image generation using poses, edges, depth, and more.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What is ControlNet?

ControlNet is a neural network architecture that adds spatial control to diffusion models. It allows you to guide image generation using various types of conditioning inputs like human poses, edge maps, depth maps, and more, while maintaining the quality and capabilities of the base model.

### How ControlNet Works

```
Input Image → Preprocessor → Control Map
                                ↓
Text Prompt → Base Model + ControlNet → Controlled Output
```

ControlNet creates a trainable copy of the diffusion model's encoder blocks, which learns to respond to specific spatial conditions while preserving the original model's generation capabilities.

## ControlNet Types

### Pose Control

#### OpenPose
Detects human body keypoints and skeleton structure.

```yaml
Purpose: Human pose transfer
Keypoints: 18-25 body points
Includes: Body, hands, face
Best for: Character consistency, pose reference
```

**Preprocessor Options**:
- `openpose_full`: Body + hands + face
- `openpose_body`: Body only
- `openpose_hand`: Hands focus
- `openpose_face`: Face landmarks

**Example Workflow**:
```python
[Reference Image] → [OpenPose Preprocessor] → [Pose Skeleton]
                                                    ↓
"A warrior in armor" → [ControlNet OpenPose] → [Posed Character]
```

#### DWPose
More accurate pose estimation with better occlusion handling.

```yaml
Advantages: Better accuracy, stable tracking
Keypoints: More detailed skeleton
Performance: Slower but more reliable
Use case: Complex poses, partial visibility
```

### Edge Detection

#### Canny Edge
Classic edge detection algorithm for clean line extraction.

```yaml
Purpose: Preserve shapes and outlines
Parameters: Low/High threshold
Output: Binary edge map
Best for: Architecture, objects, clean lines
```

**Parameter Guide**:
```python
{
    "low_threshold": 100,   # Lower = more edges
    "high_threshold": 200,  # Higher = fewer edges
}
```

#### MLSD (M-LSD)
Detects straight lines and geometric structures.

```yaml
Purpose: Architectural elements
Specialty: Straight line detection
Best for: Buildings, interiors, technical drawings
```

#### SoftEdge (HED/PIDI)
Preserves more subtle edge information.

```yaml
Methods: HED, PIDI, PidiNet
Purpose: Artistic edge preservation  
Quality: Softer, more natural edges
Best for: Organic subjects, artistic styles
```

### Depth Control

#### MiDaS
Monocular depth estimation for general scenes.

```yaml
Versions: MiDaS v2.1, v3.0
Resolution: Multiple model sizes
Output: Relative depth map
Best for: General depth control
```

#### Zoe Depth
More accurate depth estimation with metric depth.

```yaml
Accuracy: Superior to MiDaS
Type: Metric depth (actual distances)
Training: NYU Depth v2, KITTI
Best for: Realistic depth, outdoor scenes
```

#### LeReS
Learning to Recover 3D Scene Shape.

```yaml
Quality: High quality depth
Features: Handles complex scenes
Speed: Slower than MiDaS
Best for: Complex compositions
```

### Semantic Control

#### Segmentation
Uses semantic segmentation maps for region-based control.

```yaml
Models: ADE20K, COCO, custom
Classes: 150+ object categories
Control: Per-region styling
Best for: Scene composition
```

**Color Mapping Example**:
```python
segmentation_colors = {
    "sky": [134, 193, 249],
    "building": [128, 128, 128],
    "tree": [0, 128, 0],
    "person": [255, 0, 0],
    "ground": [139, 69, 19]
}
```

#### Normal Maps
Surface normal information for 3D-aware generation.

```yaml
Purpose: 3D surface orientation
Format: RGB encoded normals
Use case: 3D consistency, lighting
Best for: Products, sculptures
```

### Line Art Control

#### Anime Line Art
Extracts clean lines suitable for anime/manga style.

```yaml
Purpose: Anime/manga line extraction
Cleanliness: Very clean lines
Style: Manga-appropriate
Best for: Anime characters, manga
```

#### Scribble
Converts rough sketches to control inputs.

```yaml
Modes: Scribble, Fake Scribble
Input: Hand-drawn sketches
Tolerance: High noise tolerance
Best for: Quick ideation
```

### Special Controls

#### Shuffle
Rearranges image content while preserving style.

```yaml
Purpose: Style transfer with layout change
Method: Spatial shuffling
Randomness: Controllable
Best for: Creative variations
```

#### Tile
Enables tiled/seamless generation and upscaling.

```yaml
Purpose: Seamless textures, upscaling
Method: Overlapping tiles
Quality: Maintains consistency
Best for: Patterns, super-resolution
```

#### Inpaint
Specialized control for masked region generation.

```yaml
Input: Image + Mask
Control: Only masked areas
Blending: Seamless integration
Best for: Object removal, editing
```

## Installation and Setup

### ComfyUI Installation

```bash
# Install ControlNet models
cd ComfyUI/models/controlnet

# Download models (example)
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
```

### Required Components

```yaml
Core:
- ComfyUI-ControlNet-Aux (preprocessors)
- ControlNet models (specific to base model)
- Base diffusion model (SD 1.5, SDXL, etc.)

Optional:
- Custom preprocessors
- Additional ControlNet models
```

## Basic Workflows

### Simple ControlNet Workflow

```python
# ComfyUI nodes
[Load Image] → [ControlNet Preprocessor] → Control Image
                                               ↓
[Load ControlNet] → [Apply ControlNet] ← [Positive Prompt]
                            ↓
                    [KSampler] → [VAE Decode] → [Save]
```

### Multi-ControlNet Setup

```python
# Stack multiple ControlNets
[OpenPose Control] → [Apply ControlNet 1]
                            ↓
[Depth Control] → [Apply ControlNet 2]
                            ↓
                    [KSampler]
```

### ControlNet Parameters

```python
{
    "strength": 1.0,        # 0-2, control influence
    "start_percent": 0.0,   # When to start applying
    "end_percent": 1.0,     # When to stop applying
    "control_mode": "balanced", # balanced/prompt/control
}
```

## Advanced Techniques

### Strength Scheduling

Vary ControlNet influence during generation:

```python
# Reduce control over time for more creativity
strength_schedule = {
    0: 1.0,    # Full control at start
    0.5: 0.7,  # Reduce midway
    0.8: 0.3,  # Minimal at end
}
```

### Multiple Control Combinations

#### Pose + Depth
```python
# Character in specific pose with depth
controls = [
    {"type": "openpose", "strength": 1.0},
    {"type": "depth", "strength": 0.5}
]
```

#### Edge + Segmentation
```python
# Precise shapes with semantic regions
controls = [
    {"type": "canny", "strength": 0.8},
    {"type": "segmentation", "strength": 0.6}
]
```

### Control Mode Selection

| Mode | Description | Use Case |
|------|-------------|----------|
| Balanced | Equal weight to prompt and control | General use |
| Prompt | Prioritize text prompt | Creative freedom |
| Control | Prioritize ControlNet | Exact matching |

### Resolution Considerations

```python
# ControlNet resolution tips
if base_model == "SD1.5":
    control_res = 512
elif base_model == "SDXL":
    control_res = 1024
    
# Preprocess to match
control_image = resize_image(input_image, control_res)
```

## Preprocessing Best Practices

### Image Preparation

```python
def prepare_control_image(image, control_type):
    # Ensure correct resolution
    image = resize_to_model_resolution(image)
    
    # Enhance contrast for edge detection
    if control_type in ["canny", "mlsd"]:
        image = enhance_contrast(image)
    
    # Denoise for cleaner extraction
    if control_type == "openpose":
        image = denoise(image)
    
    return image
```

### Preprocessor Selection

| Input Quality | Recommended Preprocessor |
|--------------|-------------------------|
| Clean photo | Standard preprocessor |
| Noisy image | Robust variants (DW, LeReS+) |
| Artistic | Soft variants (HED, PIDI) |
| Technical | Precise variants (MLSD) |

### Custom Preprocessing

```python
# Create custom control maps
import cv2
import numpy as np

def custom_edge_detection(image):
    # Combine multiple edge detectors
    canny = cv2.Canny(image, 50, 150)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Weighted combination
    combined = 0.7 * canny + 0.3 * np.abs(laplacian)
    return combined.astype(np.uint8)
```

## Model Compatibility

### ControlNet Versions

| Base Model | ControlNet Version | File Pattern |
|------------|-------------------|--------------|
| SD 1.5 | v1.1 | control_v11*_sd15_*.pth |
| SD 2.1 | v1.1 SD2 | control_v11*_sd21_*.pth |
| SDXL | SDXL | controlnet-*-sdxl-*.safetensors |
| FLUX | Coming Soon | TBD |

### T2I-Adapter

Alternative to ControlNet with different characteristics:

```yaml
Advantages:
- Smaller model size (~80MB vs ~1.4GB)
- Faster inference
- Multiple adapters combinable

Disadvantages:
- Sometimes less precise
- Fewer available types
```

## Common Workflows

### Character Consistency

```python
# Maintain character across poses
workflow = {
    "reference_image": "character_reference.png",
    "preprocessor": "openpose_full",
    "prompt_template": "character_name, {pose_description}",
    "strength": 0.9,
    "seed": "fixed_for_consistency"
}
```

### Architecture Visualization

```python
# Technical drawing to render
workflow = {
    "line_drawing": "floor_plan.png",
    "preprocessor": "mlsd",
    "prompt": "modern house interior, photorealistic",
    "control_strength": 1.0,
    "cfg_scale": 7.5
}
```

### Style Transfer with Structure

```python
# Preserve composition, change style
workflow = {
    "content_image": "photo.jpg",
    "preprocessors": ["depth", "softedge"],
    "style_prompt": "oil painting in the style of van gogh",
    "control_balance": {
        "depth": 0.7,
        "softedge": 0.5
    }
}
```

## Troubleshooting

### Common Issues

#### Preprocessor Not Detecting Features
```python
# Solutions
- Increase image contrast
- Try different preprocessor variant
- Adjust detection thresholds
- Use manual annotation tools
```

#### Over-controlling Generation
```python
# Reduce control influence
{
    "strength": 0.6,  # Lower from 1.0
    "end_percent": 0.8,  # Stop control early
    "cfg_scale": 9,  # Increase prompt importance
}
```

#### Artifacts at Edges
```python
# Edge artifact mitigation
- Use softedge instead of canny
- Blur control map slightly
- Reduce control strength at boundaries
- Enable "soft" control mode
```

### Performance Optimization

```python
# Memory-efficient ControlNet
{
    "low_vram_mode": true,
    "preprocessor_device": "cpu",
    "control_net_device": "cuda",
    "offload_when_unused": true
}
```

## Creative Applications

### Hybrid Controls

```python
# Combine photo and sketch
photo_depth = extract_depth(photo)
sketch_lines = process_sketch(drawing)
combined_control = blend_controls(photo_depth, sketch_lines, alpha=0.5)
```

### Temporal Consistency

For animations:
```python
# Frame-to-frame consistency
previous_control = None
for frame in video_frames:
    current_control = extract_pose(frame)
    if previous_control:
        # Smooth between frames
        current_control = interpolate_controls(
            previous_control, current_control, 0.3
        )
    generate_frame(current_control)
    previous_control = current_control
```

### Interactive Control

```python
# Real-time adjustment
class InteractiveControl:
    def update_strength(self, value):
        self.control_strength = value
        self.regenerate()
    
    def switch_preprocessor(self, new_type):
        self.control_map = preprocess(self.input, new_type)
        self.regenerate()
```

## Best Practices

### Do's
✓ Match control resolution to model resolution  
✓ Use appropriate preprocessor for input type  
✓ Experiment with strength values  
✓ Combine multiple controls thoughtfully  
✓ Save successful control maps for reuse  

### Don'ts
✗ Don't use 100% control strength always  
✗ Don't ignore prompt importance  
✗ Don't use incompatible model versions  
✗ Don't expect perfect results immediately  
✗ Don't overstack controls (3+ rarely helpful)  

## Future Developments

### Emerging ControlNet Technologies

1. **3D-Aware Control**: Full 3D scene understanding
2. **Video ControlNet**: Temporal consistency
3. **Semantic Editing**: Natural language control
4. **Adaptive Control**: Self-adjusting strength
5. **Neural Controls**: Learned control patterns

### Integration Trends

- Real-time control preview
- Mobile-optimized controls
- Cloud-based preprocessing
- AI-assisted control creation

## Conclusion

ControlNet transforms diffusion models from probabilistic generators into precision tools. By understanding the various control types and their optimal applications, you can achieve unprecedented control over AI image generation while maintaining the creative capabilities of the base models.

The key to mastery is experimentation: try different preprocessors, adjust strengths, and combine controls creatively. As the technology evolves, ControlNet continues to bridge the gap between artistic vision and AI capabilities.