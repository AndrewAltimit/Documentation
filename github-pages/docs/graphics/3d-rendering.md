---
layout: docs
title: 3D Graphics & Rendering
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cube"
---

3D graphics rendering transforms mathematical representations of three-dimensional scenes into two-dimensional images displayed on screen. Modern rendering pipelines combine sophisticated algorithms, parallel GPU architectures, and advanced shading techniques to produce photorealistic or stylized visuals in real-time for games, simulations, and interactive applications.

## The Rendering Pipeline

### Overview

The graphics pipeline processes geometry through a series of stages:

```
Application Stage (CPU)
    │
    ▼
Geometry Processing (GPU)
├── Vertex Shader
├── Tessellation (optional)
├── Geometry Shader (optional)
└── Clipping & Culling
    │
    ▼
Rasterization
    │
    ▼
Fragment/Pixel Processing
├── Pixel Shader
├── Depth Testing
├── Stencil Testing
└── Blending
    │
    ▼
Framebuffer Output
```

### Coordinate Spaces

Transformations move vertices through coordinate systems:

| Space | Description | Transform |
|-------|-------------|-----------|
| **Object/Model** | Local coordinates relative to mesh origin | - |
| **World** | Global scene coordinates | Model Matrix |
| **View/Camera** | Relative to camera position | View Matrix |
| **Clip** | Homogeneous coordinates for clipping | Projection Matrix |
| **NDC** | Normalized Device Coordinates [-1,1] | Perspective Division |
| **Screen** | Final pixel coordinates | Viewport Transform |

### Vertex Processing

The vertex shader transforms each vertex:

```glsl
// Basic vertex shader
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(binding = 0) uniform Matrices {
    mat4 model;
    mat4 view;
    mat4 projection;
};

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

void main() {
    vec4 worldPos = model * vec4(inPosition, 1.0);
    fragPosition = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(model))) * inNormal;
    fragTexCoord = inTexCoord;

    gl_Position = projection * view * worldPos;
}
```

## Lighting and Shading

### Lighting Models

**Phong Reflection Model:**

```
I = Iₐ * kₐ + Σ[Iₗ * (kd * (N·L) + ks * (R·V)ⁿ)]

Where:
- Iₐ = Ambient light intensity
- kₐ = Ambient reflection coefficient
- Iₗ = Light intensity
- kd = Diffuse coefficient
- ks = Specular coefficient
- N = Surface normal
- L = Light direction
- R = Reflection direction
- V = View direction
- n = Shininess exponent
```

**Blinn-Phong (Optimized):**
- Uses halfway vector H = normalize(L + V)
- Specular: ks * (N·H)ⁿ
- More efficient, similar results

### Physically Based Rendering (PBR)

Modern standard for realistic materials:

**Core Parameters:**
- **Albedo/Base Color**: Surface color without lighting
- **Metallic**: Metal (1.0) vs dielectric (0.0)
- **Roughness**: Microsurface irregularity (0.0 = smooth, 1.0 = rough)
- **Normal Map**: Per-pixel surface detail
- **Ambient Occlusion**: Soft shadowing in crevices

**Cook-Torrance BRDF:**

```
f(l,v) = fᵈⁱᶠᶠᵘˢᵉ + fˢᵖᵉᶜᵘˡᵃʳ

fˢᵖᵉᶜᵘˡᵃʳ = DFG / (4(n·l)(n·v))

D = Normal Distribution Function (GGX/Trowbridge-Reitz)
F = Fresnel (Schlick approximation)
G = Geometry/Shadowing (Smith GGX)
```

### Global Illumination

Simulating indirect lighting:

**Real-Time Techniques:**
- **Screen-Space GI (SSGI)**: Sample nearby pixels
- **Voxel Cone Tracing**: Voxelize scene, trace cones
- **Light Probes**: Precomputed irradiance at points
- **Reflection Probes**: Cubemap captures for reflections
- **Ray Tracing**: Hardware-accelerated path tracing

**Unreal Engine 5 Lumen:**
- Hybrid software/hardware ray tracing
- Infinite bounces for indirect light
- Dynamic, no baking required
- Screen-space fallback for efficiency

## Shadow Rendering

### Shadow Mapping

Standard real-time shadow technique:

1. **Shadow Pass**: Render depth from light's perspective
2. **Main Pass**: Compare fragment depth to shadow map
3. **Result**: In shadow if fragment depth > shadow map depth

**Common Issues and Solutions:**
- **Shadow Acne**: Add depth bias
- **Peter Panning**: Reduce bias, use slope-scaled bias
- **Aliasing**: PCF filtering, variance shadow maps
- **Resolution**: Cascaded shadow maps for large scenes

### Cascaded Shadow Maps (CSM)

Multiple shadow maps for different distance ranges:

```
Near cascade: High resolution, close to camera
Mid cascade: Medium resolution, mid-range
Far cascade: Low resolution, distant objects

Split distances based on:
- Logarithmic distribution
- Practical split scheme (PSSM)
- Custom per-game tuning
```

### Ray-Traced Shadows

Hardware ray tracing benefits:

- Pixel-perfect accuracy
- Natural soft shadows from area lights
- No aliasing or bias issues
- Higher performance cost

## Advanced Rendering Techniques

### Deferred Rendering

Separate geometry from lighting:

**G-Buffer Contents:**
- Position (or depth for reconstruction)
- Normal
- Albedo/Diffuse
- Specular/Roughness
- Emissive (optional)

**Advantages:**
- Decouple geometry complexity from light count
- Efficient many-light scenarios
- Easy post-processing access to scene data

**Disadvantages:**
- High memory bandwidth
- Difficult transparency handling
- MSAA complications

### Forward+ Rendering

Hybrid approach:

1. **Depth Pre-Pass**: Populate depth buffer
2. **Light Culling**: Tile-based light assignment
3. **Shading**: Forward pass with culled light lists

**Benefits:**
- Supports transparency naturally
- Lower memory bandwidth than deferred
- MSAA compatible
- Efficient for moderate light counts

### Clustered Rendering

3D extension of Forward+:

- Divide view frustum into 3D clusters
- Assign lights to clusters (not just tiles)
- Better handling of depth discontinuities
- More uniform light distribution

## Post-Processing Effects

### Screen-Space Effects

**Ambient Occlusion:**
- SSAO (Screen-Space Ambient Occlusion)
- HBAO+ (Horizon-Based)
- GTAO (Ground Truth)

**Reflections:**
- SSR (Screen-Space Reflections)
- Hi-Z tracing for efficiency
- Fallback to probes for missing data

**Motion Blur:**
- Per-object velocity buffers
- Camera motion blur
- Temporal reconstruction

### Color Grading and Tone Mapping

**HDR to LDR Conversion:**

```glsl
// Reinhard tone mapping
vec3 toneMapReinhard(vec3 hdrColor) {
    return hdrColor / (hdrColor + vec3(1.0));
}

// ACES Filmic
vec3 toneMapACES(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
```

**Color Grading:**
- LUT (Look-Up Table) based
- Split toning (shadows/highlights)
- Color wheels adjustment
- Film grain and vignette

### Anti-Aliasing

**Techniques Comparison:**

| Method | Quality | Performance | Motion | Transparency |
|--------|---------|-------------|--------|--------------|
| MSAA | Good | Medium | Poor | Good |
| FXAA | Low | Fast | Poor | Good |
| SMAA | Good | Fast | Poor | Good |
| TAA | Excellent | Medium | Good | Medium |
| DLSS/FSR | Excellent | Fast* | Good | Good |

*Upscaling provides net performance gain

### Temporal Anti-Aliasing (TAA)

Modern standard approach:

1. Jitter projection matrix each frame
2. Accumulate samples over time
3. Reject samples using motion vectors
4. Apply neighborhood clamping

**Challenges:**
- Ghosting on fast motion
- Loss of fine detail
- Requires motion vectors

## GPU Architecture

### Parallelism Model

GPUs execute thousands of threads simultaneously:

```
GPU
├── Streaming Multiprocessors (SM)
│   ├── CUDA Cores / Stream Processors
│   ├── Shared Memory
│   └── L1 Cache
├── L2 Cache
├── Memory Controllers
└── Video Memory (VRAM)

Thread Hierarchy:
- Thread: Single execution unit
- Warp/Wavefront: 32/64 threads executing together
- Thread Block: Group of warps with shared memory
- Grid: All thread blocks for a dispatch
```

### Memory Hierarchy

Optimizing for GPU memory access:

| Memory Type | Latency | Scope | Size |
|-------------|---------|-------|------|
| Registers | 1 cycle | Thread | ~256 per thread |
| Shared Memory | ~20 cycles | Block | 48-96 KB |
| L1 Cache | ~20 cycles | SM | 48-128 KB |
| L2 Cache | ~200 cycles | Device | 4-6 MB |
| VRAM | ~400 cycles | Global | 8-24 GB |

### Graphics APIs

**Vulkan:**
- Low-level, explicit control
- Cross-platform
- Best for engine developers

**DirectX 12:**
- Windows/Xbox exclusive
- Similar to Vulkan
- Better tooling ecosystem

**Metal:**
- Apple platforms
- Excellent iOS/macOS integration
- Swift/Objective-C friendly

**WebGPU:**
- Browser-based 3D
- Modern API design
- Growing adoption

## Optimization Techniques

### Culling

Eliminate invisible geometry:

- **Frustum Culling**: Outside camera view
- **Occlusion Culling**: Hidden behind other objects
- **Backface Culling**: Faces pointing away from camera
- **Distance Culling**: Beyond view distance
- **Small Object Culling**: Sub-pixel geometry

### Level of Detail (LOD)

Reduce complexity with distance:

```
LOD 0: Full detail (0-50m)
LOD 1: 50% triangles (50-100m)
LOD 2: 25% triangles (100-200m)
LOD 3: 10% triangles (200m+)
Billboard: 2D impostor (very far)
```

**Modern Approaches:**
- **Nanite** (UE5): Virtualized geometry, automatic LOD
- **Mesh Shaders**: GPU-driven LOD selection
- **Continuous LOD**: Smooth transitions

### Batching and Instancing

Reduce draw calls:

- **Static Batching**: Combine static meshes
- **Dynamic Batching**: Runtime combination of small meshes
- **GPU Instancing**: Single draw call, multiple instances
- **Indirect Drawing**: GPU-driven draw commands

## Related Documentation

- [Game Development](../gamedev/index.html) - Game design and development
- [Unreal Engine](../technology/unreal.html) - UE5 rendering features
- [Performance Optimization](../optimization/index.html) - Profiling and optimization
- [VR/AR Development](../vr-ar/index.html) - Immersive rendering requirements
