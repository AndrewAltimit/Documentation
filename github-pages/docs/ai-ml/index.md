---
layout: docs
title: AI/ML Documentation
nav_order: 20
has_children: true
permalink: /docs/ai-ml/
toc: false  # Index pages typically don't need TOC
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AI/ML Documentation</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Your comprehensive guide to AI image generation, custom model training, and automated creative workflows.</p>
</div>

<div class="code-example" markdown="1">
Your guide to creating AI-generated images, training custom models, and building automated workflows. From your first generated image to training your own artistic styles and shipping them to production.
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-wave-square"></i>
    <h4>Learn the Foundations</h4>
    <p>How diffusion turns noise into images, the model stack, and how to choose a base model for your task and hardware.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-tools"></i>
    <h4>Master the Tools</h4>
    <p>Build node workflows in ComfyUI, train your own LoRAs, edit existing images, and steer composition precisely with ControlNet.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-rocket"></i>
    <h4>Ship to Production</h4>
    <p>Optimize for your hardware, compress models, automate pipelines, and run reliable, observable services with MLOps.</p>
  </div>
</div>

> **Suggested path:** start with the **Foundations**, pick up the **Tools** that match your goal, then move into **Production** workflows. Use the "Choose Your Path" table below to jump straight to your starting point.

## Why Learn AI Image Generation?

AI image generation has transformed from a research curiosity into a practical creative tool. Artists use it to explore new styles, designers prototype concepts in minutes instead of hours, and developers build automated content pipelines. The technology is accessible enough to run on consumer hardware, yet powerful enough for professional applications.

**Consider the following before diving in:**

- **What do you want to create?** Photorealistic images, artistic illustrations, anime characters, or product mockups each benefit from different approaches
- **How much control do you need?** Quick generation versus precise artistic direction require different tools and workflows
- **Will you need custom styles or subjects?** Training your own models unlocks personalized results that generic models cannot achieve

This documentation covers the practical skills you need, from understanding how the technology works to building production-ready workflows.

## Quick Start: Your First Image

The fastest way to generate an image is through ComfyUI's web interface:

```bash
# Start ComfyUI and open http://localhost:8188
docker compose up -d comfyui-server
```

Once the interface loads, you can use the default workflow immediately. Type your prompt, click "Queue Prompt," and watch your image generate.

For programmatic access or automation, the MCP API accepts JSON requests:

```bash
curl -X POST http://localhost:8189/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "generate-image", "arguments": {"prompt": "mountain landscape at sunset"}}'
```

**Hardware at a glance** — requirements scale with the models you run:

| Use Case | GPU VRAM | System RAM | Storage |
|----------|----------|------------|---------|
| Basic generation (SD 1.5) | 4-6 GB | 16 GB | 50 GB |
| Standard workflows (SDXL) | 8-12 GB | 32 GB | 200 GB |
| Advanced models (FLUX, SD3) | 16-24 GB | 64 GB | 500 GB |
| LoRA training | 8-24 GB | 32-64 GB | 100 GB |

Most modern NVIDIA GPUs work well. AMD and Apple Silicon have growing support but may require additional configuration. See the [ComfyUI Guide](comfyui-guide.html) for detailed setup, and the [Optimization & Performance](optimization-guide.html) guide if you need to fit larger models on smaller cards.

## Choose Your Path

Different goals require different starting points. Find your path below:

| Your Goal | Start Here | Then Explore |
|-----------|------------|--------------|
| Generate images quickly | [ComfyUI Guide](comfyui-guide.html) | [Base Models Comparison](base-models-comparison.html) |
| Understand the technology | [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) | [Model Types](model-types.html) |
| Train custom styles | [LoRA Training](lora-training.html) | [Advanced Techniques](advanced-techniques.html) |
| Edit existing images | [Inpainting & Editing](inpainting-editing.html) | [ControlNet](controlnet.html) |
| Control composition precisely | [ControlNet](controlnet.html) | [ComfyUI Guide](comfyui-guide.html) |
| Make it fit / run faster | [Optimization & Performance](optimization-guide.html) | [Model Compression](model-compression.html) |
| Automate generation at scale | [Production Pipelines](production-pipelines.html) | [Output Formats](output-formats.html) |
| Run reliable ML services | [MLOps & Production](mlops-production.html) | [Production Pipelines](production-pipelines.html) |

## Key Concepts

Understanding a few core ideas will help you make better decisions about models, settings, and workflows.

### How Diffusion Models Create Images

Diffusion models learn by studying how images gradually dissolve into random noise, then learning to reverse that process. When you generate an image, the model starts with pure noise and progressively refines it into a coherent picture, guided by your text prompt.

This happens in "latent space" (a compressed mathematical representation) rather than pixel-by-pixel, which is why modern models can run on consumer hardware. Each generation step removes a bit of noise while steering toward your described content.

| Generation Approach | Steps Needed | Best For |
|---------------------|--------------|----------|
| Standard diffusion | 20-50 | High quality, most control |
| LCM (Latent Consistency) | 4-8 | Fast iteration, previews |
| Turbo models | 1-4 | Real-time, interactive use |

### The Model Stack

AI image generation uses several specialized components working together:

- **Base Model** - The foundation that understands image-text relationships (SD 1.5, SDXL, FLUX)
- **VAE** - Compresses images for efficient processing, then decompresses the result
- **Text Encoder** - Translates your prompt into numbers the model understands
- **LoRA** - Small add-ons that teach the base model new styles or subjects
- **ControlNet** - Guides composition using reference images, poses, or edges

Think of the base model as a skilled artist, LoRAs as specialized training, and ControlNet as a reference sketch the artist follows.

### Choosing a Workflow Tool

Several interfaces exist for working with these models:

| Tool | Best For | Learning Curve |
|------|----------|----------------|
| ComfyUI | Complex workflows, automation, experimentation | Moderate |
| Automatic1111/Forge | Feature-rich UI, extensions ecosystem | Low |
| Fooocus | Simple generation, beginners | Very low |
| InvokeAI | Professional canvas-based editing | Low-moderate |

This documentation focuses on **ComfyUI** because its node-based approach teaches you how the components connect and enables the most advanced workflows.

### Model Generations at a Glance

The field evolves quickly. Here is how the major model families compare:

| Model | Resolution | VRAM Needed | Strengths | Best For |
|-------|------------|-------------|-----------|----------|
| SD 1.5 | 512x512 | 4-6 GB | Huge LoRA ecosystem, fast | Beginners, resource-limited setups |
| SDXL | 1024x1024 | 8-12 GB | Quality, composition | General creative work |
| SD3 | 1024x1024 | 10-16 GB | Text rendering, prompt following | Text-heavy images, precision |
| FLUX | 1024x1024+ | 12-24 GB | Photorealism, coherence | Professional quality, portraits |

Start with **SDXL** for the best balance of quality, speed, and ecosystem; use **SD 1.5** on limited hardware or for legacy LoRAs; choose **FLUX** when photorealism matters most; pick **SD3** for text-heavy or precision work. See [Base Models Comparison](base-models-comparison.html) for detailed technical differences.

## Documentation Overview

Explore the full library, grouped by purpose:

### Understanding the Foundations

<div class="command-grid">
  <div class="feature-card">
    <h4><i class="fas fa-wave-square"></i> <a href="stable-diffusion-fundamentals.html">Stable Diffusion Fundamentals</a></h4>
    <p>How diffusion models turn noise into images, and the parameters that control your results.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-cubes"></i> <a href="model-types.html">Model Types</a></h4>
    <p>The building blocks — checkpoints, LoRAs, VAEs, CLIP/T5, ControlNet, IP-Adapter — and how they fit together.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-balance-scale"></i> <a href="base-models-comparison.html">Base Models Comparison</a></h4>
    <p>SD 1.5 vs SDXL vs SD3 vs FLUX vs Pony — choosing the right foundation for your task and hardware.</p>
  </div>
</div>

### Practical Tools

<div class="command-grid">
  <div class="feature-card">
    <h4><i class="fas fa-project-diagram"></i> <a href="comfyui-guide.html">ComfyUI Guide</a></h4>
    <p>The node-based workflow builder for complex, automatable generation pipelines.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-graduation-cap"></i> <a href="lora-training.html">LoRA Training</a></h4>
    <p>Train custom models for your own styles, characters, or concepts on consumer hardware.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-sliders-h"></i> <a href="controlnet.html">ControlNet</a></h4>
    <p>Guide composition with poses, edges, depth maps, and segmentation for precise control.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-eraser"></i> <a href="inpainting-editing.html">Inpainting & Editing</a></h4>
    <p>Mask, regenerate, extend, and blend regions to edit existing images instead of rerolling them.</p>
  </div>
</div>

### Going Further

<div class="command-grid">
  <div class="feature-card">
    <h4><i class="fas fa-file-export"></i> <a href="output-formats.html">Output Formats</a></h4>
    <p>Diffusion across every medium — image, video, audio, and 3D — and how to export each.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-flask"></i> <a href="advanced-techniques.html">Advanced Techniques</a></h4>
    <p>Latent interpolation, regional prompting, flow matching, distillation, and expert optimization.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-gamepad"></i> <a href="game-ai.html">Game AI Systems</a></h4>
    <p>Pathfinding, behavior trees, steering, and ML-driven NPCs for real-time interactive AI.</p>
  </div>
</div>

### Optimization & Production

<div class="command-grid">
  <div class="feature-card">
    <h4><i class="fas fa-cog"></i> <a href="optimization-guide.html">Optimization & Performance</a></h4>
    <p>Quantization, VRAM-reduction tactics, inference speedups, and batching for diffusion models and LLMs.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-compress"></i> <a href="model-compression.html">Model Compression</a></h4>
    <p>Pruning, distillation, quantization, low-rank factorization, and edge deployment — and the accuracy you trade.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-stream"></i> <a href="production-pipelines.html">Production Pipelines</a></h4>
    <p>Headless generation at scale — batch jobs, parameter sweeps, the ComfyUI API, queues, and asset pipelines.</p>
  </div>
  <div class="feature-card">
    <h4><i class="fas fa-robot"></i> <a href="mlops-production.html">MLOps & Production</a></h4>
    <p>Reproducible training, experiment tracking, model registries, rollouts, and drift monitoring for reliable services.</p>
  </div>
</div>

## Troubleshooting

When something goes wrong, these are the most common causes and fixes.

### Out of Memory Errors

Your GPU ran out of VRAM. Try these solutions in order:

1. Use a smaller model version (fp16 instead of fp32, fp8 for FLUX)
2. Reduce image resolution
3. Enable "low VRAM" or "CPU offloading" in your workflow tool
4. Close other applications using the GPU

For systematic VRAM reduction and quantization, see [Optimization & Performance](optimization-guide.html).

### Slow Generation

Generation taking too long usually means inefficient settings:

1. Reduce sampling steps (20-30 is often sufficient)
2. Switch to a faster sampler (DPM++ 2M, Euler)
3. Verify GPU is being used (check `nvidia-smi`)
4. Ensure models are loaded once, not reloaded per image

### Poor Quality Results

When images do not match your expectations:

| Problem | Solution |
|---------|----------|
| Blurry images | Increase steps (30-50), try a different sampler |
| Wrong composition | Revise prompt structure, consider [ControlNet](controlnet.html) |
| Artifacts/glitches | Lower CFG scale, check model compatibility |
| Style not matching | Adjust LoRA strength, verify trigger words |

**Prompt tips:** put the subject first ("a knight in armor" beats "detailed, 4k, masterpiece, knight"), be specific, include lighting/setting/style, and use negative prompts to exclude what you do not want.

## Resources and Community

### Where to Find Models

- [CivitAI](https://civitai.com/) - Largest collection of LoRAs, checkpoints, and community models
- [Hugging Face](https://huggingface.co/) - Official model releases and research models

### Learning and Help

- [Reddit r/StableDiffusion](https://reddit.com/r/stablediffusion) - Active community discussions
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI) - Official documentation and issues
- [Stable Diffusion Discord](https://discord.gg/stablediffusion) - Real-time community help

### Research Papers

- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752) - Original architecture
- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206) - Latest architecture advances

<div class="see-also-card" markdown="1">
#### Related Documentation

Broader AI and machine learning concepts beyond image generation:

- [AI Fundamentals - Simplified](../technology/ai-fundamentals-simple.html) - Conceptual introduction without heavy math
- [AI Fundamentals - Complete](../technology/ai/) - Technical deep-dive into AI concepts
- [AI Documentation Hub](../artificial-intelligence/) - All AI-related documentation
- [Game AI Systems](game-ai.html) - Real-time AI for NPCs and interactive behaviors
</div>

<div class="code-example bg-yellow-000" markdown="1">
**Hardware Note**: This documentation assumes NVIDIA GPU access. AMD and Apple Silicon support is improving but may require additional configuration and have limited feature availability.
</div>
