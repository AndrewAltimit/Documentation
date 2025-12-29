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
Your guide to creating AI-generated images, training custom models, and building automated workflows. From your first generated image to training your own artistic styles.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why Learn AI Image Generation?

AI image generation has transformed from a research curiosity into a practical creative tool. Artists use it to explore new styles, designers prototype concepts in minutes instead of hours, and developers build automated content pipelines. The technology is accessible enough to run on consumer hardware, yet powerful enough for professional applications.

**Consider the following before diving in:**

- **What do you want to create?** Photorealistic images, artistic illustrations, anime characters, or product mockups each benefit from different approaches
- **How much control do you need?** Quick generation versus precise artistic direction require different tools and workflows
- **Will you need custom styles or subjects?** Training your own models unlocks personalized results that generic models cannot achieve

This documentation covers the practical skills you need, from understanding how the technology works to building production-ready workflows.

## Choose Your Path

Different goals require different starting points. Find your path below:

| Your Goal | Start Here | Then Explore |
|-----------|------------|--------------|
| Generate images quickly | [ComfyUI Guide](comfyui-guide.html) | [Base Models Comparison](base-models-comparison.html) |
| Understand the technology | [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) | [Model Types](model-types.html) |
| Train custom styles | [LoRA Training](lora-training.html) | [Advanced Techniques](advanced-techniques.html) |
| Control composition precisely | [ControlNet](controlnet.html) | [ComfyUI Guide](comfyui-guide.html) |
| Build automation pipelines | [ComfyUI Guide](comfyui-guide.html) | [Output Formats](output-formats.html) |

### Documentation Overview

**Understanding the Foundations**
- [**Stable Diffusion Fundamentals**](stable-diffusion-fundamentals.html) - How diffusion models transform noise into images
- [**Model Types**](model-types.html) - The building blocks: LoRAs, VAEs, CLIP, and how they work together
- [**Base Models Comparison**](base-models-comparison.html) - SD 1.5 vs SDXL vs FLUX: choosing the right foundation

**Practical Tools**
- [**ComfyUI Guide**](comfyui-guide.html) - Visual workflow builder for complex generation pipelines
- [**LoRA Training**](lora-training.html) - Create custom models for specific styles, characters, or concepts
- [**ControlNet**](controlnet.html) - Guide generation with poses, edges, depth maps, and more

**Going Further**
- [**Output Formats**](output-formats.html) - Working with generated content across image, video, and 3D
- [**Advanced Techniques**](advanced-techniques.html) - Expert workflows and optimization strategies

## Getting Started

Before generating your first image, you will need compatible hardware and a few software tools. The requirements scale with the complexity of your goals.

### Hardware Requirements

| Use Case | GPU VRAM | System RAM | Storage |
|----------|----------|------------|---------|
| Basic generation (SD 1.5) | 4-6 GB | 16 GB | 50 GB |
| Standard workflows (SDXL) | 8-12 GB | 32 GB | 200 GB |
| Advanced models (FLUX, SD3) | 16-24 GB | 64 GB | 500 GB |
| LoRA training | 8-24 GB | 32-64 GB | 100 GB |

Most modern NVIDIA GPUs work well. AMD and Apple Silicon have growing support but may require additional configuration.

### Software Requirements

- **Docker and Docker Compose** - Containers simplify setup and ensure consistent environments
- **NVIDIA Docker runtime** - Enables GPU acceleration inside containers
- **Python 3.10+** - Only needed for local scripts or custom development

### What You Should Already Know

No prior AI experience is required, but you should be comfortable with:
- Running commands in a terminal
- Basic file and folder operations
- Reading error messages and troubleshooting

### Quick Start: Your First Image

The fastest way to generate an image is through ComfyUI's web interface:

```bash
# Start ComfyUI and open http://localhost:8188
docker-compose up -d comfyui-server
```

Once the interface loads, you can use the default workflow immediately. Type your prompt, click "Queue Prompt," and watch your image generate.

For programmatic access or automation, the MCP API accepts JSON requests:

```bash
curl -X POST http://localhost:8005/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "generate-image", "arguments": {"prompt": "mountain landscape at sunset"}}'
```

See the [ComfyUI Guide](comfyui-guide.html) for detailed setup and workflow tutorials.

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

## Model Generations at a Glance

The field evolves quickly. Here is how the major model families compare:

| Model | Resolution | VRAM Needed | Strengths | Best For |
|-------|------------|-------------|-----------|----------|
| SD 1.5 | 512x512 | 4-6 GB | Huge LoRA ecosystem, fast | Beginners, resource-limited setups |
| SDXL | 1024x1024 | 8-12 GB | Quality, composition | General creative work |
| SD3 | 1024x1024 | 10-16 GB | Text rendering, prompt following | Text-heavy images, precision |
| FLUX | 1024x1024+ | 12-24 GB | Photorealism, coherence | Professional quality, portraits |

**When to use which:**
- Start with **SDXL** for the best balance of quality, speed, and ecosystem support
- Use **SD 1.5** if you have limited hardware or need specific legacy LoRAs
- Choose **FLUX** when photorealism and fine details matter most
- Pick **SD3** when your images include text or need precise prompt interpretation

See [Base Models Comparison](base-models-comparison.html) for detailed technical differences.

## Real-World Applications

People use AI image generation across many fields. Here are common scenarios and the approaches that work best:

### Creative Work

**Concept art and illustration** - Generate variations quickly, then refine favorites manually. Use style LoRAs to maintain visual consistency across a project.

**Character design** - Train a character LoRA from reference sketches, then generate the character in different poses and situations. Combine with ControlNet for precise posing.

**Environment art** - Generate base landscapes or interiors, use img2img for iterative refinement. ControlNet depth maps help maintain architectural consistency.

### Commercial Applications

**Product visualization** - Generate product mockups in various settings before physical prototypes exist. Works especially well for packaging and marketing concepts.

**Marketing content** - Create social media visuals, banner images, and promotional materials. Train brand-specific LoRAs for consistent visual identity.

**Game development** - Generate texture variations, background elements, and concept references. LoRAs trained on existing game art maintain style consistency.

### Technical Applications

**Dataset augmentation** - Generate training data variations for other ML models. Particularly valuable when real data is scarce or expensive to collect.

**Rapid prototyping** - Visualize ideas before committing development resources. Especially useful in early design phases.

## Best Practices

### Writing Effective Prompts

Good prompts guide the model toward your vision. Structure them with the most important elements first:

- **Subject first** - "A knight in armor" beats "detailed, 4k, masterpiece, knight"
- **Be specific** - "Golden retriever puppy" produces better results than "dog"
- **Include context** - Mention lighting, setting, camera angle, and artistic style
- **Use negative prompts** - Tell the model what to avoid (blurry, low quality, extra limbs)

### Managing System Resources

AI models consume significant GPU memory. These practices help:

- Match model size to your hardware (see requirements table above)
- Enable "low VRAM" modes in your workflow tool when needed
- Close other GPU-intensive applications during generation
- Use quantized model versions (fp16, fp8) for reduced memory usage

### Improving Output Quality

When results disappoint, try these adjustments:

| Problem | Solution |
|---------|----------|
| Blurry images | Increase steps (30-50), try different sampler |
| Wrong composition | Revise prompt structure, consider ControlNet |
| Artifacts/glitches | Lower CFG scale, check model compatibility |
| Style not matching | Adjust LoRA strength, verify trigger words |

## Troubleshooting

When something goes wrong, these are the most common causes and fixes:

### Out of Memory Errors

Your GPU ran out of VRAM. Try these solutions in order:

1. Use a smaller model version (fp16 instead of fp32, fp8 for FLUX)
2. Reduce image resolution
3. Enable "low VRAM" or "CPU offloading" in your workflow tool
4. Close other applications using the GPU

### Slow Generation

Generation taking too long usually means inefficient settings:

1. Reduce sampling steps (20-30 is often sufficient)
2. Switch to a faster sampler (DPM++ 2M, Euler)
3. Verify GPU is being used (check nvidia-smi)
4. Ensure models are loaded once, not reloaded per image

### Poor Quality Results

When images do not match expectations:

1. Review and refine your prompt (be more specific)
2. Experiment with CFG scale (try 5-10 range)
3. Increase sampling steps for more detail
4. Verify model and LoRA compatibility

## Automation and Integration

For batch processing or integration into larger systems, the MCP API provides programmatic access:

```python
import requests

# Generate an image via API
response = requests.post("http://localhost:8189/mcp/tool", json={
    "tool": "generate-image",
    "arguments": {
        "prompt": "cyberpunk city at night",
        "checkpoint": "sdxl_base.safetensors"
    }
})
```

See the [ComfyUI Guide](comfyui-guide.html) for complete API documentation and workflow submission examples.

## Resources and Community

### Where to Find Models

- [CivitAI](https://civitai.com/) - Largest collection of LoRAs, checkpoints, and community models
- [Hugging Face](https://huggingface.co/) - Official model releases and research models

### Learning and Help

- [Reddit r/StableDiffusion](https://reddit.com/r/stablediffusion) - Active community discussions
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI) - Official documentation and issues
- [Stable Diffusion Discord](https://discord.gg/stablediffusion) - Real-time community help

### Research Papers

For those interested in the underlying technology:
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752) - Original architecture
- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206) - Latest architecture advances

## Next Steps

Based on your goals, here is where to go next:

**Want to generate images now?** Start with the [ComfyUI Guide](comfyui-guide.html) for hands-on workflow building.

**Want to understand the technology?** Read [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) for the concepts behind generation.

**Want to create custom styles?** Jump to [LoRA Training](lora-training.html) to learn how to train your own models.

## Related Documentation

For broader AI and machine learning concepts:

- [AI Fundamentals - Simplified](../technology/ai-fundamentals-simple.html) - Conceptual introduction without heavy math
- [AI Fundamentals - Complete](../technology/ai.html) - Technical deep-dive into AI concepts
- [AI Documentation Hub](../artificial-intelligence/) - All AI-related documentation

<div class="code-example bg-yellow-000" markdown="1">
**Hardware Note**: This documentation assumes NVIDIA GPU access. AMD and Apple Silicon support is improving but may require additional configuration and have limited feature availability.
</div>