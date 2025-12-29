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
Create custom AI models that generate your specific styles, characters, or concepts - all without needing massive computing resources.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why Train Your Own LoRA?

Pre-made models cannot generate everything. When you need consistent characters, specific art styles, or custom objects, training a LoRA lets you teach the model exactly what you want.

**Consider the following before starting:**

- **What cannot existing models do?** If SDXL plus available LoRAs can produce what you need, training may not be necessary
- **Do you have good reference images?** Training requires 10-50+ quality images of your subject
- **Do you have the hardware?** Training needs 8GB+ VRAM (more for SDXL/FLUX)

### When Training Makes Sense

| Goal | Training Worth It? | Alternative |
|------|-------------------|-------------|
| Consistent character across many images | Yes | Use IP-Adapter (less consistent) |
| Specific art style not in existing LoRAs | Yes | Find similar LoRA, adjust prompts |
| Personal likeness (yourself, pet) | Yes | No good alternative |
| Generic style (anime, photorealistic) | Usually no | Use existing checkpoints/LoRAs |
| One-time generation | Usually no | Prompt engineering + img2img |

### What LoRA Training Actually Does

LoRA adds small adjustment layers to an existing model. Instead of changing the entire model (which would require days of training and 100GB+ of data), LoRA learns focused modifications using your small dataset.

The result: A 20-200MB file that transforms how the base model handles your specific subject while preserving everything else it knows.

## Requirements

### Hardware Needs

| Base Model | Minimum VRAM | Comfortable VRAM | Training Time (1k steps) |
|------------|--------------|------------------|-------------------------|
| SD 1.5 | 6 GB | 8 GB | 15-30 minutes |
| SDXL | 12 GB | 16 GB | 30-60 minutes |
| FLUX Dev | 16 GB | 24 GB | 60-120 minutes |

Training also needs significant system RAM (16-32GB) and storage for datasets and outputs.

### Choosing a Training Tool

Several tools can train LoRAs:

| Tool | Best For | Difficulty |
|------|----------|------------|
| Kohya SS | Most users, local training | Medium |
| AI Toolkit | Docker-based workflows | Medium |
| Cloud services | No local GPU | Easy (but costs money) |

This guide uses concepts that apply to any tool. Specific settings may vary.

## Preparing Your Dataset

The quality of your training data determines the quality of your LoRA. This is where most training success or failure happens.

### How Many Images Do You Need?

| LoRA Type | Minimum Images | Recommended | Notes |
|-----------|---------------|-------------|-------|
| Style | 10 | 20-50 | Quality matters more than quantity |
| Character | 15 | 30-100 | Need variety in poses, angles, expressions |
| Object | 10 | 20-40 | Multiple angles, lighting conditions |
| Person likeness | 20 | 40-100 | Diverse photos, different contexts |

### Image Quality Checklist

Good training images are:
- Clear and well-lit (not blurry or dark)
- High resolution (at least 512x512, 1024x1024 preferred)
- Focused on the subject you want to teach
- Varied in pose, angle, and context
- Consistent in what they show (all the same character, all the same style)

### Writing Captions

Each image needs a text file with the same name describing what is in the image:

```
my_dataset/
  image01.jpg
  image01.txt
  image02.jpg
  image02.txt
```

### Caption Format

Include a unique trigger word plus a description:

```
xyz_character woman with red hair, smiling, casual clothes, outdoor setting
```

Key principles:
- **Use a unique trigger word** - Something distinctive like "xyz_style" or "sks_person"
- **Describe what varies** - If pose changes, describe the pose
- **Keep trigger word consistent** - Same trigger in every caption
- **Match model style** - Natural language for FLUX/SD3, tag-style for SD 1.5

### Quick Caption Guide by Model

| Model | Caption Style | Example |
|-------|---------------|---------|
| SD 1.5 | Tag-based | `xyz_style, digital art, landscape, mountains, sunset, vibrant colors` |
| SDXL | Mixed | `xyz_style digital painting of mountains at sunset, vibrant colors, detailed` |
| FLUX | Natural | `A beautiful mountain landscape at sunset in the xyz_style, with vibrant orange and purple colors` |

## Training Settings

### The Essential Settings

These are the settings that matter most:

| Setting | What It Does | Start With |
|---------|--------------|------------|
| Learning rate | How fast the model learns | 0.0001 - 0.0002 |
| Steps | Total training iterations | 100 per image (e.g., 20 images = 2000 steps) |
| Rank | Complexity of the LoRA | 16-32 for most uses |
| Resolution | Training image size | Match your base model (512 or 1024) |

### Choosing the Right Rank

Rank determines how much the LoRA can learn. Higher is not always better.

| Rank | File Size | Best For |
|------|-----------|----------|
| 8-16 | 10-30 MB | Simple styles, small adjustments |
| 32 | 50-80 MB | Most character and style LoRAs |
| 64-128 | 150-300 MB | Complex subjects, maximum fidelity |

**Start with rank 32.** Increase only if results lack detail; decrease if overfitting occurs.

### Learning Rate Guidelines

| Situation | Learning Rate | Why |
|-----------|---------------|-----|
| First attempt | 0.0001 | Safe starting point |
| Not learning fast enough | 0.0002-0.0003 | Speed up learning |
| Overfitting quickly | 0.00005-0.0001 | Slow down learning |
| Using Prodigy optimizer | 1.0 | Self-adjusting rate |

### How Many Steps?

A rough formula: **100 steps per training image**

| Dataset Size | Steps | Notes |
|--------------|-------|-------|
| 10 images | 1000-1500 | Watch for overfitting |
| 20 images | 2000-2500 | Good baseline |
| 50 images | 4000-5000 | Solid training |
| 100+ images | 5000-8000 | Diminishing returns above ~8000 |

## The Training Process

### What Happens During Training

1. **Loading** - The base model and your dataset load into GPU memory
2. **Training loop** - For each step, the model sees images and adjusts weights
3. **Checkpoints** - Periodic saves let you test progress
4. **Completion** - Final LoRA file is saved

### Monitoring Training

Watch these indicators:

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Loss | Decreasing steadily | Stuck high, or dropping then rising |
| Sample images | Improving each checkpoint | Same as base model, or identical to training images |
| Training speed | Consistent steps/second | Slowing significantly |

### When to Stop

Training should stop when:
- Sample images match your intent well
- Loss has stabilized (not dropping anymore)
- You have reached your target steps

Save checkpoints periodically so you can choose the best one, not just the last one.

## Common Training Scenarios

### Training a Style LoRA

**Goal:** Capture an artistic style from example images.

**Dataset:** 15-30 images in the style you want, diverse subjects

**Settings:**
- Rank: 16-32
- Steps: 1500-3000
- Learning rate: 0.0001

**Tip:** Include variety in subjects (people, landscapes, objects) so the LoRA learns the style, not specific content.

### Training a Character LoRA

**Goal:** Generate a consistent character in different poses and situations.

**Dataset:** 20-50 images of the character, varied angles and expressions

**Settings:**
- Rank: 32-64
- Steps: 2000-4000
- Learning rate: 0.0001

**Tip:** Include the character in different outfits and settings so the LoRA learns the character, not just one specific image.

### Training a Likeness LoRA

**Goal:** Generate images of a real person or pet.

**Dataset:** 30-100 photos, diverse lighting and contexts

**Settings:**
- Rank: 32-64
- Steps: 3000-5000
- Learning rate: 0.00005-0.0001

**Tip:** Include photos from different angles, with different expressions, and in different settings. Avoid training on just one or two photos.

## Troubleshooting Training

### Common Problems and Solutions

| Problem | Symptom | Fix |
|---------|---------|-----|
| Overfitting | Generates training images exactly | Reduce steps, lower learning rate, add more training data variety |
| Underfitting | LoRA has no visible effect | Increase steps, raise learning rate, verify trigger word in prompts |
| Style bleeding | Changes things you did not intend | Improve caption specificity, use lower LoRA strength when generating |
| Memory errors | Training crashes | Enable gradient checkpointing, use fp16, reduce batch size |
| Poor quality | Results worse than base model | Check dataset quality, ensure proper resolution, verify model compatibility |

### Diagnosing from Loss Curves

| Loss Behavior | What It Means | Action |
|---------------|---------------|--------|
| Steadily decreasing | Training is working | Continue as planned |
| Flat from start | Learning too slow | Increase learning rate |
| Drops then rises | Overfitting | Stop earlier, use that checkpoint |
| Erratic/oscillating | Learning rate too high | Reduce learning rate |
| Spikes suddenly | Corrupt data or bug | Check dataset, review settings |

## Using Your Trained LoRA

### Finding the Right Strength

Start at 0.7 strength and adjust based on results:

| Effect | Adjustment |
|--------|------------|
| Too subtle | Increase strength (0.8-1.0) |
| Too strong/artifacts | Decrease strength (0.4-0.6) |
| Good but want more | Try 0.8-0.9 |
| Overpowering other content | Try 0.5-0.6 |

### Combining with Other LoRAs

When stacking multiple LoRAs, reduce each strength:
- First LoRA: 0.6-0.8
- Second LoRA: 0.4-0.6
- Third LoRA: 0.3-0.4

If LoRAs conflict (similar subjects or styles), one may override the other. Test combinations to find what works.

### Remember Your Trigger Word

Your LoRA only activates when you include the trigger word in your prompt. If results look like the base model, check that your trigger word is present.

## Best Practices Summary

### Things That Lead to Success

- Use unique trigger words (xyz_style, not just "style")
- Include varied training images
- Start with conservative settings and adjust
- Save checkpoints so you can pick the best one
- Test with prompts different from your training captions

### Common Mistakes to Avoid

- Training too long (leads to overfitting)
- Using too few images (not enough variety)
- Generic trigger words (conflict with normal vocabulary)
- Skipping captions or using poor captions
- Not checking checkpoint quality during training

## Conclusion

LoRA training gives you the ability to add anything to AI image generation - your own art style, consistent characters, specific objects, or personal likenesses. The key is quality data and patient iteration.

Start with a small dataset and simple settings. If results are not quite right, you now know how to diagnose the problem and adjust. Each training run teaches you something about what works for your specific use case.

---

## See Also
- [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) - Understanding the base models you'll train on
- [ComfyUI Guide](comfyui-guide.html) - Use trained LoRAs in advanced workflows
- [ControlNet](controlnet.html) - Combine LoRAs with ControlNet for precision control
- [Model Types](model-types.html) - Understanding LoRAs, checkpoints, and embeddings
- [Base Models Comparison](base-models-comparison.html) - Choosing the right base for training
- [Advanced Techniques](advanced-techniques.html) - Expert LoRA usage patterns
- [AI Fundamentals](../technology/ai.html) - Neural network training concepts
- [AI/ML Documentation Hub](./) - Complete AI/ML documentation index