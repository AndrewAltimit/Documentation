---
layout: docs
title: "AI & ML: Generative Models"
permalink: /docs/technology/ai/generative-models.html
toc: true
toc_sticky: true
---

[AI & Machine Learning](./) › Generative Models

Generative models learn the distribution of their training data so they can produce new, original samples from it. This page covers the dominant families—diffusion models, GANs, VAEs, and the autoregressive approach behind large language models—along with the mathematics that makes them work.

## The Mathematics Behind Modern Image Generation

Remember those AI-generated images that look impossibly real? They're created using diffusion models—a mathematical framework that seemed counterintuitive at first but has proven incredibly powerful. The key insight: instead of trying to generate images directly, we learn how to gradually remove noise from random static.

### Score-Based Generative Modeling

**Score-based diffusion models use continuous-time stochastic differential equations:**

- **Forward SDE** gradually adds noise:

$$d\mathbf{x} = f(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$$

- **Reverse SDE** runs the process backward to generate samples:

$$d\mathbf{x} = \left[f(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + g(t)\,d\bar{\mathbf{w}}$$

- **Score Matching**: learn the score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ via denoising
- **Variance schedule**: $\sigma(t) = \sigma_{\min}\left(\sigma_{\max}/\sigma_{\min}\right)^{t}$

**Key advantages:**
- Continuous time formulation enables flexible sampling
- Predictor-corrector methods improve sample quality
- Connection to neural ODEs and normalizing flows
- State-of-the-art image generation quality

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/ai/diffusion_models.py#L15">diffusion_models.py#ScoreBasedDiffusion</a>
</div>

```python
# Example usage:
from diffusion_models import ScoreBasedDiffusion

# Create score-based model
score_model = UNet(...)  # score network (predicts the score / noise)
# training loop: sample t, add noise to x_0, predict the score, backprop the loss
diffusion = ScoreBasedDiffusion(score_model, sigma_min=0.01, sigma_max=50.0)

# Training
loss = diffusion.loss_fn(batch_images)

# Sampling
samples = diffusion.sample(shape=(16, 3, 256, 256), num_steps=1000)
```

### DDPM Mathematical Framework

While score-based models work in continuous time, researchers found that discretizing the process into fixed timesteps could make training more stable and efficient. This led to DDPMs, which have become the foundation for many practical diffusion models.

**Denoising Diffusion Probabilistic Models (DDPM) use discrete timesteps:**

- **Forward process** (closed form at any step $t$):

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\, (1-\bar\alpha_t)\mathbf{I}\right)$$

- **Reverse process**: $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ learned via a neural network
- **Training objective** (predict the added noise $\varepsilon$):

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\,\mathbf{x}_0,\,\varepsilon}\left[\,\left\lVert \varepsilon - \varepsilon_\theta(\mathbf{x}_t, t)\right\rVert^2\,\right]$$

- **Variance schedule**: $\beta_t$ controls the noise level added at each step; $\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$

**Key innovations:**
- Simplified loss function (predict noise instead of data)
- Reparameterization for stable training
- DDIM: Deterministic sampling variant
- Improved schedules (cosine, learned)

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/ai/diffusion_models.py#L108">diffusion_models.py#DDPM</a>
</div>

```python
# Example usage:
from diffusion_models import DDPM

# Create DDPM model
noise_predictor = UNet(...)  # Your noise prediction network
ddpm = DDPM(noise_predictor, T=1000, beta_start=0.0001, beta_end=0.02)

# Training
loss = ddpm.loss(batch_images)

# Sampling
samples = ddpm.sample(shape=(16, 3, 256, 256))

# DDIM sampling (faster)
samples = ddpm.ddim_sample(shape=(16, 3, 256, 256), ddim_timesteps=50)
```

## Diffusion Models: Creating Art from Noise

Diffusion models are a class of generative models that revolutionized image generation and are expanding into other domains. They work by gradually adding noise to data and then learning to reverse that process, enabling high-quality sample generation.

### How Diffusion Models Work

<div class="diffusion-process">
    <div class="process-visual">
      <svg viewBox="0 0 600 200">
        <!-- Forward process -->
        <text x="300" y="30" text-anchor="middle" font-size="12" font-weight="bold">Forward Process (Adding Noise)</text>

        <!-- Original image -->
        <rect x="50" y="50" width="60" height="60" fill="url(#imageGradient)" stroke="#2c3e50" stroke-width="2" />
        <text x="80" y="130" text-anchor="middle" font-size="10">Original</text>

        <!-- Arrow -->
        <path d="M 115 80 L 145 80" stroke="#95a5a6" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Partially noisy -->
        <rect x="150" y="50" width="60" height="60" fill="url(#noisyGradient1)" stroke="#2c3e50" stroke-width="2" opacity="0.8" />
        <text x="180" y="130" text-anchor="middle" font-size="10">t = 100</text>

        <!-- Arrow -->
        <path d="M 215 80 L 245 80" stroke="#95a5a6" stroke-width="2" marker-end="url(#arrow)" />

        <!-- More noisy -->
        <rect x="250" y="50" width="60" height="60" fill="url(#noisyGradient2)" stroke="#2c3e50" stroke-width="2" opacity="0.6" />
        <text x="280" y="130" text-anchor="middle" font-size="10">t = 500</text>

        <!-- Arrow -->
        <path d="M 315 80 L 345 80" stroke="#95a5a6" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Pure noise -->
        <rect x="350" y="50" width="60" height="60" fill="#95a5a6" stroke="#2c3e50" stroke-width="2" />
        <text x="380" y="130" text-anchor="middle" font-size="10">Pure Noise</text>

        <!-- Reverse process arrow -->
        <path d="M 380 140 Q 230 160, 80 140" stroke="#e74c3c" stroke-width="3" marker-end="url(#arrow)" fill="none" />
        <text x="230" y="180" text-anchor="middle" font-size="12" fill="#e74c3c">Reverse Process (Denoising)</text>

        <!-- Gradient definitions -->
        <defs>
          <linearGradient id="imageGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#3498db;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#2ecc71;stop-opacity:1" />
          </linearGradient>
          <linearGradient id="noisyGradient1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#3498db;stop-opacity:0.7" />
            <stop offset="50%" style="stop-color:#95a5a6;stop-opacity:0.7" />
            <stop offset="100%" style="stop-color:#2ecc71;stop-opacity:0.7" />
          </linearGradient>
          <linearGradient id="noisyGradient2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#95a5a6;stop-opacity:0.8" />
            <stop offset="100%" style="stop-color:#7f8c8d;stop-opacity:0.8" />
          </linearGradient>
        </defs>
      </svg>
    </div>
</div>

The process has four stages:

1. **Forward process** — gradually adds Gaussian noise to data over many timesteps until it becomes pure noise.
2. **Reverse process** — learns to denoise step by step, recovering the original data distribution.
3. **Training** — the model learns to predict the noise added at each step.
4. **Generation** — starting from random noise, the model iteratively removes noise to produce new samples.

### Making Diffusion Practical: Latent Diffusion

The mathematical elegance of diffusion models is compelling, but early versions were too slow and computationally expensive for practical use. The key practical breakthrough was **Latent Diffusion Models (LDMs)**: rather than running the noisy diffusion process in pixel space, an autoencoder first compresses the image into a much smaller latent representation, the diffusion model operates there, and a decoder maps the result back to pixels. This cuts compute by an order of magnitude and is what makes text-to-image models like Stable Diffusion runnable on consumer GPUs. Conditioning (e.g., a text prompt) is injected into the denoising U-Net via cross-attention, and **classifier-free guidance** scales how strongly the sample follows that prompt.

<div class="notice--info">
  <p>For a full walkthrough of the latent-diffusion pipeline — VAE encode/decode, the conditioned U-Net, the scheduler, and a runnable Stable Diffusion example — see <a href="../../ai-ml/stable-diffusion-fundamentals.html">Stable Diffusion Fundamentals</a> in the AI/ML section. We keep the treatment conceptual here to avoid duplicating that pipeline.</p>
</div>

### Key Diffusion Model Architectures

#### Denoising Diffusion Probabilistic Models (DDPMs)
The foundational architecture that established the diffusion framework:
- Uses a Markov chain of diffusion steps
- Trains a neural network to predict noise at each timestep
- Achieves high sample quality but requires many denoising steps

#### Denoising Diffusion Implicit Models (DDIMs)
An improvement over DDPMs that enables:
- Deterministic sampling
- Fewer denoising steps for faster generation
- Interpolation between samples

#### Latent Diffusion Models (LDMs)
Operates in a compressed latent space:
- Significantly reduces computational requirements
- Powers Stable Diffusion and similar models
- Enables high-resolution image generation on consumer hardware

#### Score-Based Generative Models
Alternative formulation using score matching:
- Learns the gradient of the data distribution
- Provides theoretical connections to other generative models
- Enables continuous-time diffusion processes

### Real-World Impact: Applications of Diffusion Models

What started as a theoretical curiosity has become one of the most versatile tools in AI. Diffusion models aren't just creating pretty pictures—they're solving real problems across diverse fields.

#### Image Generation
- **Text-to-Image**: DALL-E 2, Stable Diffusion, Midjourney
- **Image Editing**: Inpainting, outpainting, style transfer
- **Super-Resolution**: Enhancing image quality and resolution
- **Medical Imaging**: Generating synthetic medical data, denoising scans

#### Beyond Images (representative systems, circa 2023–2024)
- **Audio Generation**: MusicGen, AudioCraft, Stable Audio, Suno AI
- **Video Generation**: Runway Gen-2, Pika Labs, Stable Video Diffusion, OpenAI Sora (preview)
- **3D Generation**: DreamGaussian, Wonder3D, Instant3D, TripoSR
- **Molecular Design**: RFDiffusion, AlphaFold 3, MoleculeGPT
- **Text-to-3D**: DreamFusion, Magic3D, Point-E, Shap-E

### Advantages of Diffusion Models

1. **Sample Quality**: Often superior to GANs in terms of fidelity and diversity
2. **Training Stability**: More stable training compared to GANs
3. **Mode Coverage**: Better at capturing the full data distribution
4. **Controllability**: Easy to incorporate conditioning information

### Challenges and Limitations

1. **Computational Cost**: Requires many denoising steps for generation
2. **Memory Requirements**: High-resolution generation needs significant resources
3. **Speed**: Slower than GANs for real-time applications
4. **Data Requirements**: Needs large datasets for training

### Recent Advances

#### Classifier-Free Guidance
Improves sample quality by combining conditional and unconditional models:
- Enables better adherence to text prompts
- Adjustable guidance scale for quality vs diversity trade-off

#### Consistency Models
New approach that enables single-step generation:
- Drastically reduces inference time
- Maintains competitive sample quality
- Promising for real-time applications

#### Cross-Attention Mechanisms
Enables better text-image alignment:
- Improved prompt following
- Fine-grained control over generation
- Used in most modern text-to-image models

## Other Generative Families: GANs, VAEs, and Autoregressive Models

Diffusion models dominate image generation today, but they're part of a broader landscape. Each family makes a different trade-off between sample quality, diversity, training stability, and speed.

### Generative Adversarial Networks (GANs)

A GAN pits two networks against each other: a **generator** tries to produce realistic samples from random noise, while a **discriminator** tries to tell real samples from generated ones. They play a minimax game,

$$\min_G \max_D \; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}[\log(1 - D(G(\mathbf{z})))],$$

driving the generator toward outputs the discriminator can no longer distinguish from real data.

- **Strengths**: Fast single-pass generation, sharp samples (StyleGAN-quality faces)
- **Weaknesses**: Notoriously unstable training; mode collapse (the generator covers only part of the distribution)
- **Notable variants**: DCGAN, StyleGAN/StyleGAN3, CycleGAN (unpaired image translation), Pix2Pix

### Variational Autoencoders (VAEs)

A VAE learns a probabilistic latent space. An **encoder** maps inputs to a distribution over latent codes, and a **decoder** reconstructs the input from a sampled code. Training maximizes the **evidence lower bound (ELBO)**,

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p(\mathbf{z})\right),$$

which balances reconstruction quality against keeping the latent distribution close to a simple prior.

- **Strengths**: Stable training, smooth and interpretable latent space, principled probabilistic framing
- **Weaknesses**: Reconstructions tend to be blurrier than GAN or diffusion outputs
- **Modern role**: The autoencoder inside a latent diffusion model is essentially a VAE — VAEs now often serve as the compression front-end rather than the final generator.

The [variational inference](architectures.html) discussion on the architectures page covers the approximation machinery VAEs rely on.

### Autoregressive Models and LLM Generation

Autoregressive models factor a joint distribution into a product of conditionals and generate one token at a time, each conditioned on everything generated so far:

$$p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i \mid x_1, \dots, x_{i-1}).$$

This is exactly how large language models generate text: given a prompt, the transformer predicts a probability distribution over the next token, a token is sampled (greedy, temperature, top-k, or nucleus/top-p sampling), it's appended to the context, and the process repeats.

- **Strengths**: Exact likelihoods, excellent at sequential/discrete data, the backbone of modern LLMs (GPT-4-class / o-series, Claude 4, Gemini)
- **Weaknesses**: Sequential generation is slow; quality depends heavily on decoding strategy
- **Beyond text**: The same principle drives autoregressive image models (PixelCNN, ImageGPT) and audio models (WaveNet)

For how these models scale and what new abilities emerge with size, see [Frontier Research & Ethics](frontier-and-ethics.html).

---

## Continue Reading

<div class="page-nav" style="display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;">
  <span>← <strong>Previous:</strong> <a href="architectures.html">Neural Network Architectures</a></span>
  <span><strong>Next:</strong> <a href="frontier-and-ethics.html">Frontier Research &amp; Ethics</a> →</span>
</div>

### See Also

- [Stable Diffusion Fundamentals](../../ai-ml/stable-diffusion-fundamentals.html) — the full latent-diffusion pipeline, hands-on
- [AI/ML Documentation Hub](../../ai-ml/) — practical generative-AI guides (ComfyUI, LoRA, ControlNet)
- [Neural Network Architectures](architectures.html) — the transformers and autoencoders these models are built from
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling laws and the frontier of generative AI
