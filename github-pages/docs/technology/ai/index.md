---
layout: docs
title: AI & Machine Learning
permalink: /docs/technology/ai/
toc: false
hide_title: true
---

<style>
/* Beginner notice styling */
.beginner-notice {
  background: #e8f4f8;
  border: 2px solid #3498db;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.beginner-notice i {
  font-size: 1.5rem;
  color: #3498db;
}

.beginner-notice p {
  margin: 0;
  flex: 1;
}

.beginner-notice a {
  color: #2980b9;
  font-weight: bold;
  text-decoration: underline;
}

.beginner-notice a:hover {
  color: #1a5276;
}
</style>

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AI &amp; Machine Learning</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Creating Intelligent Systems — from the mathematics of learning to frontier research and ethics.</p>
</div>

<div class="beginner-notice">
  <i class="fas fa-info-circle"></i>
  <p><strong>New to AI?</strong> Start with the <a href="../ai-fundamentals-simple.html">simplified version</a> — no math required. Come back here when you're ready for the technical details.</p>
</div>

Artificial Intelligence refers to computer systems that perform tasks typically requiring human intelligence — visual perception, speech recognition, decision-making, and natural language understanding. Its sub-fields nest: **machine learning** (systems that learn from data), **deep learning** (neural networks with many layers), and **NLP** (understanding human language). At its core AI is powered by mathematics; understanding it isn't just academic — it helps you build better systems, diagnose problems, and push boundaries. This reference introduces mathematical concepts as they are needed, always starting from practical motivation.

## Three Ways to Read This Material

This site covers AI at three depths. Pick the one that matches where you are:

| Page | Audience | What it covers |
|------|----------|----------------|
| [AI Fundamentals (Simplified)](../ai-fundamentals-simple.html) | Beginners | Plain-language intuitions, no math required — the gentle on-ramp. |
| **This Complete Reference** (you are here) | Practitioners | The full technical treatment: learning theory, architectures, generative models, frontier research, and ethics — split across the pages below. |
| [AI Deep Dive (Lecture)](../ai-lecture-2023.html) | Advanced | A focused lecture on transformers, LLM internals, and current research. |

For practical, hands-on generative-AI guides (Stable Diffusion, ComfyUI, LoRA training), see the [AI/ML section](../../ai-ml/). For the complete index of every AI resource on the site, see the [Artificial Intelligence hub](../../artificial-intelligence/index.html).

---

## How to Use This Hub

This reference is split into focused pages so you can go as deep as you need on any one topic without wading through the rest. Each page is self-contained, but they build on one another. If you are not sure where to begin, use these **start here** pointers:

- **New to the math of learning?** Start with [Machine Learning Foundations](ml-foundations.html) — it sets up the statistical-learning vocabulary every later page assumes.
- **Working with tabular data?** Go straight to [Core ML Algorithms](core-ml-algorithms.html); you can skip most of the deep-learning pages.
- **Here for neural networks?** Read [Deep Learning Theory](deep-learning-theory.html) for the "why," then [Deep Learning Architectures](deep-learning-architectures.html) for the "how." The older [Neural Network Architectures](architectures.html) overview ties the same ideas together in one narrative if you prefer a single pass.
- **Debugging or designing a training objective?** [Loss Functions & Objectives](loss-functions.html) is the page to bookmark — the objective is the model.
- **Adapting a pretrained model?** Jump to [Fine-Tuning & Transfer Learning](fine-tuning.html) (LoRA, instruction tuning, RLHF, DPO).
- **Building agents or reward-driven systems?** Start at [Reinforcement Learning](reinforcement-learning.html).
- **Generating images, audio, or text?** See [Generative Models](generative-models.html).
- **Thinking about scale, safety, or societal impact?** Read [Frontier Research & Ethics](frontier-and-ethics.html).

**Suggested reading order (front to back):**
[Foundations](ml-foundations.html) → [Core Algorithms](core-ml-algorithms.html) → [DL Theory](deep-learning-theory.html) → [DL Architectures](deep-learning-architectures.html) → [Loss Functions](loss-functions.html) → [Fine-Tuning](fine-tuning.html) → [Reinforcement Learning](reinforcement-learning.html) → [Generative Models](generative-models.html) → [Frontier & Ethics](frontier-and-ethics.html).

---

## Explore the Reference

| Page | Group | What it covers |
|------|-------|----------------|
| [Machine Learning Foundations](ml-foundations.html) | Foundations | Statistical learning theory, optimization, kernels and SVMs, Gaussian processes, and variational inference. |
| [Core ML Algorithms](core-ml-algorithms.html) | Foundations | The classical workhorses — regression, trees, boosting, SVMs, k-NN, and clustering — that still win most tabular problems. |
| [Deep Learning Theory](deep-learning-theory.html) | Deep Learning | Why deep networks can fit anything, how gradients flow, and what makes them generalize despite being overparameterized. |
| [Deep Learning Architectures](deep-learning-architectures.html) | Deep Learning | From the MLP to convolutions, recurrence, attention, Transformers, and the sequence models that followed. |
| [Neural Network Architectures](architectures.html) | Deep Learning | A single-narrative overview tying the foundations to CNNs, RNNs/LSTMs, Transformers, and multimodal models. |
| [Loss Functions & Objectives](loss-functions.html) | Training | Regression, classification, contrastive, ranking, and generative losses — and the why behind each. |
| [Fine-Tuning & Transfer Learning](fine-tuning.html) | Training | Adapting pretrained models — full fine-tuning, LoRA, instruction tuning, RLHF, and DPO. |
| [Reinforcement Learning](reinforcement-learning.html) | Training | Learning to act from reward — MDPs, value and policy methods, deep RL, and the algorithms behind agents and RLHF. |
| [Generative Models](generative-models.html) | Generation & Frontier | Diffusion models, GANs, VAEs, and autoregressive/LLM generation. |
| [Frontier Research & Ethics](frontier-and-ethics.html) | Generation & Frontier | Scaling laws, mechanistic interpretability, emergent abilities, AI safety/alignment, ethics, and governance. |

---

## Types of AI

**Narrow AI** (weak AI) performs specific tasks within a single domain, often surpassing human performance there, but cannot generalize across domains. Nearly every deployed AI system today is narrow:

- **IBM's Deep Blue** — chess computer that defeated world champion Garry Kasparov in 1997.
- **Google's AlphaGo** — Go-playing AI that defeated world champion Lee Sedol in 2016.
- **Amazon's Alexa** / **Apple's Siri** — voice-controlled virtual assistants.
- **OpenAI's ChatGPT (GPT-4-class / o-series)** — language models with multimodal input and step-by-step reasoning.
- **Claude 4 (Anthropic)** — Constitutional AI with strong safety alignment and coding ability.
- **Google's Gemini** — multimodal model processing text, images, audio, and video natively.

**General AI** (strong AI, or artificial general intelligence/AGI) would perform any intellectual task a human can, with broad world understanding and the ability to learn and adapt across challenges. **Status: not yet achieved — an active research area.** The central open challenges are *scalability* (handling vast knowledge and reasoning), *transfer learning* (applying knowledge from one domain to unfamiliar ones), and *commonsense reasoning* (understanding everyday situations).

## Machine Learning: Teaching Computers to Learn

Machine learning is a branch of artificial intelligence that focuses on the development of algorithms and models that can learn from data and make predictions or decisions. The primary goal of machine learning is to enable computers to improve their performance on a task over time without being explicitly programmed.

<div class="ml-section">
  <h3><i class="fas fa-graduation-cap"></i> Types of Machine Learning</h3>

  <div class="ml-types-grid">
    <div class="ml-type-card supervised">
      <div class="ml-icon"><i class="fas fa-tag"></i></div>
      <h4>Supervised Learning</h4>
      <p>The algorithm is trained on a labeled dataset, where the input features are mapped to output labels. The goal is to learn a function that can make accurate predictions for new, unseen data.</p>

      <div class="ml-visual">
        <svg viewBox="0 0 200 150">
          <!-- Training data with labels -->
          <g class="data-points">
            <circle cx="40" cy="40" r="8" fill="#3498db" />
            <text x="55" y="45" font-size="10">Cat</text>
            <circle cx="40" cy="70" r="8" fill="#e74c3c" />
            <text x="55" y="75" font-size="10">Dog</text>
            <circle cx="40" cy="100" r="8" fill="#3498db" />
            <text x="55" y="105" font-size="10">Cat</text>
          </g>

          <!-- Model -->
          <rect x="90" y="50" width="40" height="40" fill="#95a5a6" opacity="0.5" />
          <text x="110" y="75" text-anchor="middle" font-size="10">Model</text>

          <!-- Prediction -->
          <circle cx="160" cy="70" r="8" fill="#27ae60" />
          <text x="160" y="90" text-anchor="middle" font-size="10">?</text>

          <!-- Arrows -->
          <path d="M 70 70 L 85 70" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
          <path d="M 135 70 L 150 70" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
        </svg>
      </div>

      <div class="examples">
        <span class="example-tag">Regression</span>
        <span class="example-tag">Classification</span>
      </div>
    </div>

    <div class="ml-type-card unsupervised">
      <div class="ml-icon"><i class="fas fa-project-diagram"></i></div>
      <h4>Unsupervised Learning</h4>
      <p>The algorithm is trained on an unlabeled dataset, and the goal is to find patterns, relationships, or structures within the data.</p>

      <div class="ml-visual">
        <svg viewBox="0 0 200 150">
          <!-- Unlabeled data points -->
          <g class="data-points">
            <circle cx="40" cy="40" r="6" fill="#95a5a6" />
            <circle cx="60" cy="45" r="6" fill="#95a5a6" />
            <circle cx="45" cy="60" r="6" fill="#95a5a6" />
            <circle cx="140" cy="50" r="6" fill="#95a5a6" />
            <circle cx="160" cy="55" r="6" fill="#95a5a6" />
            <circle cx="145" cy="70" r="6" fill="#95a5a6" />
            <circle cx="100" cy="100" r="6" fill="#95a5a6" />
            <circle cx="90" cy="120" r="6" fill="#95a5a6" />
            <circle cx="110" cy="115" r="6" fill="#95a5a6" />
          </g>

          <!-- Discovered clusters -->
          <ellipse cx="50" cy="50" rx="30" ry="25" fill="#3498db" opacity="0.2" />
          <ellipse cx="150" cy="60" rx="30" ry="25" fill="#e74c3c" opacity="0.2" />
          <ellipse cx="100" cy="110" rx="30" ry="25" fill="#27ae60" opacity="0.2" />

          <text x="100" y="140" text-anchor="middle" font-size="10">Discovered Patterns</text>
        </svg>
      </div>

      <div class="examples">
        <span class="example-tag">Clustering</span>
        <span class="example-tag">Dimensionality Reduction</span>
      </div>
    </div>

    <div class="ml-type-card reinforcement">
      <div class="ml-icon"><i class="fas fa-robot"></i></div>
      <h4>Reinforcement Learning</h4>
      <p>The algorithm learns by interacting with an environment, receiving feedback in the form of rewards or penalties, and adjusting its actions to maximize cumulative rewards over time.</p>

      <div class="ml-visual">
        <svg viewBox="0 0 200 150">
          <!-- Agent -->
          <circle cx="50" cy="75" r="20" fill="#3498db" />
          <text x="50" y="80" text-anchor="middle" font-size="10" fill="white">Agent</text>

          <!-- Environment -->
          <rect x="120" y="40" width="70" height="70" fill="#27ae60" opacity="0.3" stroke="#27ae60" stroke-width="2" />
          <text x="155" y="80" text-anchor="middle" font-size="10">Environment</text>

          <!-- Action arrow -->
          <path d="M 70 65 Q 95 55, 120 65" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)" />
          <text x="95" y="50" text-anchor="middle" font-size="9">Action</text>

          <!-- Reward arrow -->
          <path d="M 120 85 Q 95 95, 70 85" stroke="#f39c12" stroke-width="2" marker-end="url(#arrow)" />
          <text x="95" y="105" text-anchor="middle" font-size="9">Reward</text>
        </svg>
      </div>

      <div class="examples">
        <span class="example-tag">Game Playing</span>
        <span class="example-tag">Robotics</span>
      </div>
    </div>
  </div>
</div>

<div class="dl-hierarchy">
  <h4>AI, ML, and DL Relationship</h4>
  <div class="hierarchy-visual">
    <div class="hierarchy-level ai-level">
      <span>Artificial Intelligence</span>
      <div class="hierarchy-level ml-level">
        <span>Machine Learning</span>
        <div class="hierarchy-level dl-level">
          <span>Deep Learning</span>
        </div>
      </div>
    </div>
  </div>
</div>

Deep learning—machine learning with neural networks many layers deep—is where most of today's breakthroughs happen. [Deep Learning Theory](deep-learning-theory.html) explains why these models work, and [Deep Learning Architectures](deep-learning-architectures.html) walks through the building blocks themselves; the [Machine Learning Foundations](ml-foundations.html) page covers the statistical learning theory and optimization that underpin both.

---

## Key Takeaways

- **Learning is optimization.** Training reduces to following the gradient downhill: $\theta_{t+1} = \theta_t - \eta\nabla_\theta\mathcal{L}$ — everything else is architecture and data.
- **Depth builds abstraction.** Deep networks learn hierarchical features; the transformer's self-attention made long-range, parallel modeling practical and now dominates language and vision.
- **Generative models reverse a known process.** Diffusion models learn to denoise via $\mathcal{L} = \mathbb{E}[\lVert\varepsilon - \varepsilon_\theta(\mathbf{x}_t,t)\rVert^2]$, turning random noise into structured images.
- **Scale is predictable — to a point.** Loss follows power laws in parameters and data ($L = E + A/N^\alpha + B/D^\beta$), but data quality and compute-optimal allocation matter as much as raw size.
- **Capability and responsibility scale together.** Fairness, interpretability, privacy, and safety are core engineering requirements, not optional add-ons.

---

## See Also

Reference pages in this section:

- [Machine Learning Foundations](ml-foundations.html) — statistical learning theory, optimization, kernels, GPs
- [Core ML Algorithms](core-ml-algorithms.html) — regression, trees, boosting, SVMs, clustering
- [Deep Learning Theory](deep-learning-theory.html) — expressivity, gradient flow, generalization
- [Deep Learning Architectures](deep-learning-architectures.html) — MLPs, CNNs, RNNs, Transformers
- [Neural Network Architectures](architectures.html) — single-narrative architectures overview
- [Loss Functions & Objectives](loss-functions.html) — the objective that defines the model
- [Fine-Tuning & Transfer Learning](fine-tuning.html) — LoRA, instruction tuning, RLHF, DPO
- [Reinforcement Learning](reinforcement-learning.html) — MDPs, value/policy methods, deep RL
- [Generative Models](generative-models.html) — diffusion, GANs, VAEs, autoregressive
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling, interpretability, safety, governance

Related pages:

- [AI Documentation Hub](../../artificial-intelligence/index.html) — complete index of all AI resources
- [AI Fundamentals (Simplified)](../ai-fundamentals-simple.html) — the no-math starting point
- [AI Deep Dive](../ai-lecture-2023.html) — transformers, LLM internals, and current research
- [AI/ML Documentation Hub](../../ai-ml/) — practical generative AI guides
- [AI Mathematics](../../advanced/ai-mathematics/) — theoretical foundations and proofs
- [Quantum Computing](../quantumcomputing.html) — quantum machine learning
- [AWS](../aws/) — cloud platforms for AI/ML workloads
