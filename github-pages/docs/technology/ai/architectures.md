---
layout: docs
title: "AI: ML & Deep Learning"
permalink: /docs/technology/ai/architectures.html
toc: true
toc_sticky: true
hide_title: true
---

[AI & Machine Learning](./) › ML & Deep Learning

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">ML & Deep Learning</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">From the mathematics of learning to the classical algorithms and the deep architectures that power modern AI.</p>
</div>

This is the hub for the core machine-learning and deep-learning track. It builds in reading order: first the **mathematics that makes learning possible**, then the **classical algorithms** that still win most tabular problems, then the **theory of why depth works**, and finally the **deep architectures** themselves — CNNs, RNNs, Transformers, and what came after. Read the four pages in sequence for a complete path from first principles to the Transformer, or jump straight to the one you need.

<div class="command-grid">
  <a href="ml-foundations.html" class="nav-card">
    <h4><i class="fas fa-square-root-alt"></i> 1. Machine Learning Foundations</h4>
    <p>Statistical learning theory, the bias–variance tradeoff, gradient descent and SGD, the kernel trick and SVMs, Gaussian processes, and variational inference — the mathematics that makes learning from finite data work.</p>
  </a>
  <a href="core-ml-algorithms.html" class="nav-card">
    <h4><i class="fas fa-sitemap"></i> 2. Core ML Algorithms</h4>
    <p>The classical workhorses — linear and logistic regression, decision trees, random forests, gradient boosting (XGBoost/LightGBM), SVMs, k-NN, and clustering — with runnable scikit-learn code. Still the default on tabular data.</p>
  </a>
  <a href="deep-learning-theory.html" class="nav-card">
    <h4><i class="fas fa-brain"></i> 3. Deep Learning Theory</h4>
    <p>Universal approximation, backpropagation, the optimization landscape, initialization and normalization, the neural tangent kernel, double descent, and the generalization puzzle — why deep networks can fit anything yet still generalize.</p>
  </a>
  <a href="deep-learning-architectures.html" class="nav-card">
    <h4><i class="fas fa-layer-group"></i> 4. Deep Learning Architectures</h4>
    <p>The architectural families: the multilayer perceptron, convolutional networks for vision, recurrent networks and LSTMs for sequences, attention and the Transformer, Vision Transformers and CLIP, and the post-Transformer landscape.</p>
  </a>
</div>

## How the Track Fits Together

The four pages form a deliberate progression — each assumes the one before it:

1. **[Machine Learning Foundations](ml-foundations.html)** answers *why learning from data is possible at all*. It establishes generalization, overfitting, the bias–variance tradeoff, and the optimization toolkit (gradient descent, SGD, Adam) that every later page relies on, then connects to classical tools — kernels, Gaussian processes, and variational inference — that reappear throughout deep learning.

2. **[Core ML Algorithms](core-ml-algorithms.html)** is the practical toolbox built on those foundations. If your data fits in a dataframe with named columns, start here: tree ensembles and gradient boosting are the strongest baselines on tabular data and should be beaten before a neural network is justified.

3. **[Deep Learning Theory](deep-learning-theory.html)** is the rigorous companion to the architectures. It explains what guarantees (and what mysteries) sit underneath deep networks: universal approximation bounds, how gradients flow, why overparameterized models still generalize, and the neural tangent kernel that links wide networks back to the kernel methods from the foundations page.

4. **[Deep Learning Architectures](deep-learning-architectures.html)** is the tour of the models themselves — the convolutional, recurrent, and attention-based families — with intuition, core math, and pointers to runnable code for each.

Reserve deep learning for images, audio, text, and other high-dimensional, weakly-structured signals; reach for the classical algorithms first on structured tabular data.

## See Also

- [Generative Models](generative-models.html) — diffusion, GANs, VAEs, and LLM generation built on these architectures
- [Reinforcement Learning](reinforcement-learning.html) — learning from interaction rather than labeled data
- [Fine-Tuning & Transfer Learning](fine-tuning.html) — adapting pretrained deep models to new tasks
- [Loss Functions](loss-functions.html) — the objectives these models are trained against
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling laws and interpretability of large models
- [AI Mathematics](../../advanced/ai-mathematics/) — formal proofs for the theory above
</content>
</invoke>
