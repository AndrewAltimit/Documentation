---
layout: docs
title: "AI: ML & Deep Learning"
permalink: /docs/technology/ai/architectures.html
toc: true
[AI & Machine Learning](./) › ML & Deep Learning

This is the hub for the core machine-learning and deep-learning track. It builds in reading order: first the **mathematics that makes learning possible**, then the **classical algorithms** that still win most tabular problems, then the **theory of why depth works**, and finally the **deep architectures** themselves — CNNs, RNNs, Transformers, and what came after. Read the four pages in sequence for a complete path from first principles to the Transformer, or jump straight to the one you need.

| # | Page | What it covers |
|---|------|----------------|
| 1 | [Machine Learning Foundations](ml-foundations.html) | Statistical learning theory, the bias–variance tradeoff, gradient descent and SGD, the kernel trick and SVMs, Gaussian processes, and variational inference. |
| 2 | [Core ML Algorithms](core-ml-algorithms.html) | Linear/logistic regression, decision trees, random forests, gradient boosting (XGBoost/LightGBM), SVMs, k-NN, and clustering — with runnable scikit-learn code. |
| 3 | [Deep Learning Theory](deep-learning-theory.html) | Universal approximation, backpropagation, the optimization landscape, initialization and normalization, the neural tangent kernel, double descent, and the generalization puzzle. |
| 4 | [Deep Learning Architectures](deep-learning-architectures.html) | The multilayer perceptron, convolutional networks for vision, RNNs and LSTMs for sequences, attention and the Transformer, Vision Transformers and CLIP, and the post-Transformer landscape. |

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
