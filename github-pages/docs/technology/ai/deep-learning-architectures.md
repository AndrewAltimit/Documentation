---
layout: docs
title: "AI: Deep Learning Architectures"
permalink: /docs/technology/ai/deep-learning-architectures.html
toc: true
toc_sticky: true
---

[AI & Machine Learning](./) › Deep Learning Architectures

Deep learning is the art of composing simple differentiable operations into networks deep enough to learn hierarchical representations. This page is a tour of the *architectural families* that have defined the field: the fully-connected baseline, the convolutional networks that conquered vision, the recurrent networks that first tamed sequences, the attention mechanism and the Transformer that displaced them, the Vision Transformer and CLIP that brought attention to images and to multimodal data, and finally the post-Transformer landscape of state-space models and efficient-attention variants. Each section gives the intuition, the core mathematics (ASCII display math only), and a pointer to runnable code.

For the *theory* of why depth works at all — universal approximation, the optimization landscape, and the neural tangent kernel — see the companion page on [Neural Network Architectures](architectures.html). This page focuses on the architectures themselves.

## The Multilayer Perceptron (MLP)

The multilayer perceptron is the foundational deep architecture: a stack of fully-connected (dense) layers, each applying an affine transform followed by a pointwise nonlinearity. Every architecture below can be seen as an MLP with structure baked into its weight matrices.

### Forward pass

A single layer maps an input vector $\mathbf{x}$ to an output $\mathbf{h}$ via a weight matrix $W$, a bias $\mathbf{b}$, and an activation $\phi$:

$$\mathbf{h} = \phi(W\mathbf{x} + \mathbf{b})$$

An $L$-layer MLP composes these:

$$f(\mathbf{x}) = \phi_L\!\left(W_L \,\phi_{L-1}\!\left(\cdots \phi_1(W_1 \mathbf{x} + \mathbf{b}_1) \cdots\right) + \mathbf{b}_L\right)$$

The nonlinearity is essential: without $\phi$, a composition of affine maps collapses to a single affine map, and depth buys nothing. Common choices are ReLU $\phi(z) = \max(0, z)$, its smooth relatives GELU and SiLU, and the saturating sigmoid and tanh used in older networks and in gating.

### Backpropagation

Training minimizes a loss $\mathcal{L}$ by gradient descent. Backpropagation is reverse-mode automatic differentiation: it computes $\nabla_\theta \mathcal{L}$ in a single backward sweep by applying the chain rule layer by layer. Writing the pre-activation of layer $\ell$ as $\mathbf{z}^{(\ell)} = W_\ell \mathbf{h}^{(\ell-1)} + \mathbf{b}_\ell$ and the error signal as $\boldsymbol{\delta}^{(\ell)} = \partial \mathcal{L} / \partial \mathbf{z}^{(\ell)}$, the recursion is:

$$\boldsymbol{\delta}^{(\ell)} = \left(W_{\ell+1}^{\top}\,\boldsymbol{\delta}^{(\ell+1)}\right) \odot \phi'\!\left(\mathbf{z}^{(\ell)}\right)$$

$$\frac{\partial \mathcal{L}}{\partial W_\ell} = \boldsymbol{\delta}^{(\ell)} \,\mathbf{h}^{(\ell-1)\top}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell} = \boldsymbol{\delta}^{(\ell)}$$

where $\odot$ is the elementwise (Hadamard) product. The same machinery — accumulate local Jacobians, multiply on the backward pass — drives every architecture below; only the layer operators change.

### Why MLPs are not enough

An MLP applied to an image flattens it into a vector, discarding the 2-D grid structure, and uses a dense weight for every pixel-to-unit connection. For a 224×224×3 image, the first layer alone needs about 150,000 weights *per hidden unit*. This is statistically wasteful (no parameter sharing) and ignores the strong prior that nearby pixels are related. The architectures that follow each inject a structural prior — locality, recurrence, or content-based routing — that an MLP lacks.

## Convolutional Neural Networks (CNNs)

CNNs are the architecture that made modern computer vision work. They replace the dense first layers of an MLP with **convolutions**: small learnable filters slid across the input, sharing weights across spatial positions.

<div class="arch-card cnn">
  <div class="arch-visual">
    <svg viewBox="0 0 300 150">
      <rect x="20" y="40" width="40" height="40" fill="#3498db" opacity="0.5" />
      <text x="40" y="100" text-anchor="middle" font-size="9">Input</text>
      <rect x="80" y="30" width="35" height="35" fill="#e74c3c" opacity="0.5" />
      <rect x="85" y="35" width="35" height="35" fill="#e74c3c" opacity="0.5" />
      <rect x="90" y="40" width="35" height="35" fill="#e74c3c" opacity="0.5" />
      <text x="107" y="90" text-anchor="middle" font-size="9">Conv</text>
      <rect x="145" y="45" width="25" height="25" fill="#f39c12" opacity="0.5" />
      <text x="157" y="80" text-anchor="middle" font-size="9">Pool</text>
      <circle cx="200" cy="45" r="5" fill="#27ae60" />
      <circle cx="200" cy="60" r="5" fill="#27ae60" />
      <circle cx="220" cy="52" r="5" fill="#27ae60" />
      <text x="210" y="80" text-anchor="middle" font-size="9">FC</text>
      <rect x="250" y="50" width="30" height="10" fill="#9b59b6" />
      <text x="265" y="70" text-anchor="middle" font-size="9">Classes</text>
    </svg>
  </div>
</div>

### The convolution operation

A 2-D convolution (technically a cross-correlation) computes each output activation as a dot product between a $k \times k$ filter $K$ and a local patch of the input feature map $X$:

$$Y_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} K_{m,n}\, X_{i+m,\,j+n} + b$$

In practice a layer holds many filters, each producing an output channel, and the sum also runs over input channels. Three hyperparameters shape the output: the **stride** (how far the filter hops), the **padding** (zeros added at the border), and the number of filters. For an input of spatial size $W$ with filter size $k$, padding $p$, and stride $s$, the output size is:

$$W_{\text{out}} = \left\lfloor \frac{W - k + 2p}{s} \right\rfloor + 1$$

### Why convolution is the right prior

Convolution bakes in two assumptions that match natural images:

- **Locality**: a feature (an edge, a texture) depends only on a small neighborhood, so the filter is small.
- **Translation equivariance / weight sharing**: the same filter is reused at every position, so a feature detector learned in one corner works everywhere. This slashes the parameter count and makes the network robust to where an object appears.

**Pooling** layers (max or average over a window) downsample the feature map, providing a small amount of translation *invariance* and enlarging the receptive field. Stacking conv-pool blocks builds a hierarchy: early layers detect edges, middle layers detect motifs and parts, deep layers detect whole objects.

### Landmark CNN architectures

- **LeNet-5 (1998)** — the original convolutional digit recognizer.
- **AlexNet (2012)** — ReLU, dropout, and GPU training; won ImageNet and launched the deep-learning era.
- **VGG (2014)** — showed that depth with uniform 3×3 filters is powerful.
- **ResNet (2015)** — introduced the **residual connection** $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$, letting gradients flow through identity shortcuts and enabling networks hundreds of layers deep. The residual block is now ubiquitous — Transformers use it too.
- **EfficientNet / ConvNeXt** — modern CNNs that compound width, depth, and resolution scaling and remain competitive with Vision Transformers.

The residual idea is worth isolating, because it appears everywhere. By learning a residual $\mathcal{F}(\mathbf{x})$ rather than a full transformation, each block only needs to learn the *difference* from the identity, which is far easier to optimize and keeps the backward-pass Jacobian close to the identity, mitigating vanishing gradients.

*Typical uses: image classification, object detection, segmentation.*

## Recurrent Networks: RNNs, LSTMs, and GRUs

Before attention, sequences (text, audio, time series) were handled by **recurrent neural networks**, which process one element at a time while maintaining a hidden state that summarizes everything seen so far.

<div class="arch-card rnn">
  <div class="arch-visual">
    <svg viewBox="0 0 300 150">
      <rect x="40" y="50" width="40" height="40" fill="#3498db" opacity="0.5" />
      <text x="60" y="70" text-anchor="middle" font-size="10" fill="white">h₀</text>
      <rect x="100" y="50" width="40" height="40" fill="#3498db" opacity="0.5" />
      <text x="120" y="70" text-anchor="middle" font-size="10" fill="white">h₁</text>
      <rect x="160" y="50" width="40" height="40" fill="#3498db" opacity="0.5" />
      <text x="180" y="70" text-anchor="middle" font-size="10" fill="white">h₂</text>
      <text x="220" y="70" font-size="14">...</text>
      <path d="M 80 70 L 95 70" stroke="#e74c3c" stroke-width="2" />
      <path d="M 140 70 L 155 70" stroke="#e74c3c" stroke-width="2" />
      <path d="M 200 70 L 215 70" stroke="#e74c3c" stroke-width="2" />
      <circle cx="60" cy="30" r="5" fill="#27ae60" />
      <circle cx="120" cy="30" r="5" fill="#27ae60" />
      <circle cx="180" cy="30" r="5" fill="#27ae60" />
      <text x="120" y="20" text-anchor="middle" font-size="9">Sequential Input</text>
      <circle cx="60" cy="110" r="5" fill="#f39c12" />
      <circle cx="120" cy="110" r="5" fill="#f39c12" />
      <circle cx="180" cy="110" r="5" fill="#f39c12" />
    </svg>
  </div>
</div>

### The vanilla RNN

At each step $t$ the network updates its hidden state from the previous state $\mathbf{h}_{t-1}$ and the current input $\mathbf{x}_t$:

$$\mathbf{h}_t = \tanh\!\left(W_{hh}\,\mathbf{h}_{t-1} + W_{xh}\,\mathbf{x}_t + \mathbf{b}_h\right)$$

$$\mathbf{y}_t = W_{hy}\,\mathbf{h}_t + \mathbf{b}_y$$

The same weights are reused at every time step (weight sharing across time, the temporal analogue of a CNN's spatial sharing). Training uses **backpropagation through time (BPTT)**: unroll the recurrence into a deep feed-forward graph and backpropagate.

### The vanishing/exploding gradient problem

BPTT multiplies many Jacobians together. The gradient of the loss with respect to an early state involves a product of factors $\partial \mathbf{h}_t / \partial \mathbf{h}_{t-1}$ across the whole sequence. If the dominant singular value of $W_{hh}$ is below 1, this product shrinks geometrically and gradients **vanish** — the network cannot learn long-range dependencies; if it is above 1, gradients **explode**. This is precisely the obstacle that gated recurrent units were designed to overcome.

### Long Short-Term Memory (LSTM)

The LSTM adds an explicit **cell state** $\mathbf{c}_t$ — a memory conveyor belt — that information can ride along nearly unchanged, plus three gates that learn what to forget, what to write, and what to read. Each gate is a sigmoid $\sigma$ producing values in $[0,1]$ that multiplicatively control information flow:

$$\mathbf{f}_t = \sigma\!\left(W_f\,[\mathbf{h}_{t-1},\,\mathbf{x}_t] + \mathbf{b}_f\right)$$

$$\mathbf{i}_t = \sigma\!\left(W_i\,[\mathbf{h}_{t-1},\,\mathbf{x}_t] + \mathbf{b}_i\right)$$

$$\mathbf{o}_t = \sigma\!\left(W_o\,[\mathbf{h}_{t-1},\,\mathbf{x}_t] + \mathbf{b}_o\right)$$

$$\tilde{\mathbf{c}}_t = \tanh\!\left(W_c\,[\mathbf{h}_{t-1},\,\mathbf{x}_t] + \mathbf{b}_c\right)$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Here $\mathbf{f}_t$ is the **forget gate**, $\mathbf{i}_t$ the **input gate**, $\mathbf{o}_t$ the **output gate**, and $[\cdot,\cdot]$ denotes concatenation. The crucial line is the cell-state update: because $\mathbf{c}_t$ is reached from $\mathbf{c}_{t-1}$ through *addition* gated by $\mathbf{f}_t$ (not repeated matrix multiplication), gradients can flow across hundreds of steps without vanishing whenever the forget gate stays near 1.

<div class="arch-card lstm">
  <div class="arch-visual">
    <svg viewBox="0 0 300 150">
      <rect x="100" y="40" width="100" height="70" fill="#95a5a6" opacity="0.2" stroke="#7f8c8d" stroke-width="2" />
      <circle cx="130" cy="60" r="8" fill="#e74c3c" />
      <text x="130" y="65" text-anchor="middle" font-size="8" fill="white">f</text>
      <text x="130" y="80" text-anchor="middle" font-size="8">Forget</text>
      <circle cx="150" cy="60" r="8" fill="#3498db" />
      <text x="150" y="65" text-anchor="middle" font-size="8" fill="white">i</text>
      <text x="150" y="80" text-anchor="middle" font-size="8">Input</text>
      <circle cx="170" cy="60" r="8" fill="#27ae60" />
      <text x="170" y="65" text-anchor="middle" font-size="8" fill="white">o</text>
      <text x="170" y="80" text-anchor="middle" font-size="8">Output</text>
      <line x1="90" y1="50" x2="210" y2="50" stroke="#f39c12" stroke-width="3" />
      <text x="150" y="35" text-anchor="middle" font-size="9">Cell State</text>
      <circle cx="60" cy="75" r="5" fill="#2c3e50" />
      <text x="60" y="90" text-anchor="middle" font-size="8">xₜ</text>
      <circle cx="240" cy="75" r="5" fill="#2c3e50" />
      <text x="240" y="90" text-anchor="middle" font-size="8">hₜ</text>
    </svg>
  </div>
</div>

### Gated Recurrent Unit (GRU)

The GRU is a streamlined gated cell with two gates instead of three and no separate cell state, which makes it cheaper while often matching LSTM accuracy. A **reset gate** $\mathbf{r}_t$ decides how much past state to use when forming a candidate, and an **update gate** $\mathbf{z}_t$ interpolates between the old state and the candidate:

$$\mathbf{z}_t = \sigma\!\left(W_z\,[\mathbf{h}_{t-1},\,\mathbf{x}_t]\right), \qquad \mathbf{r}_t = \sigma\!\left(W_r\,[\mathbf{h}_{t-1},\,\mathbf{x}_t]\right)$$

$$\tilde{\mathbf{h}}_t = \tanh\!\left(W_h\,[\mathbf{r}_t \odot \mathbf{h}_{t-1},\,\mathbf{x}_t]\right)$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t)\odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### The fundamental limitation of recurrence

Even with gating, RNNs share two structural weaknesses: they are **inherently sequential** (step $t$ cannot be computed until step $t-1$ finishes, so they do not parallelize across the sequence on modern hardware), and information from distant tokens must survive a long chain of state updates. Both problems are solved by attention, which lets any position read directly from any other.

*Typical uses: time series, speech recognition, sequence modeling.*

## Attention and the Transformer

The Transformer (Vaswani et al., 2017, *"Attention Is All You Need"*) replaced recurrence entirely with **attention**, a content-based routing mechanism that lets every token gather information from every other token in a single parallel step. It is the backbone of GPT, BERT, T5, and essentially every modern large language model.

<div class="arch-card transformer">
  <div class="arch-visual">
    <svg viewBox="0 0 300 150">
      <text x="150" y="20" text-anchor="middle" font-size="10">Self-Attention</text>
      <rect x="40" y="120" width="30" height="20" fill="#3498db" />
      <rect x="80" y="120" width="30" height="20" fill="#3498db" />
      <rect x="120" y="120" width="30" height="20" fill="#3498db" />
      <rect x="160" y="120" width="30" height="20" fill="#3498db" />
      <rect x="200" y="120" width="30" height="20" fill="#3498db" />
      <path d="M 55 120 Q 100 80, 55 40" stroke="#e74c3c" stroke-width="1" opacity="0.5" fill="none" />
      <path d="M 55 120 Q 100 80, 95 40" stroke="#e74c3c" stroke-width="1" opacity="0.5" fill="none" />
      <path d="M 55 120 Q 100 80, 135 40" stroke="#e74c3c" stroke-width="1" opacity="0.5" fill="none" />
      <path d="M 55 120 Q 100 80, 175 40" stroke="#e74c3c" stroke-width="1" opacity="0.5" fill="none" />
      <path d="M 55 120 Q 100 80, 215 40" stroke="#e74c3c" stroke-width="1" opacity="0.5" fill="none" />
      <rect x="40" y="30" width="30" height="20" fill="#27ae60" />
      <rect x="80" y="30" width="30" height="20" fill="#27ae60" />
      <rect x="120" y="30" width="30" height="20" fill="#27ae60" />
      <rect x="160" y="30" width="30" height="20" fill="#27ae60" />
      <rect x="200" y="30" width="30" height="20" fill="#27ae60" />
      <text x="250" y="85" font-size="9">Parallel</text>
    </svg>
  </div>
</div>

### Scaled dot-product attention

Each token is projected into three vectors: a **query** $Q$, a **key** $K$, and a **value** $V$. A token's query is compared against every key by dot product to produce attention weights, which are then used to take a weighted average of the values:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

The scaling factor $\sqrt{d_k}$ (the key dimension) keeps the dot products from growing large enough to push the softmax into saturated, low-gradient regions. Intuitively: each token *asks a question* (query), every token *advertises what it offers* (key), the match scores become a soft lookup table, and the token retrieves a blend of *contents* (values).

### Multi-head attention

Rather than one attention function, the Transformer runs $h$ of them in parallel — each with its own learned projections — so different heads can specialize (syntax, coreference, position). The heads are concatenated and projected back:

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

### The encoder block, residuals, and normalization

A Transformer layer wraps multi-head self-attention and a position-wise feed-forward MLP, each inside a residual connection with layer normalization:

$$\mathbf{z} = \text{LayerNorm}\!\left(\mathbf{x} + \text{MultiHead}(\mathbf{x}, \mathbf{x}, \mathbf{x})\right)$$

$$\mathbf{y} = \text{LayerNorm}\!\left(\mathbf{z} + \text{FFN}(\mathbf{z})\right), \qquad \text{FFN}(\mathbf{z}) = W_2\,\phi(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2$$

The residual shortcut (inherited from ResNet) and normalization are what make deep Transformer stacks trainable. Modern variants typically use **pre-normalization** (LayerNorm applied to the input of each sublayer rather than the output) for more stable training of very deep models.

### Positional encoding

Because attention is permutation-invariant — it has no built-in notion of order — position must be injected explicitly. The original Transformer added fixed sinusoidal encodings:

$$\text{PE}_{(\text{pos},\,2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \qquad \text{PE}_{(\text{pos},\,2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

Contemporary models more often use **learned**, **rotary (RoPE)**, or **ALiBi** position encodings, the latter two generalizing better to sequence lengths unseen during training.

### Encoder, decoder, and causal masking

- **Encoder-only** (BERT): bidirectional self-attention; ideal for understanding tasks.
- **Decoder-only** (GPT): a **causal mask** sets attention weights to $-\infty$ for future positions before the softmax, so each token attends only to earlier tokens — the right setup for autoregressive generation.
- **Encoder-decoder** (T5, original Transformer): the encoder reads the source, the decoder generates the target while attending to encoder outputs via **cross-attention**.

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/ai/transformer_architectures.py">transformer_architectures.py</a>
</div>

*Landmark models: BERT, GPT, T5.*

## Vision Transformers (ViT)

The Transformer's success in language raised an obvious question: could the same attention mechanism work on images? The **Vision Transformer** answered yes by treating an image as a sequence of patches.

**Vision Transformer adapts the Transformer encoder for image classification:**

- **Patch embedding**: the image is split into fixed-size patches (e.g. 16×16), each flattened and linearly projected into a token embedding. A 224×224 image yields 196 patch tokens.
- **Position embeddings**: added to patch embeddings to retain spatial layout.
- **Class token**: a special learnable `[CLS]` token is prepended; its final representation is fed to the classification head.
- **Transformer encoder**: standard multi-head self-attention over all patch tokens, so every patch can attend to every other from layer one.

For an image of side $H$ and patch side $P$, the number of patch tokens is:

$$N = \frac{H \cdot W}{P^2}$$

**Key trade-offs versus CNNs:**

- ViTs have **fewer inductive biases** than CNNs (no built-in locality or translation equivariance), so they need *more data* to reach the same accuracy. With small datasets, CNNs win; with very large pre-training sets (ImageNet-21k, JFT-300M, LAION), ViTs scale better and surpass them.
- Self-attention gives a **global receptive field** at every layer, whereas a CNN's receptive field grows only with depth.
- Hybrid and follow-up models — **Swin Transformer** (windowed, hierarchical attention), **DeiT** (data-efficient training via distillation), **DINOv2**, and **EVA** — close the data-efficiency gap and add the multi-scale structure CNNs had by design.

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/ai/transformer_architectures.py#L70">transformer_architectures.py#VisionTransformer</a>
</div>

```python
# Example usage:
from transformer_architectures import VisionTransformer

# Create ViT-Base model
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=1000
)

# Forward pass
output = model(images)  # [batch_size, num_classes]
```

## CLIP and Multimodal Models

What if a model could understand the *relationship* between images and text rather than each in isolation? **CLIP** (Contrastive Language-Image Pre-training, Radford et al., 2021) pioneered this by training two encoders — one for images, one for text — to agree in a shared embedding space.

### Contrastive learning

CLIP trains on hundreds of millions of (image, caption) pairs scraped from the web. Within a batch of $N$ pairs there are $N$ correct matches and $N^2 - N$ incorrect ones. The model learns to pull matched image-text embeddings together and push mismatched ones apart. With normalized embeddings, the similarity of image $i$ and text $j$ is the cosine $\mathbf{s}_{ij} = \mathbf{u}_i^{\top}\mathbf{v}_j$, scaled by a learnable temperature $\tau$. The symmetric InfoNCE loss is the average of an image-to-text and a text-to-image cross-entropy:

$$\mathcal{L}_{i \to t} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\mathbf{s}_{ii}/\tau)}{\sum_{j=1}^{N}\exp(\mathbf{s}_{ij}/\tau)}$$

$$\mathcal{L} = \tfrac{1}{2}\left(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i}\right)$$

### Why it matters

- **Zero-shot classification**: to classify an image, embed candidate labels as text prompts ("a photo of a {label}") and pick the one whose embedding is closest. No task-specific training needed.
- **Natural-language supervision** is a far richer and more scalable signal than fixed category labels, and it makes the model robust to distribution shift.
- **Foundation for generation**: CLIP's text encoder conditions text-to-image diffusion models (Stable Diffusion, DALL·E 2), and the contrastive recipe generalizes to other modality pairs — audio-text (CLAP), video-text, and beyond.

More recent multimodal models fuse modalities *inside* a single Transformer: **Flamingo** and **LLaVA** interleave image features into an LLM's token stream, and natively multimodal models like **GPT-4V** and **Gemini** are trained end-to-end across text, images, and more. See the [Generative Models](generative-models.html) page for how CLIP embeddings feed image generation.

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/ai/transformer_architectures.py#L190">transformer_architectures.py#CLIP</a>
</div>

```python
# Example usage:
from transformer_architectures import CLIP, VisionTransformer

# Create CLIP model
vision_encoder = VisionTransformer(num_classes=None)  # No classification head
text_encoder = TextTransformer()                      # Your text encoder
clip_model = CLIP(vision_encoder, text_encoder, embed_dim=512)

# Zero-shot classification
image_features = clip_model.encode_image(images)
text_features = clip_model.encode_text(text_prompts)
similarities = image_features @ text_features.T
```

## Beyond Transformers: Efficient Sequence Models

The Transformer's power comes at a price: self-attention costs $O(n^2)$ time and memory in the sequence length $n$, because every token attends to every other. For long documents, high-resolution images, audio, or genomics this becomes prohibitive. Two broad research directions attack this: **subquadratic sequence models** (state-space models) that replace attention, and **efficient-attention variants** that keep attention but make it cheaper.

### The quadratic bottleneck

The attention matrix $QK^{\top}$ has $n^2$ entries. Doubling the context quadruples the compute and the memory for that matrix. The goal of everything in this section is to get the modeling power of attention — global context, content-dependent mixing — at $O(n \log n)$ or $O(n)$ cost.

### State-space models and Mamba

A linear **state-space model (SSM)** describes a sequence with a continuous-time linear system, mapping input $u(t)$ to output $y(t)$ through a hidden state $\mathbf{x}(t)$:

$$\mathbf{x}'(t) = A\,\mathbf{x}(t) + B\,u(t), \qquad y(t) = C\,\mathbf{x}(t) + D\,u(t)$$

Discretizing with step size $\Delta$ yields a recurrence with matrices $\bar{A}, \bar{B}$:

$$\mathbf{x}_t = \bar{A}\,\mathbf{x}_{t-1} + \bar{B}\,u_t, \qquad y_t = C\,\mathbf{x}_t$$

The magic is **dual form**: this recurrence is also a *convolution* of the input with a fixed kernel, so the same model can be run two ways. During **training** it is computed as a long convolution — fully parallel across the sequence, like a Transformer. During **inference** it is run as a recurrence — $O(1)$ state per step and $O(n)$ total, like an RNN, with no growing key-value cache. **S4** showed that with a carefully structured (HiPPO-initialized) $A$ matrix, this captures very long-range dependencies.

The catch with classic SSMs is that $A, B, C$ are *fixed* — the model cannot decide what to remember based on content, unlike attention. **Mamba** (Gu & Dao, 2023) fixes this with a **selective** SSM: the parameters $B$, $C$, and $\Delta$ become *functions of the input*, so the model can selectively propagate or forget information per token. This input-dependence breaks the simple convolution trick, so Mamba uses a hardware-aware parallel scan instead. The result is linear $O(n)$ scaling with Transformer-quality modeling on language, audio, and genomics, and far faster inference on long sequences.

### Other linear-recurrence models

A family of related architectures revisits the RNN with modern, parallelizable formulations: **RWKV** (a linear-attention RNN that trains like a Transformer), **RetNet** (a retention mechanism with parallel, recurrent, and chunked forms), and **Hyena** (long implicit convolutions as an attention substitute). All share the goal of linear-time inference with strong long-context modeling.

### Efficient attention variants

These keep attention but reduce its cost or its memory traffic:

- **Sparse attention** (Sparse Transformer, Longformer, BigBird) restricts each token to attend to a structured subset — local windows plus a few global tokens — turning the $n^2$ pattern into roughly $O(n\sqrt{n})$ or $O(n)$. BigBird proves that local + global + random attention is still a universal sequence approximator.
- **Linear attention** removes the softmax and uses a kernel feature map $\phi$ so that attention can be re-associated. Computing $\phi(Q)\big(\phi(K)^{\top}V\big)$ instead of $\big(\phi(Q)\phi(K)^{\top}\big)V$ changes the cost from $O(n^2 d)$ to $O(n d^2)$ — linear in sequence length (Linear Transformers, Performer, Linformer).

$$\text{Attn}(Q,K,V)_i = \frac{\phi(\mathbf{q}_i)^{\top}\sum_{j}\phi(\mathbf{k}_j)\,\mathbf{v}_j^{\top}}{\phi(\mathbf{q}_i)^{\top}\sum_{j}\phi(\mathbf{k}_j)}$$

- **FlashAttention** is an *exact* attention algorithm — same math, same outputs — that is simply far faster and more memory-efficient. It computes attention in tiles that fit in fast on-chip SRAM, fusing the softmax and avoiding ever materializing the full $n \times n$ matrix in slow high-bandwidth memory. This makes attention **memory-I/O-bound rather than compute-bound** and reduces memory from $O(n^2)$ to $O(n)$, which is what makes today's long-context (100k+ token) Transformers practical. FlashAttention and SSMs are complementary: many production models now interleave attention and Mamba-style layers to get the best of both.

### Choosing an architecture

| Need | Reach for |
|------|-----------|
| Images, strong locality prior, modest data | CNN (ResNet, ConvNeXt) |
| Images at scale, global context | Vision Transformer / Swin |
| Short-to-medium sequences, top quality | Transformer (with FlashAttention) |
| Very long sequences, fast inference | Mamba / SSM, or sparse / linear attention |
| Image-text understanding, zero-shot | CLIP and multimodal Transformers |
| Streaming / online sequence with tiny state | GRU / LSTM, or a recurrent SSM |

## See Also

- [Neural Network Architectures](architectures.html) — learning theory, optimization, and the NTK underpinning these models
- [Generative Models](generative-models.html) — diffusion, GANs, VAEs, and LLM generation built on these architectures
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling laws and interpretability of large models
- [AI Deep Dive (Lecture)](../ai-lecture-2023.html) — Transformers and LLM internals in depth
- [AI Mathematics](../../advanced/ai-mathematics/) — formal proofs for the theory above
