---
layout: docs
title: "AI: Fine-Tuning & Transfer Learning"
permalink: /docs/technology/ai/fine-tuning.html
toc: true
toc_sticky: true
---

[AI & Machine Learning](./) › Fine-Tuning & Transfer Learning

Pretraining a large model from scratch costs millions of dollars and consumes enormous datasets, but the representations it learns are broadly useful. **Transfer learning** reuses those representations: instead of starting from random weights, we start from a model that already "knows" the structure of language or vision and adapt it to a specific task. This page covers the full spectrum of adaptation strategies — from updating every parameter, to surgically inserting a few million trainable weights, to aligning a model's behavior with human preferences.

## Transfer Learning: Why It Works

A deep network trained on a large, diverse corpus learns a hierarchy of features. Early layers capture generic structure (edges and textures in vision; syntax and morphology in language), while later layers capture task-specific abstractions. Transfer learning exploits the observation that the generic lower layers transfer across tasks, so only the task-specific parts need to change.

Formally, pretraining solves

$$\theta^{\ast} = \arg\min_{\theta}\ \mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{pre}}}\big[\mathcal{L}_{\text{pre}}(f_\theta(x), y)\big],$$

over a large source distribution $\mathcal{D}_{\text{pre}}$ (e.g., next-token prediction over web text). Fine-tuning then continues optimization on a much smaller target distribution $\mathcal{D}_{\text{task}}$, initialized from $\theta^{\ast}$ rather than from scratch:

$$\theta_{\text{task}} = \arg\min_{\theta}\ \mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{task}}}\big[\mathcal{L}_{\text{task}}(f_\theta(x), y)\big], \qquad \theta \;\text{initialized at}\; \theta^{\ast}.$$

Because $\theta^{\ast}$ already sits in a good region of the loss landscape, $\mathcal{D}_{\text{task}}$ can be orders of magnitude smaller than $\mathcal{D}_{\text{pre}}$ — a few thousand labeled examples often suffice where training from scratch would need millions.

**Three regimes of transfer:**

| Regime | What changes | When to use |
|--------|--------------|-------------|
| **Feature extraction** | Backbone frozen; only a new head is trained | Very little data; task close to pretraining objective |
| **Fine-tuning** | Some or all backbone weights updated | Moderate data; task somewhat different from pretraining |
| **Domain adaptation** | Continued pretraining on in-domain unlabeled text, then fine-tuning | Large domain shift (e.g., legal, biomedical) |

## Full vs. Frozen Fine-Tuning

The first design choice is *how much* of the model to update.

### Full Fine-Tuning

Every parameter is trainable. This is the most expressive option and usually gives the best task accuracy when data is plentiful, but it is also the most expensive:

- **Memory.** Training requires storing the weights, their gradients, and optimizer state. With the Adam optimizer, each parameter needs roughly its own value plus a gradient plus two moment estimates. In mixed precision the optimizer state is typically kept in 32-bit, so a model with $N$ parameters needs on the order of $16N$ bytes of optimizer/gradient/master-weight state during training — far more than the model itself.
- **Storage.** Each fine-tuned variant is a full copy of the model. Serving ten tasks means ten full checkpoints.
- **Overfitting and forgetting.** With a small dataset, updating all weights can erase pretrained knowledge (see [Catastrophic Forgetting](#catastrophic-forgetting)).

### Frozen / Partial Fine-Tuning

Here most weights are held fixed (`requires_grad = False`) and only a subset is trained — commonly just the final classifier head, or the top few transformer blocks plus the head. Freezing reduces memory (no gradients or optimizer state for frozen weights) and acts as a strong regularizer.

A common middle ground is **gradual unfreezing**: start by training only the head, then progressively unfreeze deeper layers, often with **discriminative learning rates** — smaller rates for earlier (more general) layers and larger rates for later (more specialized) layers.

<div class="code-reference">
<i class="fas fa-code"></i> Freezing a backbone and training a new head (PyTorch):
</div>

```python
import torch.nn as nn

# Freeze the pretrained backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Replace and train a fresh task head
model.head = nn.Linear(model.config.hidden_size, num_classes)

# Only the head's parameters carry gradients / optimizer state
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=1e-3
)
```

## Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of a multi-billion-parameter model is impractical on commodity hardware, and storing a full copy per task does not scale. **Parameter-efficient fine-tuning** updates only a small number of parameters (often <1% of the total) while keeping the pretrained weights frozen. The result is a tiny "delta" per task — typically a few megabytes — that can be swapped in and out cheaply.

### LoRA: Low-Rank Adaptation

The key insight behind **LoRA** is that the *update* needed to adapt a pretrained weight matrix tends to have low intrinsic rank. Rather than learning a full update matrix $\Delta W \in \mathbb{R}^{d \times k}$, LoRA factors it into two small matrices:

$$\Delta W = B A, \qquad B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d, k).$$

The adapted layer computes

$$h = W_0 x + \Delta W x = W_0 x + \tfrac{\alpha}{r}\, B A x,$$

where $W_0$ is the frozen pretrained weight, $\alpha$ is a scaling factor, and only $A$ and $B$ are trained. $A$ is initialized from a small random distribution and $B$ is initialized to zero, so at the start $\Delta W = 0$ and the adapted model exactly reproduces the base model.

The parameter savings are dramatic. A $4096 \times 4096$ projection has about 16.8M parameters; with rank $r = 8$ the LoRA delta has $2 \times 4096 \times 8 = 65{,}536$ parameters — a 256× reduction for that layer. Because $W_0$ is frozen, no gradients or optimizer state are needed for it, slashing training memory.

**Practical notes:**

- LoRA is usually applied to the attention projection matrices (query/key/value/output), and often the MLP layers too. The choice of *which* matrices to adapt matters more than raw rank.
- At inference, $B A$ can be merged into $W_0$ once ($W = W_0 + \tfrac{\alpha}{r} B A$), adding **zero** latency. Unmerged, multiple LoRA adapters can be hot-swapped on a shared base model.
- Typical ranks are 4–64; $\alpha$ is commonly set to a small multiple of $r$.

<div class="code-reference">
<i class="fas fa-code"></i> Applying LoRA with the Hugging Face PEFT library:
</div>

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                    # rank of the low-rank update
    lora_alpha=16,          # scaling factor (alpha / r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# e.g. "trainable params: 4.2M || all params: 6.7B || trainable%: 0.06"
```

### QLoRA: Quantized LoRA

**QLoRA** pushes memory efficiency further by *quantizing the frozen base model to 4-bit* while still training LoRA adapters in higher precision. This lets a model that would need ~140 GB in 16-bit fit on a single consumer or workstation GPU. Its main ingredients are:

- **4-bit NormalFloat (NF4)** — a quantization data type that is information-theoretically near-optimal for the normally-distributed weights of a neural network.
- **Double quantization** — quantizing the quantization constants themselves to save additional memory.
- **Paged optimizers** — using unified CPU/GPU memory to avoid out-of-memory spikes during gradient checkpointing.

Crucially, gradients still flow through the frozen 4-bit weights into the (16-bit) LoRA matrices, so accuracy stays close to full 16-bit fine-tuning while memory drops by roughly 3–4×.

### Adapters

**Adapter modules** insert small bottleneck feed-forward blocks between the layers of a frozen transformer. Each adapter projects the hidden state down to a small dimension $m$, applies a nonlinearity, projects back up, and adds a residual connection:

$$h \leftarrow h + W_{\text{up}}\,\sigma\!\big(W_{\text{down}}\, h\big),\qquad W_{\text{down}} \in \mathbb{R}^{m \times d},\ W_{\text{up}} \in \mathbb{R}^{d \times m},\ m \ll d.$$

Only the adapter weights are trained. Adapters were the first widely-used PEFT method; their downside relative to LoRA is that the extra modules add a small but nonzero inference cost (they cannot be merged away). Variants like **AdapterFusion** combine multiple task adapters, and **(IA)³** rescales activations with learned vectors for an even smaller footprint.

### Prefix Tuning and Prompt Tuning

Instead of modifying weights, these methods prepend trainable vectors to the input or to each layer's key/value cache, steering a frozen model through its own attention mechanism.

- **Prompt tuning** prepends a small number of trainable "soft prompt" embeddings to the input sequence. The model's weights are entirely frozen; only these continuous prompt vectors are learned. It is the most parameter-thrifty method (often a few thousand to a few hundred thousand parameters) and becomes competitive with full fine-tuning as the base model grows very large.
- **Prefix tuning** is similar but prepends trainable key/value vectors at *every* transformer layer rather than only at the input embedding. The extra depth of intervention makes it more expressive than prompt tuning at the cost of slightly more parameters.
- **P-tuning v2** generalizes prefix tuning to make it robust across model scales and tasks.

These approaches treat the frozen model as a fixed function and search only over a compact continuous "instruction" that conditions it — conceptually a learned, differentiable counterpart to hand-written prompts.

### Choosing a PEFT Method

| Method | Trainable params | Inference overhead | Notes |
|--------|------------------|--------------------|-------|
| **Full fine-tuning** | 100% | None | Best accuracy with abundant data; expensive to train and store |
| **LoRA** | ~0.01–1% | None (mergeable) | The default for most LLM adaptation today |
| **QLoRA** | ~0.01–1% | None (mergeable) | LoRA on a 4-bit base; fits huge models on one GPU |
| **Adapters** | ~0.5–4% | Small | Modular, fusable; slight added latency |
| **Prefix tuning** | ~0.1–1% | Small (longer context) | Per-layer soft prefixes |
| **Prompt tuning** | <0.1% | Small (longer input) | Simplest; shines at very large scale |

## Instruction Tuning and Preference Alignment

A base language model trained only on next-token prediction is a powerful *completion* engine, but it does not natively follow instructions or behave helpfully and safely. Turning a base model into a useful assistant typically proceeds in stages.

### Supervised Fine-Tuning (Instruction Tuning)

**Instruction tuning** is supervised fine-tuning on a dataset of (instruction, desired response) pairs spanning many tasks. The model learns the *format* of following instructions — answering questions, summarizing, writing code, refusing unsafe requests — by imitating high-quality demonstrations. The training objective is ordinary next-token cross-entropy, but typically masked so the loss is computed only on the response tokens, not the prompt:

$$\mathcal{L}_{\text{SFT}} = -\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\sum_{t} \log \pi_\theta\big(y_t \mid x,\ y_{<t}\big).$$

Instruction tuning on a *diverse mixture* of tasks (the approach behind models like FLAN and InstructGPT) dramatically improves **zero-shot** generalization: the model follows instructions for tasks it never saw during tuning.

### RLHF: Reinforcement Learning from Human Feedback

Supervised demonstrations capture *one* good answer, but for open-ended tasks there are many acceptable responses and it is easier for humans to *compare* outputs than to write ideal ones. **RLHF** turns human comparisons into a training signal in three steps:

1. **Supervised fine-tuning (SFT).** Start from an instruction-tuned policy $\pi_{\text{SFT}}$.
2. **Reward modeling.** Collect human preference data: for a prompt $x$, annotators rank two responses, marking the preferred one $y_w$ over the rejected one $y_l$. Train a **reward model** $r_\phi$ under the Bradley–Terry model so that preferred responses score higher:

$$\mathcal{L}_{\text{RM}} = -\,\mathbb{E}_{(x,\,y_w,\,y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big].$$

3. **Policy optimization.** Optimize the policy $\pi_\theta$ to maximize the reward model's score, with a KL penalty that keeps it from drifting too far from the SFT reference $\pi_{\text{ref}}$ (preventing reward hacking and degenerate text):

$$\max_{\pi_\theta}\ \mathbb{E}_{x,\ y\sim\pi_\theta}\Big[r_\phi(x, y)\Big] \;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).$$

This last step is classically solved with the **PPO** (Proximal Policy Optimization) algorithm. RLHF is what made models like InstructGPT and ChatGPT markedly more helpful and aligned than their base models, but the pipeline is complex: it requires training and serving a separate reward model and running an unstable on-policy RL loop.

### DPO: Direct Preference Optimization

**Direct Preference Optimization** observes that the constrained reward-maximization objective above has a closed-form optimal policy, which can be rearranged to express the reward *implicitly* in terms of the policy itself. Substituting that into the Bradley–Terry loss eliminates the explicit reward model and the RL loop entirely. DPO trains directly on preference pairs with a simple classification-style loss:

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,\,y_w,\,y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

Intuitively, DPO increases the policy's likelihood of preferred responses and decreases it for rejected ones, *relative to* the frozen reference model, with $\beta$ controlling how aggressively it deviates. Because it is a single supervised-style objective with no reward model and no sampling loop, DPO is far simpler and more stable to train than PPO-based RLHF while often matching its quality. Related variants include **IPO** (which addresses an overfitting pathology in DPO), **KTO** (which learns from non-paired thumbs-up/thumbs-down signals), and **ORPO** (which folds preference optimization into the SFT stage).

| Method | Needs reward model? | Needs RL loop? | Relative complexity |
|--------|---------------------|----------------|---------------------|
| **SFT / instruction tuning** | No | No | Low |
| **RLHF (PPO)** | Yes | Yes | High |
| **DPO** | No | No | Low–moderate |

For how alignment fits into the broader safety landscape, see [Frontier Research & Ethics](frontier-and-ethics.html).

## Catastrophic Forgetting

When a model is fine-tuned on a narrow new task, gradient updates can overwrite the weights that encoded its general capabilities — the model becomes good at the new task but loses competence on everything it knew before. This is **catastrophic forgetting** (also called catastrophic interference), and it is the central tension of all fine-tuning: adapt enough to learn the new task, but not so much that you erase the old knowledge.

**Why it happens.** Neural networks store knowledge in shared, distributed weights. Because the new task's loss says nothing about preserving old behavior, unconstrained gradient descent freely moves weights into regions that minimize new-task loss while raising old-task loss.

**Mitigations:**

- **Lower learning rates and fewer epochs.** Small steps near $\theta^{\ast}$ stay in the good region. Over-training on a small dataset is the most common cause of forgetting.
- **Parameter-efficient methods.** LoRA, adapters, and prompt tuning keep $\theta^{\ast}$ frozen by construction, so the base model's knowledge is *physically* preserved — only the small delta changes.
- **Regularization toward the pretrained weights.** Add a penalty $\lambda\,\lVert\theta - \theta^{\ast}\rVert^2$, or weight that penalty per-parameter by how important each weight was to old tasks (the idea behind **Elastic Weight Consolidation**, using a Fisher-information estimate of importance).
- **Rehearsal / data mixing.** Mix a fraction of the original pretraining or instruction data back into the fine-tuning set so the old distribution is still represented in the gradient.
- **KL regularization.** The $\beta\,D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})$ term in RLHF and DPO is partly a forgetting safeguard — it explicitly anchors the new policy to the reference model's behavior.

## Data and Evaluation

The quality of a fine-tuned model is bounded by the quality of its data and the rigor of its evaluation.

### Data

- **Quality over quantity.** For instruction tuning, a few thousand carefully curated, diverse, high-quality examples often beat hundreds of thousands of noisy ones. Demonstrations should reflect exactly the format and behavior you want.
- **Diversity and coverage.** The data must span the range of inputs the model will see in production, including edge cases and the unsafe requests you want it to refuse.
- **Loss masking.** For instruction data, compute the loss only over response tokens so the model learns to *produce* answers rather than to *predict* prompts.
- **Contamination control.** Ensure your evaluation sets are not present in the fine-tuning data, or reported gains will be illusory.
- **Splits.** Hold out a validation set drawn from the same distribution as the target task to tune hyperparameters and detect overfitting early.

### Evaluation

No single number captures a fine-tuned model. Combine:

- **Task metrics.** Accuracy / F1 for classification, exact-match or pass@k for code, ROUGE/BLEU as rough proxies for summarization and translation (with the caveat that they correlate poorly with human judgment).
- **Held-out benchmarks.** Standardized suites for knowledge and reasoning to verify that adaptation did not degrade general capability — a direct check against catastrophic forgetting.
- **Preference / pairwise evaluation.** For open-ended generation, have humans (or a strong **LLM-as-a-judge**) compare outputs head-to-head and report win rates against a baseline. This mirrors how preference data is collected for RLHF/DPO.
- **Calibration and safety.** Measure refusal behavior, hallucination rate, and whether confidence tracks correctness.

A disciplined loop — curate data, fine-tune with a method matched to your compute and forgetting constraints, evaluate on both the target task *and* held-out general benchmarks, and iterate — is what separates a model that genuinely improved from one that merely memorized its training set.

---

## Key Takeaways

- **Transfer learning reuses pretrained representations**, so a new task needs orders of magnitude less data than training from scratch.
- **Full fine-tuning** is the most expressive but the most expensive in memory and storage; **freezing** most layers regularizes and saves resources.
- **PEFT methods** (LoRA, QLoRA, adapters, prefix/prompt tuning) train <1% of parameters; LoRA's low-rank update $\Delta W = \tfrac{\alpha}{r}BA$ merges into the base model with zero inference cost.
- **Alignment is staged**: instruction tuning teaches format via SFT; **RLHF** optimizes a reward model with PPO under a KL leash; **DPO** reaches similar quality directly from preference pairs without a reward model or RL loop.
- **Catastrophic forgetting** is the core risk — mitigate with small learning rates, PEFT, weight regularization, rehearsal, and KL anchoring.
- **Data quality and honest, multi-faceted evaluation** (task metrics + held-out benchmarks + pairwise preference) determine whether fine-tuning actually helped.

---

## See Also

- [Neural Network Architectures](architectures.html) — the transformers and optimization these methods build on
- [Generative Models](generative-models.html) — autoregressive LLM generation and decoding strategies
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling laws, alignment, and AI safety
- [AI/ML Documentation Hub](../../ai-ml/) — hands-on LoRA training and ComfyUI guides
- [AI Mathematics](../../advanced/ai-mathematics/) — the optimization and learning theory underneath
- [AI Documentation Hub](../../artificial-intelligence/index.html) — complete index of AI resources
