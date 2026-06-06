---
layout: docs
title: "AI: Reinforcement Learning"
permalink: /docs/technology/ai/reinforcement-learning.html
toc: true
toc_sticky: true
---

[AI & Machine Learning](./) › Reinforcement Learning

Supervised learning maps inputs to labels; reinforcement learning (RL) learns **behavior**. An *agent* interacts with an *environment*, observes states, takes actions, and receives scalar *rewards*. There is no labeled "correct action" — the only feedback is reward, which may be sparse, delayed, and noisy. The agent's job is to discover a policy that maximizes cumulative reward over time. This page builds the formalism (Markov decision processes), the classical solution methods (value and policy iteration), the model-free workhorses (Q-learning, policy gradients, actor-critic), the deep-learning era (DQN, A2C, PPO), and the applications that made RL famous, from Atari and Go to robotics and the reinforcement-learning-from-human-feedback (RLHF) pipeline that aligns modern language models.

## The Reinforcement Learning Problem

The defining feature of RL is the **interaction loop**. At each time step $t$ the agent is in state $s_t$, selects an action $a_t$, and the environment responds with a new state $s_{t+1}$ and a reward $r_{t+1}$. The agent never sees the environment's internal mechanics directly; it learns purely from this stream of experience.

```
        ┌─────────────────────────────────────┐
        │            ENVIRONMENT              │
        └─────────────────────────────────────┘
            ▲   state s_t, reward r_t   │
   action  │                           │  next state s_{t+1},
   a_t     │                           ▼  reward r_{t+1}
        ┌─────────────────────────────────────┐
        │               AGENT                 │
        │   policy π(a | s)  →  chooses a_t   │
        └─────────────────────────────────────┘
```

This loop introduces challenges absent from supervised learning:

- **Delayed reward (credit assignment).** A move early in a chess game may only pay off twenty moves later. The agent must attribute distant reward to the actions that caused it.
- **Exploration vs. exploitation.** To find the best policy the agent must try actions whose value it does not yet know (explore), but it also wants to use what it already knows to earn reward (exploit). Over-exploiting locks in a mediocre strategy; over-exploring wastes reward.
- **Non-stationarity from the agent's own learning.** As the policy changes, the distribution of states the agent visits changes too, so the "dataset" is a moving target.

## Markov Decision Processes

The standard mathematical model of the RL problem is the **Markov decision process (MDP)**, a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$ — the set of states.
- $\mathcal{A}$ — the set of actions.
- $P(s' \mid s, a)$ — the transition probability of reaching $s'$ after taking action $a$ in state $s$.
- $R(s, a)$ — the expected immediate reward.
- $\gamma \in [0, 1]$ — the discount factor.

The defining **Markov property** is that the future depends on the present state alone, not the full history:

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t).$$

This is what makes the state a sufficient statistic and makes the problem tractable. When the agent cannot observe the full state (only a noisy or partial observation), the problem becomes a **partially observable MDP (POMDP)**, usually handled by maintaining a belief state or by feeding a history of observations into a recurrent network.

### Returns and the Discount Factor

The agent maximizes the **return** $G_t$, the discounted sum of future rewards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1} = r_{t+1} + \gamma \, r_{t+2} + \gamma^2 r_{t+3} + \cdots$$

The discount $\gamma$ does double duty. Mathematically, $\gamma < 1$ keeps the infinite sum finite (bounded rewards give a return bounded by $r_{\max}/(1-\gamma)$). Conceptually, it expresses a preference for sooner rewards: $\gamma \to 0$ makes the agent myopic, caring only about immediate reward, while $\gamma \to 1$ makes it far-sighted, weighting distant rewards almost as heavily as near ones. Typical values lie between 0.9 and 0.999.

### Policies and Value Functions

A **policy** $\pi(a \mid s)$ is a (possibly stochastic) mapping from states to a distribution over actions — it *is* the agent's behavior. Everything RL does is, ultimately, search over policies.

To compare policies we use **value functions**, which measure expected return. The **state-value function** of a policy $\pi$ is

$$V^\pi(s) = \mathbb{E}_\pi\!\left[ G_t \mid s_t = s \right],$$

the expected return starting from state $s$ and following $\pi$ thereafter. The **action-value function** (the "Q-function") adds a committed first action:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[ G_t \mid s_t = s, \, a_t = a \right].$$

$Q^\pi$ is what you act on: if you know $Q^\pi(s, \cdot)$, picking the highest-valued action is straightforward.

### The Bellman Equations

Value functions obey a recursive consistency relation — the **Bellman equation** — which is the engine of nearly every RL algorithm. Splitting the return into the immediate reward plus the discounted return from the next state gives

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \, V^\pi(s') \right].$$

The corresponding equation for $Q^\pi$ is

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a') \right].$$

There is always at least one **optimal policy** $\pi^*$ that achieves the maximum value in every state. Its value functions $V^*$ and $Q^*$ satisfy the **Bellman optimality equations**, which replace the average-over-actions with a max:

$$V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \, V^*(s') \right],$$

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \max_{a'} Q^*(s', a') \right].$$

Once $Q^*$ is known, the optimal policy is **greedy**: $\pi^*(s) = \arg\max_a Q^*(s, a)$. The entire field can be read as different ways of approximating these equations.

## Dynamic Programming: Value and Policy Iteration

When the MDP is fully known — transitions $P$ and rewards $R$ are available — the Bellman optimality equations can be solved by **dynamic programming (DP)**. DP is rarely usable at scale (it needs the model and sweeps the whole state space), but it is the conceptual ancestor of every model-free method.

### Policy Evaluation

Given a fixed policy $\pi$, **policy evaluation** computes $V^\pi$ by turning the Bellman equation into an iterative update and applying it until convergence:

$$V_{k+1}(s) \leftarrow \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \, V_k(s') \right].$$

Because the Bellman operator is a $\gamma$-contraction (it shrinks the distance between any two value estimates by a factor $\gamma$), this iteration converges to the unique fixed point $V^\pi$.

### Policy Iteration

**Policy iteration** alternates two steps until the policy stops changing:

1. **Policy evaluation** — compute $V^\pi$ for the current policy.
2. **Policy improvement** — make the policy greedy with respect to that value:

$$\pi'(s) = \arg\max_{a} \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \, V^\pi(s') \right].$$

The **policy improvement theorem** guarantees $V^{\pi'} \ge V^\pi$ everywhere, so each round strictly improves (or leaves unchanged) the policy. With finitely many deterministic policies, it converges to $\pi^*$ in a finite number of iterations.

### Value Iteration

**Value iteration** fuses evaluation and improvement into a single update by applying the Bellman *optimality* operator directly, never waiting for full evaluation to converge:

$$V_{k+1}(s) \leftarrow \max_{a} \sum_{s'} P(s' \mid s, a)\left[ R(s, a) + \gamma \, V_k(s') \right].$$

It converges to $V^*$, and the greedy policy is read off at the end. Value iteration is usually faster in wall-clock terms because each sweep is cheap.

```python
import numpy as np

def value_iteration(P, R, gamma=0.99, theta=1e-8):
    """
    P[s, a, s'] = transition probability, R[s, a] = expected reward.
    Returns the optimal value function and greedy policy.
    """
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            # Q(s, a) for every action, then take the best.
            q = R[s] + gamma * P[s].dot(V)   # shape (n_actions,)
            best = q.max()
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    policy = np.argmax(R + gamma * np.einsum("sap,p->sa", P, V), axis=1)
    return V, policy
```

The catch: DP requires the model and visits every state, so it is confined to small, fully known MDPs. The rest of RL is about lifting both restrictions — learning from sampled experience without a model, and generalizing across states with function approximation.

## Model-Free Prediction and Control

When the model is unknown, the agent must learn from **sampled experience**. Two ideas dominate.

**Monte Carlo (MC)** methods estimate value by averaging actual returns observed over complete episodes. They are unbiased but high-variance and need episodes to terminate.

**Temporal-difference (TD)** learning is the central innovation of RL: it updates value estimates from *other* estimates (bootstrapping) without waiting for the episode to end. The simplest version, TD(0), nudges $V(s_t)$ toward the observed reward plus the discounted value of the next state:

$$V(s_t) \leftarrow V(s_t) + \alpha\left[ \underbrace{r_{t+1} + \gamma\, V(s_{t+1})}_{\text{TD target}} - V(s_t) \right].$$

The bracketed quantity is the **TD error** $\delta_t$, the surprise between prediction and bootstrapped target. TD learns online, from incomplete episodes, and trades a little bias (from bootstrapping) for much lower variance than MC. TD($\lambda$) and eligibility traces interpolate smoothly between TD(0) and MC.

### Q-Learning

**Q-learning** (Watkins, 1989) is the landmark *off-policy* TD control algorithm. It learns $Q^*$ directly, regardless of the policy actually generating the data, by bootstrapping on the *maximum* next-state value:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right].$$

Because the update uses $\max_{a'}$ rather than the action the behavior policy actually took, Q-learning converges to $Q^*$ even while exploring with, say, an $\varepsilon$-greedy policy — this is what "off-policy" means. The closely related **SARSA** algorithm is *on-policy*: it bootstraps on the action $a_{t+1}$ actually chosen, $r_{t+1} + \gamma\, Q(s_{t+1}, a_{t+1})$, learning the value of the policy it is following (typically yielding more conservative behavior near hazards).

```python
import numpy as np

def q_learning(env, n_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.n_states, env.n_actions))
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            # epsilon-greedy action selection (exploration)
            if np.random.rand() < epsilon:
                a = np.random.randint(env.n_actions)
            else:
                a = np.argmax(Q[s])
            s_next, r, done = env.step(a)
            # off-policy TD update: bootstrap on the BEST next action
            td_target = r + gamma * np.max(Q[s_next]) * (not done)
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
    return Q
```

Tabular Q-learning stores one entry per state-action pair, which is hopeless for large or continuous state spaces — a single Atari frame has more configurations than there are atoms in the universe. The fix is **function approximation**: replace the table with a parameterized $Q_\theta(s, a)$.

## Deep Q-Networks (DQN)

The **Deep Q-Network** (Mnih et al., DeepMind, 2015) was the breakthrough that launched deep RL: a single architecture learned to play 49 Atari games from raw pixels at human level. It approximates $Q^*(s, a)$ with a convolutional network $Q_\theta$ and trains it by minimizing the squared TD error toward a bootstrapped target:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s')}\!\left[\left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right].$$

Naively combining Q-learning with a neural network is famously unstable — the "deadly triad" of bootstrapping, off-policy learning, and function approximation can diverge. DQN introduced two stabilizers that became standard:

- **Experience replay.** Transitions $(s, a, r, s')$ are stored in a replay buffer and sampled in random mini-batches. This breaks the temporal correlation between consecutive samples (which violates the i.i.d. assumption of SGD) and reuses each experience many times, improving data efficiency.
- **Target network.** The bootstrap target uses a separate network $Q_{\theta^-}$ whose weights $\theta^-$ are frozen copies of $\theta$, synced only every few thousand steps. Holding the target fixed stops the network from "chasing its own tail," dramatically improving stability.

A family of refinements followed, often combined in the "Rainbow" agent:

- **Double DQN** decouples action selection from evaluation to cut the overestimation bias introduced by the $\max$ operator: the action is chosen by the online network but evaluated by the target network.
- **Dueling DQN** splits the network into a state-value stream $V(s)$ and an advantage stream $A(s, a)$, recombined as $Q(s, a) = V(s) + (A(s, a) - \text{mean}_a A(s, a))$, learning state values even when action choice barely matters.
- **Prioritized experience replay** samples surprising (high-TD-error) transitions more often.
- **Distributional RL (C51)** learns the full distribution of returns rather than just its mean.

DQN and its relatives are powerful for discrete action spaces, but the $\max_{a'}$ becomes intractable when actions are continuous (a robot joint torque). That motivates the policy-gradient family.

## Policy Gradient Methods

Value-based methods learn $Q$ and derive a policy implicitly. **Policy-gradient methods** instead parameterize the policy $\pi_\theta(a \mid s)$ directly — typically a neural network whose softmax (discrete) or Gaussian (continuous) output defines the action distribution — and optimize its parameters by gradient ascent on expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$. This handles continuous actions naturally and can represent stochastic policies, which are sometimes optimal (e.g., in partially observed or adversarial settings).

### The Policy Gradient Theorem and REINFORCE

The **policy gradient theorem** gives a remarkably clean expression for the gradient of return, free of the (unknown) transition dynamics:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t \right].$$

The intuition is "trial and error with a gradient": each action's log-probability is pushed up in proportion to the return that followed it. Good trajectories make their actions more likely; bad ones make theirs less likely. The Monte Carlo estimator of this gradient is the **REINFORCE** algorithm (Williams, 1992):

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t.$$

REINFORCE is unbiased but suffers from **high variance**, because $G_t$ is a noisy single-sample return. The standard variance-reduction trick subtracts a state-dependent **baseline** $b(s_t)$, which leaves the gradient unbiased (its expectation is zero) but shrinks its variance:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big) \right].$$

The natural choice of baseline is the state value $V(s_t)$, which leads directly to actor-critic.

## Actor-Critic Methods

**Actor-critic** methods combine the two families: an **actor** $\pi_\theta(a \mid s)$ proposes actions, and a **critic** $V_\phi(s)$ estimates value and supplies a low-variance learning signal. With $V_\phi$ as the baseline, the term $G_t - V(s_t)$ becomes an estimate of the **advantage**

$$A(s, a) = Q(s, a) - V(s),$$

which measures how much better action $a$ is than the policy's average behavior in state $s$. The one-step advantage estimate is just the TD error of the critic, $\hat{A}_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, and the actor update becomes

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t.$$

The critic is trained by the usual TD regression toward its bootstrapped target. **Generalized advantage estimation (GAE)** interpolates between high-bias one-step and high-variance Monte Carlo advantage estimates with a parameter $\lambda$, and is used in most modern implementations.

### A2C and A3C

**A3C** (Asynchronous Advantage Actor-Critic) ran many actor-learners in parallel, each on its own copy of the environment, asynchronously updating shared parameters; the diversity of experience decorrelated updates and removed the need for a replay buffer. **A2C** is the synchronous variant — gather a batch of transitions from parallel environments, then perform one combined update — which is simpler and often performs as well on modern hardware.

### Proximal Policy Optimization (PPO)

**PPO** (Schulman et al., OpenAI, 2017) is the most widely used policy-gradient algorithm today, prized for being robust, sample-efficient enough, and simple to tune. The problem it solves: a raw policy-gradient step can move the policy too far, collapsing performance. PPO restricts each update to a **trust region** around the current policy, but does so with a cheap clipped objective rather than the second-order constraint of its predecessor, TRPO.

Defining the probability ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$, PPO maximizes the clipped surrogate

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[ \min\!\Big( r_t(\theta)\,\hat{A}_t,\; \text{clip}\big(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon\big)\,\hat{A}_t \Big) \right].$$

The clip (typically $\epsilon = 0.2$) removes the incentive to push the ratio far from 1: once an advantageous action's probability has increased by more than $\epsilon$, the objective flattens, so a single batch cannot be exploited into a destructively large update. This lets PPO safely reuse each batch of data for several gradient epochs. Crucially for what follows, PPO is the optimizer at the heart of the RLHF pipeline.

```python
# PPO clipped objective (the core of an update step), PyTorch-style pseudocode
import torch

def ppo_loss(logp_new, logp_old, advantages, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)         # pi_new / pi_old
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    # negative because optimizers minimize
    return -torch.min(unclipped, clipped).mean()
```

For continuous control, **DDPG**, **TD3**, and **SAC** form a parallel off-policy actor-critic lineage. **Soft Actor-Critic (SAC)** in particular augments the reward with a policy-entropy bonus, maximizing $\mathbb{E}[\sum_t r_t + \alpha\,\mathcal{H}(\pi(\cdot \mid s_t))]$; the entropy term encourages exploration and makes SAC one of the most sample-efficient and stable algorithms for robotics.

## Exploration

Balancing exploration and exploitation is the problem unique to RL. A handful of strategies recur:

- **$\varepsilon$-greedy.** Act greedily with probability $1 - \varepsilon$, randomly otherwise. Simple and surprisingly effective; $\varepsilon$ is usually annealed from near 1 down to a small floor.
- **Boltzmann (softmax) exploration.** Sample actions in proportion to $\exp(Q(s, a)/\tau)$; the temperature $\tau$ tunes randomness and decays over training.
- **Optimism / upper confidence bounds (UCB).** Add an exploration bonus to actions tried fewer times — "optimism in the face of uncertainty" — which underlies the bandit-style guarantees and the exploration in tree search.
- **Entropy regularization.** Adding $\beta\,\mathcal{H}(\pi(\cdot \mid s))$ to the objective (as in A2C, PPO, and SAC) keeps the policy stochastic and discourages premature collapse to a deterministic, possibly suboptimal, strategy.
- **Intrinsic motivation / curiosity.** When extrinsic reward is sparse, the agent invents its own bonus for visiting novel states — via prediction error (the agent is rewarded for surprising itself), state-visitation counts, or random network distillation. This is what lets agents make progress in games like Montezuma's Revenge, where reward is almost never seen by a random policy.

## Model-Based Reinforcement Learning

Everything above is **model-free**: the agent learns values or a policy directly from experience without ever modeling the environment's dynamics. **Model-based RL** instead learns (or is given) a model $\hat{P}(s' \mid s, a)$ and $\hat{R}(s, a)$, then uses it to plan or to generate imagined experience. The payoff is **sample efficiency** — planning in a learned model can replace many costly real-environment interactions — at the cost of complexity and the risk that the agent exploits errors in an imperfect model.

- **Dyna** interleaves real experience with simulated experience drawn from a learned model, using both to update the same value function.
- **Model predictive control (MPC)** uses the model to plan a short lookahead at every step, executes the first action, then re-plans — standard in robotics and control.
- **Monte Carlo Tree Search (MCTS)** builds a search tree of futures by sampling, balancing exploration and exploitation with a UCB rule. It is the planning engine behind AlphaGo and AlphaZero.
- **World models** (e.g., the **Dreamer** family) learn a compact latent dynamics model and train the policy almost entirely "in imagination," rolling out trajectories inside the learned model. **MuZero** learns a model purely in terms of quantities relevant to planning — reward, value, and policy — without ever reconstructing the raw observation, and matched AlphaZero on Go, chess, and shogi while also mastering Atari, all without being told the rules.

## Applications

### Games

Games are RL's proving ground: clear rewards, unlimited self-play data, and superhuman benchmarks.

- **Atari (DQN, 2015)** — learning 49 games from pixels with one architecture was the spark of the deep-RL era.
- **AlphaGo (2016)** combined supervised learning from human games, self-play policy-gradient training, and MCTS to defeat world champion Lee Sedol — a milestone long thought decades away. **AlphaGo Zero** then surpassed it using self-play *only*, with no human games, and **AlphaZero** generalized the same algorithm to chess and shogi.
- **OpenAI Five** (Dota 2) and **AlphaStar** (StarCraft II) scaled self-play RL to long-horizon, partially observed, multi-agent strategy games.

### Robotics and Control

RL controls continuous, high-dimensional systems where hand-designed controllers struggle: dexterous manipulation, legged locomotion, autonomous driving, and data-center cooling. The central obstacle is the **sim-to-real gap** — policies trained in simulation (cheap, safe, massively parallel) must transfer to the messy real world. **Domain randomization** — randomizing physics, textures, lighting, and sensor noise in simulation so reality looks like just another variation — is the standard remedy and powered results such as OpenAI's robotic-hand Rubik's-cube manipulation and modern quadruped locomotion.

### Reinforcement Learning from Human Feedback (RLHF)

RLHF is how RL reaches everyday AI: it is the alignment stage that turns a raw, next-token-predicting language model into a helpful, harmless assistant. Because "helpfulness" has no programmable reward function, RLHF *learns* one from human preferences. The pipeline has three stages:

1. **Supervised fine-tuning (SFT).** Fine-tune the pretrained model on high-quality human demonstrations of the desired behavior.
2. **Reward modeling.** Collect human comparisons (given a prompt, which of two responses is better?) and train a **reward model** $r_\psi$ to predict human preference, typically under the Bradley–Terry model: for a preferred response $y_w$ over $y_l$,

$$\mathcal{L}(\psi) = -\,\mathbb{E}_{(x,\, y_w,\, y_l)}\Big[ \log \sigma\big( r_\psi(x, y_w) - r_\psi(x, y_l) \big) \Big].$$

3. **RL fine-tuning.** Treat the language model as a policy and optimize it with **PPO** against the learned reward model, with a KL-divergence penalty that keeps the policy close to the SFT model so it does not drift into degenerate text that merely games the reward:

$$\max_\theta \; \mathbb{E}_{x,\, y \sim \pi_\theta}\!\Big[ r_\psi(x, y) - \beta \, \mathrm{KL}\big( \pi_\theta(y \mid x) \,\|\, \pi_{\text{SFT}}(y \mid x) \big) \Big].$$

The KL term is doing the same job as PPO's clip: bounding how far each update strays from a trusted reference. RLHF was the technique behind InstructGPT and the assistant behavior of systems such as ChatGPT, Claude, and Gemini. Newer variants simplify the pipeline: **Direct Preference Optimization (DPO)** skips the explicit reward model and optimizes the policy on preference data with a single classification-style loss, and **RLAIF** (including Constitutional AI) replaces or augments human labels with AI-generated preference judgments to scale feedback collection.

## Key Takeaways

- **RL is optimization under interaction.** The agent learns a policy that maximizes discounted return $G_t = \sum_k \gamma^k r_{t+k+1}$ from reward alone, fighting delayed credit assignment and the exploration–exploitation trade-off.
- **The Bellman equation is the engine.** Value and policy iteration solve it when the model is known; TD learning and Q-learning approximate it from sampled experience when it is not.
- **Deep RL = function approximation + stabilizers.** DQN made neural Q-learning work with experience replay and a target network; policy-gradient and actor-critic methods (A2C, PPO) scale to continuous actions and stochastic policies.
- **PPO is the modern default.** Its clipped surrogate keeps each update inside a trust region, making it robust enough to be the optimizer behind both game-playing agents and RLHF.
- **From Atari to alignment.** The same principles drive AlphaZero's self-play, sim-to-real robotics, and the RLHF pipeline (SFT → reward model → PPO) that aligns today's large language models.

---

## See Also

- [Neural Network Architectures](architectures.html) — the CNNs and networks that DQN, actors, and critics are built from
- [Generative Models](generative-models.html) — the autoregressive LLMs that RLHF aligns
- [Frontier Research & Ethics](frontier-and-ethics.html) — scaling, alignment, and AI safety
- [AI & Machine Learning](./) — the section hub and full reference index
- [AI Mathematics](../../advanced/ai-mathematics/) — formal foundations, including optimization theory
- [AI Documentation Hub](../../artificial-intelligence/index.html) — complete index of all AI resources
