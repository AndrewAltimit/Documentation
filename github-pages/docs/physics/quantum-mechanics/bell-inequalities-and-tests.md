---
layout: docs
title: "Quantum Mechanics: Bell's Theorem & Experimental Tests"
permalink: /docs/physics/quantum-mechanics/bell-inequalities-and-tests.html
toc: true
toc_sticky: true
hide_title: true
---

## Bell's Theorem & Experimental Tests

[Quantum Mechanics](./) &raquo; Bell's Theorem &amp; Experimental Tests

**Why this page exists.** Entanglement is introduced on the [Systems & Phenomena](systems-and-phenomena.html) page as a correlation with "no classical analog." That phrase is not rhetoric — it is a *theorem*. Bell's inequality turns the philosophical EPR debate into a sharp, falsifiable prediction, and decades of increasingly airtight experiments have decided it. This page develops the argument from first principles: the EPR setup, the local-hidden-variable assumption, the CHSH inequality and its derivation, the quantum prediction and Tsirelson's bound, the landmark experiments, the loopholes that kept the question open until 2015, and the consequences for no-signaling and device-independent technology.

## The EPR Argument

In 1935, Einstein, Podolsky, and Rosen (EPR) published a challenge to the completeness of quantum mechanics. They did not dispute that the theory's predictions were correct; they argued that the wave function could not be the *whole* story.

Their reasoning rests on two premises that seemed beyond reproach:

- **Locality.** Two systems sufficiently far apart cannot influence each other instantaneously. A measurement performed on one cannot, in EPR's words, disturb the other "in any way."
- **Reality (the EPR criterion).** "If, without in any way disturbing a system, we can predict with certainty the value of a physical quantity, then there exists an element of physical reality corresponding to that quantity."

Consider two spin-½ particles prepared in the **singlet state** and flying apart to distant laboratories run by Alice and Bob:

$$
|\Psi^-\rangle = \frac{1}{\sqrt{2}}\left(|\!\uparrow\rangle_A|\!\downarrow\rangle_B - |\!\downarrow\rangle_A|\!\uparrow\rangle_B\right).
$$

This state is **rotationally invariant**: it has the same form along *any* common axis. Consequently the spins are perfectly anti-correlated no matter which direction is chosen — if Alice measures spin-up along some axis, Bob is certain to find spin-down along that same axis.

Now run EPR's logic. Alice is free to measure her particle's spin along $z$ *or* along $x$.

- If she measures along $z$ and gets $+\tfrac{\hbar}{2}$, she can predict with certainty that Bob's $z$-spin is $-\tfrac{\hbar}{2}$. By the reality criterion, Bob's $z$-spin is an *element of reality*.
- Had she instead chosen $x$, she could equally have predicted Bob's $x$-spin with certainty, making Bob's $x$-spin an element of reality.

Because (by locality) Alice's choice cannot disturb Bob's distant particle, **both** $z$-spin and $x$-spin must be simultaneously real properties of Bob's particle. But quantum mechanics forbids assigning simultaneous definite values to non-commuting observables — $[\hat S_x, \hat S_z] \neq 0$. EPR concluded that the wave function is therefore an **incomplete** description: there must be additional "hidden" variables, not captured by $|\psi\rangle$, that fix the outcomes in advance.

**What EPR were and were not claiming.** EPR did not claim quantum mechanics was *wrong*. They argued it was *incomplete* — that a deeper, deterministic theory respecting locality could underlie it and supply the "elements of reality" the wave function omits. Bohr's reply rejected the reality criterion as inapplicable to entangled systems, but the debate stayed philosophical for nearly thirty years. Everyone assumed local-hidden-variable theories and quantum mechanics made the *same* predictions, so no experiment could distinguish them. Bell's achievement was to prove that assumption false.

## Local Hidden Variables

To turn EPR's intuition into mathematics, suppose the outcomes are predetermined by some **hidden variable** $\lambda$ — a list of instructions the particles carry from the source. $\lambda$ may be anything: a single number, a high-dimensional vector, a whole field configuration. It is distributed according to some probability density $\rho(\lambda)$ with $\int \rho(\lambda)\,d\lambda = 1$.

A **local hidden variable (LHV)** theory makes two assumptions precise:

1. **Realism (determinism / value-definiteness).** Each measurement outcome is a function of the local setting and $\lambda$ alone. If Alice measures along direction $\mathbf{a}$ and Bob along $\mathbf{b}$, their results are
$$
A(\mathbf{a}, \lambda) = \pm 1, \qquad B(\mathbf{b}, \lambda) = \pm 1.
$$
2. **Locality (parameter independence).** Alice's outcome does not depend on Bob's setting and vice versa: $A$ is not a function of $\mathbf{b}$, and $B$ is not a function of $\mathbf{a}$. The choices are made independently, and no influence propagates between the labs.

A third, often-implicit assumption is **measurement independence** (or "free choice"): the settings $\mathbf{a}, \mathbf{b}$ are chosen independently of $\lambda$, so $\rho(\lambda)$ does not depend on the experimenters' choices. Violating this is the **freedom-of-choice loophole** discussed below.

Within an LHV theory, the **correlation** — the expectation value of the product of the two outcomes — is obtained by averaging over the hidden variable:

$$
E(\mathbf{a}, \mathbf{b}) = \int \rho(\lambda)\, A(\mathbf{a}, \lambda)\, B(\mathbf{b}, \lambda)\, d\lambda.
$$

This single equation encodes the entire LHV worldview. Bell's theorem is the statement that correlations of this form must obey an inequality that quantum mechanics violates.

## The CHSH Inequality

Bell's original 1964 inequality assumed perfect anti-correlation. The 1969 **CHSH inequality** (Clauser, Horne, Shimony, Holt) is the experimentally robust version and the one used by essentially every modern test. It involves **two** settings on each side.

Let Alice choose between directions $\mathbf{a}$ and $\mathbf{a}'$, and Bob between $\mathbf{b}$ and $\mathbf{b}'$. Each measurement yields $\pm 1$. Define the **CHSH quantity**:

$$
S = E(\mathbf{a}, \mathbf{b}) - E(\mathbf{a}, \mathbf{b}') + E(\mathbf{a}', \mathbf{b}) + E(\mathbf{a}', \mathbf{b}').
$$

### Derivation

The derivation is elementary algebra, which is precisely what makes the theorem so powerful — it relies on no quantum mechanics at all. For a *single* value of $\lambda$, write the four outcomes as $A = A(\mathbf{a}, \lambda)$, $A' = A(\mathbf{a}', \lambda)$, $B = B(\mathbf{b}, \lambda)$, $B' = B(\mathbf{b}', \lambda)$, each equal to $\pm 1$. Group the terms:

$$
A B - A B' + A' B + A' B' = A(B - B') + A'(B + B').
$$

Because $B$ and $B'$ are each $\pm 1$, one of the two combinations $(B - B')$ and $(B + B')$ is $0$ and the other is $\pm 2$. Hence the right-hand side is either $\pm 2 A$ or $\pm 2 A'$ — in every case it equals exactly $\pm 2$:

$$
A(B - B') + A'(B + B') = \pm 2.
$$

Now average over $\lambda$ with weight $\rho(\lambda) \ge 0$. The average of a quantity bounded by $[-2, +2]$ is itself bounded by $[-2, +2]$. Since averaging is linear and each averaged term is one of the correlations $E$, we arrive at the **CHSH inequality**:

$$
|S| = \left| E(\mathbf{a}, \mathbf{b}) - E(\mathbf{a}, \mathbf{b}') + E(\mathbf{a}', \mathbf{b}) + E(\mathbf{a}', \mathbf{b}') \right| \le 2.
$$

**What the derivation actually used.** Read it back: the only ingredients were (i) each outcome is a definite ±1 (realism), (ii) Alice's outcome does not depend on Bob's setting so the four outcomes can be evaluated for one common $\lambda$ (locality), and (iii) a single distribution $\rho(\lambda)$ that does not depend on the settings (measurement independence). That is the entire content of "local hidden variables." Any theory satisfying these three assumptions — Newtonian or otherwise — is bound by $|S| \le 2$. A violation therefore falsifies the conjunction of those three assumptions, not quantum mechanics.

## The Quantum Prediction and Tsirelson's Bound

Quantum mechanics does not assign predetermined values; it computes correlations directly from the entangled state. For the singlet state, measuring spin along unit vectors $\mathbf{a}$ and $\mathbf{b}$ (using observables $\boldsymbol{\sigma}\cdot\mathbf{a}$ and $\boldsymbol{\sigma}\cdot\mathbf{b}$ with outcomes $\pm 1$) gives the famous result

$$
E_{\text{QM}}(\mathbf{a}, \mathbf{b}) = \langle \Psi^- |\, (\boldsymbol{\sigma}\cdot\mathbf{a}) \otimes (\boldsymbol{\sigma}\cdot\mathbf{b}) \,| \Psi^- \rangle = -\,\mathbf{a}\cdot\mathbf{b} = -\cos\theta_{ab},
$$

where $\theta_{ab}$ is the angle between the two measurement axes. The minus sign reflects the singlet's anti-correlation: at $\theta = 0$ the spins are perfectly anti-aligned ($E = -1$), at $\theta = \pi$ perfectly aligned ($E = +1$).

### Choosing the optimal angles

Pick coplanar settings separated by successive $45^\circ$ angles: $\mathbf{a}$ at $0^\circ$, $\mathbf{b}$ at $45^\circ$, $\mathbf{a}'$ at $90^\circ$, $\mathbf{b}'$ at $135^\circ$. Then

$$
E(\mathbf{a}, \mathbf{b}) = -\cos 45^\circ, \quad E(\mathbf{a}, \mathbf{b}') = -\cos 135^\circ, \quad E(\mathbf{a}', \mathbf{b}) = -\cos 45^\circ, \quad E(\mathbf{a}', \mathbf{b}') = -\cos 45^\circ.
$$

Substituting into $S$ (with $\cos 45^\circ = \tfrac{1}{\sqrt 2}$ and $\cos 135^\circ = -\tfrac{1}{\sqrt 2}$):

$$
S_{\text{QM}} = -\frac{1}{\sqrt 2} - \frac{1}{\sqrt 2} - \frac{1}{\sqrt 2} - \frac{1}{\sqrt 2} = -\frac{4}{\sqrt 2} = -2\sqrt{2}.
$$

So $|S_{\text{QM}}| = 2\sqrt{2} \approx 2.828$, **decisively exceeding** the LHV bound of $2$. No local hidden variable theory can reproduce this; quantum mechanics predicts a correlation no classical instruction set can match.

### Tsirelson's bound

How far can quantum mechanics push the violation? The maximum is **Tsirelson's bound** (Cirel'son, 1980):

$$
|S| \le 2\sqrt{2} \approx 2.828.
$$

The $45^\circ$ configuration above already saturates it. The bound follows from operator algebra: writing the CHSH operator as $\hat{S} = \hat A \hat B - \hat A \hat B' + \hat A' \hat B + \hat A' \hat B'$ with Hermitian $\hat A, \hat A', \hat B, \hat B'$ that square to the identity, one finds

$$
\hat{S}^2 = 4\,\mathbb{1} - [\hat A, \hat A']\,[\hat B, \hat B'].
$$

Since the operator norm of each commutator is at most $2$, $\|\hat S^2\| \le 4 + 4 = 8$, giving $\|\hat S\| \le 2\sqrt 2$.

**Why isn't the bound 4?** Algebraically, $|S|$ could be as large as $4$ — and there *are* hypothetical "super-quantum" correlations (Popescu–Rohrlich boxes) that reach $S = 4$ while still forbidding faster-than-light signaling. Nature does not use them. Quantum mechanics stops precisely at $2\sqrt 2$. Understanding *why* physical correlations are limited to Tsirelson's bound, rather than the larger no-signaling bound of 4, is an active question in quantum foundations and information theory (principles like "information causality" reproduce $2\sqrt 2$).

The hierarchy of bounds is worth memorizing:

| Theory | Maximum $|S|$ |
|--------|---------------|
| Local hidden variables (classical) | $2$ |
| Quantum mechanics (Tsirelson) | $2\sqrt{2} \approx 2.83$ |
| No-signaling (Popescu–Rohrlich) | $4$ |

## Bell's Theorem

Combining the two preceding sections gives the statement now called **Bell's theorem**:

> No physical theory based on local hidden variables can reproduce all of the predictions of quantum mechanics.

It is a *no-go* theorem. It does not, by itself, say which assumption nature violates — only that **realism + locality + measurement independence** cannot all hold. The standard reading, vindicated by experiment, is that **local realism** fails: the world cannot be described by predetermined local properties. One can keep realism by abandoning locality (as in Bohmian mechanics, an explicitly non-local hidden-variable theory) or rethink the meaning of measurement outcomes entirely, but the comfortable EPR picture of local, predetermined values is excluded.

```python
import numpy as np

# Singlet correlation E(a, b) = -cos(theta_ab) and the CHSH quantity S.
def E(theta_a, theta_b):
    return -np.cos(np.radians(theta_a - theta_b))

# Optimal settings: a=0, a'=90 for Alice; b=45, b'=135 for Bob
a, a_prime = 0.0, 90.0
b, b_prime = 45.0, 135.0

S = E(a, b) - E(a, b_prime) + E(a_prime, b) + E(a_prime, b_prime)

print(f"Classical (LHV) bound:   |S| <= 2")
print(f"Quantum prediction:      S  = {S:.4f}")        # -2.8284
print(f"Tsirelson bound:         |S| <= {2*np.sqrt(2):.4f}")
# |S| = 2.83 > 2  ->  local hidden variables are ruled out
```

## Experimental Tests

A theorem is only as good as its experimental verdict. The history of Bell tests is a fifty-year campaign to make the violation visible while closing every conceivable way an LHV theory could fake it.

### Freedman–Clauser (1972)

The first experimental test, by Stuart Freedman and John Clauser at Berkeley, used **polarization-entangled photon pairs** from a calcium atomic cascade. They measured a violation of a Bell-type inequality by about six standard deviations, in agreement with quantum mechanics. The result was suggestive but used static polarizers and modest statistics, leaving large loopholes open.

### Aspect (1981–1982)

Alain Aspect's group in Orsay performed the experiments that made Bell violations famous. Their decisive 1982 experiment introduced **time-varying analyzers**: acousto-optic switches redirected each photon to one of two differently-oriented polarizers every ~10 ns, *while the photons were in flight*. Because the switching was faster than the light-travel time between the two stations, the choice of setting on Alice's side could not — without superluminal influence — have been "known" to Bob's particle when it left the source.

- Measured: $S = 2.697 \pm 0.015$, violating the classical bound of $2$ by tens of standard deviations.
- Significance: a first serious attack on the **locality loophole** by enforcing space-like separation between the *setting choices* and the distant measurement.

Aspect's switching was quasi-periodic rather than truly random, so a determined local-realist could still object, but the experiment shifted the consensus firmly toward quantum mechanics.

### Weihs (1998)

Gregor Weihs and Anton Zeilinger's group in Innsbruck closed the locality loophole far more rigorously. Using polarization-entangled photons sent through **400 m of optical fiber**, they used genuine **fast quantum random number generators** to select each measurement setting *independently and at random* on each side, with the entire choose-and-measure event on one side kept **space-like separated** from the other.

- Measured: a CHSH violation by roughly 30 standard deviations.
- Significance: the first test enforcing strict **Einstein locality** with truly random, relativistically-spacelike setting choices. After Weihs, the locality loophole was essentially closed for photons — but the **detection loophole** remained, because photon detectors are inefficient and most pairs were never registered.

**Two stubborn loopholes, two platforms.** By the late 1990s the field faced a frustrating split. **Photon** experiments could enforce locality (photons fly far, fast) but suffered the detection loophole (detectors miss most photons). **Trapped-ion and atom** experiments had near-perfect detection but could not be separated far enough to enforce locality. Each platform closed one loophole while leaving the other open, so a die-hard local realist could always retreat to the gap. Closing both at once was the grand challenge of the 2010s.

## The Detection and Locality Loopholes

A loophole is any auxiliary assumption that, if false, would let an LHV model mimic a Bell violation. The two principal ones:

### The locality loophole

If a signal could travel from Alice's apparatus to Bob's during the experiment, then Bob's outcome *could* depend on Alice's setting without violating relativity — and an LHV model could reproduce the quantum correlations. **Closing it** requires that the choice of setting and the recording of the outcome on each side be **space-like separated** from the corresponding events on the other side, so that no sub-luminal influence can connect them. This is what Aspect attacked and Weihs settled for photons.

### The detection (fair-sampling) loophole

Real detectors miss particles. If only a fraction of pairs are detected, one must *assume* the detected sample is representative ("fair sampling"). But an LHV model can exploit inefficiency: each particle's hidden variable could decide *whether* it gets detected as a function of the local setting, biasing the detected subensemble to fake a violation. **Closing it** requires the **overall detection efficiency to exceed a threshold** — for the CHSH inequality with maximally entangled states, $\eta > 2/3 \approx 82.8\%$ (the Eberhard limit lowers this to ~$2/3$ for non-maximally entangled states with a modified inequality).

**The tension between the loopholes.** The two loopholes pull in opposite directions. Closing locality wants the detectors *far apart*, which favors photons (they travel well) but worsens losses. Closing detection wants *near-perfect efficiency*, which favors massive particles like ions or electron spins, but those are hard to separate over the kilometers needed for space-like separation. Any single platform tended to close one loophole only by reopening the other. This is why no experiment before 2015 was simultaneously free of both.

A third, subtler loophole is the **freedom-of-choice (measurement-independence) loophole**: if the hidden variable $\lambda$ could influence the experimenters' setting choices, the derivation's assumption $\rho(\lambda) \perp (\mathbf{a},\mathbf{b})$ fails. It is closed by deriving the settings from sources space-like separated from the entanglement source — or, dramatically, from cosmic photons emitted by distant quasars billions of years ago (the "Cosmic Bell" tests, 2017–2018).

## The Loophole-Free Experiments (2015+)

In 2015, three groups independently performed Bell tests closing the **detection and locality loopholes simultaneously** — the long-sought "loophole-free" experiments.

### Hensen et al. (Delft, 2015) — electron spins

The Delft group (Hanson lab) used **electron spins in nitrogen-vacancy (NV) centers** in diamond, located in two labs **1.3 km apart**. The trick was **entanglement swapping by event-ready detection**: each NV electron spin is entangled with an emitted photon; the photons meet at a midpoint station, and a joint (Bell-state) measurement there *heralds* successful spin–spin entanglement. Because the entanglement is confirmed *before* the spin readout, and the spin measurement (using settings chosen by fast random generators) is fast and **deterministic at near-unit efficiency**, both loopholes close at once.

- Distance: 1.3 km, ensuring space-like separation of the setting choices and readouts (closing **locality**).
- Detection: spin readout is essentially deterministic — every heralded event is recorded (closing **detection**).
- Result: $S = 2.42 \pm 0.20$, a violation of the classical bound with a p-value around $0.039$.

The price of using event-ready entanglement was a very low rate — 245 trials over weeks — but it was the first single experiment free of both major loopholes.

### Giustina et al. (Vienna) and Shalm et al. (NIST), 2015 — photons

Two photon experiments published the same year closed both loopholes using **high-efficiency superconducting transition-edge detectors** (efficiency above ~90%, beating the Eberhard threshold) together with **fast random setting selection** and sufficient separation:

- **Giustina et al. (Vienna):** polarization-entangled photons, violation with a p-value on the order of $10^{-7}$.
- **Shalm et al. (NIST/Boulder):** entangled photons over ~185 m, with settings driven partly by physical and pseudo-random sources, p-value around $10^{-8}$.

These delivered overwhelming statistical significance at much higher event rates than the NV experiment, using non-maximally entangled states and the Eberhard inequality to tolerate the remaining loss.

### The "Big Bell Test" (2016) and Cosmic Bell (2017–2018)

Subsequent experiments attacked the **freedom-of-choice loophole**:

- **The BIG Bell Test (2016):** ~100,000 human volunteers worldwide generated the setting bits by playing an online game, replacing machine randomness with unpredictable human choices.
- **Cosmic Bell tests (2017–2018):** setting choices were derived from the color of photons emitted by **high-redshift quasars** — light that left its source billions of years ago. For a hidden-variable conspiracy to fix the settings, the correlation would have had to be established in the early universe, pushing the freedom-of-choice loophole back to cosmological time scales.

**The verdict.** By 2018, Bell violations had been demonstrated with the locality, detection, and freedom-of-choice loopholes simultaneously closed (to within cosmological limits), across photons, electron spins, and atoms. **Local realism is experimentally dead.** The 2022 Nobel Prize in Physics went to Alain Aspect, John Clauser, and Anton Zeilinger "for experiments with entangled photons, establishing the violation of Bell inequalities and pioneering quantum information science." No remaining loophole is taken seriously as a real description of nature; the residual ones (e.g. superdeterminism) require abandoning the experimenters' free choice itself.

## Implications

### No-signaling: violation without communication

A natural worry: if measuring Alice's particle "instantly affects" Bob's, doesn't that signal faster than light? It does not. The **no-signaling theorem** guarantees that Bob's local statistics — the marginal probabilities of his outcomes — are completely independent of Alice's *choice of setting*. Formally, for any quantum state,

$$
\sum_{b} P(a, b \mid \mathbf{a}, \mathbf{b}) = \sum_{b} P(a, b \mid \mathbf{a}, \mathbf{b}'),
$$

so summing over (i.e. ignoring) Alice's result, Bob sees the same distribution regardless of what Alice does. The correlations only become visible *after* the two parties bring their result lists together over a classical channel. Quantum non-locality is non-local in *correlations*, not in *signals* — fully compatible with relativistic causality. (See the no-communication discussion on the [Quantum Computing](qm-computing.html#entanglement-as-a-resource) page.)

### Device independence

The most powerful practical consequence is **device-independent (DI) certification**. Because a Bell violation can *only* be produced by genuinely entangled systems and cannot be faked by any local (classical) machinery, observing $S > 2$ certifies entanglement and quantumness **without trusting the internal workings of the devices** — treating them as black boxes characterized only by inputs (settings) and outputs (results). This underpins:

- **Device-independent quantum key distribution (DIQKD).** The amount by which $|S|$ exceeds $2$ bounds an eavesdropper's possible information, so two parties can establish a provably secret key even using untrusted, possibly adversary-built hardware. First experimental demonstrations appeared in 2022 (trapped ions; atoms).
- **Device-independent randomness generation/expansion.** A Bell violation guarantees the outcomes are *intrinsically random* — not pre-set by any hidden mechanism — certifying true randomness from the laws of physics rather than from trust in a hardware vendor. NIST and others have built certified-randomness beacons on this principle.
- **Self-testing.** Saturating Tsirelson's bound, $|S| = 2\sqrt 2$, *uniquely* certifies (up to local isometry) that the underlying state is a maximally entangled pair and the measurements are the optimal anti-commuting ones — the statistics alone pin down the physics.

**The conceptual payoff.** Bell's theorem converted a 1935 philosophical dispute about the "completeness" of the wave function into a quantitative, technological resource. The same inequality that rules out local realism also serves as a *witness*: a number you can measure to certify that a device is exploiting genuine quantum entanglement, with security guarantees that hold even against a manufacturer you do not trust. Foundations became engineering.

## Key Takeaways

- **EPR posed the question.** Perfect correlations in the singlet state seemed to demand predetermined local "elements of reality," implying quantum mechanics is incomplete.
- **Bell made it testable.** Any local-hidden-variable theory obeys $|S| \le 2$ (CHSH); the derivation is pure algebra with no quantum input.
- **Quantum mechanics breaks it.** The singlet gives $E(\mathbf{a},\mathbf{b}) = -\cos\theta_{ab}$ and $|S| = 2\sqrt 2$, up to Tsirelson's bound and no further.
- **Loopholes mattered.** The locality and detection loopholes kept the question technically open; photon and matter platforms each closed only one until 2015.
- **2015 closed both at once.** Delft (NV spins, 1.3 km), Vienna, and NIST (high-efficiency photons) performed loophole-free violations; the 2022 Nobel followed.
- **It is a resource.** No-signaling preserves relativity, while device-independence turns the violation into certified security and randomness.

## See Also

- [Systems & Phenomena](systems-and-phenomena.html) — the singlet state, entanglement, and the experiments this page expands on.
- [States, Operators & Dynamics](formalism.html) — spin operators, Pauli matrices, and the measurement postulate behind the correlation $E(\mathbf{a},\mathbf{b})$.
- [Quantum Computing](qm-computing.html) — the four Bell states, the no-communication theorem, and entanglement as a computational resource.
- [Computing, Information & Advanced Formalism](computing-and-advanced.html) — density matrices and the quantum-information machinery underlying device-independent protocols.
- [Research Frontiers](qm-research-frontiers.html) — Wigner's-friend tests, objective-collapse searches, and the foundational program Bell experiments belong to.
- [Quantum Mechanics Hub](./) — the postulates and core formalism these arguments rest on.
