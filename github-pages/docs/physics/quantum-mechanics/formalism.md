---
layout: docs
title: "Quantum Mechanics: States, Operators & Dynamics"
permalink: /docs/physics/quantum-mechanics/formalism.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">States, Operators &amp; Dynamics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The working machinery of quantum mechanics: the Schrödinger equation, states and observables, measurement, angular momentum, time evolution, and perturbation theory.</p>
</div>

[Quantum Mechanics](./) &raquo; States, Operators &amp; Dynamics

## The Schrödinger Equation
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://www.fisica.net/mecanica-quantica/Schrodinger_1926.pdf"> Paper: <b><i>An Undulatory Theory of the Mechanics of Atoms and Molecules</i></b> - Erwin Schrödinger</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/schrcn.html"> Article: <b><i>The Schrödinger Equation - HyperPhysics</i></b></a></p>

### Time-Dependent Schrödinger Equation

The fundamental equation of quantum mechanics:

$$i\hbar\,\frac{\partial \psi}{\partial t} = \hat{H}\psi$$

Where $\hat{H}$ is the Hamiltonian operator:
$$\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(x,t)$$

It plays the role for quantum mechanics that $F = ma$ plays for classical mechanics: given the state now, it determines the state at every future time.

### Time-Independent Schrödinger Equation

For stationary states with definite energy:

$$\hat{H}\psi = E\psi$$

Or explicitly:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

## Quantum States and Operators

### Dirac Notation

Quantum states are represented as vectors in Hilbert space:
- **Ket:** $|\psi\rangle$ represents a quantum state
- **Bra:** $\langle\psi|$ represents the complex conjugate (dual vector)
- **Inner product:** $\langle\phi|\psi\rangle$ gives a probability amplitude
- **Outer product:** $|\phi\rangle\langle\psi|$ represents an operator

### Observable Quantities

Physical quantities are represented by Hermitian operators:

**Position operator:** $\hat{x} = x$

**Momentum operator:** $\hat{p} = -i\hbar\,\dfrac{\partial}{\partial x}$

**Energy operator (Hamiltonian):** $\hat{H} = \dfrac{\hat{p}^2}{2m} + V(\hat{x})$

**Angular momentum:** $\hat{\mathbf{L}} = \hat{\mathbf{r}} \times \hat{\mathbf{p}}$

### Eigenvalues and Eigenstates

Measurement of an observable $\hat{A}$ yields eigenvalues:

$$\hat{A}|\psi_n\rangle = a_n|\psi_n\rangle$$

The probability of measuring eigenvalue $a_n$ is:
$$P(a_n) = |\langle\psi_n|\psi\rangle|^2$$

## Decoherence

Decoherence is a concrete, calculable physical process — not an interpretation. It follows directly from the standard linear, unitary dynamics applied to a system *together with* its environment, and it explains why macroscopic superpositions are never observed in practice. Crucially, decoherence makes definite experimental predictions (timescales, loss of interference fringe visibility) that have been measured in the laboratory. Keep it firmly separated from the *interpretational* questions discussed in the next section: decoherence tells you why interference disappears, but on its own it does **not** answer why a single definite outcome is observed.

### How Decoherence Arises

**Decoherence** explains why we don't see quantum superpositions in everyday life:

1. **Environment interaction**: System entangles with environment
2. **Information leakage**: Quantum information spreads to environment
3. **Apparent collapse**: System appears classical to local observers

A system prepared in a superposition $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ does not stay isolated. Each branch becomes correlated with a distinct environmental state, $|0\rangle|E_0\rangle + |1\rangle|E_1\rangle$. Once $\langle E_0|E_1\rangle \to 0$, the off-diagonal (coherence) terms of the system's reduced density matrix vanish and the system behaves, *locally*, like a classical statistical mixture. No interference between $|0\rangle$ and $|1\rangle$ can be recovered without re-collecting the information that leaked into the environment — which is, for a macroscopic environment, practically impossible.

**Decoherence timescales:**
- Electron in vacuum: ~10¹⁰ years
- Dust particle in air: ~10⁻³¹ seconds
- Schrödinger's cat: ~10⁻²³ seconds

The enormous range is the whole point: it is what makes a dust grain or a cat effectively classical while leaving an isolated electron coherent for cosmological times. This explains why cats are never observed alive-and-dead but electrons readily display interference.

### Mathematical Framework

The system-environment interaction Hamiltonian:
$$\hat{H}_{\text{int}} = \sum_\alpha g_\alpha\, \hat{S}_\alpha \otimes \hat{E}_\alpha$$
Where $\hat{S}_\alpha$ are system operators and $\hat{E}_\alpha$ are environment operators.

Tracing out the environment yields the reduced density matrix $\rho_S$, whose evolution follows a master equation:
$$\frac{\partial \rho_S}{\partial t} = -\frac{i}{\hbar}[\hat{H}_S, \rho_S] - \sum_\alpha \gamma_\alpha\,[\hat{S}_\alpha, [\hat{S}_\alpha, \rho_S]]$$
Where $\gamma_\alpha$ are decoherence rates determined by environmental coupling strengths and correlation times. The first (commutator) term is ordinary unitary evolution; the second (double-commutator) term suppresses the off-diagonal elements in the pointer basis selected by $\hat{S}_\alpha$, driving $\rho_S$ toward a diagonal, classical-looking mixture at rate $\gamma_\alpha$.

### Quantum Zeno Effect

Frequent measurements can "freeze" quantum evolution:
- Continuous observation prevents transitions
- Used in quantum error correction
- Demonstrated with trapped ions

**Example**: Watched pot never boils... quantum mechanically!

## The Measurement Problem & Interpretations

The previous section is settled physics; this one is not. Decoherence explains why interference disappears, but it leaves the *measurement problem* untouched: even after the reduced density matrix has become diagonal, the full system-plus-environment state is still a superposition, and unitary evolution alone never picks out the single outcome we actually observe. The questions below are about what the formalism *means* — they are genuinely open, and different answers correspond to different **interpretations** rather than different experimental predictions.

### The Measurement Problem

One of the most profound mysteries in quantum mechanics is measurement. When we measure a quantum system:

1. **Before measurement**: System in superposition $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$
2. **During measurement**: Wave function "collapses" to eigenstate of measured observable
3. **After measurement**: System in definite state $|0\rangle$ with probability $|\alpha|^2$ OR $|1\rangle$ with probability $|\beta|^2$

**Key Questions:**
- What constitutes a measurement?
- Why do we see definite outcomes, not superpositions?
- Is collapse real or apparent?

The linear, unitary Schrödinger evolution that governs every other process *never* produces step 2 on its own. Reconciling this with our experience of single, definite outcomes is precisely the measurement problem, and it is what the interpretations below attempt to address.

### Interpretations of Quantum Mechanics

The mathematics of quantum mechanics is not in dispute — every interpretation makes the *same* experimental predictions. What they disagree about is what the formalism *means*: is the wave function a real physical object, or just a bookkeeping device for our knowledge? Does collapse actually happen, or only appear to? These are questions about the measurement problem (postulates 3 and 4), and they remain genuinely open. The table contrasts the major positions before the descriptions below.

| Interpretation | Is the wave function real? | Does collapse happen? | Determinism | Notable feature |
|----------------|---------------------------|-----------------------|-------------|-----------------|
| Copenhagen | Epistemic / agnostic | Yes (on measurement) | No | The orthodox "textbook" view |
| Many-Worlds | Yes | No — all branches persist | Yes | No collapse; the universe splits |
| Pilot Wave (de Broglie–Bohm) | Yes, plus hidden particle positions | No | Yes (but nonlocal) | Definite trajectories restored |
| QBism | No — it encodes an agent's beliefs | Belief update, not physical | No | Observer-centric, subjective |

<div class="tip-card">
  <h4>Why this debate persists</h4>
  <p>Because all interpretations agree on every measurable outcome, no experiment yet devised can decisively choose between them — the disagreement is about ontology, not predictions. They are best judged on parsimony, how naturally they handle the measurement problem, and whether they can be extended to relativity and gravity. For doing physics you can stay agnostic; for understanding what quantum mechanics says about reality, the choice matters.</p>
</div>

- **Copenhagen:** Wave function collapses upon measurement; complementarity; no definite reality until measurement.
- **Many-Worlds:** All possible outcomes occur in parallel branches; no collapse; deterministic evolution.
- **Pilot Wave (de Broglie–Bohm):** Particles have definite positions guided by a pilot wave; non-local hidden variables; deterministic but non-local.
- **QBism (Quantum Bayesianism):** Wave functions represent an agent's subjective beliefs; measurement updates those beliefs; observer-centric.

<div class="tip-card">
  <h4>Decoherence is not a substitute for an interpretation</h4>
  <p>A common confusion is to treat decoherence as "solving" the measurement problem. It does not: it explains the <em>disappearance of interference</em> and the emergence of a preferred (pointer) basis, but the global state remains a superposition and the question of why <em>one</em> outcome is realized still requires an interpretation. Decoherence is therefore an ingredient that every interpretation uses, not a replacement for them.</p>
</div>

## Angular Momentum

### Orbital Angular Momentum

**Operators:**
$$\begin{aligned}
\hat{L}^2 |\ell,m\rangle &= \hbar^2\ell(\ell+1)|\ell,m\rangle \\
\hat{L}_z |\ell,m\rangle &= \hbar m\,|\ell,m\rangle
\end{aligned}$$

**Commutation relations:**
$$[\hat{L}_i, \hat{L}_j] = i\hbar\,\varepsilon_{ijk}\hat{L}_k$$

### Spin

Intrinsic angular momentum of particles:

**Spin-½ particles (fermions):**
- Electrons, protons, neutrons
- Pauli matrices represent spin operators

**Pauli Matrices:**
$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

**Spin states:**
- Spin up: $|\!\uparrow\rangle = |\tfrac{1}{2}, \tfrac{1}{2}\rangle$
- Spin down: $|\!\downarrow\rangle = |\tfrac{1}{2}, -\tfrac{1}{2}\rangle$

## Time Evolution

### Schrödinger Picture

States evolve in time according to:

$$|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle$$

Where the time evolution operator is:
$$\hat{U}(t) = e^{-i\hat{H}t/\hbar}$$
Note: this form assumes a time-independent Hamiltonian $\hat{H}$.

### Heisenberg Picture

Operators evolve while states remain fixed:

$$\hat{A}(t) = \hat{U}^\dagger(t)\,\hat{A}(0)\,\hat{U}(t)$$

**Heisenberg equation of motion:**
$$\frac{d\hat{A}}{dt} = \frac{i}{\hbar}[\hat{H},\hat{A}] + \frac{\partial \hat{A}}{\partial t}$$

## Perturbation Theory

### Time-Independent Perturbation Theory

For $\hat{H} = \hat{H}_0 + \lambda\hat{V}$:

**First-order energy correction:**
$$E_n^{(1)} = \langle n^{(0)}|\hat{V}|n^{(0)}\rangle$$

**First-order wave function correction:**
$$
|n^{(1)}\rangle = \sum_{m \neq n} \frac{\langle m^{(0)}|\hat{V}|n^{(0)}\rangle}{E_n^{(0)} - E_m^{(0)}} |m^{(0)}\rangle
$$

If the perturbation series diverges, check that the perturbation is genuinely small ($|\langle\hat{V}\rangle| \ll |\langle\hat{H}_0\rangle|$), that the unperturbed states are orthogonal, and whether the level is degenerate — degenerate levels require degenerate perturbation theory.

### Time-Dependent Perturbation Theory

**Transition probability (Fermi's Golden Rule):**
$$
P_{i \to f} = \frac{2\pi}{\hbar} |\langle f|\hat{V}|i\rangle|^2 \delta(E_f - E_i)
$$

---

## Continue Reading

- **Next:** [Systems & Phenomena](systems-and-phenomena.html) — the box, oscillator, and hydrogen atom; tunneling, entanglement, and superposition.
- **Up:** [Quantum Mechanics Hub](./)

## See Also

- [Quantum Field Theory](../quantum-field-theory.html) — the Heisenberg picture and operator dynamics extended to fields.
- [Classical Mechanics](../classical-mechanics/) — the Hamiltonian and the classical limit of these equations.
