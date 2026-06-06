---
layout: docs
title: "Quantum Mechanics: Quantum Computing"
permalink: /docs/physics/quantum-mechanics/qm-computing.html
toc: true
toc_sticky: true
hide_title: true
---

## Quantum Computing

[Quantum Mechanics](./) &raquo; Quantum Computing

**Scope: the physics, not the engineering.** This page treats quantum computing as applied quantum mechanics — what a qubit *is* as a two-level system, why entanglement is the resource that makes algorithms super-classical, and how decoherence sets the rules. The hardware stacks, software frameworks, and computer-science side of algorithms live in the [Quantum Computing](../../quantum-computing/) technology hub. Everything below follows directly from the five postulates assembled on the [Quantum Mechanics Hub](./).

## Classical vs. Quantum Information

A classical **bit** takes one of two definite values, $0$ or $1$. A **qubit** is the state of any quantum two-level system, and quantum mechanics permits coherent superpositions:

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1
$$

with $\alpha, \beta \in \mathbb{C}$. The Born rule makes the amplitudes physical:

- $|\alpha|^2$ = probability of measuring $0$,
- $|\beta|^2$ = probability of measuring $1$.

The state $|\psi\rangle$ lives in a two-dimensional complex Hilbert space $\mathcal{H} = \mathbb{C}^2$, with the **computational basis** $\{|0\rangle, |1\rangle\}$ identified with the standard basis vectors

$$
|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \qquad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.
$$

A single qubit therefore carries a *continuum* of states, but a measurement extracts only one classical bit. The power of quantum computing comes not from "storing more" in one qubit, but from how superposition and entanglement compose across **many** qubits: $n$ qubits span a Hilbert space of dimension $2^n$, and a generic state requires $2^n$ complex amplitudes to describe — exponentially more than the $n$ bits of a classical register.

### Physical Qubit Implementations

Any quantum system with a well-isolated pair of energy levels can serve as a qubit. The dominant platforms differ in coherence time (how long a superposition survives) and gate speed:

1. **Superconducting qubits** (Google, IBM)
   - Josephson junctions create an anharmonic LC oscillator; the lowest two levels are the qubit, and the anharmonicity prevents leakage to higher levels.
   - Coherence time: ~100 μs. Gate time: ~10–100 ns.

2. **Trapped ions** (IonQ, Quantinuum)
   - Internal electronic or hyperfine states of ions held by oscillating electric fields; gates use laser-driven transitions.
   - Coherence time: seconds to minutes. Gate time: ~10–100 μs.

3. **Neutral atoms** (QuEra, Atom Computing)
   - Atoms held in optical tweezers; Rydberg excitations mediate interactions.
   - Highly scalable arrays; long coherence times.

4. **Photonic qubits** (Xanadu, PsiQuantum)
   - Photon presence/polarization/path; photons are naturally isolated from the environment.
   - Challenge: photons barely interact, so two-qubit gates are hard (measurement-induced nonlinearity).

5. **Topological qubits** (Microsoft)
   - Non-abelian anyons store information non-locally, giving inherent protection from local noise. Still experimental.

## Qubits and the Bloch Sphere

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://en.wikipedia.org/wiki/Bloch_sphere"> Article: <b><i>The Bloch Sphere Representation - Wikipedia</i></b></a></p>

A pure single-qubit state has four real parameters ($\alpha, \beta$ each complex), but normalization removes one and an unobservable global phase removes another. The remaining **two** real parameters map every pure qubit state onto the surface of a unit sphere — the **Bloch sphere**:

$$
|\psi\rangle = \cos\!\left(\frac{\theta}{2}\right)|0\rangle + e^{i\varphi}\sin\!\left(\frac{\theta}{2}\right)|1\rangle, \qquad 0 \le \theta \le \pi, \;\; 0 \le \varphi < 2\pi.
$$

The corresponding **Bloch vector** is

$$
\mathbf{r} = (\sin\theta\cos\varphi,\; \sin\theta\sin\varphi,\; \cos\theta),
$$

and the density operator is $\hat\rho = \tfrac12(\mathbb{1} + \mathbf{r}\cdot\boldsymbol{\sigma})$, where $\boldsymbol{\sigma} = (\hat\sigma_x, \hat\sigma_y, \hat\sigma_z)$ are the Pauli matrices.

- The **north pole** ($\theta=0$) is $|0\rangle$; the **south pole** ($\theta=\pi$) is $|1\rangle$.
- States on the **equator**, like $|+\rangle = \tfrac{1}{\sqrt2}(|0\rangle + |1\rangle)$, are equal superpositions differing only in phase $\varphi$.
- **Pure states** lie on the surface ($|\mathbf{r}| = 1$); **mixed states** (from decoherence or partial information) lie strictly inside ($|\mathbf{r}| < 1$). The center $\mathbf{r} = 0$ is the maximally mixed state $\hat\rho = \tfrac12\mathbb{1}$.

This geometric picture is invaluable: every single-qubit gate is a **rotation** of the Bloch sphere, and decoherence shrinks the Bloch vector toward the center.

**Why "half-angle"?** The factor of θ/2 reflects the two-to-one (spin-1/2) relationship between the Bloch sphere SO(3) and the state space SU(2): rotating a spin-1/2 by 2π returns it to *minus* itself. Antipodal points on the Bloch sphere (e.g. |0⟩ and |1⟩) are *orthogonal* states, not opposite ones — the sphere is a projective picture, not the raw Hilbert space.

## Quantum Gates

The third postulate says closed-system evolution is **unitary**: $|\psi'\rangle = \hat U|\psi\rangle$ with $\hat U^\dagger\hat U = \mathbb{1}$. A quantum gate is just a unitary acting on one or more qubits. Unitarity has a deep consequence — gates are **reversible** (the inverse gate is $\hat U^\dagger$), unlike irreversible classical gates such as AND.

### Single-Qubit Gates

The Pauli matrices are themselves gates (bit flip, phase flip, and their combination):

$$
\hat\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \qquad
\hat\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \qquad
\hat\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.
$$

The **Hadamard gate** creates equal superpositions and is the workhorse for moving between the $Z$- and $X$-bases:

$$
\hat H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix},
\qquad \hat H|0\rangle = |+\rangle, \quad \hat H|1\rangle = |-\rangle.
$$

The **phase gate** $\hat S$ and the **$T$ gate** add relative phases:

$$
\hat S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \qquad
\hat T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}.
$$

Geometrically, an arbitrary single-qubit gate is a rotation by angle $\gamma$ about a Bloch-sphere axis $\hat{\mathbf{n}}$:

$$
\hat U = e^{-i\gamma\, \hat{\mathbf{n}}\cdot\boldsymbol{\sigma}/2}
= \cos\!\left(\frac{\gamma}{2}\right)\mathbb{1} - i\sin\!\left(\frac{\gamma}{2}\right)\hat{\mathbf{n}}\cdot\boldsymbol{\sigma}.
$$

### Two-Qubit Gates and Universality

Single-qubit rotations alone only ever produce **product** states. To generate entanglement we need an entangling two-qubit gate. The canonical choice is **CNOT** (controlled-NOT): it flips the target qubit if and only if the control is $|1\rangle$,

$$
\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix},
$$

acting on the ordered basis $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$.

**Universality.** Any unitary on n qubits can be approximated to arbitrary accuracy by a finite set of gates. A standard **universal set** is {H, T, CNOT}. The single-qubit rotations supply the continuum of the Bloch sphere (the Solovay–Kitaev theorem guarantees efficient approximation), while CNOT supplies entanglement. This is the quantum analog of NAND being universal for classical logic.

## Entanglement as a Resource

Entanglement is the feature with no classical analog, and it is precisely what lets quantum algorithms outrun classical ones. Apply $\hat H$ to the control, then CNOT, starting from $|00\rangle$:

$$
\text{CNOT}\,(\hat H\otimes \mathbb{1})\,|00\rangle
= \text{CNOT}\,\frac{|00\rangle + |10\rangle}{\sqrt2}
= \frac{|00\rangle + |11\rangle}{\sqrt2} \equiv |\Phi^+\rangle.
$$

The four maximally entangled **Bell states** are

$$
|\Phi^{\pm}\rangle = \frac{|00\rangle \pm |11\rangle}{\sqrt2}, \qquad
|\Psi^{\pm}\rangle = \frac{|01\rangle \pm |10\rangle}{\sqrt2}.
$$

A state is **entangled** when it cannot be written as a product $|\psi_A\rangle\otimes|\psi_B\rangle$. The operational signature is the **reduced density matrix**: tracing out qubit $B$ from $|\Phi^+\rangle$ gives

$$
\hat\rho_A = \text{Tr}_B\,|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\mathbb{1},
$$

the maximally mixed state. A subsystem of a pure entangled state is itself mixed — the information lives in the **correlations**, not in either qubit alone. The **von Neumann entropy** $S(\hat\rho_A) = -\text{Tr}(\hat\rho_A\ln\hat\rho_A)$ quantifies this; it is $0$ for product states and maximal ($\ln 2$ per qubit) for Bell states.

**Entanglement does not transmit information.** Measuring one half of a Bell pair instantly fixes the correlated outcome for the other — but the marginal statistics on each side are unchanged until the two parties compare results over a classical channel. The **no-communication theorem** forbids signaling. Entanglement is a resource for *correlation* and *computation*, not for faster-than-light messaging.

### Where the speedup comes from

A common misconception is that quantum computers "try all answers in parallel." They do create a superposition over all $2^n$ inputs, but measurement collapses it to a single random outcome. The real trick is **interference**: a well-designed algorithm uses entanglement and phase manipulation so that amplitudes for wrong answers **cancel** while amplitudes for the right answer **reinforce**. No entanglement, no super-classical speedup — for pure-state circuits, a quantum computation that never entangles can be simulated efficiently classically.

## Quantum Algorithms

The following algorithms are sketched from the physics side — the focus is *which quantum-mechanical feature* delivers the advantage.

### Shor's Algorithm (1994)

**Purpose:** Factor a large integer $N$ exponentially faster than the best known classical method.

**Physics behind it:** Factoring reduces to **period finding**. Pick a random $a < N$ coprime to $N$; the function $f(x) = a^x \bmod N$ is periodic with some period $r$. Prepare a superposition over $x$, evaluate $f$ into a second register (entangling the two), and apply the **Quantum Fourier Transform** (QFT). The QFT is exactly the change of basis under which a periodic state concentrates its amplitude on multiples of $1/r$ — interference makes the period readable in a single measurement. Once $r$ is known (and is even with $a^{r/2}\not\equiv -1$), $\gcd(a^{r/2}\pm 1,\, N)$ yields a nontrivial factor.

$$
|x\rangle \xrightarrow{\text{QFT}} \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i\, xk/2^n}\,|k\rangle
$$

**Speedup:** roughly $O((\log N)^3)$ versus the sub-exponential general number field sieve.

```python
# Structure of Shor's algorithm (the quantum part is the period finding)
def shors_algorithm(N):
    # 1. Choose random a < N with gcd(a, N) = 1
    # 2. Use the QFT to find the period r of a^x mod N  (quantum subroutine)
    # 3. If r is even and a^(r/2) != -1 mod N:
    #        factors = gcd(a^(r/2) +/- 1, N)
    # 4. Otherwise retry with a new a
    pass
```

**Impact:** breaks RSA and discrete-log cryptography, motivating post-quantum cryptography.

### Grover's Algorithm (1996)

**Purpose:** Find a marked item in an unstructured search space of size $N$.

**Physics behind it:** **Amplitude amplification.** Start in the uniform superposition

$$
|s\rangle = \frac{1}{\sqrt{N}}\sum_{x} |x\rangle,
$$

then repeatedly apply the Grover operator $\hat G = (2|s\rangle\langle s| - \mathbb{1})\,\hat O$, where the oracle $\hat O$ flips the phase of the marked state. Each iteration is a **rotation** in the two-dimensional plane spanned by the marked state and its orthogonal complement, turning the state vector a little closer to the answer. After about $\tfrac{\pi}{4}\sqrt{N}$ iterations the amplitude of the marked item is near $1$.

**Speedup:** quadratic, $O(\sqrt{N})$ versus $O(N)$. This is provably optimal for unstructured search — interference can do no better than quadratic here, in contrast to Shor's exponential gain from structure.

### Variational Quantum Eigensolver (VQE)

**Purpose:** Estimate the ground-state energy of a molecule or lattice Hamiltonian.

**Physics behind it:** the **variational principle** — for any trial state $|\psi(\boldsymbol{\theta})\rangle$, the expectation $\langle\psi(\boldsymbol{\theta})|\hat H|\psi(\boldsymbol{\theta})\rangle \ge E_0$. A parameterized quantum circuit (the *ansatz*) prepares the trial state, the quantum device measures the energy expectation by sampling the Hamiltonian's Pauli terms, and a **classical optimizer** updates $\boldsymbol{\theta}$ to lower it. This hybrid loop keeps circuits shallow, making VQE a leading near-term application on noisy hardware.

```python
def vqe_iteration(hamiltonian, ansatz, params):
    # 1. Prepare quantum state |psi(theta)>  on the device
    # 2. Measure <psi(theta)| H |psi(theta)> by sampling Pauli terms
    # 3. Classical optimizer updates theta to lower the energy
    # 4. Repeat until convergence; the minimum bounds E_0 from above
    pass
```

**Current use:** quantum chemistry, drug discovery, materials science.

### Quantum Approximate Optimization Algorithm (QAOA)

**Purpose:** Approximate solutions to combinatorial optimization (MaxCut, scheduling, portfolio selection).

**Physics behind it:** alternate between a **problem Hamiltonian** $\hat H_C$ (encoding the cost function) and a **mixing Hamiltonian** $\hat H_B$, applying $e^{-i\gamma \hat H_C}$ and $e^{-i\beta \hat H_B}$ for tunable angles. This is a discretized, variational cousin of **adiabatic** quantum computation: a $p$-layer circuit interpolates between an easy ground state and the cost ground state.

## Decoherence: Why Quantum Computers Are Hard

The clean unitary story above assumes a **closed** system. Real qubits couple to their environment, and that coupling destroys the very superpositions the algorithms exploit. This is **decoherence**, treated rigorously by [open quantum systems](qm-advanced-formalism.html#open-quantum-systems-and-the-lindblad-equation) and the Lindblad master equation.

Two time scales govern a qubit:

- **$T_1$ (energy relaxation):** the time for the excited state $|1\rangle$ to decay to $|0\rangle$ by emitting energy into the environment. On the Bloch sphere this drives $\mathbf{r}$ toward the north pole.
- **$T_2$ (phase coherence / dephasing):** the time for a definite *phase* between $|0\rangle$ and $|1\rangle$ to randomize. This shrinks the equatorial component of $\mathbf{r}$, turning a pure superposition into a classical mixture even when no energy is lost. The bound $T_2 \le 2T_1$ always holds, since energy loss necessarily destroys phase too.

A qubit initialized as $|+\rangle$ undergoes pure dephasing as

$$
\hat\rho(t) = \frac{1}{2}\begin{pmatrix} 1 & e^{-t/T_2} \\ e^{-t/T_2} & 1 \end{pmatrix}
\;\xrightarrow{\;t \gg T_2\;}\; \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix},
$$

the off-diagonal **coherences** decaying away to leave a diagonal, classical state. Decoherence is the physical mechanism behind the quantum-to-classical transition: the environment continually "measures" the qubit in the energy basis. Every gate therefore races against $T_2$, and this is why coherence time versus gate speed is the central figure of merit for hardware.

## Quantum Error Correction

If decoherence is unavoidable, fault-tolerant computing requires actively correcting errors faster than they accumulate. Classical error correction copies bits for redundancy — but the **no-cloning theorem** forbids copying an unknown quantum state, and measuring a qubit collapses it. Quantum error correction (QEC) solves both problems at once.

**The key idea:** spread the information of one **logical** qubit across many **physical** qubits, and measure carefully chosen joint observables (**stabilizers**) that reveal *whether* an error occurred and *where* — without ever measuring the encoded data itself.

The simplest illustration is the three-qubit bit-flip code, $|0\rangle_L = |000\rangle$, $|1\rangle_L = |111\rangle$. Measuring the parity operators $\hat Z_1\hat Z_2$ and $\hat Z_2\hat Z_3$ detects a single bit flip and identifies which qubit flipped — and crucially these parities commute with the logical state, so they leak no information about $\alpha$ or $\beta$. A correcting flip restores the state. Shor's nine-qubit code extends this to correct *any* single-qubit error (bit flips, phase flips, and their combination), because correcting the two generators of the Pauli group suffices to correct an arbitrary error by linearity.

**Code notation** $[[n, k, d]]$:

- $n$ = number of physical qubits,
- $k$ = number of logical qubits encoded,
- $d$ = code distance (it can correct up to $\lfloor (d-1)/2 \rfloor$ errors).

**Surface codes** are the leading approach: a logical qubit is stored in a 2D lattice of physical qubits, with stabilizers being local plaquette and vertex parity checks. They tolerate a relatively high error threshold (~1%) and need only nearest-neighbor coupling, matching superconducting hardware. The cost is overhead — on the order of $10^3$ physical qubits per logical qubit at useful distances.

**The threshold theorem.** If the physical error rate per gate is below a code-dependent **threshold** p<sub>th</sub> (≈ 1% for surface codes), then arbitrarily long, arbitrarily accurate computation is possible — the logical error rate falls exponentially as the code distance grows. This is the theoretical guarantee that fault-tolerant quantum computing is possible *in principle*; the engineering challenge is staying below threshold while scaling to millions of physical qubits.

## NISQ, Supremacy, and Advantage

**NISQ era** (Noisy Intermediate-Scale Quantum): today's machines have roughly 50–1000 physical qubits with **no** full error correction. Decoherence limits circuit depth, so the practical algorithms are shallow, error-mitigated, variational methods like VQE and QAOA.

**Quantum supremacy** (Google, 2019): the 53-qubit "Sycamore" processor sampled random circuits in ~200 seconds, a task argued to take classical supercomputers far longer. The benchmark is contrived — random circuit sampling has no practical use, and classical methods have since narrowed the gap — but it demonstrated a quantum device doing *something* beyond easy classical reach.

**Quantum advantage** is the still-pursued goal of solving a *useful* problem faster or cheaper than any classical method. Leading candidates are quantum-chemistry simulation (the natural home of VQE), certain optimization problems, and cryptanalysis (Shor, once enough error-corrected qubits exist).

**Recent milestones (2023–2024):**

- IBM Condor: 1,121 superconducting qubits.
- Atom Computing: 1,180 neutral-atom qubits.
- Google: demonstrated *below-threshold* surface-code error correction — logical error rate dropping as distance increased.

## See Also

- [Systems & Phenomena](systems-and-phenomena.html) — superposition, entanglement, and the experiments (Bell tests, Stern–Gerlach) that the qubit picture rests on.
- [States, Operators & Dynamics](formalism.html) — the Schrödinger equation, unitary evolution, and measurement postulate that gates and readout implement.
- [Computing, Information & Advanced Formalism](computing-and-advanced.html) — density matrices, open-system Lindblad dynamics, and the full quantum-information formalism behind decoherence and entanglement measures.
- [Quantum Computing](../../quantum-computing/) — the engineering and computer-science side: hardware stacks, software frameworks, and algorithms as a computing discipline.
- [Quantum Field Theory](../quantum-field-theory.html) — second quantization, the formalism underlying many physical qubit platforms.
