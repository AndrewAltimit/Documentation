---
layout: docs
title: "Quantum Mechanics: Research Frontiers"
permalink: /docs/physics/quantum-mechanics/qm-research-frontiers.html
toc: true
toc_sticky: true
---

## Research Frontiers

[Quantum Mechanics](./) &raquo; Research Frontiers

**Level and scope.** This page collects the active research directions that grow directly out of the quantum-mechanical formalism. It is graduate-level reference material and assumes comfort with Dirac notation, density matrices, and second quantization (all developed on the [States, Operators & Dynamics](formalism.html) and [Computing, Information & Advanced Formalism](computing-and-advanced.html) pages). The emphasis here is on *why* these are frontiers — what is understood, what is not, and where the field is moving.

Most of an introductory course presents quantum mechanics as a finished theory: postulates, the Schrödinger equation, a handful of solvable systems. The frontier is what happens when those rules are pushed into regimes where they are hard to solve, where new organizing principles emerge, or where the foundational assumptions themselves are stress-tested. Five themes dominate the contemporary landscape, and they are the subject of this page:

| Frontier | Core question | Key concept |
|----------|---------------|-------------|
| [Many-body physics](#many-body-quantum-physics) | How do $N \sim 10^{23}$ interacting quanta organize themselves? | Entanglement, emergence, Hilbert-space scaling |
| [Geometric &amp; Berry phases](#geometric-and-berry-phases) | What does a state remember about the path it took? | Holonomy, Berry curvature, gauge structure |
| [Topological matter](#topological-quantum-matter) | Which properties survive any smooth deformation? | Topological invariants, protected edge states |
| [Measurement-induced phenomena](#measurement-induced-phenomena) | What does monitoring a quantum system <em>do</em> to it? | Quantum trajectories, entanglement phase transitions |
| [Foundations &amp; open questions](#foundations-and-open-questions) | Is the formalism complete, and what does it mean? | Measurement problem, quantum thermodynamics |

## Many-Body Quantum Physics

A single particle in a box is an exercise; $10^{23}$ interacting particles is a research program. The difficulty is not conceptual but combinatorial: the Hilbert space of $N$ spin-½ particles has dimension $2^N$, so even storing a generic state of 300 spins would require more complex amplitudes than there are atoms in the observable universe. Many-body quantum mechanics is, at bottom, the study of how nature confines itself to the tiny, physically relevant corner of this exponentially large space — and how we can follow it there.

### Second Quantization

The natural language for many identical particles is **second quantization**, which builds the many-body Hilbert space (Fock space) from particle-number sectors:

$$
\mathcal{F} = \bigoplus_{n=0}^{\infty} \mathcal{H}^{(n)}
$$

States are created and destroyed by ladder operators whose (anti)commutation relations encode the particle statistics:

- Bosons: $[\hat a_i, \hat a_j^\dagger] = \delta_{ij}$
- Fermions: $\{\hat a_i, \hat a_j^\dagger\} = \delta_{ij}$

Field operators assemble these into position-space objects,

$$
\hat{\psi}(x) = \sum_k \phi_k(x)\, \hat{a}_k, \qquad \hat{\psi}^\dagger(x) = \sum_k \phi_k^*(x)\, \hat{a}_k^\dagger,
$$

and the generic interacting Hamiltonian — kinetic energy, external potential, and a two-body interaction — becomes

$$
\hat{H} = \int dx\; \hat{\psi}^\dagger(x)\left[-\frac{\hbar^2 \nabla^2}{2m} + V(x)\right]\hat{\psi}(x) + \frac{1}{2}\iint dx\, dy\; \hat{\psi}^\dagger(x)\hat{\psi}^\dagger(y)\, U(x-y)\, \hat{\psi}(y)\hat{\psi}(x).
$$

The antisymmetry of fermion operators automatically enforces the Pauli exclusion principle; the symmetry of boson operators allows macroscopic occupation of a single mode, the mathematical seed of Bose–Einstein condensation and superfluidity.

### The Entanglement Perspective

The modern viewpoint reframes many-body physics around **entanglement** rather than wave-function amplitudes. The key empirical fact is that the ground states of physically reasonable (local, gapped) Hamiltonians are not generic: they obey an **area law**. The entanglement entropy of a region $A$ scales with the size of its *boundary* $\partial A$, not its volume,

$$
S(\rho_A) = -\text{Tr}(\rho_A \ln \rho_A) \sim |\partial A|,
$$

whereas a randomly chosen state in the $2^N$-dimensional space would have volume-law entanglement, $S \sim |A|$. This is the deep reason simulation is ever possible: area-law states live on a thin, low-entanglement manifold that efficient representations can track.

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://arxiv.org/pdf/0906.4725.pdf"> Review: <b><i>Area Laws in Quantum Systems</i></b> - Eisert, Cramer, Plenio</a></p>

### Tensor Networks

The area law has a constructive payoff: **tensor networks** are wave-function ansätze whose entanglement is bounded by construction. A **Matrix Product State (MPS)** represents an $N$-site state as a chain of rank-3 tensors with a *bond dimension* $D$ controlling the maximum entanglement across any cut:

$$
|\Psi\rangle = \sum_{\{s\}} \text{Tr}\!\left(A^{s_1} A^{s_2}\cdots A^{s_N}\right) |s_1 s_2 \cdots s_N\rangle .
$$

The **Density Matrix Renormalization Group (DMRG)** — Steven White's 1992 algorithm, now understood as variational optimization over MPS — is the workhorse for one-dimensional systems, routinely reaching machine-precision ground-state energies. Higher-dimensional generalizations (PEPS, MERA) are an active research area precisely because their area-law structure differs and contraction is far harder.

```python
import numpy as np
import tensornetwork as tn

def create_mps(N, d, D):
    """
    Matrix Product State for a 1D chain.
    N : number of sites
    d : local (physical) dimension, e.g. 2 for spin-1/2
    D : bond dimension (the entanglement budget)
    """
    tensors = []
    for i in range(N):
        if i == 0:
            shape = (d, D)            # left boundary
        elif i == N - 1:
            shape = (D, d)            # right boundary
        else:
            shape = (D, d, D)         # bulk tensor
        tensors.append(np.random.randn(*shape))

    nodes = [tn.Node(t) for t in tensors]
    for i in range(N - 1):            # connect virtual bonds
        if i == 0:
            nodes[i][1] ^ nodes[i + 1][0]
        else:
            nodes[i][2] ^ nodes[i + 1][0]
    return nodes

def dmrg_step(mps, mpo, site):
    """
    One DMRG sweep step at a given site:
      1. contract the environment into an effective local Hamiltonian
      2. solve the local eigenvalue problem for the ground state
      3. update the on-site tensor and shift the orthogonality center
    """
    pass
```

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/ITensor/ITensors.jl"> Library: <b><i>ITensor - Tensor Network Calculations</i></b></a></p>

### Quantum Monte Carlo

Where the sign problem is mild (bosons, sign-free spin models), **quantum Monte Carlo** estimates ground-state and thermal properties by stochastic sampling. Variational Monte Carlo evaluates the energy of a trial wave function by Metropolis sampling of configurations:

```python
import numpy as np

def variational_monte_carlo(psi_trial, H, n_samples=10000):
    """Estimate <H> for a trial wave function via Metropolis sampling."""
    energy_samples = []
    config = initialize_random_config()

    for _ in range(n_samples):
        new_config = propose_move(config)
        # |psi(new)|^2 / |psi(old)|^2 is the Metropolis ratio
        prob_ratio = abs(psi_trial(new_config) / psi_trial(config))**2
        if np.random.rand() < prob_ratio:
            config = new_config
        energy_samples.append(calculate_local_energy(config, psi_trial, H))

    mean = np.mean(energy_samples)
    err  = np.std(energy_samples) / np.sqrt(n_samples)
    return mean, err
```

The unsolved obstacle that defines the frontier is the **fermion sign problem**: for fermions and frustrated magnets the sampling weights become negative, the variance grows exponentially, and the method fails. Whether a polynomial-time sign-free reformulation exists for general fermionic systems is known to be NP-hard in the worst case — one of the sharpest open problems in computational quantum physics.

### Strong Correlation and Emergence

The recurring lesson of many-body physics is **emergence**: the collective behavior is qualitatively new and not deducible by inspecting the constituents. Superconductivity (BCS pairing), the fractional quantum Hall effect (electrons binding flux to become anyons), Mott insulators, and high-$T_c$ cuprates are all phenomena with no single-particle analog. The frontier here is the regime of **strong correlation**, where perturbation theory around free particles fails outright and the very notion of a "particle" must be replaced by emergent quasiparticles — or, in non-Fermi liquids, may not survive at all.

## Geometric and Berry Phases

Run a quantum system slowly around a closed loop in parameter space and return it exactly to where it started. Does the state return unchanged? Up to its normal dynamical phase, you might expect yes. Michael Berry's 1984 insight was that it generally does **not**: the state picks up an extra, purely *geometric* phase that depends only on the shape of the loop, not on how fast it was traversed. This **Berry phase** turned out to be a unifying thread running through the Aharonov–Bohm effect, molecular physics, and — decisively — topological matter.

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://michaelberryphysics.files.wordpress.com/2013/07/berry187.pdf"> Paper: <b><i>Quantal Phase Factors Accompanying Adiabatic Changes</i></b> - Michael Berry</a></p>

### The Adiabatic Theorem and the Berry Phase

For a Hamiltonian $\hat H(R)$ depending on slowly varying parameters $R(t)$, the adiabatic theorem guarantees a system started in eigenstate $|n(R)\rangle$ stays in that instantaneous eigenstate. Carrying $R$ around a closed loop $C$ in parameter space, the accumulated geometric phase is

$$
\gamma_n = i\oint_C \langle n(R)|\nabla_R|n(R)\rangle \cdot dR .
$$

The integrand $\mathcal{A}_n(R) = i\langle n(R)|\nabla_R|n(R)\rangle$ is the **Berry connection**, and it behaves exactly like a gauge potential: a change of phase convention $|n\rangle \to e^{i\chi(R)}|n\rangle$ shifts $\mathcal{A}_n \to \mathcal{A}_n - \nabla_R \chi$, leaving the loop integral $\gamma_n$ invariant (mod $2\pi$). The Berry phase is therefore a genuine, measurable, gauge-invariant property of the path.

### Berry Curvature

By Stokes' theorem the loop integral equals the flux of a curl, defining the **Berry curvature**:

$$
\Omega_n(R) = \nabla_R \times \mathcal{A}_n(R) = \nabla_R \times \langle u_n(R)|\, i\nabla_R\, |u_n(R)\rangle .
$$

The Berry curvature acts like a "magnetic field in parameter space." For Bloch electrons it appears as an *anomalous velocity* term, $\dot{r} = \partial_k \varepsilon_n(k)/\hbar - \dot{k} \times \Omega_n(k)$, which underlies the intrinsic anomalous Hall effect. Crucially, integrating the curvature over a *closed* parameter manifold (such as the Brillouin zone of a 2D crystal) yields a quantized integer — the bridge from geometry to topology developed below.

### The Aharonov–Bohm Effect

The earliest and most striking geometric phase is the **Aharonov–Bohm effect**: a charged particle traveling around a region of magnetic flux $\Phi$ acquires a phase even though it never enters any field,

$$
\Delta\phi = \frac{e}{\hbar}\oint \mathbf{A}\cdot d\mathbf{l} = \frac{e}{\hbar}\Phi .
$$

This established that the vector potential $\mathbf{A}$ — not merely the field $\mathbf{B}$ — carries physical, observable consequences in quantum mechanics, and it is the prototype for all subsequent geometric-phase phenomena. The frontier today extends this to **non-Abelian geometric phases**, where degenerate states transform under a matrix-valued holonomy rather than a scalar phase — the working principle behind proposals for geometric (holonomic) quantum gates that are intrinsically robust to timing errors.

## Topological Quantum Matter

For most of the twentieth century, phases of matter were classified by **symmetry breaking** in the Landau paradigm: a liquid breaks no symmetry, a crystal breaks translation, a magnet breaks rotation. The discovery of the quantum Hall effect in 1980 broke the paradigm itself. Here were distinct phases with *identical* symmetry, distinguished instead by a **topological invariant** — a global integer that cannot change under any smooth deformation that keeps the energy gap open. Topological matter is the study of these phases, and it is among the most active frontiers in all of physics.

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://en.wikipedia.org/wiki/Topological_insulator"> Article: <b><i>Topological Insulator - Wikipedia</i></b></a></p>

### Topological Invariants

The central invariant for a 2D band insulator is the **Chern number**, the integral of the Berry curvature over the Brillouin zone:

$$
C = \frac{1}{2\pi}\int_{BZ} \Omega_n(k)\; d^2k \in \mathbb{Z}.
$$

It is necessarily an integer — the Berry curvature is the curvature of a fiber bundle over the torus-shaped Brillouin zone, and its integral counts the bundle's twist (a Chern–Gauss–Bonnet statement). In the integer quantum Hall effect this integer *is* the Hall conductance in units of $e^2/h$, which is why that conductance is quantized to better than one part in a billion regardless of sample disorder. For systems with time-reversal symmetry the relevant invariant is instead a $\mathbb{Z}_2$ index (even or odd), classifying topological insulators.

### Bulk–Boundary Correspondence

The defining experimental signature is the **bulk–boundary correspondence**: wherever a topologically nontrivial region meets a trivial one (or the vacuum), the invariant must change, and it can only do so by closing the gap *at the boundary*. The result is **protected edge or surface states** — conducting channels living on the surface of an insulating bulk, immune to backscattering because there is no available state to scatter into without violating the topology. These edge modes are the robust, useful face of topology.

### The Topological Zoo

| System | Invariant | Protected feature |
|--------|-----------|-------------------|
| Integer quantum Hall | Chern number $C \in \mathbb{Z}$ | Chiral edge channels; quantized $\sigma_{xy}$ |
| Topological insulator | $\mathbb{Z}_2$ index | Helical (spin-momentum-locked) surface states |
| Topological superconductor | $\mathbb{Z}$ / $\mathbb{Z}_2$ | Majorana zero modes at boundaries/vortices |
| Fractional quantum Hall | Fractional filling | Anyons with fractional charge and statistics |

Two entries point to the deepest frontiers. **Majorana zero modes** are their own antiparticles and obey non-Abelian exchange statistics; braiding them implements quantum gates that are topologically protected from local noise — the basis of the topological quantum-computing program. **Anyons** in fractional quantum Hall states carry fractional charge ($e/3$ in the $\nu = 1/3$ state, observed directly via shot noise) and accumulate fractional statistical phases under exchange, neither bosonic nor fermionic. Both are at the boundary of current experimental capability, with claims and retractions making this one of the most closely watched areas of condensed-matter research.

## Measurement-Induced Phenomena

Standard quantum mechanics treats measurement as a one-off event: the wave function collapses, an outcome is recorded, done. But what if a system is monitored *continuously*, weakly and repeatedly, throughout its evolution? This question — sharpened by experiments that can now track individual quantum systems in real time — has opened a genuinely new frontier in which measurement is not a postscript to dynamics but a *competing dynamical process* in its own right.

### Continuous Monitoring and Quantum Trajectories

A weakly, continuously monitored system does not jump discontinuously; it diffuses. Each experimental record corresponds to a **quantum trajectory**, a particular stochastic unraveling of the open-system dynamics. Averaging over all possible measurement records recovers the deterministic, decohering **Lindblad master equation**,

$$
\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat H, \hat\rho] + \sum_k \gamma_k\left(\hat L_k \hat\rho\, \hat L_k^\dagger - \tfrac{1}{2}\{\hat L_k^\dagger \hat L_k, \hat\rho\}\right),
$$

but the individual trajectories contain far more: each is a pure-state evolution conditioned on what was actually observed. This trajectory picture is now routine in circuit-QED and trapped-ion experiments, where one can watch a single qubit's state drift under continuous readout and even *reverse* an incipient collapse by feedback.

### Measurement-Induced Phase Transitions

The most surprising recent development concerns *many* monitored qubits. Take a circuit of entangling unitary gates interleaved with local projective measurements applied at rate $p$. The unitaries build entanglement; the measurements destroy it. These two tendencies compete, and as $p$ is tuned across a critical value the steady state undergoes a genuine **entanglement phase transition**:

- **Low measurement rate** — a *volume-law* phase: entanglement entropy scales with subsystem size, the system stays highly entangled and effectively scrambles information.
- **High measurement rate** — an *area-law* phase: measurements continually disentangle the system, entropy scales only with the boundary, and quantum information is rapidly destroyed.

This transition is invisible to the averaged density matrix — it lives in the structure of individual trajectories and in nonlinear functions of the state like the entanglement entropy. Discovered theoretically around 2018–2019 and now confirmed on noisy quantum processors, it established **measurement itself as a tunable phase-transition parameter**, a conceptual addition to the toolkit that did not exist a decade ago.

### Decoherence and the Measurement Boundary

Underlying all of this is **decoherence**: the loss of phase coherence as a system entangles with an unmonitored environment. It explains *why* superpositions of macroscopically distinct states are never observed — interference terms are suppressed exponentially fast — and it is the modern bridge across the measurement problem, converting quantum superpositions into apparent classical mixtures without invoking a conscious observer. The relevant timescales are

$$
T_2^{*} \;\leq\; T_2 \;\leq\; 2\,T_1 ,
$$

with $T_1$ the energy-relaxation time and $T_2$ the phase-coherence time. Decoherence is the enemy of quantum computing (it sets the error budget) and simultaneously the resource that makes the classical world emerge — a duality that keeps it at the center of both applied and foundational research.

## Foundations and Open Questions

The frontiers above are largely about *new physics* extracted from the standard formalism. A parallel frontier asks whether the formalism is **complete and correctly interpreted** — questions that were once dismissed as philosophy but are now driven by precision experiments and by the demands of quantum information.

### Quantum Thermodynamics

What do work, heat, and entropy mean for a single quantum system that may be coherent and far from equilibrium? **Quantum thermodynamics** extends the laws of thermodynamics to this regime. Work done by changing a Hamiltonian is

$$
W = \text{Tr}(\hat\rho_i \hat H_f) - \text{Tr}(\hat\rho_i \hat H_i),
$$

and fluctuation theorems (Jarzynski, Crooks) survive in quantum form. The live questions are whether quantum **coherence** and **entanglement** can be exploited as thermodynamic resources — driving heat engines beyond their classical strokes, or charging "quantum batteries" faster collectively than individually — and how the second law is to be understood for systems strongly coupled to their baths.

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://arxiv.org/pdf/1409.3435.pdf"> Review: <b><i>Quantum Thermodynamics</i></b> - Goold et al.</a></p>

### Thermalization and Its Failure

How does an isolated, unitarily evolving quantum system reach thermal equilibrium when its evolution is in principle reversible? The **Eigenstate Thermalization Hypothesis (ETH)** answers that individual energy eigenstates already look thermal for local observables, so the long-time average reproduces the microcanonical ensemble. The frontier lies in the systems that *violate* ETH and fail to thermalize:

- **Many-body localization (MBL)** — strong disorder can localize a system so completely that it never thermalizes, retaining memory of its initial state indefinitely; whether true MBL survives in the thermodynamic limit is hotly contested.
- **Quantum many-body scars** — special non-thermal eigenstates embedded in an otherwise chaotic spectrum, producing persistent revivals (first seen in Rydberg-atom arrays).
- **Floquet and time-crystal phases** — periodically driven systems that break time-translation symmetry, oscillating at a fraction of the drive frequency without heating to infinite temperature.

### The Measurement Problem and Interpretations

The unresolved conceptual core remains the **measurement problem**: unitary Schrödinger evolution never produces the single definite outcome that experiment records. Decoherence explains the *appearance* of classicality but not the selection of one branch. The competing responses are not idle metaphysics — some make distinct predictions and motivate experiments:

- **Copenhagen / standard** — collapse is a primitive, measurement is outside the dynamics.
- **Many-worlds (Everett)** — no collapse; every outcome occurs in a branch of a universal wave function.
- **Pilot-wave (de Broglie–Bohm)** — particles have definite positions guided by the wave function; explicitly nonlocal.
- **Objective collapse (GRW, CSL)** — modifies the Schrödinger equation with spontaneous localization; *experimentally testable* and being actively bounded by optomechanics experiments.
- **QBism / relational QM** — the state is information relative to an agent or system, not an objective object.

Modern foundational experiments — delayed-choice quantum erasers, Leggett–Garg and Wigner's-friend tests, loophole-free Bell violations — keep narrowing the space of viable interpretations, and the search for an objective-collapse signal is a concrete, ongoing experimental program.

### Quantum Biology

A speculative but vigorous frontier asks whether evolution exploits quantum coherence in "warm, wet, and noisy" biological environments where naïve estimates predict decoherence in femtoseconds. Candidate phenomena include:

- **Photosynthetic energy transfer** — evidence for room-temperature coherence (lasting beyond a picosecond) aiding near-unit-efficiency excitation transport in light-harvesting complexes, though its functional role is debated.
- **Avian magnetoreception** — the radical-pair mechanism in cryptochrome proteins, where electron-spin entanglement plausibly underlies birds' magnetic compass.
- **Olfaction and enzyme catalysis** — proposals of vibration-assisted electron and proton tunneling.

The unifying theoretical insight is counterintuitive: environmental noise need not only destroy coherence. **Environment-assisted quantum transport** shows that coupling to a noisy bath can *enhance* transfer efficiency by lifting destructive interference and dephasing trapping states — a mechanism actively studied with open-quantum-system methods.

## Connections to Other Fields

These frontiers do not respect disciplinary boundaries. The same mathematics recurs across physics:

- **[Quantum Field Theory](../quantum-field-theory.html)** — second quantization, vacuum fluctuations, and the renormalization group are the field-theoretic continuation of many-body quantum mechanics.
- **[Statistical Mechanics](../statistical-mechanics/)** — quantum statistics, partition functions $Z = \text{Tr}(e^{-\beta \hat H})$, and quantum phase transitions (driven by quantum, not thermal, fluctuations) tie thermalization to equilibrium theory.
- **[Condensed Matter Physics](../condensed-matter/)** — band theory, BCS superconductivity, quantum magnetism, and the topological phases above are where these ideas meet the laboratory.
- **[Quantum Computing](../../quantum-computing/)** — error correction, topological qubits, and measurement-based computation turn these frontiers into engineering targets.
- **Cosmology** — quantum fluctuations seeding cosmic structure and Hawking radiation, $T_H = \hbar c^3 / (8\pi G M k_B)$, extend the formalism to gravity.

## Research-Level Resources

### Graduate Textbooks
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.cambridge.org/highereducation/books/modern-quantum-mechanics/AAE1925F1A0963C6124421B03D7801AE"> Book: <b><i>Modern Quantum Mechanics</i></b> - J.J. Sakurai</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://link.springer.com/book/10.1007/0-306-47120-5"> Book: <b><i>Quantum Theory: Concepts and Methods</i></b> - Asher Peres</a></p>

### Research Papers and Reviews
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://arxiv.org/pdf/0906.4725.pdf"> Review: <b><i>Area Laws in Quantum Systems</i></b> - Eisert, Cramer, Plenio</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://arxiv.org/pdf/1409.3435.pdf"> Review: <b><i>Quantum Thermodynamics</i></b> - Goold et al.</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://michaelberryphysics.files.wordpress.com/2013/07/berry187.pdf"> Paper: <b><i>Quantal Phase Factors Accompanying Adiabatic Changes</i></b> - Michael Berry</a></p>

### Advanced Courses
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/play-btn-fill.svg" class="icon"><a href="https://ocw.mit.edu/courses/physics/8-05-quantum-physics-ii-fall-2013/"> Course: <b><i>MIT 8.05 Quantum Physics II</i></b></a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/play-btn-fill.svg" class="icon"><a href="https://perimeterinstitute.ca/online-courses"> Course: <b><i>Perimeter Institute - Online Physics Courses</i></b></a></p>

### Computational Resources
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/ITensor/ITensors.jl"> Library: <b><i>ITensor - Tensor Network Calculations</i></b></a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/qutip/qutip"> Library: <b><i>QuTiP - Open Quantum Systems in Python</i></b></a></p>

## See Also

- [Computing, Information &amp; Advanced Formalism](computing-and-advanced.html) — the Hilbert-space, density-matrix, and path-integral machinery these frontiers build on.
- [Systems &amp; Phenomena](systems-and-phenomena.html) — the solvable systems and quantum effects that anchor the many-body and topological discussions.
- [States, Operators &amp; Dynamics](formalism.html) — the core formalism (measurement, operators, time evolution) underlying continuous monitoring and decoherence.
- [Quantum Field Theory](../quantum-field-theory.html) — second quantization and the renormalization group developed in full.
- [Statistical Mechanics](../statistical-mechanics/) — thermalization, partition functions, and quantum phase transitions.
- [Condensed Matter Physics](../condensed-matter/) — where topological matter and strong correlation are realized experimentally.
- [Quantum Mechanics Hub](./) — back to the section overview.
