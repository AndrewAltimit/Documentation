---
layout: docs
title: "Quantum Mechanics: Computational Methods"
permalink: /docs/physics/quantum-mechanics/qm-computational-methods.html
toc: true
toc_sticky: true
hide_title: true
---

## Computational Methods

[Quantum Mechanics](./) &raquo; Computational Methods

**Level and scope.** This page is a reference for the numerical methods used to solve quantum problems that have no closed-form answer. It assumes the linear-algebra and operator language developed on the [Formalism](formalism.html) page, and complements the conceptual [Computing & Advanced Formalism](computing-and-advanced.html) page. The recurring theme is the *curse of dimensionality*: the Hilbert space of an N-particle system grows exponentially, so every method below is, at heart, a strategy for taming or sidestepping that exponential.

## The Curse of Dimensionality

A single spin-$\tfrac{1}{2}$ lives in a $2$-dimensional Hilbert space. Two spins live in $2^2 = 4$ dimensions, and $N$ spins in $2^N$. By $N = 50$ the state vector has $2^{50} \approx 10^{15}$ complex amplitudes — a petabyte of memory just to store one wavefunction. The Hamiltonian is a $2^N \times 2^N$ matrix.

$$
\dim \mathcal{H} = d^N
$$

where $d$ is the local Hilbert-space dimension. Every computational method in quantum mechanics is a response to this exponential wall:

- **Exact diagonalization** confronts it head-on, exploiting symmetry and sparsity to push the accessible $N$ as far as memory allows (typically $N \lesssim 20$–$40$ spins).
- **Tensor networks / DMRG** exploit the fact that physical ground states occupy a tiny, low-entanglement corner of Hilbert space, parameterizing the state with polynomially many numbers.
- **Quantum Monte Carlo** never stores the full state; it samples it stochastically.
- **Time-propagation methods** integrate the Schrödinger equation step by step, avoiding any global diagonalization.

The right tool depends on dimensionality, the entanglement structure, whether the model has a sign problem, and whether you want statics or dynamics.

## Exact Diagonalization

Exact diagonalization (ED) builds the Hamiltonian matrix in a finite basis and diagonalizes it numerically. It is the gold standard for small systems: it gives the full spectrum (or, with iterative solvers, the low-lying eigenstates) with no approximation beyond the truncation of the basis itself.

### Building the Hamiltonian

For a spin chain the basis states are the $2^N$ computational-basis strings $|s_1 s_2 \cdots s_N\rangle$. The Hamiltonian acts on these by local operators. For the transverse-field Ising model,

$$
\hat{H} = -J \sum_{i} \hat{\sigma}^z_i \hat{\sigma}^z_{i+1} - h \sum_i \hat{\sigma}^x_i .
$$

The $\hat{\sigma}^z\hat{\sigma}^z$ terms are diagonal; the $\hat{\sigma}^x$ terms flip a single spin and connect basis states differing in one bit. The matrix is therefore extremely **sparse** — each row has only $O(N)$ nonzero entries out of $2^N$.

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# Single-site Pauli matrices
sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
I2 = sp.identity(2, format="csr", dtype=complex)

def op_at(op, site, N):
    """Embed a single-site operator at `site` in an N-site chain via Kronecker products."""
    mats = [op if i == site else I2 for i in range(N)]
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out

def tfim_hamiltonian(N, J=1.0, h=1.0, periodic=True):
    """Transverse-field Ising model as a sparse 2^N x 2^N matrix."""
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    bonds = range(N) if periodic else range(N - 1)
    for i in bonds:
        j = (i + 1) % N
        H = H - J * (op_at(sz, i, N) @ op_at(sz, j, N))
    for i in range(N):
        H = H - h * op_at(sx, i, N)
    return H

N = 12
H = tfim_hamiltonian(N, J=1.0, h=1.0)

# Lanczos: get the lowest few eigenpairs without dense diagonalization
evals, evecs = eigsh(H, k=4, which="SA")  # smallest algebraic
print("Ground-state energy per site:", evals[0] / N)
print("Lowest 4 energies:", np.round(evals, 5))
```

### Lanczos and Sparse Solvers

Dense diagonalization (`numpy.linalg.eigh`) costs $O((d^N)^3)$ time and $O(d^{2N})$ memory — hopeless beyond $N \approx 14$ spins. The **Lanczos algorithm** instead builds a small Krylov subspace,

$$
\mathcal{K}_m = \mathrm{span}\{ |v\rangle, \hat{H}|v\rangle, \hat{H}^2|v\rangle, \ldots, \hat{H}^{m-1}|v\rangle \},
$$

in which $\hat{H}$ projects to a tridiagonal $m \times m$ matrix whose extreme eigenvalues converge to those of $\hat{H}$ after only $m \sim 100$ iterations. Each iteration needs only a sparse matrix–vector product, costing $O(N\, d^N)$ — this is what `scipy.sparse.linalg.eigsh` (and ARPACK underneath) does. Lanczos delivers the ground state and a few excited states; it does not give the full spectrum.

### Exploiting Symmetry

Conserved quantities block-diagonalize $\hat{H}$. If $[\hat{H}, \hat{Q}] = 0$, the eigenstates organize by the eigenvalue of $\hat{Q}$, and each symmetry sector can be diagonalized independently — shrinking the largest matrix you ever store. Common reductions:

- **Magnetization** $\hat{S}^z_{\text{tot}} = \sum_i \hat{\sigma}^z_i$: for a Heisenberg model this splits the $2^N$ space into sectors of fixed total $S^z$, the largest being $\binom{N}{N/2}$ — roughly a factor of $\sqrt{N}$ smaller.
- **Translation invariance**: momentum $k$ blocks reduce dimension by a factor of $N$.
- **Parity / reflection / spin-flip**: additional factors of $2$.

Combining symmetries routinely pushes ED from $N \approx 20$ to $N \approx 40$ for the largest sector — the difference between intractable and a coffee break.

**When to use ED.** Small systems, when you need the *exact* answer (full spectrum, degeneracies, dynamical correlation functions, finite-temperature properties via the full eigenbasis), or as the ground-truth benchmark for approximate methods. The hard ceiling is memory: the state vector alone is `16 × 2^N` bytes in complex128.

## Tensor Networks and DMRG

The exponential wall has a loophole: the ground states of *local, gapped* Hamiltonians are not generic vectors in Hilbert space. They obey an **area law** for entanglement entropy — the entanglement between a region and its complement scales with the boundary, not the volume. In one dimension this means a gapped ground state can be represented to high accuracy by a **Matrix Product State (MPS)** with only polynomially many parameters.

### Matrix Product States

An MPS factorizes the $d^N$ amplitudes of a state into a chain of small tensors:

$$
|\psi\rangle = \sum_{s_1 \cdots s_N} A^{s_1}_{(1)} A^{s_2}_{(2)} \cdots A^{s_N}_{(N)} \; |s_1 s_2 \cdots s_N\rangle ,
$$

where each $A^{s_i}_{(i)}$ is a $D \times D$ matrix (smaller at the boundaries) and the **bond dimension** $D$ controls how much entanglement the state can carry. The number of parameters is $O(N d D^2)$ — linear in $N$ rather than exponential. $D$ is the single knob that trades accuracy for cost: $D \to d^{N/2}$ recovers any state exactly, but in practice $D$ of a few hundred captures gapped ground states to machine precision.

The diagram for a $5$-site MPS, with physical legs pointing down and bond (virtual) legs connecting neighbors:

<div style="overflow-x:auto;">
<svg viewBox="0 0 520 160" xmlns="http://www.w3.org/2000/svg" style="max-width:560px; width:100%; height:auto; font-family:sans-serif;">
  <!-- bond legs -->
  <line x1="60" y1="70" x2="140" y2="70" stroke="#11998e" stroke-width="2.5"/>
  <line x1="140" y1="70" x2="220" y2="70" stroke="#11998e" stroke-width="2.5"/>
  <line x1="220" y1="70" x2="300" y2="70" stroke="#11998e" stroke-width="2.5"/>
  <line x1="300" y1="70" x2="380" y2="70" stroke="#11998e" stroke-width="2.5"/>
  <!-- nodes -->
  <g>
    <circle cx="60"  cy="70" r="18" fill="#38ef7d" stroke="#0b6b62" stroke-width="2"/>
    <circle cx="140" cy="70" r="18" fill="#38ef7d" stroke="#0b6b62" stroke-width="2"/>
    <circle cx="220" cy="70" r="18" fill="#38ef7d" stroke="#0b6b62" stroke-width="2"/>
    <circle cx="300" cy="70" r="18" fill="#38ef7d" stroke="#0b6b62" stroke-width="2"/>
    <circle cx="380" cy="70" r="18" fill="#38ef7d" stroke="#0b6b62" stroke-width="2"/>
  </g>
  <!-- physical legs -->
  <g stroke="#0b6b62" stroke-width="2.5">
    <line x1="60"  y1="88" x2="60"  y2="130"/>
    <line x1="140" y1="88" x2="140" y2="130"/>
    <line x1="220" y1="88" x2="220" y2="130"/>
    <line x1="300" y1="88" x2="300" y2="130"/>
    <line x1="380" y1="88" x2="380" y2="130"/>
  </g>
  <g fill="#0b6b62" font-size="14" text-anchor="middle">
    <text x="60"  y="150">s₁</text>
    <text x="140" y="150">s₂</text>
    <text x="220" y="150">s₃</text>
    <text x="300" y="150">s₄</text>
    <text x="380" y="150">s₅</text>
  </g>
  <text x="100" y="60" fill="#0b6b62" font-size="13" text-anchor="middle">D</text>
  <text x="440" y="74" fill="#555" font-size="13">bond dim D</text>
</svg>
</div>

```python
import numpy as np
import tensornetwork as tn

def create_mps(N, d, D):
    """
    Create a (random) Matrix Product State for ground-state calculation.
    N: number of sites
    d: local (physical) dimension
    D: bond dimension
    """
    # Initialize random MPS tensors (open boundary conditions)
    tensors = []
    for i in range(N):
        if i == 0:
            shape = (d, D)
        elif i == N - 1:
            shape = (D, d)
        else:
            shape = (D, d, D)
        tensors.append(np.random.randn(*shape))

    # Create tensor-network nodes
    nodes = [tn.Node(tensor) for tensor in tensors]

    # Connect virtual (bond) legs between neighbors
    for i in range(N - 1):
        if i == 0:
            nodes[i][1] ^ nodes[i + 1][0]
        else:
            nodes[i][2] ^ nodes[i + 1][0]

    return nodes
```

### Matrix Product Operators

Operators get the same treatment. A **Matrix Product Operator (MPO)** writes the Hamiltonian as a chain of rank-4 tensors. Local Hamiltonians have remarkably small MPOs — the transverse-field Ising model needs bond dimension $3$. The MPO encodes a finite-state automaton: as you sweep along the chain it is in one of a few "states" (identity-so-far, just-placed-the-first-half-of-a-bond, done), which is why nearest-neighbor models compress so tightly.

### The DMRG Algorithm

The **Density Matrix Renormalization Group (DMRG)** finds the ground-state MPS by variational sweeping. It optimizes one (or two) tensors at a time while holding the rest fixed, sweeping left-to-right and back until the energy converges. Each local update is a small eigenvalue problem solved by Lanczos.

The two-site DMRG sweep:

1. Contract everything to the left of the active sites into a left **environment** $L$, and everything to the right into a right environment $R$.
2. Form the effective Hamiltonian $H_{\text{eff}}$ acting only on the two active-site tensor, sandwiched between $L$, the local MPO tensors, and $R$.
3. Find the ground state of $H_{\text{eff}}$ by Lanczos — this is a tiny $\sim d^2 D^2$ eigenproblem.
4. **SVD** the optimized two-site tensor to split it back into two single-site tensors, truncating the bond to dimension $D$ by keeping the largest singular values. The discarded weight is a rigorous error estimate.
5. Shift one site and repeat; reverse direction at the chain ends.

$$
H_{\text{eff}} = L \;\cdot\; W_{(i)} \;\cdot\; W_{(i+1)} \;\cdot\; R
$$

The variational energy is monotonically non-increasing, and the truncation error tells you when to grow $D$. DMRG is essentially exact for 1D gapped systems and is the workhorse for quantum chemistry and condensed-matter ground states.

```python
# Variational optimization using DMRG (schematic two-site update)
def dmrg_step(mps, mpo, site):
    """
    Single two-site DMRG optimization step.

    1. Build the left/right environments around `site` and `site+1`.
    2. Form the effective Hamiltonian from (L, W[site], W[site+1], R).
    3. Lanczos-solve for the local ground state.
    4. SVD-split and truncate the two-site tensor to bond dimension D.
    5. Carry the singular values into the next site (canonical form).
    """
    L = build_left_environment(mps, mpo, site)
    R = build_right_environment(mps, mpo, site + 1)
    theta = contract_two_sites(mps, site)
    H_eff = effective_hamiltonian(L, mpo[site], mpo[site + 1], R)
    theta_gs = lanczos_ground_state(H_eff, theta)
    A, S, B = truncated_svd(theta_gs, max_bond=mps.D)
    update_mps_tensors(mps, site, A, S, B)
    return discarded_weight(S)  # convergence / truncation diagnostic
```

**Beyond 1D.** The MPS area law is special to one dimension. In 2D, ground-state entanglement grows with the boundary length, so an MPS needs `D` exponential in the width — DMRG on cylinders works for moderate widths but degrades fast. Genuinely 2D networks (PEPS, Projected Entangled Pair States) restore the area law but are far costlier to contract. The [ITensor](https://github.com/ITensor/ITensors.jl) library implements all of these.

## Quantum Monte Carlo

Quantum Monte Carlo (QMC) sidesteps the exponential wall entirely: it never stores the wavefunction, but samples observables stochastically. The error falls as $1/\sqrt{N_{\text{samples}}}$ independent of system size, so QMC scales to thousands of particles where ED and even DMRG cannot reach — at the cost of statistical noise and, sometimes, the sign problem.

### Variational Monte Carlo

Variational Monte Carlo (VMC) estimates the energy of a *trial* wavefunction $\psi_T$ by importance sampling. Writing the energy expectation in the position basis,

$$
E = \frac{\langle \psi_T | \hat{H} | \psi_T \rangle}{\langle \psi_T | \psi_T \rangle}
  = \sum_{\mathbf{x}} \frac{|\psi_T(\mathbf{x})|^2}{\sum_{\mathbf{x}'} |\psi_T(\mathbf{x}')|^2} \; E_{\text{loc}}(\mathbf{x}),
$$

with the **local energy**

$$
E_{\text{loc}}(\mathbf{x}) = \frac{\hat{H}\psi_T(\mathbf{x})}{\psi_T(\mathbf{x})} .
$$

We sample configurations $\mathbf{x}$ from the probability distribution $p(\mathbf{x}) = |\psi_T(\mathbf{x})|^2 / Z$ using the **Metropolis algorithm** and average $E_{\text{loc}}$ over the samples. The variance vanishes as $\psi_T$ approaches an exact eigenstate (the *zero-variance principle*), making $E_{\text{loc}}$ a sensitive measure of trial-function quality.

```python
import numpy as np

def variational_monte_carlo(psi_trial, H, n_samples=10000):
    """
    Variational Monte Carlo for quantum systems.

    psi_trial:  callable, psi_trial(config) -> amplitude
    H:          Hamiltonian (used inside calculate_local_energy)
    Returns the mean energy and its standard error.
    """
    energy_samples = []

    # Metropolis sampling from |psi_trial|^2
    config = initialize_random_config()

    for _ in range(n_samples):
        # Propose a move
        new_config = propose_move(config)

        # Acceptance probability = |psi(new)/psi(old)|^2
        prob_ratio = abs(psi_trial(new_config) / psi_trial(config)) ** 2

        if np.random.rand() < prob_ratio:
            config = new_config

        # Local energy at the current configuration
        E_local = calculate_local_energy(config, psi_trial, H)
        energy_samples.append(E_local)

    mean_energy = np.mean(energy_samples)
    std_error = np.std(energy_samples) / np.sqrt(n_samples)
    return mean_energy, std_error
```

### Projector and Diffusion Monte Carlo

VMC is only as good as the trial function. **Projector** methods improve on it by filtering the trial state toward the true ground state through imaginary-time evolution. Since

$$
e^{-\tau \hat{H}} |\psi_T\rangle = \sum_n e^{-\tau E_n} c_n |n\rangle \xrightarrow{\tau \to \infty} c_0 e^{-\tau E_0} |0\rangle ,
$$

the highest-energy components decay fastest, leaving the ground state. **Diffusion Monte Carlo (DMC)** realizes $e^{-\tau \hat{H}}$ as a branching random walk: the kinetic term becomes diffusion of walkers, and the potential term becomes a birth/death (branching) process. DMC is essentially exact for bosons and for the fixed-node approximation in fermionic systems.

### The Sign Problem

For fermions (and frustrated spins), the imaginary-time weights are not positive — wavefunction antisymmetry forces sign changes, and the Monte Carlo average becomes a difference of large nearly-cancelling positive and negative contributions. The statistical error then grows *exponentially* with system size and inverse temperature:

$$
\frac{\sigma}{\langle \mathrm{sign} \rangle} \sim e^{+\beta N \, \Delta f},
$$

where $\Delta f$ is a free-energy difference. This **sign problem** is the central obstacle of QMC; it is NP-hard in general, with no universal cure. The standard workarounds — the fixed-node approximation in DMC, or restricting to sign-problem-free (stoquastic) models like unfrustrated bosons and bipartite antiferromagnets — define the boundary of what QMC can reach.

**When to use QMC.** Large systems in any dimension, especially bosons, unfrustrated spins, and continuum electronic-structure problems where a good trial wavefunction exists. Avoid it (or accept the fixed-node bias) for frustrated magnets and generic fermions, where the sign problem bites.

## Time Propagation Methods

Static methods find eigenstates; **dynamics** asks how a state evolves. The formal solution of the time-dependent Schrödinger equation,

$$
i\hbar \, \frac{\partial}{\partial t} |\psi(t)\rangle = \hat{H}(t)\, |\psi(t)\rangle ,
$$

is the propagator $|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle$. For a time-independent $\hat{H}$, $\hat{U}(t) = e^{-i\hat{H}t/\hbar}$; the numerical task is to apply this exponential without ever forming the full matrix exponential.

### Direct Integration

The most direct route is to treat the Schrödinger equation as a system of coupled ODEs and hand it to a standard integrator (Runge–Kutta, adaptive step). This is flexible — it handles arbitrary time-dependence — but a naive RK scheme is not unitary, so the norm drifts and energy is not conserved over long times unless steps are kept small.

```python
import numpy as np
from scipy.integrate import solve_ivp

def schrodinger_rhs(t, psi, H):
    """d|psi>/dt = -(i/hbar) H |psi>, with hbar = 1."""
    return -1j * (H @ psi)

def integrate_state(H, psi0, t_span, t_eval):
    sol = solve_ivp(
        schrodinger_rhs, t_span, psi0, t_eval=t_eval,
        args=(H,), method="RK45", rtol=1e-9, atol=1e-11,
    )
    return sol.t, sol.y  # columns are |psi(t)>
```

### Split-Operator and Trotter Methods

For a Hamiltonian that splits into kinetic and potential parts, $\hat{H} = \hat{T} + \hat{V}$, the **split-operator** method applies each piece where it is diagonal — $\hat{V}$ in position space, $\hat{T}$ in momentum space — alternating via the FFT. The Suzuki–Trotter decomposition,

$$
e^{-i\hat{H}\,\Delta t/\hbar} \approx e^{-i\hat{V}\Delta t/2\hbar}\; e^{-i\hat{T}\Delta t/\hbar}\; e^{-i\hat{V}\Delta t/2\hbar} + O(\Delta t^3),
$$

is **exactly unitary** at every step (each factor is the exponential of a Hermitian operator), so the norm is conserved by construction. The symmetric (Strang) form above is second-order accurate per step. The same Trotter idea underlies **TEBD** (Time-Evolving Block Decimation), which propagates an MPS by applying two-site gates and re-truncating — the dynamical counterpart of DMRG.

```python
import numpy as np

def split_operator_step(psi, V, k2, dt, hbar=1.0, m=1.0):
    """
    One symmetric (Strang) split-operator step for H = p^2/2m + V(x)
    on a uniform grid. `k2` is the array of k^2 momentum values.
    """
    # Half step in potential (position space)
    psi = np.exp(-1j * V * dt / (2 * hbar)) * psi
    # Full step in kinetic energy (momentum space via FFT)
    psi_k = np.fft.fft(psi)
    psi_k = np.exp(-1j * hbar * k2 * dt / (2 * m)) * psi_k
    psi = np.fft.ifft(psi_k)
    # Half step in potential again
    psi = np.exp(-1j * V * dt / (2 * hbar)) * psi
    return psi
```

### Krylov-Subspace Propagation

When $\hat{H}$ is large and sparse, the most accurate per-step method is **Krylov-subspace (Lanczos) propagation**: build a small Krylov space from $|\psi(t)\rangle$, exponentiate the resulting tiny tridiagonal matrix exactly, and project back. A subspace of dimension $m \sim 20$–$40$ typically reaches machine precision for a single time step, with the error controllable a posteriori. This is what production codes (e.g. `expokit`, `scipy.sparse.linalg.expm_multiply`) use to apply $e^{-i\hat{H}t}|\psi\rangle$ without forming the exponential.

```python
import numpy as np
from scipy.sparse.linalg import expm_multiply

def krylov_propagate(H, psi0, times):
    """
    Apply exp(-i H t) |psi0> at a sequence of times using a Krylov method.
    `H` is a sparse Hamiltonian (hbar = 1).
    """
    # expm_multiply efficiently builds exp(A) @ b across a time grid
    start, stop = times[0], times[-1]
    num = len(times)
    return expm_multiply(-1j * H, psi0, start=start, stop=stop, num=num)
```

### Driven Systems and Floquet Theory

When the Hamiltonian is periodic in time, $\hat{H}(t + T) = \hat{H}(t)$, **Floquet theory** is the time-domain analogue of Bloch's theorem. The one-period propagator $\hat{U}(T)$ has eigenvalues $e^{-i\varepsilon_n T/\hbar}$ defining **quasienergies** $\varepsilon_n$ (defined modulo $\hbar\omega = 2\pi\hbar/T$) and time-periodic **Floquet modes**. Diagonalizing $\hat{U}(T)$ once captures the long-time stroboscopic dynamics of a periodically driven system — the basis for Floquet engineering of effective Hamiltonians.

```python
import numpy as np
from scipy import linalg
import qutip as qt

def time_dependent_hamiltonian(t, args):
    """Driven two-level / oscillator Hamiltonian H0 + H1 cos(omega t)."""
    H0 = args["H0"]
    H1 = args["H1"]
    omega = args["omega"]
    return H0 + H1 * np.cos(omega * t)

def floquet_modes(H_func, T, args):
    """
    Compute Floquet modes and quasienergies for a T-periodic Hamiltonian.
    """
    # One-period propagator U(T)
    U = qt.propagator(H_func, T, args=args)

    # Diagonalize the Floquet operator
    evals, evecs = linalg.eig(U.full())

    # Quasienergies (defined modulo hbar*omega = 2*pi/T)
    epsilon = -np.angle(evals) / T

    return epsilon, evecs
```

### Open Systems and Lindblad Dynamics

Closed-system propagation conserves purity; real systems couple to an environment. For Markovian dissipation the density matrix obeys the **Lindblad master equation**,

$$
\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}] + \sum_k \gamma_k \left( \hat{L}_k \hat{\rho}\hat{L}_k^\dagger - \frac{1}{2}\{ \hat{L}_k^\dagger \hat{L}_k, \hat{\rho} \} \right),
$$

which is propagated either by integrating the $d^2$-dimensional vectorized $\hat{\rho}$ directly (`mesolve` below) or by averaging over **quantum trajectories** — stochastic pure-state evolutions punctuated by random jumps, which scale as $d$ rather than $d^2$.

## A Worked Example with QuTiP

[QuTiP](https://github.com/qutip/qutip) bundles many of these methods behind a clean interface. The example below builds a two-level Hamiltonian, evolves an initial state, and tracks an expectation value — exercising the open-system `mesolve` propagator on a trivially small system so the mechanics are visible.

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/qutip/qutip"> Library: <b><i>QuTiP - Quantum Toolbox in Python</i></b></a></p>

```python
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Create a two-level system (qubit)
N = 2
a = destroy(N)

# Define the Hamiltonian
w0 = 1.0  # frequency
g = 0.1   # coupling strength
H = w0 * a.dag() * a + g * (a + a.dag())

# Initial state (ground state)
psi0 = basis(N, 0)

# Time evolution
times = np.linspace(0, 50, 500)
result = mesolve(H, psi0, times, [], [])

# Expectation value of the number operator
n_exp = expect(a.dag() * a, result.states)

# Visualize the evolution
plt.figure(figsize=(10, 6))
plt.plot(times, n_exp)
plt.xlabel("Time")
plt.ylabel("Excitation Probability")
plt.title("Driven Two-Level Evolution")
plt.grid(True)
plt.show()
```

### Visualizing Wave Functions

The harmonic-oscillator eigenstates are a useful sanity check: they have a closed form, so plotting the numerically evaluated $\psi_n(x)$ confirms a code's normalization and basis conventions before turning it loose on harder problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial

def quantum_harmonic_oscillator(x, n, m=1, w=1, hbar=1):
    """Wave function of the quantum harmonic oscillator.
    Returns a properly normalized psi_n with integral |psi|^2 dx = 1.
    """
    # Length scale
    x0 = np.sqrt(hbar / (m * w))

    # Normalization constant
    C = 1 / np.sqrt(2**n * factorial(n)) * (m * w / (np.pi * hbar)) ** 0.25

    # Hermite polynomial
    H = hermite(n)

    # Wave function
    psi = C * np.exp(-m * w * x**2 / (2 * hbar)) * H(x / x0)
    return psi

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(12, 8))
for n in range(5):
    psi = quantum_harmonic_oscillator(x, n)
    plt.subplot(2, 3, n + 1)
    plt.plot(x, psi, "b", linewidth=2)
    plt.fill_between(x, 0, psi, alpha=0.3)
    plt.title(f"n = {n}")
    plt.xlabel("Position")
    plt.ylabel("psi(x)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)

plt.tight_layout()
plt.show()
```

<center>
<p class="referenceBoxes type2">
<a href="https://qutip.org/docs/latest/guide/guide-basics.html">
<img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"> Tutorial: <b><i>QuTiP Basics - Quantum System Simulation</i></b></a>
</p>
</center>

## Choosing a Method

<table>
  <thead>
    <tr><th>Method</th><th>Best for</th><th>Cost / limit</th><th>Approximation</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Exact diagonalization</td>
      <td>Small systems; full spectrum; ground-truth benchmarks</td>
      <td>Exponential in N (memory ceiling ~20–40 spins with symmetry)</td>
      <td>None (basis truncation only)</td>
    </tr>
    <tr>
      <td>DMRG / MPS</td>
      <td>1D gapped ground states &amp; low-lying spectrum; quantum chemistry</td>
      <td>Polynomial in N; cost grows with entanglement (bond dim D)</td>
      <td>Controlled by truncation / discarded weight</td>
    </tr>
    <tr>
      <td>TEBD / tensor-network dynamics</td>
      <td>1D real-time evolution at short-to-moderate times</td>
      <td>Fails once entanglement grows linearly in time</td>
      <td>Truncation error per step</td>
    </tr>
    <tr>
      <td>Quantum Monte Carlo</td>
      <td>Large systems, any D; bosons, unfrustrated spins, electronic structure</td>
      <td>Statistical error ~1/√N_samples; sign problem for fermions/frustration</td>
      <td>Statistical; fixed-node bias for fermions</td>
    </tr>
    <tr>
      <td>Time propagation (split-op / Krylov)</td>
      <td>Real- and imaginary-time dynamics; driven and open systems</td>
      <td>Per-step cost of a sparse matrix–vector product</td>
      <td>Trotter / Krylov truncation, controllable</td>
    </tr>
  </tbody>
</table>

## Computational Resources

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/qutip/qutip"> Library: <b><i>QuTiP - Quantum Toolbox in Python</i></b></a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/ITensor/ITensors.jl"> Library: <b><i>ITensor - Tensor Network &amp; DMRG Calculations</i></b></a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/tenpy/tenpy"> Library: <b><i>TeNPy - Tensor Networks in Python (DMRG, TEBD)</i></b></a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://arxiv.org/pdf/1008.3477.pdf"> Review: <b><i>The density-matrix renormalization group in the age of matrix product states</i></b> - Ulrich Schollwöck</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://arxiv.org/pdf/0906.4725.pdf"> Review: <b><i>Area Laws in Quantum Systems</i></b> - Eisert, Cramer, Plenio</a></p>

---

## See Also

- [Computing, Information &amp; Advanced Formalism](computing-and-advanced.html) — quantum computing, density matrices, and the graduate mathematical machinery these methods rest on.
- [Systems &amp; Phenomena](systems-and-phenomena.html) — the solvable systems used to benchmark these numerical methods.
- [Formalism](formalism.html) — the Hilbert-space and operator language underlying every algorithm here.
- [Quantum Mechanics Hub](./) — the full quantum-mechanics reference.
- [Statistical Mechanics](../statistical-mechanics/) — partition functions and finite-temperature methods that extend these techniques to thermal ensembles.
