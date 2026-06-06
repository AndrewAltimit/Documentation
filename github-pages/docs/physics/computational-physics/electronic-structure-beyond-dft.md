---
layout: docs
title: "Computational Physics: Electronic Structure Beyond DFT"
permalink: /docs/physics/computational-physics/electronic-structure-beyond-dft.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Electronic Structure Beyond DFT</p>

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Electronic Structure Beyond DFT</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Wavefunction methods — Hartree-Fock, MP2, configuration interaction, coupled cluster — and excited states with TD-DFT, each trading cost for accuracy.</p>
</div>

<div class="notice--info">
  <p>This page complements <a href="quantum-methods.html">Quantum Computational Methods</a>, which covers density functional theory (DFT) and real-time wavefunction propagation. Here we focus on the <em>ab initio</em> wavefunction hierarchy that systematically improves on (or benchmarks) DFT, plus the time-dependent DFT route to excited states. Read the DFT material first if you have not — the two approaches are complementary, and practitioners routinely cross-check one against the other.</p>
</div>

## Why Go Beyond DFT?

Density functional theory is the workhorse of computational materials science and quantum chemistry because it delivers useful accuracy at a cost that scales as roughly $O(N^3)$ in the number of electrons. But DFT has a structural weakness: the **exact exchange-correlation functional is unknown**, and every practical functional (LDA, PBE, B3LYP, …) is an approximation with uncontrolled error. You cannot, in general, make a DFT result more accurate by spending more compute — there is no convergent ladder to the exact answer.

Wavefunction methods take the opposite stance. They start from a well-defined approximation (Hartree-Fock) and add electron correlation through a **systematically improvable hierarchy**. Spend more compute, get provably closer to the exact solution of the electronic Schrödinger equation within a given basis. The price is steep scaling — anywhere from $O(N^5)$ to $O(N^7)$ or worse — so these methods are reserved for small molecules, careful benchmarks, and cases where DFT is known to fail (dispersion-dominated complexes, bond breaking, near-degenerate states, excited states with charge transfer).

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-layer-group"></i>
    <h4>A convergent hierarchy</h4>
    <p>HF → MP2 → CCSD → CCSD(T) → Full CI climbs systematically toward the exact answer in a fixed basis, unlike DFT's non-convergent functional zoo.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-coins"></i>
    <h4>Cost buys accuracy</h4>
    <p>Each rung adds correlation but multiplies cost: $O(N^4)$ for HF, $O(N^5)$ for MP2, $O(N^7)$ for CCSD(T), factorial for Full CI.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-puzzle-piece"></i>
    <h4>Basis sets matter</h4>
    <p>Correlation energy converges slowly with basis size; correlation-consistent sets and extrapolation are essential for benchmark accuracy.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-bolt"></i>
    <h4>Excited states need new tools</h4>
    <p>Ground-state DFT says nothing about spectra; TD-DFT, EOM-CC, and CI give absorption energies and oscillator strengths.</p>
  </div>
</div>

## The Electronic Structure Problem

Within the Born-Oppenheimer approximation the nuclei are clamped and we solve for the electrons in their field. The non-relativistic electronic Hamiltonian (atomic units, with $\hbar = m_e = e = 1$) is

$$
\hat{H} = -\frac{1}{2}\sum_i \nabla_i^2 \;-\; \sum_{i,A}\frac{Z_A}{r_{iA}} \;+\; \sum_{i<j}\frac{1}{r_{ij}}
$$

The first term is the electronic kinetic energy, the second the electron-nucleus attraction, and the third the electron-electron repulsion. That last term is the whole difficulty: it couples every electron to every other, so the exact wavefunction is not a product of one-electron orbitals. The exact ground state $\Psi(\mathbf{r}_1, \dots, \mathbf{r}_N)$ lives in $3N$-dimensional space and cannot be solved analytically for more than one electron.

Every method below is a strategy for approximating $\Psi$ and the energy $E = \langle \Psi | \hat{H} | \Psi \rangle$. The variational principle guarantees that for any normalized trial $\Psi$,

$$
E[\Psi] = \langle \Psi | \hat{H} | \Psi \rangle \;\geq\; E_0
$$

so a method that minimizes this expectation over a class of trial functions gives an upper bound to the true ground-state energy $E_0$.

## Hartree-Fock: The Mean-Field Starting Point

Hartree-Fock (HF) approximates the many-electron wavefunction by a **single Slater determinant** — an antisymmetrized product of one-electron spin-orbitals $\chi_i$:

$$
\Psi_{\mathrm{HF}}(\mathbf{x}_1, \dots, \mathbf{x}_N) = \frac{1}{\sqrt{N!}}
\begin{vmatrix}
\chi_1(\mathbf{x}_1) & \chi_2(\mathbf{x}_1) & \cdots & \chi_N(\mathbf{x}_1) \\
\chi_1(\mathbf{x}_2) & \chi_2(\mathbf{x}_2) & \cdots & \chi_N(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\chi_1(\mathbf{x}_N) & \chi_2(\mathbf{x}_N) & \cdots & \chi_N(\mathbf{x}_N)
\end{vmatrix}
$$

The determinant form enforces antisymmetry (the Pauli principle) automatically: swapping two electrons swaps two rows and flips the sign. Minimizing the energy over the orbitals subject to orthonormality yields the **Hartree-Fock equations**, a set of coupled eigenvalue problems for an effective one-electron operator, the **Fock operator** $\hat{F}$:

$$
\hat{F}\,\chi_i = \varepsilon_i\,\chi_i, \qquad
\hat{F} = \hat{h} + \sum_j \left( \hat{J}_j - \hat{K}_j \right)
$$

Here $\hat{h}$ is the one-electron (kinetic + nuclear attraction) operator, $\hat{J}_j$ is the **Coulomb operator** (the classical electrostatic repulsion from the charge cloud of orbital $j$), and $\hat{K}_j$ is the **exchange operator**, a purely quantum term with no classical analogue that arises from antisymmetry. Because $\hat{F}$ depends on the occupied orbitals it solves for, HF is a **self-consistent field (SCF)** problem, solved iteratively much like the Kohn-Sham loop in [DFT](quantum-methods.html#density-functional-theory-dft).

In a finite basis of $K$ atomic functions, the HF equations become the matrix **Roothaan-Hall equations**

$$
\mathbf{F}\,\mathbf{C} = \mathbf{S}\,\mathbf{C}\,\boldsymbol{\varepsilon}
$$

where $\mathbf{F}$ is the Fock matrix, $\mathbf{S}$ the overlap matrix (atomic orbitals are not orthogonal), $\mathbf{C}$ the orbital coefficients, and $\boldsymbol{\varepsilon}$ the orbital energies. This is a generalized eigenvalue problem solved at each SCF iteration.

### What Hartree-Fock Captures and Misses

HF accounts for **exchange exactly** but treats electron-electron repulsion only in a **mean-field** way: each electron moves in the *average* field of the others. It misses the instantaneous correlation in their motion — electrons "dodging" one another. The difference between the exact non-relativistic energy (in a complete basis) and the HF energy is, by definition, the **correlation energy**:

$$
E_{\mathrm{corr}} = E_{\mathrm{exact}} - E_{\mathrm{HF}}
$$

It is always negative (correlation lowers the energy) and typically only about 1% of the total electronic energy — but that small fraction is decisive for chemistry: bond energies, reaction barriers, and intermolecular interactions all live in the correlation energy. Recovering $E_{\mathrm{corr}}$ efficiently is the entire business of post-HF methods.

```python
# Hartree-Fock with PySCF: the canonical starting point
from pyscf import gto, scf

# Define the water molecule (geometry in Angstrom)
mol = gto.M(
    atom='''
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
    ''',
    basis='cc-pVDZ',   # correlation-consistent double-zeta basis
    verbose=0,
)

# Restricted Hartree-Fock (closed shell)
mf = scf.RHF(mol)
e_hf = mf.kernel()
print(f"Hartree-Fock energy: {e_hf:.6f} Hartree")

# The converged MO coefficients and energies feed every post-HF method
mo_energies = mf.mo_energy      # orbital energies (eps_i)
mo_coeff = mf.mo_coeff          # AO -> MO transformation matrix
homo = mol.nelectron // 2 - 1
print(f"HOMO-LUMO gap: {mo_energies[homo+1] - mo_energies[homo]:.4f} Hartree")
```

## MP2: Perturbative Correlation, Cheaply

The cheapest way to add correlation is **Møller-Plesset perturbation theory**, treating the difference between the true electron repulsion and the HF mean field as a perturbation $\hat{V} = \hat{H} - \hat{F}$. The first nonvanishing correction to the energy appears at **second order (MP2)**:

$$
E^{(2)} = \sum_{i<j}^{\mathrm{occ}} \sum_{a<b}^{\mathrm{virt}}
\frac{\left| \langle ij \,\|\, ab \rangle \right|^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
$$

The sum runs over pairs of occupied orbitals $i,j$ excited into pairs of virtual (unoccupied) orbitals $a,b$. The numerator $\langle ij \| ab \rangle$ is an antisymmetrized two-electron integral coupling the occupied and virtual orbitals; the denominator is the orbital-energy gap for the double excitation. Physically, MP2 sums the leading **double excitations** — the lowest-order way electrons correlate their motion.

MP2 scales as $O(N^5)$ and recovers a large fraction (often 80-90%) of the correlation energy for well-behaved closed-shell systems. It is the workhorse for **non-covalent interactions and dispersion**, where standard DFT functionals struggle. Its weaknesses: it can badly overestimate dispersion for large $\pi$ systems, it diverges for systems with small HOMO-LUMO gaps (the denominator approaches zero), and as a non-variational method it does not bound the true energy.

```python
from pyscf import mp

# MP2 correlation on top of the converged HF reference
mp2 = mp.MP2(mf)
e_corr_mp2, t2 = mp2.kernel()
print(f"MP2 correlation energy: {e_corr_mp2:.6f} Hartree")
print(f"Total MP2 energy:       {e_hf + e_corr_mp2:.6f} Hartree")
```

## Configuration Interaction

**Configuration interaction (CI)** abandons the single-determinant picture entirely. It writes the wavefunction as a linear combination of the HF determinant $\Phi_0$ and excited determinants built by promoting electrons from occupied to virtual orbitals:

$$
\Psi_{\mathrm{CI}} = c_0 \Phi_0 + \sum_{ia} c_i^a \Phi_i^a + \sum_{i<j,\,a<b} c_{ij}^{ab}\,\Phi_{ij}^{ab} + \cdots
$$

The coefficients $c$ are found variationally by diagonalizing the Hamiltonian in the space of determinants — so CI energies are rigorous **upper bounds** to the exact energy.

- **CISD** (singles + doubles) truncates the expansion at double excitations. It scales as $O(N^6)$ and recovers much of the correlation energy, but suffers from a critical flaw: it is **not size-consistent**. The energy of two non-interacting fragments computed together does not equal the sum of the fragment energies, so CISD errors grow with system size. This defect is why CISD has largely been superseded by coupled cluster.

- **Full CI (FCI)** includes *every* possible excitation. It gives the **exact** answer within the chosen basis — the gold-standard benchmark — but the number of determinants grows factorially. FCI is feasible only for tiny systems (a handful of electrons in a modest basis) and exists mainly to calibrate cheaper methods.

Modern **selected CI** and **density matrix renormalization group (DMRG)** methods push the FCI frontier much further by including only the most important determinants, making near-exact treatment of strongly correlated, multireference systems (transition-metal complexes, bond dissociation) tractable.

```python
from pyscf import fci

# Full CI in an active space (here, all orbitals for a small molecule)
cisolver = fci.FCI(mf)
e_fci, fcivec = cisolver.kernel()
print(f"Full CI energy: {e_fci:.6f} Hartree (exact within the basis)")
print(f"Correlation recovered vs HF: {e_fci - e_hf:.6f} Hartree")
```

## Coupled Cluster: The Gold Standard

**Coupled cluster (CC)** theory fixes CI's size-consistency problem with an **exponential ansatz**:

$$
\Psi_{\mathrm{CC}} = e^{\hat{T}}\,\Phi_0, \qquad
\hat{T} = \hat{T}_1 + \hat{T}_2 + \cdots
$$

The cluster operator $\hat{T}$ generates excitations, and the exponential automatically produces *products* of excitations (so-called disconnected terms) that make the method **size-consistent and size-extensive** even when $\hat{T}$ is truncated. Truncating at doubles gives **CCSD** (coupled cluster with singles and doubles), scaling as $O(N^6)$.

The defining method of modern quantum chemistry is **CCSD(T)** — CCSD plus a perturbative estimate of connected triple excitations, scaling as $O(N^7)$. It is so reliable for ground-state thermochemistry of single-reference molecules that it is called the **"gold standard"**: with a large basis it routinely reaches **"chemical accuracy"** (errors below 1 kcal/mol ≈ 0.0016 Hartree). The amplitudes in CCSD are found by projecting the Schrödinger equation onto excited determinants, giving coupled nonlinear equations:

$$
\langle \Phi_{ij}^{ab} | e^{-\hat{T}}\,\hat{H}\,e^{\hat{T}} | \Phi_0 \rangle = 0
$$

The caveat: CCSD(T) assumes a **single dominant reference determinant**. For strongly correlated systems — stretched bonds, biradicals, many transition metals — multiple determinants matter and CCSD(T) can fail dramatically, sometimes giving nonsensical (even imaginary) triples corrections. There the multireference methods (CASSCF, CASPT2, DMRG, selected CI) take over.

```python
from pyscf import cc

# CCSD on the HF reference
mycc = cc.CCSD(mf)
e_corr_ccsd, t1, t2 = mycc.kernel()
print(f"CCSD correlation energy:  {e_corr_ccsd:.6f} Hartree")
print(f"Total CCSD energy:        {e_hf + e_corr_ccsd:.6f} Hartree")

# Perturbative triples -> CCSD(T), the gold standard
e_t = mycc.ccsd_t()
print(f"(T) triples correction:   {e_t:.6f} Hartree")
print(f"Total CCSD(T) energy:     {e_hf + e_corr_ccsd + e_t:.6f} Hartree")
```

## Excited States: TD-DFT and Beyond

Everything above targets the **ground state**. Spectroscopy — UV/Vis absorption, fluorescence, photochemistry — needs **excited states**, which ground-state DFT and ground-state CCSD(T) do not provide directly.

### Time-Dependent DFT (TD-DFT)

The most widely used excited-state method is **time-dependent density functional theory (TD-DFT)**. In its standard linear-response formulation, one looks at how the ground-state density responds to a weak oscillating perturbation; the poles of the response function are the excitation energies. In a basis this reduces to the non-Hermitian **Casida eigenvalue equation**:

$$
\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^{*} & \mathbf{A}^{*} \end{pmatrix}
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}
= \omega
\begin{pmatrix} \mathbf{1} & \mathbf{0} \\ \mathbf{0} & -\mathbf{1} \end{pmatrix}
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}
$$

where the eigenvalues $\omega$ are the excitation energies and the eigenvectors $(\mathbf{X}, \mathbf{Y})$ give the transition densities (hence oscillator strengths and spectra). The matrices $\mathbf{A}$ and $\mathbf{B}$ are built from orbital-energy differences and exchange-correlation kernel couplings. Setting $\mathbf{B} = 0$ gives the cheaper, often more robust **Tamm-Dancoff approximation (TDA)**.

TD-DFT scales like ground-state DFT (roughly $O(N^3)$-$O(N^4)$) and is the default for organic chromophores and large molecules. Its failures are well-catalogued and mirror DFT's: standard functionals badly underestimate **charge-transfer** and **Rydberg** excitations (range-separated hybrids like CAM-B3LYP fix much of this), struggle with **double excitations** (absent from the adiabatic approximation), and inherit any functional-dependent error.

The real-time alternative — propagating the density under a time-dependent field and Fourier-transforming the dipole response — is covered alongside wavefunction propagation in [Quantum Computational Methods](quantum-methods.html#time-dependent-schrödinger-equation).

```python
from pyscf import dft, tddft

# Ground-state DFT reference (B3LYP hybrid functional)
mf_dft = dft.RKS(mol)
mf_dft.xc = 'b3lyp'
mf_dft.kernel()

# Linear-response TD-DFT for the lowest excited states
td = tddft.TDDFT(mf_dft)
td.nstates = 5
excitation_energies = td.kernel()[0]

# Convert Hartree to eV and report with oscillator strengths
hartree_to_ev = 27.2114
osc = td.oscillator_strength()
for i, (e, f) in enumerate(zip(excitation_energies, osc), start=1):
    print(f"State {i}: {e * hartree_to_ev:6.3f} eV   f = {f:.4f}")
```

### Wavefunction Excited States: EOM-CC and Friends

When TD-DFT is not trustworthy, **equation-of-motion coupled cluster (EOM-CCSD)** provides benchmark excited states. It applies an excitation operator to the CCSD ground state and diagonalizes the similarity-transformed Hamiltonian in the space of excited determinants, scaling as $O(N^6)$. Different flavors target different states: **EE-EOM** for neutral excitations, **IP/EA-EOM** for ionization and electron attachment. Multireference perturbation theories (**CASPT2**, **NEVPT2**) built on a **CASSCF** reference handle excited states that are inherently multiconfigurational, such as those in transition-metal photochemistry.

## Cost vs. Accuracy: The Whole Picture

The unifying theme is a **cost-accuracy trade-off**. Each step up the hierarchy recovers more correlation energy but raises the formal scaling with system size $N$ (number of basis functions / electrons).

| Method | Recovers correlation? | Formal scaling | Variational? | Size-consistent? | Typical use |
|--------|----------------------|----------------|--------------|------------------|-------------|
| Hartree-Fock | No (mean field only) | $O(N^4)$ | Yes | Yes | Reference for all post-HF methods |
| DFT (KS) | Approximately | $O(N^3)$ | Yes | Yes | Default for materials & large molecules |
| MP2 | ~80-90% (single ref) | $O(N^5)$ | No | Yes | Dispersion, non-covalent interactions |
| CISD | Partial | $O(N^6)$ | Yes | **No** | Mostly historical |
| CCSD | Most | $O(N^6)$ | No | Yes | Reliable correlated energies |
| CCSD(T) | Near-exact (single ref) | $O(N^7)$ | No | Yes | "Gold standard" thermochemistry |
| Full CI | Exact (in basis) | factorial | Yes | Yes | Benchmark only, tiny systems |
| TD-DFT | Approximately | $\sim O(N^3$-$N^4)$ | — | — | Excited states, spectra |
| EOM-CCSD | Most (excited) | $O(N^6)$ | — | Yes | Benchmark excited states |

Two practical caveats sit beneath this table:

- **Basis-set convergence.** Correlation energy converges painfully slowly with basis size — roughly as $X^{-3}$ for a correlation-consistent set of cardinal number $X$ (cc-pVDZ, cc-pVTZ, cc-pVQZ, …). Benchmark CCSD(T) numbers almost always involve **complete-basis-set (CBS) extrapolation** from at least two basis sets, often combined with explicitly correlated **F12** methods that converge far faster.

- **Single- vs. multireference.** The whole single-reference hierarchy (MP2, CCSD(T)) presumes one determinant dominates. The diagnostic $T_1$ from a CCSD calculation flags trouble: $T_1 > 0.02$ warns that the system is multireference and CCSD(T) may be unreliable, signalling a switch to CASSCF/CASPT2, DMRG, or selected CI.

<div class="notice--success">
  <p><strong>Rule of thumb.</strong> For a routine ground-state energy of an organic molecule, DFT is the pragmatic default. When DFT is suspect (dispersion complexes, reaction barriers, conformers within a few kcal/mol), validate against MP2 or CCSD(T)/CBS. For excited states, start with TD-DFT and a range-separated hybrid; escalate to EOM-CCSD or CASPT2 only when TD-DFT's known failure modes apply.</p>
</div>

## The Software Ecosystem

The methods above are implemented in a rich ecosystem of quantum-chemistry packages. The major distinction is **Gaussian-type orbital (GTO)** codes — dominant in molecular quantum chemistry — versus **plane-wave / projector-augmented-wave** codes used for periodic solids.

| Package | License | Strengths | Notes |
|---------|---------|-----------|-------|
| **PySCF** | Open source | HF, DFT, MP2, CC, CASSCF, FCI; Python-native, scriptable | Used in the examples here; excellent for prototyping and research |
| **Psi4** | Open source | CCSD(T), SAPT, DFT; clean Python API | Strong for non-covalent interaction analysis |
| **ORCA** | Free (academic) | DLPNO-CCSD(T), multireference, broad method coverage | Local-correlation CC scales to hundreds of atoms |
| **Gaussian** | Commercial | Long-standing, comprehensive; widely cited | The historical reference implementation |
| **MOLPRO** | Commercial | High-accuracy CC, explicitly correlated F12, multireference | Benchmark-quality wavefunction methods |
| **NWChem** | Open source | Massively parallel HF/DFT/CC | Built for HPC and large systems |
| **Quantum ESPRESSO / VASP** | Open / Commercial | Plane-wave DFT for periodic solids | The materials-science counterpart to the molecular codes |

A key modern development is **local-correlation coupled cluster** (DLPNO-CCSD(T) in ORCA, LCCSD in others), which exploits the short range of correlation to reduce the scaling of "gold-standard" CC from $O(N^7)$ toward near-linear for large molecules — bringing CCSD(T)-quality energies to systems of hundreds of atoms that were unthinkable a decade ago. Together with F12 explicitly correlated methods (which slash basis-set error), local CC has dramatically widened the range of chemistry where benchmark accuracy is affordable.

## See Also

- [Quantum Computational Methods](quantum-methods.html) — DFT, the Kohn-Sham self-consistent field, and real-time wavefunction propagation that this page builds on.
- [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) — variational and diffusion quantum Monte Carlo, an alternative route to correlation energies.
- [Parallel Computing &amp; Machine Learning](hpc-and-ml.html) — the HPC techniques and ML potentials that scale these methods to large systems.
- [Quantum Mechanics](../quantum-mechanics/) — the variational principle, perturbation theory, and the many-body Schrödinger equation underpinning these methods.
- [Condensed Matter Physics](../condensed-matter/) — electronic structure and correlation in periodic solids.
