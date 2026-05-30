---
layout: docs
title: "Condensed Matter: Graduate-Level Formalism & Experiment"
permalink: /docs/physics/condensed-matter/advanced-formalism.html
toc: true
toc_sticky: true
hide_title: true
---

[Condensed Matter Physics](./)

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Graduate-Level Formalism &amp; Experiment</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Many-body theory, experimental probes, and research frontiers</p>
</div>

## Experimental Techniques

### Transport Measurements
- Resistivity: four-probe method
- Hall effect: extract carrier density and mobility
- Quantum oscillations: map Fermi surface

### Spectroscopy
- ARPES: angle-resolved photoemission
- STM/STS: scanning tunneling microscopy/spectroscopy
- Neutron scattering: magnetic structure and excitations
- X-ray scattering: crystal structure

### Thermodynamic Measurements
- Specific heat: identify phase transitions
- Magnetization: magnetic properties
- Thermal expansion: coupling to lattice

## Current Research Areas

### 2D Materials
- Graphene: Dirac fermions
- Transition metal dichalcogenides
- van der Waals heterostructures
- Moiré superlattices

### Quantum Materials
- Weyl and Dirac semimetals
- Axion insulators
- Quantum spin liquids
- Majorana fermions

### Non-equilibrium Physics
- Floquet engineering
- Many-body localization
- Time crystals
- Driven-dissipative systems

## Graduate-Level Mathematical Formalism

### Second Quantization in Condensed Matter

**Field operators for fermions:**

$$\psi(\mathbf{r}) = \sum_k \phi_k(\mathbf{r}) c_k$$

$$\psi^\dagger(\mathbf{r}) = \sum_k \phi_k^*(\mathbf{r}) c_k^\dagger$$

**Anticommutation relations:**

$$\{\psi(\mathbf{r}), \psi^\dagger(\mathbf{r}')\} = \delta(\mathbf{r} - \mathbf{r}')$$

$$\{\psi(\mathbf{r}), \psi(\mathbf{r}')\} = \{\psi^\dagger(\mathbf{r}), \psi^\dagger(\mathbf{r}')\} = 0$$

**General Hamiltonian:**

$$H = \int d\mathbf{r} \, \psi^\dagger(\mathbf{r})\left[-\frac{\hbar^2\nabla^2}{2m} + V(\mathbf{r})\right]\psi(\mathbf{r}) + \frac{1}{2}\int d\mathbf{r} \, d\mathbf{r}' \, \psi^\dagger(\mathbf{r})\psi^\dagger(\mathbf{r}')U(\mathbf{r}-\mathbf{r}')\psi(\mathbf{r}')\psi(\mathbf{r})$$

### Many-Body Green's Functions

#### Definitions

**Single-particle Green's function:**

$$G(\mathbf{r},t;\mathbf{r}',t') = -i\langle T[\psi(\mathbf{r},t)\psi^\dagger(\mathbf{r}',t')]\rangle$$

**Spectral function:**

$$A(\mathbf{k},\omega) = -2\text{Im}[G^R(\mathbf{k},\omega)]$$

**Dyson equation:**

$$G = G_0 + G_0 \Sigma G$$

Where $\Sigma$ is the self-energy.

#### Matsubara Formalism

**Imaginary time:**

$$G(\mathbf{r},\tau;\mathbf{r}',\tau') = -\langle T_\tau[\psi(\mathbf{r},\tau)\psi^\dagger(\mathbf{r}',\tau')]\rangle$$

**Matsubara frequencies:**
- Fermions: $\omega_n = (2n+1)\pi/\beta$
- Bosons: $\omega_n = 2n\pi/\beta$

**Analytic continuation:** $i\omega_n \to \omega + i\delta$

### Advanced Band Theory

#### k·p Method

Near band extrema:

$$H = E_0 + \frac{\hbar^2 k^2}{2m^*} + \frac{\hbar}{m_0}\sum_i k_i p_i + O(k^2)$$

**Kane model for narrow gap semiconductors:**

$$H = \begin{pmatrix}
E_c + \frac{\hbar^2 k^2}{2m_c} & Pk \\
Pk & E_v - \frac{\hbar^2 k^2}{2m_v}
\end{pmatrix}$$

#### Wannier Functions

**Construction from Bloch states:**

$$w_n(\mathbf{r} - \mathbf{R}) = \frac{V}{(2\pi)^3} \int_{BZ} d\mathbf{k} \, e^{-i\mathbf{k} \cdot \mathbf{R}} \psi_{n\mathbf{k}}(\mathbf{r})$$

**Maximally localized Wannier functions:** Minimize spread

$$\Omega = \sum_n\left[\langle w_n|\mathbf{r}^2|w_n\rangle - \langle w_n|\mathbf{r}|w_n\rangle^2\right]$$

#### Topological Band Theory

**Berry connection:**

$$\mathbf{A}_n(\mathbf{k}) = i\langle u_{n\mathbf{k}}|\nabla_\mathbf{k}|u_{n\mathbf{k}}\rangle$$

**Berry curvature:**

$$\boldsymbol{\Omega}_n(\mathbf{k}) = \nabla_\mathbf{k} \times \mathbf{A}_n(\mathbf{k}) = i\sum_{m\neq n} \frac{\langle u_{n\mathbf{k}}|\nabla_\mathbf{k} H|u_{m\mathbf{k}}\rangle \times \langle u_{m\mathbf{k}}|\nabla_\mathbf{k} H|u_{n\mathbf{k}}\rangle}{(E_n - E_m)^2}$$

**Z₂ invariant:**

$$(-1)^\nu = \prod_{i=1}^4 \frac{\text{Pf}[w(\mathbf{k}_i)]}{\sqrt{\det[w(\mathbf{k}_i)]}}$$

Where $w_{mn} = \langle u_{m\mathbf{k}}|-i\partial_{k_\mu}|u_{n\mathbf{k}}\rangle$

### Superconductivity: Advanced Theory

#### Bogoliubov-de Gennes Formalism

**BdG Hamiltonian:**

$$H_{BdG} = \begin{pmatrix}
H_0(\mathbf{k}) & \Delta(\mathbf{k}) \\
\Delta^*(\mathbf{k}) & -H_0^*(-\mathbf{k})
\end{pmatrix}$$

**Nambu spinor:** $\Psi = (c_{\mathbf{k}\uparrow}, c_{-\mathbf{k}\downarrow}^\dagger)^T$

**Quasiparticle spectrum:**

$$E_\mathbf{k} = \pm\sqrt{\xi_\mathbf{k}^2 + |\Delta_\mathbf{k}|^2}$$

#### Ginzburg-Landau Theory

**GL functional:**

$$F = \int d^3\mathbf{r} \left[\alpha|\psi|^2 + \frac{\beta}{2}|\psi|^4 + \frac{1}{2m^*}|(-i\hbar\nabla - e^*\mathbf{A})\psi|^2 + \frac{B^2}{2\mu_0}\right]$$

**GL equations:**

$$\alpha\psi + \beta|\psi|^2\psi + \frac{1}{2m^*}(-i\hbar\nabla - e^*\mathbf{A})^2\psi = 0$$

$$\mathbf{j} = \frac{e\hbar}{2m^*i}(\psi^*\nabla\psi - \psi\nabla\psi^*) - \frac{4e^2}{m^*}|\psi|^2\mathbf{A}$$

**Coherence length:** $\xi = \frac{\hbar}{\sqrt{2m^*|\alpha|}}$

**Penetration depth:** $\lambda = \sqrt{\frac{m^*}{\mu_0 4e^2 n_s}}$

#### Josephson Effects

**Josephson relations:**

$$I = I_c \sin(\phi)$$

$$\frac{\partial\phi}{\partial t} = \frac{2eV}{\hbar}$$

**RCSJ model:**

$$C \frac{d^2\phi}{dt^2} + \frac{1}{R}\frac{d\phi}{dt} + I_c \sin(\phi) = I$$

**Shapiro steps:** $V_n = \frac{n\hbar\omega}{2e}$

### Quantum Hall Physics

#### Landau Levels

**Single particle states:**

$$\psi_{n,m}(z) = (z - z_m)^n e^{-|z - z_m|^2/(4l_B^2)}$$

Where $l_B = \sqrt{\hbar/(eB)}$ is magnetic length.

**Projected density operators:**

$$\rho_q = \sum_k c_{k+q}^\dagger c_k e^{iq \times k l_B^2/2}$$

#### Composite Fermion Theory

**CF transformation:**

$$\Psi_{CF} = P_{LLL} \prod_{i<j}(z_i - z_j)^2 \Phi_{fermions}$$

**Effective magnetic field:**

$$B_{eff} = B - 2\phi_0\rho$$

Where $\phi_0 = h/e$ is flux quantum.

#### Chern-Simons Theory

**Effective action:**

$$S = \int d^3x \left[\frac{\epsilon^{\mu\nu\lambda}}{4\pi} a_\mu\partial_\nu a_\lambda + j^\mu a_\mu\right]$$

**Statistical transmutation:** Fermions $\leftrightarrow$ Bosons + flux

### Strongly Correlated Electrons

#### Hubbard Model Extensions

**t-J model (large U limit):**

$$H = -t\sum_{\langle ij\rangle,\sigma} P(c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.})P + J\sum_{\langle ij\rangle}\left(\mathbf{S}_i \cdot \mathbf{S}_j - \frac{n_i n_j}{4}\right)$$

Where $P$ projects out double occupancy.

**Anderson model (impurity):**

$$H = \sum_{k\sigma}\epsilon_k c_{k\sigma}^\dagger c_{k\sigma} + \sum_\sigma \epsilon_d d_\sigma^\dagger d_\sigma + Un_{d\uparrow}n_{d\downarrow} + V\sum_{k\sigma}(c_{k\sigma}^\dagger d_\sigma + \text{h.c.})$$

#### Dynamical Mean-Field Theory (DMFT)

**Self-consistency equations:**

$$G_{loc}(\omega) = \sum_k G(\mathbf{k},\omega)$$

$$G^{-1}(\mathbf{k},\omega) = \omega + \mu - \epsilon_\mathbf{k} - \Sigma(\omega)$$

$$\Gamma(\omega) = G_0^{-1}(\omega) - G_{loc}^{-1}(\omega)$$

**Anderson impurity problem:**

$$H_{imp} = \epsilon_d d^\dagger d + Un_{d\uparrow}n_{d\downarrow} + \sum_k V_k(c_k^\dagger d + \text{h.c.}) + \sum_k \epsilon_k c_k^\dagger c_k$$

#### Slave Particle Methods

**Slave boson representation:**

$$c_{i\sigma} = b_i^\dagger f_{i\sigma}$$

Constraint: $b_i^\dagger b_i + \sum_\sigma f_{i\sigma}^\dagger f_{i\sigma} = 1$

**Mean-field decoupling:** $\langle b_i\rangle \neq 0$ describes coherent quasiparticles

### Topological Phases: Advanced Topics

#### Topological Field Theory

**Chern-Simons term:**

$$S_{CS} = \frac{k}{4\pi} \int d^3x \, \epsilon^{\mu\nu\lambda} A_\mu\partial_\nu A_\lambda$$

**BF theory:**

$$S_{BF} = \frac{K_{IJ}}{2\pi} \int d^3x \, \epsilon^{\mu\nu\lambda} a_\mu^I\partial_\nu a_\lambda^J$$

#### Topological Order

**Ground state degeneracy on torus:** Depends on topology

**Modular matrices:** $S$ and $T$ characterize anyon statistics

$$S_{ab} = \frac{1}{\mathcal{M}} \sum_c \frac{N_{ab}^c d_c}{d_a d_b}$$

**Topological entanglement entropy:**

$$S = \alpha L - \gamma$$

Where $\gamma = \ln(\mathcal{M})$ is universal.

#### Symmetry-Protected Topological Phases

**Classification by cohomology:** $H^{d+1}(G, U(1))$

**Matrix product state representation:**

$$|\psi\rangle = \sum_{s_1...s_N} \text{Tr}[A^{s_1}...A^{s_N}]|s_1...s_N\rangle$$

Symmetry: $u(g)A^s u^\dagger(g) = \sum_{s'} U(g)_{ss'} A^{s'}$

### Quantum Criticality

#### Scaling Theory

**Dynamic scaling:** $z$ = dynamic critical exponent

$$\omega \sim k^z$$

**Finite-size scaling:**

$$M(t,h,L) = L^{-\beta/\nu}f(tL^{1/\nu}, hL^{y_h/\nu})$$

#### Quantum-to-Classical Mapping

d-dimensional quantum $\leftrightarrow$ (d+1)-dimensional classical

**Effective temperature:** $T_{eff} \sim \hbar\omega$

#### Deconfined Quantum Criticality

**Néel-VBS transition:**

$$S = \int d^2x \, d\tau \left[|(\partial_\tau - ia_\tau)z|^2 + |(\nabla - i\mathbf{a})z|^2 + s|z|^2 + u|z|^4\right]$$

Emergent gauge field $a_\mu$ mediates transition.

### Modern Experimental Probes

#### ARPES (Angle-Resolved Photoemission)

**Intensity:**

$$I(\mathbf{k},\omega) \propto |M_{fi}|^2 f(\omega) A(\mathbf{k},\omega)$$

Where $M_{fi}$ is matrix element, $f(\omega)$ is Fermi function.

**Self-energy extraction:**

$$\Sigma'(\mathbf{k},\omega) = \omega - \epsilon_\mathbf{k}^0 - \text{Re}[\Sigma(\mathbf{k},\omega)]$$

$$\Sigma''(\mathbf{k},\omega) = \text{Im}[\Sigma(\mathbf{k},\omega)]$$

#### Quantum Oscillations

**Lifshitz-Kosevich formula:**

$$M \propto \left(\frac{T}{B}\right)^{1/2} R_T R_D R_S \sin\left(\frac{2\pi F}{B} + \phi\right)$$

Where:
- $R_T$ = thermal damping
- $R_D$ = Dingle factor
- $R_S$ = spin factor
- $F$ = oscillation frequency

**Fermiology:** Extract Fermi surface, effective mass, scattering rate

#### STM/STS

**Tunneling current:**

$$I \propto \int_{-eV}^0 d\omega \, \rho_s(\omega)\rho_t(\mathbf{r},\omega+eV)T(\omega,eV)$$

**Differential conductance:**

$$\frac{dI}{dV} \propto \rho_s(E_F)\rho_t(\mathbf{r},eV)$$

**Quasiparticle interference:** Fourier transform reveals $\mathbf{q} = \mathbf{k}_f - \mathbf{k}_i$

### Computational Methods

#### Density Functional Theory for Solids

**Kohn-Sham equations:**

$$\left[-\frac{\hbar^2\nabla^2}{2m} + v_{eff}(\mathbf{r})\right]\phi_i(\mathbf{r}) = \epsilon_i\phi_i(\mathbf{r})$$

**Exchange-correlation functionals:**
- LDA: $\epsilon_{xc}[n] = \epsilon_{xc}(n)$
- GGA: $\epsilon_{xc}[n,\nabla n]$
- Hybrid: Mix exact exchange

**Band structure calculations:** Plane wave basis, pseudopotentials

#### Quantum Monte Carlo

**Variational QMC:**

$$E = \frac{\langle\Psi_T|H|\Psi_T\rangle}{\langle\Psi_T|\Psi_T\rangle}$$

**Diffusion QMC:** Project out ground state

$$|\Psi_0\rangle = \lim_{t\to\infty} e^{-Ht}|\Psi_T\rangle$$

**Sign problem:** Constrains fermionic/frustrated systems

#### Tensor Network Methods

**iPEPS for 2D systems:**

$$|\Psi\rangle = \sum_s \text{tTr}[A^{s_{1,1}}...A^{s_{N,N}}]|s\rangle$$

**Corner transfer matrix:** Compute observables

**Time evolution:** TEBD, MPO methods

```python
import numpy as np
from scipy.linalg import expm

def tebd_step(psi, U_bonds, chi_max):
    """Time-evolving block decimation step"""
    for bond in range(0, len(psi)-1, 2):  # Even bonds
        psi = apply_two_site_gate(psi, U_bonds[bond], bond, chi_max)
    for bond in range(1, len(psi)-1, 2):  # Odd bonds  
        psi = apply_two_site_gate(psi, U_bonds[bond], bond, chi_max)
    return psi

def apply_two_site_gate(psi, U, bond, chi_max):
    """Apply two-site gate with truncation"""
    # Contract tensors
    theta = np.tensordot(psi[bond], psi[bond+1], axes=([2],[0]))
    theta = np.tensordot(U, theta, axes=([2,3],[0,2]))
    
    # SVD and truncate
    theta = theta.transpose(0,2,1,3).reshape(d*chi_l, d*chi_r)
    u, s, vh = np.linalg.svd(theta, full_matrices=False)
    
    # Truncate to chi_max
    chi_new = min(len(s), chi_max)
    u = u[:, :chi_new]
    s = s[:chi_new]
    vh = vh[:chi_new, :]
    
    # Update MPS tensors
    psi[bond] = u.reshape(chi_l, d, chi_new)
    psi[bond+1] = (np.diag(s) @ vh).reshape(chi_new, d, chi_r)
    
    return psi
```

## Research Frontiers

### Quantum Materials Design

**Materials informatics:** Machine learning for materials discovery

**Heterostructure engineering:** Designer quantum phases

**Moiré systems:** Tunable strongly correlated physics

### Non-equilibrium Phenomena

**Floquet engineering:** Light-induced topological phases

$$H_F = H_0 + V \cos(\omega t)$$

**Ultrafast spectroscopy:** Pump-probe dynamics

**Many-body localization:** Breakdown of thermalization

### Quantum Technologies

**Topological quantum computing:** Anyonic braiding

**Quantum sensors:** NV centers, SQUIDs

**Coherent quantum devices:** Josephson junctions, quantum dots

### Unconventional Superconductivity

**Iron-based superconductors:** Multi-orbital physics

**Heavy fermion superconductors:** Quantum criticality

**Organic superconductors:** Low dimensionality

**Interface superconductivity:** STO/LAO, FeSe/STO

### Correlated Topology

**Twisted bilayer graphene:** Flat bands and superconductivity

**Magnetic topological insulators:** Quantum anomalous Hall effect

**Weyl-Kondo semimetals:** Topology meets strong correlations

## References and Further Reading

### Classic Textbooks
1. **Ashcroft & Mermin** - *Solid State Physics*
2. **Kittel** - *Introduction to Solid State Physics*
3. **Mahan** - *Many-Particle Physics*
4. **Abrikosov, Gorkov & Dzyaloshinski** - *Methods of Quantum Field Theory in Statistical Physics*

### Advanced Monographs
1. **Coleman** - *Introduction to Many-Body Physics*
2. **Wen** - *Quantum Field Theory of Many-Body Systems*
3. **Bernevig & Hughes** - *Topological Insulators and Topological Superconductors*
4. **Tinkham** - *Introduction to Superconductivity*

### Specialized Topics
1. **Giamarchi** - *Quantum Physics in One Dimension*
2. **Sachdev** - *Quantum Phase Transitions*
3. **Girvin & Yang** - *Modern Condensed Matter Physics*
4. **Phillips** - *Advanced Solid State Physics*

### Recent Reviews
1. **Keimer et al.** - *From quantum matter to high-temperature superconductivity in copper oxides* (2015)
2. **Armitage, Mele & Vishwanath** - *Weyl and Dirac semimetals in three-dimensional solids* (2018)
3. **Balents et al.** - *Superconductivity and strong correlations in moiré flat bands* (2020)
4. **Khajetoorians et al.** - *Creating designer quantum states of matter atom-by-atom* (2019)

---

<div class="page-nav" style="display: flex; justify-content: space-between; margin-top: 2rem;">
  <a href="emergent-phases.html">&larr; Superconductivity, Quantum Hall &amp; Topological Phases</a>
  <a href="./">Condensed Matter Physics (Hub) &rarr;</a>
</div>

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="emergent-phases.html">Superconductivity, Quantum Hall &amp; Topological Phases</a> — the narrative treatment of these phases.</li>
    <li><a href="./">Condensed Matter Physics (Hub)</a> — crystal structure, band theory, and magnetism.</li>
    <li><a href="../quantum-field-theory.html">Quantum Field Theory</a> — field-theoretic methods for collective excitations.</li>
    <li><a href="../computational-physics/">Computational Physics</a> — DFT, Monte Carlo, and tensor-network simulations.</li>
  </ul>
</div>
