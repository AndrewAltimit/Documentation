---
layout: docs
title: "Statistical Mechanics: Phase Transitions & Graduate Formalism"
permalink: /docs/physics/statistical-mechanics/phase-transitions-and-advanced.html
toc: true
toc_sticky: true
---

<!-- Custom styles are now loaded via main.scss -->

[Statistical Mechanics](./)

Critical phenomena, fluctuations, non-equilibrium dynamics, and advanced field-theoretic methods.

## Phase Transitions

<div class="phase-transitions-section">
  <div class="classification-grid">
    <h3><i class="fas fa-layer-group"></i> Classification</h3>
    
    <div class="transition-types">
      <div class="transition-card first-order">
        <h4><i class="fas fa-cut"></i> First Order</h4>
        <p>Discontinuous change in first derivative of free energy</p>
        
        <svg viewBox="0 0 400 260" class="transition-plot" style="max-width: 500px; width: 100%;">
          <!-- Background -->
          <rect x="5" y="5" width="390" height="250" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

          <!-- Title -->
          <text x="200" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">First-Order Phase Transition</text>

          <!-- Define arrow markers -->
          <defs>
            <marker id="arrowFirst" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
            </marker>
          </defs>

          <!-- Axes -->
          <line x1="60" y1="200" x2="360" y2="200" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowFirst)"/>
          <line x1="60" y1="200" x2="60" y2="50" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowFirst)"/>

          <!-- Axis labels -->
          <text x="210" y="235" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Temperature T</text>
          <text x="25" y="125" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50" transform="rotate(-90, 25, 125)">Order Parameter</text>

          <!-- Critical temperature line -->
          <line x1="180" y1="50" x2="180" y2="200" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="8,4"/>
          <text x="180" y="220" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">T_c</text>

          <!-- High-order phase (left of Tc) -->
          <path d="M 60 70 L 170 85" stroke="#c0392b" stroke-width="4"/>

          <!-- Discontinuous jump (at Tc) -->
          <line x1="175" y1="87" x2="175" y2="155" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,3"/>

          <!-- Low-order phase (right of Tc) -->
          <path d="M 180 160 L 340 175" stroke="#c0392b" stroke-width="4"/>

          <!-- Endpoints at discontinuity -->
          <circle cx="172" cy="87" r="6" fill="#c0392b" stroke="#922b21" stroke-width="2"/>
          <circle cx="178" cy="158" r="6" fill="white" stroke="#c0392b" stroke-width="3"/>

          <!-- Jump annotation -->
          <path d="M 195 120 L 220 90" fill="none" stroke="#27ae60" stroke-width="2"/>
          <text x="240" y="85" font-size="14" fill="#27ae60" font-weight="bold">Discontinuous</text>
          <text x="240" y="102" font-size="14" fill="#27ae60" font-weight="bold">jump at T_c</text>

          <!-- Latent heat annotation -->
          <rect x="250" y="140" width="110" height="40" rx="5" fill="#f1c40f" opacity="0.3" stroke="#d4ac0d" stroke-width="1"/>
          <text x="305" y="162" text-anchor="middle" font-size="13" fill="#9a7d0a" font-weight="bold">Latent heat</text>
          <text x="305" y="176" text-anchor="middle" font-size="13" fill="#9a7d0a">released/absorbed</text>
        </svg>
        
        <div class="examples">
          <span class="example-tag">Ice ↔ Water</span>
          <span class="example-tag">Boiling</span>
        </div>
      </div>
      
      <div class="transition-card second-order">
        <h4><i class="fas fa-wave-square"></i> Second Order</h4>
        <p>Continuous first derivative, discontinuous second derivative</p>
        
        <svg viewBox="0 0 400 260" class="transition-plot" style="max-width: 500px; width: 100%;">
          <!-- Background -->
          <rect x="5" y="5" width="390" height="250" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

          <!-- Title -->
          <text x="200" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Second-Order Phase Transition</text>

          <!-- Define arrow markers -->
          <defs>
            <marker id="arrowSecond" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
            </marker>
          </defs>

          <!-- Axes -->
          <line x1="60" y1="200" x2="360" y2="200" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowSecond)"/>
          <line x1="60" y1="200" x2="60" y2="50" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowSecond)"/>

          <!-- Axis labels -->
          <text x="210" y="235" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Temperature T</text>
          <text x="25" y="125" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50" transform="rotate(-90, 25, 125)">Order Parameter m</text>

          <!-- Critical temperature line -->
          <line x1="200" y1="50" x2="200" y2="200" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="8,4"/>
          <text x="200" y="220" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">T_c</text>

          <!-- Continuous curve (order parameter vanishes smoothly) -->
          <path d="M 60 70 Q 100 72, 140 90 Q 170 120, 195 180 Q 198 195, 200 200" stroke="#2980b9" stroke-width="4" fill="none"/>

          <!-- Zero line after Tc -->
          <path d="M 200 200 L 340 200" stroke="#2980b9" stroke-width="4"/>

          <!-- Critical point marker -->
          <circle cx="200" cy="200" r="7" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
          <text x="205" y="195" font-size="13" fill="#c0392b" font-weight="bold">Critical point</text>

          <!-- Power law annotation -->
          <path d="M 150 110 Q 170 100, 190 105" fill="none" stroke="#27ae60" stroke-width="2"/>
          <text x="115" y="95" font-size="14" fill="#27ae60" font-weight="bold">m ~ |T - T_c|^beta</text>
          <text x="115" y="112" font-size="13" fill="#27ae60">(power-law behavior)</text>

          <!-- Ordered vs disordered regions -->
          <rect x="65" y="45" width="90" height="30" rx="5" fill="#2980b9" opacity="0.2" stroke="#2980b9" stroke-width="1"/>
          <text x="110" y="65" text-anchor="middle" font-size="14" fill="#1a5276" font-weight="bold">Ordered</text>

          <rect x="260" y="165" width="90" height="30" rx="5" fill="#95a5a6" opacity="0.3" stroke="#7f8c8d" stroke-width="1"/>
          <text x="305" y="185" text-anchor="middle" font-size="14" fill="#555" font-weight="bold">Disordered</text>
        </svg>
        
        <div class="examples">
          <span class="example-tag">Magnetization</span>
          <span class="example-tag">Superconductivity</span>
        </div>
      </div>
    </div>
  </div>
</div>

### Critical Phenomena

Near a critical point, observables follow power laws in the reduced temperature $t = (T-T_c)/T_c$:

| Exponent | Observable | Scaling |
|----------|-----------|---------|
| $\alpha$ | Specific heat $C$ | $C \sim \lvert t\rvert^{-\alpha}$ |
| $\beta$ | Order parameter $m$ | $m \sim \lvert t\rvert^{\beta}$ (for $t < 0$) |
| $\gamma$ | Susceptibility $\chi$ | $\chi \sim \lvert t\rvert^{-\gamma}$ |
| $\nu$ | Correlation length $\xi$ | $\xi \sim \lvert t\rvert^{-\nu}$ |

### Universality

Systems with the same dimensionality and symmetry share identical critical exponents (examples of universality classes: 2D Ising, 3D XY, percolation). The exponents are not independent but obey **scaling relations**:

- **Rushbrooke:** $\alpha + 2\beta + \gamma = 2$
- **Widom:** $\gamma = \beta(\delta - 1)$
- **Fisher:** $\gamma = \nu(2 - \eta)$

## Fluctuations

### Gaussian Fluctuations
For energy fluctuations:
$$\langle (\Delta E)^2 \rangle = k_B T^2 C_V$$

For particle number:
$$\langle (\Delta N)^2 \rangle = k_B T \left(\frac{\partial N}{\partial \mu}\right)_{T,V}$$

### Fluctuation-Dissipation Theorem
Connects response functions to equilibrium fluctuations:
$$\chi(\omega) = \frac{1}{k_B T} \int_0^{\infty} \langle A(t)A(0) \rangle e^{i\omega t} dt$$

### Einstein's Relations
- Diffusion: $D = \mu k_B T$ (mobility $\mu$)
- Conductivity: $\sigma = ne^2\tau/m$ (Drude model)

## Non-equilibrium Statistical Mechanics

### Boltzmann Equation
Evolution of distribution function:
$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_r f + \frac{\mathbf{F}}{m} \cdot \nabla_v f = \left(\frac{\partial f}{\partial t}\right)_{coll}$$

### H-theorem
Boltzmann's H-function decreases:
$$H = \int f \ln f d^3v$$
$$\frac{dH}{dt} \leq 0$$

### Linear Response Theory
For small perturbation $F(t)$:
$$\langle A(t) \rangle = \langle A \rangle_0 + \int_{-\infty}^t \chi(t-t') F(t') dt'$$

Kubo formula for conductivity:
$$\sigma = \lim_{\omega \to 0} \frac{1}{\omega} \int_0^{\infty} dt e^{i\omega t} \langle J(t)J(0) \rangle$$

## Applications

### Condensed Matter Physics
- Electronic properties of solids
- Superconductivity (BCS theory)
- Quantum Hall effects
- Topological phases

### Soft Matter
- Polymer physics
- Liquid crystals
- Colloids
- Biological membranes

### Cosmology
- Early universe thermodynamics
- Dark matter freeze-out
- Cosmic microwave background

### Quantum Information
- Thermal states in quantum computing
- Entanglement entropy
- Quantum thermodynamics

## Computational Methods

### Monte Carlo
- Metropolis algorithm
- Cluster algorithms (Wolff, Swendsen-Wang)
- Quantum Monte Carlo

### Molecular Dynamics
- Verlet algorithm
- Nosé-Hoover thermostat
- Parrinello-Rahman barostat

### Density Functional Theory
Hohenberg-Kohn theorem: Ground state density determines all properties.

Kohn-Sham equations:
$$\left[-\frac{\hbar^2}{2m}\nabla^2 + v_{\text{eff}}\left[n\right](r)\right]\psi_i(r) = \epsilon_i\psi_i(r)$$

## Advanced Topics

### Renormalization Group
- Block spin transformations
- Fixed points and universality
- Epsilon expansion
- Functional renormalization

### Conformal Field Theory
At critical points, systems exhibit conformal symmetry.

Central charge characterizes universality class.

### AdS/CFT Correspondence
Connects strongly coupled field theories to weakly coupled gravity.

Applications to quark-gluon plasma and condensed matter.

## Graduate-Level Mathematical Formalism

The remaining sections collect the field-theoretic and many-body machinery of modern statistical mechanics in a terse, reference-style format. They are reference material rather than a self-contained exposition, and can be skipped on a first read.

### Information Theory and Statistical Mechanics

**Shannon entropy:**
$$S = -k_B \sum_i p_i \ln p_i$$

**Maximum entropy principle:** The equilibrium distribution maximizes entropy subject to constraints.

**Canonical ensemble from MaxEnt:**
Maximize S subject to:
- Normalization: $\sum_i p_i = 1$
- Energy constraint: $\sum_i p_i E_i = \langle E \rangle$

Using Lagrange multipliers:
$$p_i = \frac{e^{-\beta E_i}}{Z}$$

**Jaynes' principle:** Statistical mechanics as inference theory

**Relative entropy (Kullback-Leibler divergence):**
$$D_{KL}(p||q) = \sum_i p_i \ln\left(\frac{p_i}{q_i}\right) \geq 0$$

### Advanced Ensemble Theory

#### Generalized Ensembles

**Tsallis statistics:**
$$S_q = \frac{k_B (1 - \sum_i p_i^q)}{q - 1}$$

**Pressure ensemble:** (NPT)
$$\Delta(N,P,T) = \int_0^{\infty} dV \, e^{-\beta PV} Z(N,V,T)$$

**Isothermal-isobaric partition function:**
$$\Delta = \frac{k_B T}{P} Z(N,\langle V \rangle,T) e^{\beta P \langle V \rangle}$$

#### Jarzynski Equality and Fluctuation Theorems

**Jarzynski equality:**
$$\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$$

**Crooks fluctuation theorem:**
$$\frac{P_F(W)}{P_R(-W)} = e^{\beta(W - \Delta F)}$$

**Work distribution:** Gaussian near equilibrium
$$P(W) \approx (2\pi\sigma^2)^{-1/2} \exp\left[-\frac{(W - \langle W \rangle)^2}{2\sigma^2}\right]$$

### Path Integral Formulation

**Quantum partition function:**
$$Z = \text{Tr}(e^{-\beta H}) = \int \mathcal{D}[q] \exp(-S_E[q]/\hbar)$$

**Euclidean action:**
$$S_E = \int_0^{\beta\hbar} d\tau \left[\frac{m\dot{x}^2}{2} + V(q)\right]$$

**Feynman-Kac formula:** Connection to diffusion
$$\langle q_f|e^{-\beta H}|q_i\rangle = \int_{q(0)=q_i}^{q(\beta\hbar)=q_f} \mathcal{D}[q] \, e^{-S_E[q]/\hbar}$$

**Effective action at finite temperature:**
$$\Gamma[q_c] = -k_B T \ln Z[J] + \int d\tau \, J(\tau)q_c(\tau)$$

### Field Theoretic Methods

#### Hubbard-Stratonovich Transformation

For interaction term:
$$\exp\left[\frac{\beta}{2} \sum_{ij} J_{ij}s_i s_j\right] = \int \mathcal{D}[\phi] \exp\left[-\frac{\beta}{2} \sum_{ij} \phi_i(J^{-1})_{ij}\phi_j + \beta\sum_i \phi_i s_i\right]$$

#### Replica Method

For disordered systems:
$$\langle \ln Z \rangle = \lim_{n\to 0} \frac{\langle Z^n \rangle - 1}{n}$$

**Replica symmetry breaking:** Order parameter $q_{ab}$

#### Functional Integral Representation

**Grand canonical ensemble:**
$$\Xi = \int \mathcal{D}[\psi^*, \psi] \exp(-S[\psi^*, \psi])$$

**Action for bosons:**
$$S = \int_0^{\beta} d\tau \int d^dr \left[\psi^*(\partial_\tau - \mu)\psi + \frac{\hbar^2}{2m}|\nabla\psi|^2 + U(\psi^*\psi)\right]$$

### Critical Phenomena: Advanced Treatment

#### Scaling Theory

**Scaling hypothesis:** Near $T_c$, singular part of free energy:
$$f_s(t, h) = b^{-d}f_s(b^{y_t}t, b^{y_h}h)$$

Where $y_t = 1/\nu$, $y_h = d - \beta/\nu$

**Scaling relations derivation:**
- From $f_s$: $\alpha = 2 - d\nu$
- From $m = -\partial f/\partial h$: $\beta = (d - y_h)\nu$
- From $\chi = \partial^2f/\partial h^2$: $\gamma = (2y_h - d)\nu$

**Data collapse:** Plot $m/|t|^\beta$ vs $h/|t|^{\beta\delta}$

#### Renormalization Group: Field Theory

**$\phi^4$ theory action:**
$$S = \int d^dx \left[\frac{1}{2}(\nabla\phi)^2 + \frac{r}{2}\phi^2 + \frac{u}{4!}\phi^4\right]$$

**RG flow equations (one-loop):**
$$\frac{dr}{dl} = (2 - \eta)r + Au \frac{r^2}{1 + r}$$
$$\frac{du}{dl} = \varepsilon u - Bu^2 + \frac{Cu^3}{(1 + r)^2}$$

**Fixed points:**
- Gaussian: $(r^*, u^*) = (0, 0)$
- Wilson-Fisher: $(r^*, u^*) = (-\varepsilon/A, \varepsilon/B)$

**Critical exponents ($\varepsilon$-expansion):**
$$\nu = \frac{1}{2} + \frac{\varepsilon}{12} + O(\varepsilon^2)$$
$$\eta = \frac{\varepsilon^2}{54} + O(\varepsilon^3)$$

#### Conformal Field Theory at Criticality

**Conformal algebra in 2D:** Virasoro algebra
$$[L_m, L_n] = (m - n)L_{m+n} + \frac{c}{12} m(m^2 - 1)\delta_{m+n,0}$$

**Central charge:** Characterizes universality class
- Ising: $c = 1/2$
- XY model: $c = 1$
- Potts model (q states): $c = 1 - 6/[q(q+1)]$

**Operator product expansion:**
$$\phi_i(z)\phi_j(0) = \sum_k C_{ijk}z^{h_k-h_i-h_j}\phi_k(0)$$

### Exact Solutions

#### 2D Ising Model (Onsager Solution)

**Transfer matrix method:**
$$Z = \text{Tr}(T^N)$$

**Critical temperature:**
$$\sinh\left(\frac{2J}{k_B T_c}\right) = 1$$

**Free energy per site:**
$$f = -k_B T \ln(2\cosh(2\beta J)) - \frac{k_B T}{2\pi} \int_0^\pi d\theta \, \ln\left[1 + \sqrt{1 - \kappa^2\sin^2\theta}\right]$$

Where $\kappa = 2\sinh(2\beta J)/\cosh^2(2\beta J)$

**Magnetization ($T < T_c$):**
$$m = \left[1 - \sinh^{-4}(2\beta J)\right]^{1/8}$$

#### Bethe Ansatz

**1D Heisenberg chain:**
$$H = J\sum_i \boldsymbol{\sigma}_i \cdot \boldsymbol{\sigma}_{i+1}$$

**Bethe equations:**
$$k_j L = 2\pi I_j - \sum_{k\neq j} \theta(k_j - k_k)$$

**Ground state energy:**
$$\frac{E_0}{N} = -J \ln 2 + \frac{J}{4}$$

### Non-equilibrium Field Theory

#### Keldysh Formalism

**Contour ordering:** Forward and backward branches

**Green's functions:**
$$G^{++}(t,t') = -i\langle T\phi(t)\phi(t')\rangle$$
$$G^{--}(t,t') = -i\langle \tilde{T}\phi(t)\phi(t')\rangle$$
$$G^{+-}(t,t') = -i\langle \phi(t')\phi(t)\rangle$$
$$G^{-+}(t,t') = -i\langle \phi(t)\phi(t')\rangle$$

**Keldysh rotation:**
$$G^R = G^{++} - G^{+-}$$
$$G^A = G^{++} - G^{-+}$$
$$G^K = G^{++} + G^{--} - G^{+-} - G^{-+}$$

#### Langevin Dynamics

**Stochastic equation:**
$$\partial_t\phi = -\Gamma\frac{\delta F}{\delta\phi} + \eta$$

**Noise correlations:**
$$\langle \eta(x,t)\eta(x',t') \rangle = 2\Gamma k_B T\delta(x-x')\delta(t-t')$$

**Martin-Siggia-Rose formalism:** Path integral with response field
$$Z = \int \mathcal{D}[\phi, \tilde{\phi}] \exp(iS[\phi, \tilde{\phi}])$$

### Quantum Many-Body Systems

#### Fermi Liquid Theory

**Quasiparticle concept:** Landau parameters $f^s$, $f^a$

**Effective mass:**
$$\frac{m^*}{m} = 1 + \frac{F_1^s}{3}$$

**Compressibility:**
$$\frac{\kappa}{\kappa_0} = (1 + F_0^s)^{-1}$$

**Collective modes:** Zero sound velocity
$$s = v_F\sqrt{1 + \frac{F_0^s}{3}}$$

#### BCS Theory of Superconductivity

**BCS Hamiltonian:**
$$H = \sum_k \varepsilon_k c^\dagger_{k\sigma}c_{k\sigma} - g\sum_{kk'} c^\dagger_{k\uparrow}c^\dagger_{-k\downarrow}c_{-k'\downarrow}c_{k'\uparrow}$$

**Gap equation:**
$$\Delta = g\sum_k \frac{\Delta}{2E_k} \tanh(\beta E_k/2)$$

Where $E_k = \sqrt{\varepsilon_k^2 + |\Delta|^2}$

**Critical temperature:**
$$k_B T_c = 1.14\hbar\omega_D \exp(-1/N(0)g)$$

#### Luttinger Liquids (1D)

**Bosonization:** Fermion operators to Boson fields
$$\psi(x) \sim \exp[i\phi(x)]$$

**Luttinger parameter:** $K < 1$ repulsive, $K > 1$ attractive

**Power-law correlations:**
$$\langle \psi^\dagger(x)\psi(0) \rangle \sim x^{-1/(2K)}$$

### Modern Developments

#### Tensor Network Methods

**Matrix Product States (MPS):**
$$|\psi\rangle = \sum_{s_1...s_N} \text{Tr}(A^{s_1}...A^{s_N})|s_1...s_N\rangle$$

**DMRG algorithm:** Variational optimization of MPS

**Area law entanglement:** $S \sim L^{d-1}$ for ground states

#### Machine Learning in Statistical Mechanics

**Neural network representation of states:**
$$\psi(s) = \exp\left[\sum_i a_i s_i + \sum_{ij} W_{ij}h_i(s)s_j + ...\right]$$

**Variational Monte Carlo with NNs:**
$$E = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}$$

**Unsupervised learning of phases:**
- Principal component analysis
- Autoencoders
- Diffusion maps

#### Quantum Thermalization

**Eigenstate Thermalization Hypothesis (ETH):**
$$\langle E_n|O|E_m\rangle = O(E)\delta_{nm} + e^{-S(E)/2}f_O(E,\omega)R_{nm}$$

**Many-body localization:** Failure of thermalization

**Floquet systems:** Time-periodic Hamiltonians

### Stochastic Processes and Field Theory

#### Doi-Peliti Formalism

**Creation/annihilation operators for classical particles:**
$$a^\dagger|n\rangle = |n+1\rangle$$
$$a|n\rangle = n|n-1\rangle$$

**Master equation to "Schrodinger" equation:**
$$\partial_t|\psi\rangle = H|\psi\rangle$$

**Coherent state path integral:**
$$P(n,t) = \int \mathcal{D}[\phi^*,\phi] \exp(-S[\phi^*,\phi])$$

#### Active Matter

**Toner-Tu equations:** Flocking
$$\partial_t\rho + \nabla\cdot(\rho\mathbf{v}) = 0$$
$$\partial_t\mathbf{v} + \lambda(\mathbf{v}\cdot\nabla)\mathbf{v} = \alpha\mathbf{v} - \beta|\mathbf{v}|^2\mathbf{v} - \nabla P + \nu\nabla^2\mathbf{v} + \mathbf{f}$$

**Motility-induced phase separation:**
$$\partial_t\rho = \nabla\cdot[(D(\rho) + D_t)\nabla\rho]$$

### Advanced Computational Methods

#### Quantum Monte Carlo

**Path integral Monte Carlo:**
$$\rho(R,R';\beta) = (2\pi\lambda\beta)^{-3N/2}\sum_P (\pm)^P \exp\left[-\beta\sum_i V(R_i)\right]$$

**Sign problem:** Fermionic systems, frustrated magnets

**Continuous-time algorithms:** Worm algorithm, CT-QMC

#### Machine Learning Acceleration

```python
import torch
import torch.nn as nn

class VariationalWavefunction(nn.Module):
    def __init__(self, L, hidden_dim=100):
        super().__init__()
        self.L = L
        self.net = nn.Sequential(
            nn.Linear(L, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Real and imaginary parts
        )
    
    def forward(self, states):
        """states: (batch_size, L) binary spin configurations"""
        out = self.net(states.float())
        log_amp = out[:, 0]
        phase = out[:, 1]
        return log_amp, phase
    
    def sample(self, n_samples):
        """Metropolis sampling from |psi|^2"""
        states = torch.randint(0, 2, (n_samples, self.L))
        # Implement Metropolis-Hastings...
        return states
```

## Research Frontiers

### Quantum Information and Statistical Mechanics

**Entanglement entropy scaling:**
- Volume law: S ∼ L^d (thermal, excited states)
- Area law: S ∼ L^{d-1} (ground states)
- Logarithmic: S ∼ log L (1D critical)

**Tensor network representations:**
- MPS, PEPS, MERA
- Entanglement renormalization

### Non-equilibrium Quantum Systems

**Prethermalization:** Quasi-stationary states

**Dynamical phase transitions:** Non-analytic behavior in Loschmidt echo

**Floquet engineering:** Designer Hamiltonians

### Machine Learning and Physics

**Reverse engineering Hamiltonians:** Learning from data

**Accelerating simulations:** Neural network quantum states

**Discovering order parameters:** Unsupervised learning

### Topological Phases

**Symmetry-protected topological phases:**
- Classification by cohomology
- Edge states

**Topological order:**
- Anyonic excitations
- Topological entanglement entropy

### Many-Body Localization

**Phenomenology:**
- Area law entanglement
- Emergent integrability
- l-bits (localized integrals of motion)

**Transitions:**
- Thermal to MBL
- MBL to ergodic

## References and Further Reading

### Classic Textbooks
1. **Pathria & Beale** - *Statistical Mechanics*
2. **Kardar** - *Statistical Physics of Particles & Fields*
3. **Landau & Lifshitz** - *Statistical Physics* (Parts 1 & 2)
4. **Huang** - *Statistical Mechanics*

### Advanced Monographs
1. **Altland & Simons** - *Condensed Matter Field Theory*
2. **Sachdev** - *Quantum Phase Transitions*
3. **Nishimori & Ortiz** - *Elements of Phase Transitions and Critical Phenomena*
4. **Täuber** - *Critical Dynamics*

### Specialized Topics
1. **Gogolin, Nersesyan & Tsvelik** - *Bosonization and Strongly Correlated Systems*
2. **Schollwöck** - *The density-matrix renormalization group in the age of matrix product states*
3. **Eisert, Cramer & Plenio** - *Colloquium: Area laws for the entanglement entropy*
4. **Carleo & Troyer** - *Solving the quantum many-body problem with artificial neural networks*

### Recent Reviews
1. **Nandkishore & Huse** - *Many-body localization and thermalization* (2015)
2. **Calabrese, Cardy & Doyon** - *Special issue on quantum integrability in out of equilibrium systems* (2016)
3. **Abanin et al.** - *Colloquium: Many-body localization, thermalization, and entanglement* (2019)
4. **Carrasquilla** - *Machine learning for quantum matter* (2020)

---

## See Also

- [Statistical Mechanics Hub](./) — overview, microstates/macrostates, and ensembles.
- [Classical & Quantum Statistical Mechanics](classical-and-quantum.html) — Previous: partition functions, quantum statistics, and ideal and interacting gases.
- [Condensed Matter Physics](../condensed-matter/) — many-body applications to solids and phase transitions.
- [Quantum Field Theory](../quantum-field-theory.html) — finite-temperature field theory and the path-integral link.

**Previous:** ← [Classical & Quantum Statistical Mechanics](classical-and-quantum.html)
