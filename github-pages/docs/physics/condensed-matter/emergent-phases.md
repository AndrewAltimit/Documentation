---
layout: docs
title: "Condensed Matter: Superconductivity, Quantum Hall & Topological Phases"
permalink: /docs/physics/condensed-matter/emergent-phases.html
toc: true
toc_sticky: true
hide_title: true
---

[Condensed Matter Physics](./)

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Superconductivity, Quantum Hall &amp; Topological Phases</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The emergent and topological states of matter</p>
</div>

## Superconductivity

<div class="superconductivity-section">
  <div class="phenomenology">
    <h3><i class="fas fa-snowflake"></i> Phenomenology</h3>
    
    <div class="phenomenon-cards">
      <div class="phenomenon-card">
        <i class="fas fa-bolt"></i>
        <h4>Zero Resistance</h4>
        <p>Below $T_c$</p>
        <div class="mini-plot">
          <svg viewBox="0 0 180 140" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 6px;">
            <!-- Title -->
            <text x="90" y="15" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Resistance vs Temperature</text>

            <!-- Axes -->
            <line x1="30" y1="115" x2="165" y2="115" stroke="#333" stroke-width="2" />
            <line x1="30" y1="115" x2="30" y2="25" stroke="#333" stroke-width="2" />

            <!-- Axis labels -->
            <text x="165" y="130" font-size="12" font-weight="bold" fill="#333">T</text>
            <text x="18" y="70" font-size="12" font-weight="bold" fill="#333" transform="rotate(-90 18 70)">R</text>

            <!-- Normal state (linear) -->
            <path d="M 30 40 L 90 40" stroke="#7f8c8d" stroke-width="3" stroke-dasharray="4,2" />
            <text x="60" y="35" text-anchor="middle" font-size="10" fill="#7f8c8d">Normal</text>

            <!-- Superconducting transition -->
            <path d="M 30 40 L 90 40 L 95 110" stroke="#e74c3c" stroke-width="3" fill="none" />

            <!-- Superconducting state (R=0) -->
            <path d="M 95 110 L 160 110" stroke="#3498db" stroke-width="3" />
            <text x="130" y="100" text-anchor="middle" font-size="10" fill="#2980b9" font-weight="bold">R = 0</text>

            <!-- Tc marker -->
            <line x1="95" y1="115" x2="95" y2="105" stroke="#27ae60" stroke-width="2" />
            <text x="95" y="128" text-anchor="middle" font-size="11" font-weight="bold" fill="#27ae60">T_c</text>
          </svg>
        </div>
      </div>

      <div class="phenomenon-card">
        <i class="fas fa-magnet"></i>
        <h4>Meissner Effect</h4>
        <p>Expulsion of magnetic field</p>
        <div class="meissner-visual">
          <svg viewBox="0 0 180 140" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 6px;">
            <!-- Title -->
            <text x="90" y="15" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Magnetic Field Expulsion</text>

            <!-- Superconductor body -->
            <ellipse cx="90" cy="80" rx="45" ry="30" fill="#3498db" opacity="0.6" stroke="#2980b9" stroke-width="2" />
            <text x="90" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a5276">SC</text>

            <!-- Magnetic field lines bending around superconductor -->
            <!-- Left side incoming -->
            <path d="M 10 30 Q 25 50, 40 70 Q 45 80, 40 90 Q 25 110, 10 130" stroke="#e74c3c" stroke-width="2.5" fill="none" />
            <!-- Arrow head -->
            <polygon points="10,30 15,38 5,38" fill="#e74c3c" />

            <!-- Right side outgoing -->
            <path d="M 170 30 Q 155 50, 140 70 Q 135 80, 140 90 Q 155 110, 170 130" stroke="#e74c3c" stroke-width="2.5" fill="none" />
            <!-- Arrow head -->
            <polygon points="170,130 165,122 175,122" fill="#e74c3c" />

            <!-- Field lines going around top -->
            <path d="M 30 40 Q 50 35, 90 30 Q 130 35, 150 40" stroke="#e74c3c" stroke-width="2" fill="none" />

            <!-- Field lines going around bottom -->
            <path d="M 30 120 Q 50 125, 90 130 Q 130 125, 150 120" stroke="#e74c3c" stroke-width="2" fill="none" />

            <!-- B=0 inside label -->
            <text x="90" y="100" text-anchor="middle" font-size="10" fill="#1a5276" font-weight="bold">B = 0</text>

            <!-- External B field label -->
            <text x="25" y="80" font-size="10" fill="#c0392b" font-weight="bold">B</text>
          </svg>
        </div>
      </div>

      <div class="phenomenon-card">
        <i class="fas fa-ring"></i>
        <h4>Flux Quantization</h4>
        <p>$\Phi = n\frac{h}{2e}$</p>
        <div class="flux-quantum">
          <svg viewBox="0 0 180 140" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 6px;">
            <!-- Title -->
            <text x="90" y="15" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Quantized Flux</text>

            <!-- Superconducting ring (torus cross-section) -->
            <circle cx="90" cy="75" r="40" fill="none" stroke="#3498db" stroke-width="12" opacity="0.7" />
            <circle cx="90" cy="75" r="40" fill="none" stroke="#2980b9" stroke-width="2" />
            <circle cx="90" cy="75" r="28" fill="none" stroke="#2980b9" stroke-width="2" />

            <!-- Inner hole -->
            <circle cx="90" cy="75" r="22" fill="#fafbfc" />

            <!-- Flux through hole -->
            <circle cx="90" cy="75" r="15" fill="#e74c3c" opacity="0.2" />
            <text x="90" y="80" text-anchor="middle" font-size="14" font-weight="bold" fill="#c0392b">Phi_0</text>

            <!-- Flux quantum value -->
            <text x="90" y="128" text-anchor="middle" font-size="11" fill="#555">Phi_0 = h/2e</text>
            <text x="90" y="140" text-anchor="middle" font-size="10" fill="#777">= 2.07 x 10^-15 Wb</text>

            <!-- SC label -->
            <text x="130" y="60" font-size="10" fill="#2980b9" font-weight="bold">SC ring</text>
          </svg>
        </div>
      </div>
    </div>
  </div>
  
  <div class="theories-grid">
    <div class="theory-card gl-theory">
      <h3><i class="fas fa-wave-square"></i> Ginzburg-Landau Theory</h3>
      <p>Order parameter $\psi(\mathbf{r})$:</p>
      
      <div class="gl-content">
        <p>Free energy:</p>
        <div class="equation-box scrollable" markdown="1">
$$F = \int d^3r \left[\alpha|\psi|^2 + \frac{\beta}{2}|\psi|^4 + \frac{1}{2m^*}|(-i\hbar\nabla - e^*\mathbf{A})\psi|^2 + \frac{B^2}{2\mu_0}\right]$$
</div>
        
        <div class="length-scales">
          <div class="scale-item">
            <span class="scale-name">Coherence length:</span>
            <span class="scale-eq">$\xi = \sqrt{\frac{\hbar^2}{2m^*|\alpha|}}$</span>
          </div>
          <div class="scale-item">
            <span class="scale-name">Penetration depth:</span>
            <span class="scale-eq">$\lambda = \sqrt{\frac{m^*}{e^{*2}\mu_0 n_s}}$</span>
          </div>
        </div>
        
        <div class="type-classification">
          <p class="classification-note">Type I: $\kappa = \lambda/\xi < 1/\sqrt{2}$</p>
          <p class="classification-note">Type II: $\kappa = \lambda/\xi > 1/\sqrt{2}$</p>
        </div>
      </div>
    </div>
    
    <div class="theory-card bcs-theory">
      <h3><i class="fas fa-link"></i> BCS Theory</h3>
      <p>Cooper pair wavefunction:</p>
      
      <div class="bcs-content">
        <div class="equation-box" markdown="1">
$$|\text{BCS}\rangle = \prod_k (u_k + v_k c_{k\uparrow}^\dagger c_{-k\downarrow}^\dagger)|0\rangle$$
</div>
        
        <p>Gap equation:</p>
        <div class="equation-box" markdown="1">
$$\Delta_k = -\sum_{k'} V_{kk'} \frac{\Delta_{k'}}{2E_{k'}} \tanh\left(\frac{E_{k'}}{2k_B T}\right)$$
</div>
        
        <p>Where $E_k = \sqrt{\epsilon_k^2 + |\Delta_k|^2}$</p>
        
        <div class="cooper-pair-visual">
          <svg viewBox="0 0 400 180" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
            <!-- Title -->
            <text x="200" y="22" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Cooper Pair Formation</text>

            <!-- Lattice background (ionic lattice) -->
            <g opacity="0.4">
              <circle cx="50" cy="100" r="12" fill="#95a5a6" />
              <circle cx="100" cy="100" r="12" fill="#95a5a6" />
              <circle cx="150" cy="100" r="12" fill="#95a5a6" />
              <circle cx="200" cy="100" r="12" fill="#95a5a6" />
              <circle cx="250" cy="100" r="12" fill="#95a5a6" />
              <circle cx="300" cy="100" r="12" fill="#95a5a6" />
              <circle cx="350" cy="100" r="12" fill="#95a5a6" />
              <text x="50" y="130" text-anchor="middle" font-size="10" fill="#555">Ion</text>
            </g>

            <!-- Lattice distortion visualization -->
            <path d="M 100 88 Q 130 75, 160 88" stroke="#27ae60" stroke-width="2" fill="none" stroke-dasharray="4,2" />
            <text x="130" y="68" text-anchor="middle" font-size="10" fill="#27ae60" font-weight="bold">Phonon</text>

            <!-- First electron (spin up) -->
            <circle cx="110" cy="80" r="15" fill="#3498db" stroke="#2980b9" stroke-width="3" />
            <text x="110" y="75" text-anchor="middle" font-size="18" fill="white" font-weight="bold">e-</text>
            <text x="110" y="90" text-anchor="middle" font-size="12" fill="white" font-weight="bold">spin-up</text>

            <!-- Second electron (spin down) -->
            <circle cx="290" cy="80" r="15" fill="#e74c3c" stroke="#c0392b" stroke-width="3" />
            <text x="290" y="75" text-anchor="middle" font-size="18" fill="white" font-weight="bold">e-</text>
            <text x="290" y="90" text-anchor="middle" font-size="12" fill="white" font-weight="bold">spin-down</text>

            <!-- Pairing interaction (phonon-mediated) -->
            <path d="M 125 80 Q 200 35, 275 80" stroke="#9b59b6" stroke-width="3" stroke-dasharray="6,3" fill="none" />
            <text x="200" y="45" text-anchor="middle" font-size="12" font-weight="bold" fill="#8e44ad">Attractive Interaction</text>
            <text x="200" y="58" text-anchor="middle" font-size="11" fill="#8e44ad">(phonon-mediated)</text>

            <!-- Momentum labels -->
            <text x="110" y="55" text-anchor="middle" font-size="13" font-weight="bold" fill="#2980b9">k</text>
            <text x="290" y="55" text-anchor="middle" font-size="13" font-weight="bold" fill="#c0392b">-k</text>

            <!-- Cooper pair bracket -->
            <path d="M 95 110 L 95 120 L 305 120 L 305 110" stroke="#2c3e50" stroke-width="2" fill="none" />
            <text x="200" y="140" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Cooper Pair: (k spin-up, -k spin-down)</text>

            <!-- Coherence length indicator -->
            <line x1="110" y1="160" x2="290" y2="160" stroke="#555" stroke-width="1.5" />
            <line x1="110" y1="155" x2="110" y2="165" stroke="#555" stroke-width="1.5" />
            <line x1="290" y1="155" x2="290" y2="165" stroke="#555" stroke-width="1.5" />
            <text x="200" y="175" text-anchor="middle" font-size="11" fill="#555">Coherence length xi ~ 100-1000 nm</text>
          </svg>
        </div>
      </div>
    </div>
  </div>
  
  <div class="josephson-effects">
    <h3><i class="fas fa-exchange-alt"></i> Josephson Effects</h3>
    
    <div class="josephson-grid">
      <div class="josephson-type">
        <h4>DC Josephson</h4>
        <div class="equation-box" markdown="1">
$$I = I_c \sin\phi$$
</div>
        <p class="effect-desc">Supercurrent without voltage</p>
      </div>
      
      <div class="josephson-type">
        <h4>AC Josephson</h4>
        <div class="equation-box" markdown="1">
$$\frac{d\phi}{dt} = \frac{2eV}{\hbar}$$
</div>
        <p class="effect-desc">Oscillating current with DC voltage</p>
      </div>
    </div>
    
    <div class="josephson-junction">
      <svg viewBox="0 0 500 220" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Title -->
        <text x="250" y="22" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Josephson Junction Structure</text>

        <!-- Arrow marker -->
        <defs>
          <marker id="josephson-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 L2,5 Z" fill="#e74c3c" />
          </marker>
          <marker id="tunnel-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 L2,4 Z" fill="#27ae60" />
          </marker>
        </defs>

        <!-- Left Superconductor -->
        <rect x="40" y="60" width="130" height="80" rx="5" fill="#3498db" opacity="0.7" stroke="#2980b9" stroke-width="3" />
        <text x="105" y="95" text-anchor="middle" font-size="16" font-weight="bold" fill="white">Superconductor 1</text>
        <text x="105" y="115" text-anchor="middle" font-size="14" fill="white">Phase: phi_1</text>

        <!-- Barrier (insulator/normal metal) -->
        <rect x="170" y="60" width="60" height="80" rx="3" fill="#f39c12" opacity="0.5" stroke="#d68910" stroke-width="2" />
        <text x="200" y="95" text-anchor="middle" font-size="13" font-weight="bold" fill="#7d5a29">Barrier</text>
        <text x="200" y="112" text-anchor="middle" font-size="11" fill="#7d5a29">(~1-2 nm)</text>

        <!-- Right Superconductor -->
        <rect x="230" y="60" width="130" height="80" rx="5" fill="#3498db" opacity="0.7" stroke="#2980b9" stroke-width="3" />
        <text x="295" y="95" text-anchor="middle" font-size="16" font-weight="bold" fill="white">Superconductor 2</text>
        <text x="295" y="115" text-anchor="middle" font-size="14" fill="white">Phase: phi_2</text>

        <!-- Tunneling Cooper pairs -->
        <g transform="translate(0, -5)">
          <line x1="155" y1="95" x2="175" y2="95" stroke="#27ae60" stroke-width="2.5" marker-end="url(#tunnel-arrow)" />
          <line x1="225" y1="105" x2="245" y2="105" stroke="#27ae60" stroke-width="2.5" marker-end="url(#tunnel-arrow)" />
          <text x="200" y="75" text-anchor="middle" font-size="11" fill="#27ae60" font-weight="bold">Cooper pair</text>
          <text x="200" y="87" text-anchor="middle" font-size="11" fill="#27ae60" font-weight="bold">tunneling</text>
        </g>

        <!-- Current flow indicator -->
        <path d="M 105 155 Q 200 145, 295 155" stroke="#e74c3c" stroke-width="3" fill="none" marker-end="url(#josephson-arrow)" />
        <text x="200" y="175" text-anchor="middle" font-size="14" font-weight="bold" fill="#c0392b">Supercurrent I</text>

        <!-- Phase difference annotation -->
        <rect x="375" y="55" width="115" height="90" fill="white" stroke="#ddd" stroke-width="1" rx="5" />
        <text x="432" y="75" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Phase difference:</text>
        <text x="432" y="95" text-anchor="middle" font-size="14" font-weight="bold" fill="#8e44ad">phi = phi_2 - phi_1</text>
        <text x="432" y="115" text-anchor="middle" font-size="11" fill="#555">Critical current:</text>
        <text x="432" y="132" text-anchor="middle" font-size="13" font-weight="bold" fill="#c0392b">I = I_c sin(phi)</text>

        <!-- Junction types note -->
        <text x="200" y="200" text-anchor="middle" font-size="12" fill="#555">Types: SIS (superconductor-insulator-superconductor), SNS, SCS</text>
      </svg>
    </div>
  </div>
</div>

## Quantum Hall Effects

Confine electrons to a plane, cool them down, and crank up a perpendicular magnetic field, and something extraordinary happens: the transverse (Hall) conductance locks onto exact multiples of $e^2/h$, reproducible to better than one part in a billion regardless of sample shape or disorder. That precision is no accident — it is the first laboratory signature of **topology** in a material. The conductance counts a topological invariant that cannot change under smooth deformation, which is why it is immune to the messy details of any real sample. (The quantum Hall resistance now defines the SI ohm.)

### Integer Quantum Hall Effect
The magnetic field bunches the electron energies into massively degenerate **Landau levels**:

$$E_n = \hbar\omega_c\left(n + \tfrac{1}{2}\right), \qquad \omega_c = \frac{eB}{m}.$$

When an integer number $n$ of these levels is exactly filled, the bulk is gapped and insulating, while current flows along dissipationless **edge channels**. The Hall conductance is then quantized:

$$\sigma_{xy} = \frac{n e^2}{h}.$$

### Fractional Quantum Hall Effect
At *fractional* filling the single-particle picture fails — the plateaus appear only because of strong electron-electron interactions, which organize the electrons into an incompressible quantum fluid with **fractionally charged** excitations.

Occurs at fractional filling $\nu = \frac{1}{3}, \frac{2}{5}, \frac{5}{2}, ...$

Laughlin wavefunction for $\nu = 1/m$:
$$\Psi = \prod_{i<j}(z_i - z_j)^m e^{-\sum_i |z_i|^2/4l_B^2}$$

Composite fermions: electrons bound to flux quanta.

## Topological Phases

For most of the 20th century, Landau's paradigm classified phases by **symmetry breaking** — a magnet picks a direction, a crystal breaks translation symmetry. Topological phases break this mold: they are distinguished not by any local order parameter but by a *global*, integer-valued invariant of their wavefunctions. Two insulators can look identical locally yet be topologically distinct, and that distinction is robust — it cannot change without closing the energy gap. The price (or the gift) of a nontrivial invariant is protected, conducting states at the boundary.

### Berry Phase
The mathematical engine behind topological phases is the **Berry phase** — the geometric phase a quantum state accumulates when its Hamiltonian is carried slowly around a closed loop in parameter space:

$$\gamma = i\oint \langle n|\nabla_{\mathbf{R}}|n\rangle \cdot d\mathbf{R}.$$

Berry curvature:
$$\Omega_n(\mathbf{k}) = \nabla_k \times \langle n|\nabla_k|n\rangle$$

### Topological Insulators
Bulk insulator with conducting surface states protected by time-reversal symmetry.

Z₂ invariant distinguishes from ordinary insulators:
$$(-1)^{\nu} = \prod_{i=1}^{4} \text{Pf}[w(\Gamma_i)]/\sqrt{\det[w(\Gamma_i)]}$$

Effective Hamiltonian for surface:
$$H = v_F(\sigma_x k_y - \sigma_y k_x)$$

**3D Topological Insulator Surface States:**
- Linear dispersion (Dirac cone)
- Spin-momentum locking
- Protected crossing at TRIM points
- Absence of backscattering

### Chern Insulators
Characterized by Chern number:
$$C = \frac{1}{2\pi} \int_{BZ} d^2k \, \Omega(\mathbf{k})$$

Non-zero Chern number implies chiral edge states.

## Strongly Correlated Systems

Band theory quietly assumes electrons move independently in an average potential. That assumption breaks down spectacularly when the Coulomb repulsion between electrons rivals their kinetic energy. In these **strongly correlated** systems, band theory can be qualitatively *wrong* — predicting a metal where experiment finds an insulator — and the richest phenomena in condensed matter (high-$T_c$ superconductivity, heavy fermions, quantum magnetism) live here.

### Hubbard Model
The minimal model of correlation keeps just two competing terms: electrons gain energy $t$ by hopping between neighboring sites, but pay an energy penalty $U$ whenever two of them (opposite spins) sit on the same site:

$$H = -t\sum_{\langle ij\rangle,\sigma} c_{i\sigma}^\dagger c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow}.$$

When hopping wins ($U \ll t$) the system is a conventional metal. When repulsion wins ($U \gg t$) at half-filling, electrons localize one-per-site to avoid the penalty — a **Mott insulator**, insulating purely because of interactions, not band structure. The competition between these limits drives the **Mott metal–insulator transition** and is widely believed to hold the key to high-temperature superconductivity.

### Heavy Fermions
In certain rare-earth and actinide compounds, conduction electrons hybridize with localized $f$-electrons via the **Kondo effect**, dressing them into quasiparticles with enormous effective mass — $m^* \gg m_e$, sometimes by a factor of hundreds. Despite this, they often remain well-described as a (very heavy) Fermi liquid at low temperature, a striking validation of Landau's framework even in a strongly interacting setting.

### High-Temperature Superconductivity
The cuprates are quasi-2D copper-oxide layers that superconduct at temperatures far above the BCS expectation, with an unconventional **$d$-wave** pairing symmetry. Their phase diagram is a battleground of competing orders — antiferromagnetic insulator, mysterious pseudogap, and superconducting dome — as a function of doping. Explaining it from a model as simple as the Hubbard Hamiltonian remains one of the central unsolved problems in physics.

## Soft Condensed Matter

### Liquid Crystals
- Nematic: orientational order
- Smectic: orientational + 1D positional order
- Cholesteric: twisted nematic

Frank free energy:
$$F = \frac{1}{2}\int d^3r [K_1(\nabla \cdot \mathbf{n})^2 + K_2(\mathbf{n} \cdot \nabla \times \mathbf{n})^2 + K_3(\mathbf{n} \times \nabla \times \mathbf{n})^2]$$

### Polymers
Random walk model: $\langle R^2 \rangle = Nl^2$

Flory radius in good solvent: $R_F \sim N^{3/5}$

### Colloids
DLVO theory: balance of van der Waals attraction and electrostatic repulsion.

Debye screening length: $\lambda_D = \sqrt{\frac{\epsilon k_B T}{2e^2 n_0}}$

---

<div class="page-nav" style="display: flex; justify-content: space-between; margin-top: 2rem;">
  <a href="./">&larr; Condensed Matter Physics (Hub)</a>
  <a href="advanced-formalism.html">Graduate-Level Formalism &amp; Experiment &rarr;</a>
</div>

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="advanced-formalism.html">Graduate-Level Formalism &amp; Experiment</a> — Bogoliubov-de Gennes, Chern-Simons, DMFT, and experimental probes.</li>
    <li><a href="./">Condensed Matter Physics (Hub)</a> — crystal structure, band theory, and magnetism.</li>
    <li><a href="../quantum-field-theory.html">Quantum Field Theory</a> — field-theoretic methods for collective excitations.</li>
  </ul>
</div>
