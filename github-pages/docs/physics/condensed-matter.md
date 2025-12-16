---
layout: docs
title: Condensed Matter Physics
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section">
  <div class="hero-content">
    
    <p class="hero-subtitle">Exploring the Quantum World of Materials</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Condensed matter physics studies the physical properties of matter in its condensed phases, primarily solids and liquids. It is the largest field of contemporary physics, encompassing phenomena from superconductivity to topological insulators.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-cube"></i>
      <h4>Crystal Structure</h4>
      <p>Periodic arrangements and their properties</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-bolt"></i>
      <h4>Electronic Properties</h4>
      <p>Band theory and quantum behavior</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-magnet"></i>
      <h4>Emergent Phenomena</h4>
      <p>Superconductivity and magnetism</p>
    </div>
  </div>
</div>

## Crystal Structure

<div class="crystal-section">
  <h3><i class="fas fa-gem"></i> Bravais Lattices</h3>
  <p>14 distinct lattice types in 3D, characterized by lattice vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$</p>
  
  <div class="lattice-visual">
    <div class="equation-highlight">
      <p>Position vector:</p>
      $$\mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3$$
    </div>
    
    <div class="lattice-examples">
      <svg viewBox="0 0 600 200" class="lattice-diagram">
        <!-- Simple Cubic -->
        <g transform="translate(0,0)">
          <text x="100" y="20" text-anchor="middle" font-size="12">Simple Cubic</text>
          <!-- Unit cell -->
          <line x1="50" y1="50" x2="150" y2="50" stroke="#3498db" stroke-width="2" />
          <line x1="50" y1="50" x2="50" y2="150" stroke="#3498db" stroke-width="2" />
          <line x1="50" y1="50" x2="80" y2="30" stroke="#3498db" stroke-width="2" />
          <line x1="150" y1="50" x2="150" y2="150" stroke="#3498db" stroke-width="2" />
          <line x1="150" y1="50" x2="180" y2="30" stroke="#3498db" stroke-width="2" />
          <line x1="50" y1="150" x2="150" y2="150" stroke="#3498db" stroke-width="2" />
          <line x1="50" y1="150" x2="80" y2="130" stroke="#3498db" stroke-width="2" />
          <line x1="80" y1="30" x2="180" y2="30" stroke="#3498db" stroke-width="2" />
          <line x1="80" y1="30" x2="80" y2="130" stroke="#3498db" stroke-width="2" />
          <line x1="180" y1="30" x2="180" y2="130" stroke="#3498db" stroke-width="2" />
          <line x1="150" y1="150" x2="180" y2="130" stroke="#3498db" stroke-width="2" />
          <line x1="80" y1="130" x2="180" y2="130" stroke="#3498db" stroke-width="2" />
          <!-- Atoms -->
          <circle cx="50" cy="50" r="5" fill="#e74c3c" />
          <circle cx="150" cy="50" r="5" fill="#e74c3c" />
          <circle cx="50" cy="150" r="5" fill="#e74c3c" />
          <circle cx="150" cy="150" r="5" fill="#e74c3c" />
          <circle cx="80" cy="30" r="5" fill="#e74c3c" />
          <circle cx="180" cy="30" r="5" fill="#e74c3c" />
          <circle cx="80" cy="130" r="5" fill="#e74c3c" />
          <circle cx="180" cy="130" r="5" fill="#e74c3c" />
        </g>
        
        <!-- FCC -->
        <g transform="translate(200,0)">
          <text x="100" y="20" text-anchor="middle" font-size="12">Face-Centered Cubic</text>
          <!-- Similar structure with face centers -->
          <!-- Base cube -->
          <line x1="50" y1="50" x2="150" y2="50" stroke="#27ae60" stroke-width="2" />
          <line x1="50" y1="50" x2="50" y2="150" stroke="#27ae60" stroke-width="2" />
          <line x1="50" y1="50" x2="80" y2="30" stroke="#27ae60" stroke-width="2" />
          <!-- Corner atoms -->
          <circle cx="50" cy="50" r="5" fill="#f39c12" />
          <circle cx="150" cy="50" r="5" fill="#f39c12" />
          <!-- Face center atoms -->
          <circle cx="100" cy="50" r="5" fill="#9b59b6" />
          <circle cx="50" cy="100" r="5" fill="#9b59b6" />
          <circle cx="100" cy="100" r="5" fill="#9b59b6" />
        </g>
        
        <!-- BCC -->
        <g transform="translate(400,0)">
          <text x="100" y="20" text-anchor="middle" font-size="12">Body-Centered Cubic</text>
          <!-- Base cube with body center -->
          <line x1="50" y1="50" x2="150" y2="50" stroke="#e67e22" stroke-width="2" />
          <line x1="50" y1="50" x2="50" y2="150" stroke="#e67e22" stroke-width="2" />
          <!-- Corner atoms -->
          <circle cx="50" cy="50" r="5" fill="#2c3e50" />
          <circle cx="150" cy="50" r="5" fill="#2c3e50" />
          <!-- Body center atom -->
          <circle cx="100" cy="90" r="5" fill="#c0392b" />
        </g>
      </svg>
    </div>
  </div>
  
  <div class="reciprocal-lattice">
    <h3><i class="fas fa-sync-alt"></i> Reciprocal Lattice</h3>
    <p>Defined by vectors satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\delta_{ij}$:</p>
    
    <div class="equation-box">
      $$\mathbf{b}_1 = 2\pi \frac{\mathbf{a}_2 \times \mathbf{a}_3}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}$$
    </div>
    
    <div class="brillouin-zone">
      <p class="note">First Brillouin zone: Wigner-Seitz cell of reciprocal lattice</p>
      <svg viewBox="0 0 300 200" class="brillouin-diagram">
        <!-- 2D Brillouin zone example -->
        <polygon points="150,50 220,100 150,150 80,100" fill="#3498db" opacity="0.3" stroke="#2980b9" stroke-width="2" />
        <text x="150" y="100" text-anchor="middle" font-size="12" fill="#2c3e50">1st BZ</text>
        <!-- High symmetry points -->
        <circle cx="150" cy="100" r="3" fill="#e74c3c" />
        <text x="150" y="115" text-anchor="middle" font-size="10">Γ</text>
        <circle cx="220" cy="100" r="3" fill="#e74c3c" />
        <text x="230" y="100" font-size="10">X</text>
        <circle cx="185" cy="75" r="3" fill="#e74c3c" />
        <text x="195" y="75" font-size="10">M</text>
      </svg>
    </div>
  </div>
  
  <div class="xray-diffraction">
    <h3><i class="fas fa-radiation"></i> X-ray Diffraction</h3>
    
    <div class="bragg-law">
      <p>Bragg's law:</p>
      <div class="equation-box highlighted">$$2d\sin\theta = n\lambda$$</div>
      
      <svg viewBox="0 0 350 200" class="bragg-diagram">
        <!-- Crystal planes -->
        <line x1="50" y1="50" x2="250" y2="50" stroke="#95a5a6" stroke-width="2" />
        <line x1="50" y1="100" x2="250" y2="100" stroke="#95a5a6" stroke-width="2" />
        <line x1="50" y1="150" x2="250" y2="150" stroke="#95a5a6" stroke-width="2" />
        <text x="260" y="100" font-size="10">d</text>
        
        <!-- Incident rays -->
        <path d="M 20 20 L 100 50" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)" />
        <path d="M 20 70 L 100 100" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)" />
        
        <!-- Reflected rays -->
        <path d="M 100 50 L 180 20" stroke="#3498db" stroke-width="2" marker-end="url(#arrow)" />
        <path d="M 100 100 L 180 70" stroke="#3498db" stroke-width="2" marker-end="url(#arrow)" />
        
        <!-- Angle -->
        <path d="M 80 50 Q 100 40, 120 50" fill="none" stroke="#2c3e50" stroke-width="1" />
        <text x="100" y="35" text-anchor="middle" font-size="10">θ</text>
      </svg>
    </div>
    
    <div class="structure-factor">
      <p>Structure factor:</p>
      <div class="equation-box">$$F_{\mathbf{G}} = \sum_j f_j e^{i\mathbf{G} \cdot \mathbf{r}_j}$$</div>
    </div>
  </div>
</div>

## Electronic Band Theory

<div class="band-theory-section">
  <div class="bloch-theorem">
    <h3><i class="fas fa-wave-square"></i> Bloch's Theorem</h3>
    <p>Wavefunctions in periodic potential:</p>
    
    <div class="equation-box bloch">
      $$\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k} \cdot \mathbf{r}} u_{n\mathbf{k}}(\mathbf{r})$$
    </div>
    
    <p class="note">Where $u_{n\mathbf{k}}(\mathbf{r})$ has lattice periodicity</p>
    
    <div class="bloch-visual">
      <svg viewBox="0 0 400 200">
        <!-- Periodic potential -->
        <path d="M 20 150 Q 40 100, 60 150 T 100 150 T 140 150 T 180 150 T 220 150 T 260 150 T 300 150 T 340 150 T 380 150" 
              fill="none" stroke="#95a5a6" stroke-width="2" />
        <text x="200" y="180" text-anchor="middle" font-size="10">V(x)</text>
        
        <!-- Bloch wave envelope -->
        <path d="M 20 100 Q 200 50, 380 100" fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,2" />
        <text x="40" y="90" font-size="10" fill="#e74c3c">e^{ikx}</text>
        
        <!-- Modulated wave -->
        <path d="M 20 100 Q 30 80, 40 100 T 60 100 Q 70 75, 80 100 T 100 100 Q 110 70, 120 100 T 140 100" 
              fill="none" stroke="#3498db" stroke-width="2" />
        <text x="100" y="60" font-size="10" fill="#3498db">ψ(x)</text>
      </svg>
    </div>
  </div>
  
  <div class="band-models">
    <div class="model-card nfe">
      <h3><i class="fas fa-chart-line"></i> Nearly Free Electron Model</h3>
      <p>Weak periodic potential creates band gaps at Brillouin zone boundaries</p>
      
      <div class="gap-equation">
        <p>Gap size:</p>
        <div class="equation-box">$$\Delta E = 2|V_{\mathbf{G}}|$$</div>
        <p class="variable-note">where $V_{\mathbf{G}}$ is Fourier component of potential</p>
      </div>
      
      <svg viewBox="0 0 250 200" class="band-diagram">
        <!-- Energy vs k plot -->
        <line x1="30" y1="170" x2="220" y2="170" stroke="#2c3e50" stroke-width="2" />
        <line x1="30" y1="170" x2="30" y2="30" stroke="#2c3e50" stroke-width="2" />
        <text x="125" y="190" text-anchor="middle" font-size="10">k</text>
        <text x="10" y="100" text-anchor="middle" font-size="10" transform="rotate(-90 10 100)">E</text>
        
        <!-- Parabolic bands with gap -->
        <path d="M 30 150 Q 90 100, 115 80" fill="none" stroke="#3498db" stroke-width="2" />
        <path d="M 135 100 Q 160 80, 220 50" fill="none" stroke="#3498db" stroke-width="2" />
        
        <!-- Band gap -->
        <line x1="115" y1="80" x2="135" y2="100" stroke="#e74c3c" stroke-width="1" stroke-dasharray="3,3" />
        <text x="125" y="95" text-anchor="middle" font-size="9" fill="#e74c3c">ΔE</text>
        
        <!-- BZ boundary -->
        <line x1="125" y1="170" x2="125" y2="30" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3" />
        <text x="125" y="25" text-anchor="middle" font-size="9">π/a</text>
      </svg>
    </div>
    
    <div class="model-card tight-binding">
      <h3><i class="fas fa-link"></i> Tight-Binding Model</h3>
      <p>Start from atomic orbitals:</p>
      
      <div class="equation-box">
        $$\psi_{\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{R}} e^{i\mathbf{k} \cdot \mathbf{R}} \phi(\mathbf{r} - \mathbf{R})$$
      </div>
      
      <p>Dispersion relation:</p>
      <div class="equation-box">
        $$E(\mathbf{k}) = \epsilon_0 - 2t[\cos(k_xa) + \cos(k_ya) + \cos(k_za)]$$
      </div>
      
      <svg viewBox="0 0 250 200" class="tb-diagram">
        <!-- Cosine band -->
        <line x1="30" y1="170" x2="220" y2="170" stroke="#2c3e50" stroke-width="2" />
        <line x1="30" y1="170" x2="30" y2="30" stroke="#2c3e50" stroke-width="2" />
        <text x="125" y="190" text-anchor="middle" font-size="10">k</text>
        <text x="10" y="100" text-anchor="middle" font-size="10" transform="rotate(-90 10 100)">E</text>
        
        <!-- Cosine curve -->
        <path d="M 30 100 Q 70 60, 125 50 Q 180 60, 220 100" fill="none" stroke="#27ae60" stroke-width="3" />
        
        <!-- Band width -->
        <line x1="25" y1="50" x2="35" y2="50" stroke="#2c3e50" stroke-width="1" />
        <line x1="25" y1="100" x2="35" y2="100" stroke="#2c3e50" stroke-width="1" />
        <text x="15" y="75" text-anchor="middle" font-size="9" transform="rotate(-90 15 75)">4t</text>
      </svg>
    </div>
  </div>
  
  <div class="density-of-states">
    <h3><i class="fas fa-chart-area"></i> Density of States</h3>
    
    <div class="equation-box">
      $$g(E) = \sum_n \int \frac{d^3k}{(2\pi)^3} \delta(E - E_n(\mathbf{k}))$$
    </div>
    
    <p class="singularity-note">Van Hove singularities occur where $\nabla_k E_n(\mathbf{k}) = 0$</p>
    
    <div class="dos-plots">
      <svg viewBox="0 0 500 200">
        <!-- 1D DOS -->
        <g transform="translate(0,0)">
          <text x="75" y="20" text-anchor="middle" font-size="11">1D DOS</text>
          <line x1="30" y1="150" x2="120" y2="150" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="150" x2="30" y2="50" stroke="#2c3e50" stroke-width="2" />
          <path d="M 30 140 Q 50 120, 60 50 Q 70 120, 90 140 L 120 140" 
                fill="#3498db" opacity="0.3" stroke="#3498db" stroke-width="2" />
        </g>
        
        <!-- 2D DOS -->
        <g transform="translate(170,0)">
          <text x="75" y="20" text-anchor="middle" font-size="11">2D DOS</text>
          <line x1="30" y1="150" x2="120" y2="150" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="150" x2="30" y2="50" stroke="#2c3e50" stroke-width="2" />
          <path d="M 30 140 L 60 140 L 60 100 L 90 100 L 90 60 L 120 60" 
                fill="#e74c3c" opacity="0.3" stroke="#e74c3c" stroke-width="2" />
        </g>
        
        <!-- 3D DOS -->
        <g transform="translate(340,0)">
          <text x="75" y="20" text-anchor="middle" font-size="11">3D DOS</text>
          <line x1="30" y1="150" x2="120" y2="150" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="150" x2="30" y2="50" stroke="#2c3e50" stroke-width="2" />
          <path d="M 30 140 Q 60 130, 75 100 T 120 50" 
                fill="#27ae60" opacity="0.3" stroke="#27ae60" stroke-width="2" />
        </g>
      </svg>
    </div>
  </div>
</div>

## Semiconductors

### Band Structure
- Valence band maximum (VBM)
- Conduction band minimum (CBM)
- Direct gap: VBM and CBM at same k-point
- Indirect gap: VBM and CBM at different k-points

### Carrier Statistics
Intrinsic carrier concentration:
$$n_i = \sqrt{N_c N_v} e^{-E_g/2k_BT}$$

Where $N_c$, $N_v$ are effective densities of states.

### Doping
- n-type: Donors provide electrons
- p-type: Acceptors provide holes

Mass action law: $np = n_i^2$

### p-n Junction
Built-in potential:
$$V_{bi} = \frac{k_BT}{e} \ln\left(\frac{N_A N_D}{n_i^2}\right)$$

Depletion width:
$$W = \sqrt{\frac{2\epsilon_s V_{bi}}{e}\left(\frac{N_A + N_D}{N_A N_D}\right)}$$

### Recent Advances in 2D Semiconductors (2023-2024)
- **Moiré Engineering**: Twisted bilayer TMDs showing correlated insulator states
- **Valleytronics**: Valley-selective optical excitation in monolayer WSe₂
- **Exciton Condensates**: Room-temperature exciton-polariton BEC in perovskites
- **Quantum Emitters**: Single-photon sources in hBN defects

## Metals and Fermi Liquids

### Drude Model
Conductivity: $\sigma = \frac{ne^2\tau}{m}$

Hall coefficient: $R_H = -\frac{1}{ne}$

### Sommerfeld Model
Free electron gas with Fermi-Dirac statistics.

Fermi energy: $E_F = \frac{\hbar^2}{2m}(3\pi^2n)^{2/3}$

Electronic specific heat: $C_V = \gamma T$ where $\gamma = \frac{\pi^2 k_B^2 g(E_F)}{3}$

### Fermi Liquid Theory
Quasiparticles with effective mass $m^*$ and interactions.

Landau parameters describe quasiparticle interactions:
$$\delta E = \sum_{k\sigma} \epsilon_k n_{k\sigma} + \frac{1}{2V}\sum_{kk'\sigma\sigma'} f_{kk'}^{\sigma\sigma'} n_{k\sigma} n_{k'\sigma'}$$

## Magnetism

### Paramagnetism
Curie law: $\chi = \frac{C}{T}$ where $C = \frac{N\mu_0\mu_B^2 g^2 J(J+1)}{3k_B}$

Pauli paramagnetism (metals): $\chi = \mu_0\mu_B^2 g(E_F)$

### Ferromagnetism
Mean field theory:
$$M = Ng\mu_B J B_J\left(\frac{g\mu_B J(H + \lambda M)}{k_B T}\right)$$

Curie temperature: $T_C = \frac{g\mu_B J(J+1)\lambda}{3k_B}$

### Antiferromagnetism
Néel temperature marks onset of staggered magnetization.

Two-sublattice model gives susceptibility:
$$\chi = \frac{2C}{T + T_N}$$

### Spin Waves
Low-energy excitations in ordered magnets.

Dispersion for ferromagnet: $\omega_k = \frac{2JS}{\hbar}(1 - \cos(ka))$

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
          <svg viewBox="0 0 100 80">
            <line x1="10" y1="70" x2="90" y2="70" stroke="#2c3e50" stroke-width="2" />
            <line x1="10" y1="70" x2="10" y2="10" stroke="#2c3e50" stroke-width="2" />
            <path d="M 10 20 L 50 20 L 50 65" stroke="#e74c3c" stroke-width="3" fill="none" />
            <text x="50" y="78" text-anchor="middle" font-size="8">Tc</text>
            <text x="5" y="40" font-size="7" transform="rotate(-90 5 40)">R</text>
          </svg>
        </div>
      </div>
      
      <div class="phenomenon-card">
        <i class="fas fa-magnet"></i>
        <h4>Meissner Effect</h4>
        <p>Expulsion of magnetic field</p>
        <div class="meissner-visual">
          <svg viewBox="0 0 100 80">
            <!-- Superconductor -->
            <ellipse cx="50" cy="50" rx="30" ry="20" fill="#3498db" opacity="0.5" />
            <!-- Field lines -->
            <path d="M 10 20 Q 20 30, 20 50 Q 20 70, 10 80" stroke="#e74c3c" stroke-width="2" fill="none" />
            <path d="M 90 20 Q 80 30, 80 50 Q 80 70, 90 80" stroke="#e74c3c" stroke-width="2" fill="none" />
            <path d="M 10 35 Q 25 40, 50 40 Q 75 40, 90 35" stroke="#e74c3c" stroke-width="2" fill="none" />
            <path d="M 10 65 Q 25 60, 50 60 Q 75 60, 90 65" stroke="#e74c3c" stroke-width="2" fill="none" />
            <text x="50" y="50" text-anchor="middle" font-size="8" fill="white">SC</text>
          </svg>
        </div>
      </div>
      
      <div class="phenomenon-card">
        <i class="fas fa-ring"></i>
        <h4>Flux Quantization</h4>
        <p>$\Phi = n\frac{h}{2e}$</p>
        <div class="flux-quantum">
          <svg viewBox="0 0 100 80">
            <circle cx="50" cy="40" r="25" fill="none" stroke="#27ae60" stroke-width="3" />
            <text x="50" y="45" text-anchor="middle" font-size="10">Φ₀</text>
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
        <div class="equation-box scrollable">
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
        <div class="equation-box">
          $$|\text{BCS}\rangle = \prod_k (u_k + v_k c_{k\uparrow}^\dagger c_{-k\downarrow}^\dagger)|0\rangle$$
        </div>
        
        <p>Gap equation:</p>
        <div class="equation-box">
          $$\Delta_k = -\sum_{k'} V_{kk'} \frac{\Delta_{k'}}{2E_{k'}} \tanh\left(\frac{E_{k'}}{2k_B T}\right)$$
        </div>
        
        <p>Where $E_k = \sqrt{\epsilon_k^2 + |\Delta_k|^2}$</p>
        
        <div class="cooper-pair-visual">
          <svg viewBox="0 0 200 100">
            <!-- Cooper pair illustration -->
            <circle cx="70" cy="50" r="8" fill="#3498db" />
            <text x="70" y="55" text-anchor="middle" font-size="10" fill="white">↑</text>
            <circle cx="130" cy="50" r="8" fill="#e74c3c" />
            <text x="130" y="55" text-anchor="middle" font-size="10" fill="white">↓</text>
            <path d="M 78 50 Q 100 30, 122 50" stroke="#95a5a6" stroke-width="2" stroke-dasharray="3,3" fill="none" />
            <text x="100" y="80" text-anchor="middle" font-size="10">k↑, -k↓</text>
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
        <div class="equation-box">$$I = I_c \sin\phi$$</div>
        <p class="effect-desc">Supercurrent without voltage</p>
      </div>
      
      <div class="josephson-type">
        <h4>AC Josephson</h4>
        <div class="equation-box">$$\frac{d\phi}{dt} = \frac{2eV}{\hbar}$$</div>
        <p class="effect-desc">Oscillating current with DC voltage</p>
      </div>
    </div>
    
    <div class="josephson-junction">
      <svg viewBox="0 0 300 150">
        <!-- Junction diagram -->
        <rect x="50" y="50" width="80" height="50" fill="#3498db" opacity="0.5" />
        <text x="90" y="80" text-anchor="middle" font-size="10" fill="white">SC1</text>
        <rect x="170" y="50" width="80" height="50" fill="#3498db" opacity="0.5" />
        <text x="210" y="80" text-anchor="middle" font-size="10" fill="white">SC2</text>
        <rect x="130" y="50" width="40" height="50" fill="#95a5a6" opacity="0.3" />
        <text x="150" y="80" text-anchor="middle" font-size="8">Barrier</text>
        <!-- Current flow -->
        <path d="M 90 120 Q 150 110, 210 120" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)" />
        <text x="150" y="130" text-anchor="middle" font-size="10">I</text>
      </svg>
    </div>
  </div>
</div>

## Quantum Hall Effects

### Integer Quantum Hall Effect
Quantized Hall conductance: $\sigma_{xy} = \frac{ne^2}{h}$

Landau levels: $E_n = \hbar\omega_c(n + \frac{1}{2})$

### Fractional Quantum Hall Effect
Occurs at fractional filling $\nu = \frac{1}{3}, \frac{2}{5}, \frac{5}{2}, ...$

Laughlin wavefunction for $\nu = 1/m$:
$$\Psi = \prod_{i<j}(z_i - z_j)^m e^{-\sum_i |z_i|^2/4l_B^2}$$

Composite fermions: electrons bound to flux quanta.

## Topological Phases

### Berry Phase
$$\gamma = i\oint \langle n|\nabla_{\mathbf{R}}|n\rangle \cdot d\mathbf{R}$$

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

### Hubbard Model
$$H = -t\sum_{\langle ij\rangle,\sigma} c_{i\sigma}^\dagger c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow}$$

Mott transition occurs when $U \gg t$.

### Heavy Fermions
Effective mass $m^* \gg m_e$ due to Kondo effect.

Low-temperature behavior dominated by f-electron hybridization.

### High-Temperature Superconductivity
Cuprates: quasi-2D systems with d-wave pairing.

Phase diagram includes antiferromagnetic, pseudogap, and superconducting phases.

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

$$F = \int d^3\mathbf{r} \left[\alpha|\psi|^2 + \frac{\beta}{2}|\psi|^4 + \frac{1}{2m^*}|(i\hbar\nabla - 2e\mathbf{A})\psi|^2 + \frac{B^2}{2\mu_0}\right]$$

**GL equations:**

$$\alpha\psi + \beta|\psi|^2\psi + \frac{1}{2m^*}(i\hbar\nabla - 2e\mathbf{A})^2\psi = 0$$

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

## See Also

### Core Physics Topics:
- [Quantum Mechanics](quantum-mechanics.html) - Quantum foundations and wave functions
- [Statistical Mechanics](statistical-mechanics.html) - Many-body theory and phase transitions
- [Quantum Field Theory](quantum-field-theory.html) - Field theoretic methods in condensed matter
- [Thermodynamics](thermodynamics.html) - Macroscopic properties and phase diagrams

### Related Topics:
- [Classical Mechanics](classical-mechanics.html) - Lattice dynamics and phonons
- [Computational Physics](computational-physics.html) - DFT, Monte Carlo, and MD simulations
- [String Theory](string-theory.html) - AdS/CMT correspondence and holographic duality
- [Relativity](relativity.html) - Relativistic effects in graphene and Weyl semimetals