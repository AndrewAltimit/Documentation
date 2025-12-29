---
layout: docs
title: Statistical Mechanics
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section">
  <div class="hero-content">
    
    <p class="hero-subtitle">Bridging the Microscopic and Macroscopic Worlds</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Statistical mechanics provides the microscopic foundation for thermodynamics by connecting the behavior of individual particles to macroscopic observables. It explains how the laws of thermodynamics emerge from the statistical behavior of large ensembles of particles.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-dice"></i>
      <h4>Probabilistic Nature</h4>
      <p>Macroscopic properties emerge from statistical averages</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-layer-group"></i>
      <h4>Ensembles</h4>
      <p>Different statistical descriptions for different constraints</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-exchange-alt"></i>
      <h4>Phase Transitions</h4>
      <p>Critical phenomena and universality</p>
    </div>
  </div>
</div>

## Fundamental Principles

<div class="principle-section">
  <h3>Microstates and Macrostates</h3>
  
  <div class="concept-grid">
    <div class="concept-card microstate">
      <h4><i class="fas fa-atom"></i> Microstate</h4>
      <p>Complete specification of the quantum state of every particle</p>
      <div class="visual-example">
        <svg viewBox="0 0 200 100">
          <!-- Particles with individual states -->
          <circle cx="30" cy="30" r="5" fill="#3498db" />
          <text x="30" y="20" text-anchor="middle" font-size="10">↑</text>
          <circle cx="60" cy="50" r="5" fill="#e74c3c" />
          <text x="60" y="40" text-anchor="middle" font-size="10">↓</text>
          <circle cx="90" cy="40" r="5" fill="#3498db" />
          <text x="90" y="30" text-anchor="middle" font-size="10">↑</text>
          <circle cx="120" cy="60" r="5" fill="#e74c3c" />
          <text x="120" y="50" text-anchor="middle" font-size="10">↓</text>
          <circle cx="150" cy="35" r="5" fill="#3498db" />
          <text x="150" y="25" text-anchor="middle" font-size="10">↑</text>
          <text x="100" y="90" text-anchor="middle" font-size="12">Individual states</text>
        </svg>
      </div>
    </div>
    
    <div class="concept-card macrostate">
      <h4><i class="fas fa-temperature-high"></i> Macrostate</h4>
      <p>Specification of macroscopic variables (T, P, V, N, E)</p>
      <div class="visual-example">
        <svg viewBox="0 0 200 100">
          <!-- Container with properties -->
          <rect x="20" y="20" width="160" height="60" fill="none" stroke="#2c3e50" stroke-width="2" />
          <text x="100" y="50" text-anchor="middle" font-size="14">T = 300K</text>
          <text x="100" y="65" text-anchor="middle" font-size="14">P = 1 atm</text>
          <text x="100" y="95" text-anchor="middle" font-size="12">Bulk properties</text>
        </svg>
      </div>
    </div>
  </div>
  
  <div class="fundamental-postulate">
    <i class="fas fa-balance-scale"></i>
    <h4>Fundamental Postulate</h4>
    <p>All accessible microstates are equally probable</p>
  </div>
</div>

### Statistical Ensembles

<div class="ensemble-container">
  <div class="ensemble-card microcanonical">
    <h4><i class="fas fa-lock"></i> Microcanonical Ensemble (NVE)</h4>
    <p class="ensemble-desc">Isolated system with fixed energy, volume, and particle number</p>
    
    <div class="ensemble-visual">
      <svg viewBox="0 0 200 150">
        <!-- Isolated system -->
        <rect x="50" y="30" width="100" height="80" fill="#ecf0f1" stroke="#34495e" stroke-width="3" />
        <text x="100" y="70" text-anchor="middle" font-size="12">E, V, N fixed</text>
        <!-- Barrier symbols -->
        <path d="M 40 20 L 40 120" stroke="#e74c3c" stroke-width="4" />
        <path d="M 160 20 L 160 120" stroke="#e74c3c" stroke-width="4" />
        <text x="100" y="130" text-anchor="middle" font-size="10">No exchange</text>
      </svg>
    </div>
    
    <div class="ensemble-equations">
      <p><strong>Partition function:</strong> $\Omega(E,V,N)$ = number of microstates</p>
      <p><strong>Entropy:</strong> $S = k_B \ln \Omega$</p>
    </div>
  </div>
  
  <div class="ensemble-card canonical">
    <h4><i class="fas fa-thermometer-half"></i> Canonical Ensemble (NVT)</h4>
    <p class="ensemble-desc">System in thermal equilibrium with heat bath at temperature T</p>
    
    <div class="ensemble-visual">
      <svg viewBox="0 0 200 150">
        <!-- System and heat bath -->
        <rect x="60" y="40" width="80" height="60" fill="#3498db" stroke="#2980b9" stroke-width="2" opacity="0.8" />
        <text x="100" y="75" text-anchor="middle" font-size="11" fill="white">System</text>
        <rect x="20" y="20" width="160" height="100" fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" />
        <text x="100" y="135" text-anchor="middle" font-size="10">Heat bath (T)</text>
        <!-- Energy exchange arrows -->
        <path d="M 90 100 L 90 115" stroke="#27ae60" stroke-width="2" marker-end="url(#arrowhead)" />
        <path d="M 110 115 L 110 100" stroke="#27ae60" stroke-width="2" marker-end="url(#arrowhead)" />
      </svg>
    </div>
    
    <div class="ensemble-equations">
      <p><strong>Partition function:</strong></p>
      <div class="equation-box">$$Z = \sum_i e^{-\beta E_i} = \text{Tr}(e^{-\beta H})$$</div>
      <p>Where $\beta = \frac{1}{k_B T}$</p>
      <p><strong>Helmholtz free energy:</strong> $F = -k_B T \ln Z$</p>
    </div>
  </div>
  
  <div class="ensemble-card grand-canonical">
    <h4><i class="fas fa-exchange-alt"></i> Grand Canonical Ensemble (μVT)</h4>
    <p class="ensemble-desc">System can exchange particles and energy with reservoir</p>
    
    <div class="ensemble-visual">
      <svg viewBox="0 0 200 150">
        <!-- System and reservoir -->
        <rect x="60" y="40" width="80" height="60" fill="#9b59b6" stroke="#8e44ad" stroke-width="2" opacity="0.8" />
        <text x="100" y="75" text-anchor="middle" font-size="11" fill="white">System</text>
        <rect x="20" y="20" width="160" height="100" fill="none" stroke="#f39c12" stroke-width="2" stroke-dasharray="3,3" />
        <text x="100" y="135" text-anchor="middle" font-size="10">Reservoir (T, μ)</text>
        <!-- Particle and energy exchange -->
        <circle cx="75" cy="110" r="3" fill="#2ecc71" />
        <circle cx="85" cy="110" r="3" fill="#2ecc71" />
        <path d="M 80 100 L 80 115" stroke="#2ecc71" stroke-width="2" />
        <path d="M 120 115 L 120 100" stroke="#e74c3c" stroke-width="2" />
      </svg>
    </div>
    
    <div class="ensemble-equations">
      <p><strong>Grand partition function:</strong></p>
      <div class="equation-box">$$\mathcal{Z} = \sum_{N=0}^{\infty} \sum_i e^{-\beta(E_i - \mu N)}$$</div>
      <p><strong>Grand potential:</strong> $\Omega = -k_B T \ln \mathcal{Z}$</p>
    </div>
  </div>
</div>

<defs>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#27ae60" />
  </marker>
</defs>

## Classical Statistical Mechanics

<div class="classical-mechanics-section">
  <h3><i class="fas fa-chart-line"></i> Phase Space</h3>
  
  <div class="phase-space-visual">
    <p>6N-dimensional space of positions and momenta for N particles</p>
    
    <svg viewBox="0 0 400 200" class="phase-diagram">
      <!-- Phase space axes -->
      <line x1="50" y1="150" x2="350" y2="150" stroke="#2c3e50" stroke-width="2" />
      <line x1="50" y1="150" x2="50" y2="50" stroke="#2c3e50" stroke-width="2" />
      <text x="200" y="170" text-anchor="middle" font-size="12">Position (q)</text>
      <text x="30" y="100" text-anchor="middle" font-size="12" transform="rotate(-90 30 100)">Momentum (p)</text>
      
      <!-- Phase space trajectory -->
      <path d="M 100 120 Q 150 80, 200 100 T 300 90" fill="none" stroke="#3498db" stroke-width="2" />
      <circle cx="100" cy="120" r="3" fill="#e74c3c" />
      <circle cx="300" cy="90" r="3" fill="#27ae60" />
      
      <!-- Volume element -->
      <rect x="180" y="90" width="40" height="30" fill="#f39c12" opacity="0.3" stroke="#f39c12" />
      <text x="200" y="105" text-anchor="middle" font-size="10">dΓ</text>
    </svg>
    
    <div class="equation-highlight">
      <p>Phase space volume element:</p>
      $$d\Gamma = \prod_{i=1}^{N} d^3\mathbf{r}_i d^3\mathbf{p}_i$$
    </div>
  </div>
  
  <div class="theorem-box liouville">
    <h3><i class="fas fa-balance-scale"></i> Liouville's Theorem</h3>
    <p>Phase space density is conserved along trajectories:</p>
    <div class="equation-box">$$\frac{d\rho}{dt} = \frac{\partial \rho}{\partial t} + \{\rho, H\} = 0$$</div>
    
    <div class="visual-interpretation">
      <svg viewBox="0 0 300 150">
        <!-- Flow in phase space -->
        <ellipse cx="80" cy="75" rx="40" ry="30" fill="#3498db" opacity="0.3" />
        <text x="80" y="75" text-anchor="middle" font-size="10">t = 0</text>
        <path d="M 120 75 L 160 75" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
        <ellipse cx="220" cy="75" rx="30" ry="40" fill="#3498db" opacity="0.3" />
        <text x="220" y="75" text-anchor="middle" font-size="10">t = τ</text>
        <text x="150" y="130" text-anchor="middle" font-size="11">Volume preserved</text>
      </svg>
    </div>
  </div>
  
  <div class="partition-function-box">
    <h3><i class="fas fa-calculator"></i> Classical Partition Function</h3>
    <div class="equation-box">$$Z = \frac{1}{N!h^{3N}} \int e^{-\beta H(\mathbf{r},\mathbf{p})} d\Gamma$$</div>
    <p class="note">The factor $1/N!$ accounts for indistinguishability (Gibbs correction)</p>
  </div>
  
  <div class="equipartition-box">
    <h3><i class="fas fa-equals"></i> Equipartition Theorem</h3>
    <p>Each quadratic term in the energy contributes $\frac{1}{2}k_B T$ to the average energy</p>
    
    <div class="example-grid">
      <div class="example-card">
        <h4>Harmonic Oscillator</h4>
        <p>$\langle E \rangle = k_B T$</p>
        <span class="detail">(kinetic + potential)</span>
      </div>
      <div class="example-card">
        <h4>Ideal Gas Molecule</h4>
        <p>$\langle E_{trans} \rangle = \frac{3}{2}k_B T$</p>
        <span class="detail">(3 translational DOF)</span>
      </div>
    </div>
  </div>
</div>

## Quantum Statistical Mechanics

<div class="quantum-stat-section">
  <div class="density-matrix-box">
    <h3><i class="fas fa-th"></i> Density Matrix</h3>
    <p>For a mixed state:</p>
    <div class="equation-box">$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$</div>
    <p>Canonical density matrix: $\rho = \frac{e^{-\beta H}}{Z}$</p>
    
    <div class="matrix-visual">
      <svg viewBox="0 0 200 150">
        <!-- Density matrix representation -->
        <rect x="50" y="30" width="100" height="100" fill="none" stroke="#2c3e50" stroke-width="2" />
        <!-- Matrix elements -->
        <rect x="60" y="40" width="30" height="30" fill="#3498db" opacity="0.8" />
        <rect x="110" y="90" width="30" height="30" fill="#3498db" opacity="0.8" />
        <rect x="85" y="65" width="30" height="30" fill="#e74c3c" opacity="0.4" />
        <text x="100" y="20" text-anchor="middle" font-size="12">ρ</text>
      </svg>
    </div>
  </div>
  
  <div class="partition-function">
    <h3><i class="fas fa-sum"></i> Quantum Partition Function</h3>
    <div class="equation-box">$$Z = \text{Tr}(e^{-\beta H}) = \sum_n e^{-\beta E_n}$$</div>
  </div>
  
  <div class="statistics-comparison">
    <div class="stat-card fermi-dirac">
      <h3><i class="fas fa-minus-circle"></i> Fermi-Dirac Statistics</h3>
      <p class="particle-type">For fermions (half-integer spin)</p>
      
      <div class="occupation-formula">
        <p>Average occupation number:</p>
        <div class="equation-box">$$\langle n_i \rangle = \frac{1}{e^{\beta(\epsilon_i - \mu)} + 1}$$</div>
      </div>
      
      <div class="distribution-plot">
        <svg viewBox="0 0 250 150">
          <!-- Fermi-Dirac distribution -->
          <line x1="30" y1="120" x2="220" y2="120" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="120" x2="30" y2="20" stroke="#2c3e50" stroke-width="2" />
          <text x="125" y="140" text-anchor="middle" font-size="10">ε/μ</text>
          <text x="10" y="70" text-anchor="middle" font-size="10" transform="rotate(-90 10 70)">⟨n⟩</text>
          
          <!-- Fermi function curve -->
          <path d="M 30 30 Q 100 30, 125 70 T 220 110" fill="none" stroke="#e74c3c" stroke-width="3" />
          <line x1="125" y1="20" x2="125" y2="120" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3" />
          <text x="125" y="15" text-anchor="middle" font-size="9">μ</text>
          <text x="200" y="50" font-size="9" fill="#e74c3c">T > 0</text>
          
          <!-- T=0 step function -->
          <path d="M 30 30 L 125 30 L 125 110 L 220 110" fill="none" stroke="#2c3e50" stroke-width="2" stroke-dasharray="5,2" />
          <text x="60" y="45" font-size="9" fill="#2c3e50">T = 0</text>
        </svg>
      </div>
      
      <p class="special-note">At T = 0, becomes a step function at the Fermi energy</p>
    </div>
    
    <div class="stat-card bose-einstein">
      <h3><i class="fas fa-plus-circle"></i> Bose-Einstein Statistics</h3>
      <p class="particle-type">For bosons (integer spin)</p>
      
      <div class="occupation-formula">
        <p>Average occupation number:</p>
        <div class="equation-box">$$\langle n_i \rangle = \frac{1}{e^{\beta(\epsilon_i - \mu)} - 1}$$</div>
      </div>
      
      <div class="distribution-plot">
        <svg viewBox="0 0 250 150">
          <!-- Bose-Einstein distribution -->
          <line x1="30" y1="120" x2="220" y2="120" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="120" x2="30" y2="20" stroke="#2c3e50" stroke-width="2" />
          <text x="125" y="140" text-anchor="middle" font-size="10">ε/μ</text>
          <text x="10" y="70" text-anchor="middle" font-size="10" transform="rotate(-90 10 70)">⟨n⟩</text>
          
          <!-- Bose function curve -->
          <path d="M 40 110 Q 80 90, 100 60 Q 120 30, 125 20" fill="none" stroke="#3498db" stroke-width="3" />
          <path d="M 125 70 Q 160 85, 220 95" fill="none" stroke="#3498db" stroke-width="3" />
          <line x1="125" y1="20" x2="125" y2="120" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3" />
          <text x="125" y="15" text-anchor="middle" font-size="9">μ</text>
          
          <!-- BEC region -->
          <rect x="30" y="20" width="95" height="100" fill="#3498db" opacity="0.1" />
          <text x="77" y="60" text-anchor="middle" font-size="9" fill="#2980b9">BEC</text>
        </svg>
      </div>
      
      <p class="special-note">Allows for Bose-Einstein condensation when $\mu \to 0^-$</p>
    </div>
  </div>
</div>

## Ideal Gases

### Classical Ideal Gas
Partition function:
$$Z = \frac{V^N}{N!\lambda^{3N}}$$

Where $\lambda = \sqrt{\frac{2\pi\hbar^2}{mk_BT}}$ is the thermal de Broglie wavelength.

Equation of state: $PV = Nk_BT$

### Quantum Ideal Gases

#### Fermi Gas
At low temperature, forms a Fermi sphere in momentum space.

Fermi energy: $E_F = \frac{\hbar^2}{2m}(3\pi^2n)^{2/3}$

Specific heat at low T: $C_V \propto T$

#### Bose Gas
Below critical temperature:
$$T_c = \frac{2\pi\hbar^2}{mk_B}\left(\frac{n}{2.612}\right)^{2/3}$$

Bose-Einstein condensation occurs.

## Interacting Systems

### Virial Expansion
For weakly interacting gas:
$$\frac{PV}{Nk_BT} = 1 + B_2(T)n + B_3(T)n^2 + ...$$

Second virial coefficient:
$$B_2(T) = -\frac{1}{2V}\int (e^{-\beta u(r)} - 1)d^3r$$

### Mean Field Theory
Approximate interactions by average field.

Example - Ising model magnetization:
$$m = \tanh\left(\frac{m z J}{k_B T}\right)$$

Critical temperature: $T_c = \frac{zJ}{k_B}$

### Correlation Functions
Two-point correlation:
$$G(r) = \langle s_i s_j \rangle - \langle s_i \rangle\langle s_j \rangle$$

Near critical point: $G(r) \sim \frac{e^{-r/\xi}}{r^{d-2+\eta}}$

## Phase Transitions

<div class="phase-transitions-section">
  <div class="classification-grid">
    <h3><i class="fas fa-layer-group"></i> Classification</h3>
    
    <div class="transition-types">
      <div class="transition-card first-order">
        <h4><i class="fas fa-cut"></i> First Order</h4>
        <p>Discontinuous change in first derivative of free energy</p>
        
        <svg viewBox="0 0 200 150" class="transition-plot">
          <!-- First order transition -->
          <line x1="30" y1="120" x2="170" y2="120" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="120" x2="30" y2="30" stroke="#2c3e50" stroke-width="2" />
          <text x="100" y="140" text-anchor="middle" font-size="10">T</text>
          <text x="10" y="75" text-anchor="middle" font-size="10" transform="rotate(-90 10 75)">Order param</text>
          
          <!-- Discontinuous jump -->
          <path d="M 30 40 L 90 50" stroke="#e74c3c" stroke-width="3" />
          <path d="M 90 50 L 90 90" stroke="#e74c3c" stroke-width="1" stroke-dasharray="3,3" />
          <path d="M 90 90 L 170 100" stroke="#e74c3c" stroke-width="3" />
          <circle cx="90" cy="50" r="3" fill="#e74c3c" />
          <circle cx="90" cy="90" r="3" fill="white" stroke="#e74c3c" stroke-width="2" />
          <text x="90" y="115" text-anchor="middle" font-size="9">Tc</text>
        </svg>
        
        <div class="examples">
          <span class="example-tag">Ice ↔ Water</span>
          <span class="example-tag">Boiling</span>
        </div>
      </div>
      
      <div class="transition-card second-order">
        <h4><i class="fas fa-wave-square"></i> Second Order</h4>
        <p>Continuous first derivative, discontinuous second derivative</p>
        
        <svg viewBox="0 0 200 150" class="transition-plot">
          <!-- Second order transition -->
          <line x1="30" y1="120" x2="170" y2="120" stroke="#2c3e50" stroke-width="2" />
          <line x1="30" y1="120" x2="30" y2="30" stroke="#2c3e50" stroke-width="2" />
          <text x="100" y="140" text-anchor="middle" font-size="10">T</text>
          <text x="10" y="75" text-anchor="middle" font-size="10" transform="rotate(-90 10 75)">Order param</text>
          
          <!-- Continuous curve -->
          <path d="M 30 40 Q 80 45, 100 75 T 170 100" stroke="#3498db" stroke-width="3" fill="none" />
          <line x1="100" y1="30" x2="100" y2="120" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3" />
          <text x="100" y="115" text-anchor="middle" font-size="9">Tc</text>
        </svg>
        
        <div class="examples">
          <span class="example-tag">Magnetization</span>
          <span class="example-tag">Superconductivity</span>
        </div>
      </div>
    </div>
  </div>
  
  <div class="critical-phenomena">
    <h3><i class="fas fa-chart-line"></i> Critical Phenomena</h3>
    <p class="intro">Near critical point, observables follow power laws:</p>
    
    <div class="critical-exponents">
      <div class="exponent-card">
        <div class="symbol">C</div>
        <div class="name">Specific heat</div>
        <div class="scaling">$C \sim |t|^{-\alpha}$</div>
      </div>
      <div class="exponent-card">
        <div class="symbol">m</div>
        <div class="name">Order parameter</div>
        <div class="scaling">$m \sim |t|^{\beta}$ <span class="condition">(t < 0)</span></div>
      </div>
      <div class="exponent-card">
        <div class="symbol">χ</div>
        <div class="name">Susceptibility</div>
        <div class="scaling">$\chi \sim |t|^{-\gamma}$</div>
      </div>
      <div class="exponent-card">
        <div class="symbol">ξ</div>
        <div class="name">Correlation length</div>
        <div class="scaling">$\xi \sim |t|^{-\nu}$</div>
      </div>
    </div>
    
    <div class="reduced-temp">
      <p>Reduced temperature: $t = \frac{T-T_c}{T_c}$</p>
    </div>
  </div>
  
  <div class="universality-box">
    <h3><i class="fas fa-infinity"></i> Universality</h3>
    <p>Systems with same dimensionality and symmetry have identical critical exponents</p>
    
    <div class="scaling-relations">
      <h4>Scaling Relations</h4>
      <div class="relation-grid">
        <div class="relation">
          <span class="name">Rushbrooke:</span>
          <span class="equation">$\alpha + 2\beta + \gamma = 2$</span>
        </div>
        <div class="relation">
          <span class="name">Widom:</span>
          <span class="equation">$\gamma = \beta(\delta - 1)$</span>
        </div>
        <div class="relation">
          <span class="name">Fisher:</span>
          <span class="equation">$\gamma = \nu(2 - \eta)$</span>
        </div>
      </div>
    </div>
    
    <div class="universality-classes">
      <h4>Universality Classes</h4>
      <div class="class-examples">
        <span class="class-tag">2D Ising</span>
        <span class="class-tag">3D XY</span>
        <span class="class-tag">Percolation</span>
      </div>
    </div>
  </div>
</div>

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

## See Also

<div class="see-also-grid">
  <a href="thermodynamics.html" class="see-also-card">
    <i class="fas fa-temperature-high"></i>
    <h4>Thermodynamics</h4>
    <p>Macroscopic consequences</p>
  </a>
  <a href="quantum-mechanics.html" class="see-also-card">
    <i class="fas fa-atom"></i>
    <h4>Quantum Mechanics</h4>
    <p>Quantum foundation</p>
  </a>
  <a href="condensed-matter.html" class="see-also-card">
    <i class="fas fa-cube"></i>
    <h4>Condensed Matter Physics</h4>
    <p>Applications to solids</p>
  </a>
  <a href="quantum-field-theory.html" class="see-also-card">
    <i class="fas fa-wave-square"></i>
    <h4>Quantum Field Theory</h4>
    <p>Field theoretic methods</p>
  </a>
</div>