---
layout: docs
title: Statistical Mechanics
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
hide_title: true
---

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Statistical Mechanics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Bridging the Microscopic and Macroscopic Worlds</p>
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
        <svg viewBox="0 0 420 180" style="max-width: 500px; width: 100%;">
          <!-- Background container -->
          <rect x="10" y="10" width="400" height="130" rx="8" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>

          <!-- Title -->
          <text x="210" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Spin Configuration of 5 Particles</text>

          <!-- Particles with individual spin states -->
          <g transform="translate(50, 80)">
            <!-- Particle 1: Spin Up -->
            <circle cx="0" cy="0" r="18" fill="#2980b9" stroke="#1a5276" stroke-width="2"/>
            <text x="0" y="6" text-anchor="middle" font-size="20" fill="white" font-weight="bold">+1/2</text>
            <text x="0" y="-28" text-anchor="middle" font-size="16" fill="#1a5276" font-weight="bold">s=+1/2</text>
            <text x="0" y="45" text-anchor="middle" font-size="14" fill="#555">n=1</text>
          </g>

          <g transform="translate(130, 80)">
            <!-- Particle 2: Spin Down -->
            <circle cx="0" cy="0" r="18" fill="#c0392b" stroke="#922b21" stroke-width="2"/>
            <text x="0" y="6" text-anchor="middle" font-size="20" fill="white" font-weight="bold">-1/2</text>
            <text x="0" y="-28" text-anchor="middle" font-size="16" fill="#922b21" font-weight="bold">s=-1/2</text>
            <text x="0" y="45" text-anchor="middle" font-size="14" fill="#555">n=2</text>
          </g>

          <g transform="translate(210, 80)">
            <!-- Particle 3: Spin Up -->
            <circle cx="0" cy="0" r="18" fill="#2980b9" stroke="#1a5276" stroke-width="2"/>
            <text x="0" y="6" text-anchor="middle" font-size="20" fill="white" font-weight="bold">+1/2</text>
            <text x="0" y="-28" text-anchor="middle" font-size="16" fill="#1a5276" font-weight="bold">s=+1/2</text>
            <text x="0" y="45" text-anchor="middle" font-size="14" fill="#555">n=3</text>
          </g>

          <g transform="translate(290, 80)">
            <!-- Particle 4: Spin Down -->
            <circle cx="0" cy="0" r="18" fill="#c0392b" stroke="#922b21" stroke-width="2"/>
            <text x="0" y="6" text-anchor="middle" font-size="20" fill="white" font-weight="bold">-1/2</text>
            <text x="0" y="-28" text-anchor="middle" font-size="16" fill="#922b21" font-weight="bold">s=-1/2</text>
            <text x="0" y="45" text-anchor="middle" font-size="14" fill="#555">n=4</text>
          </g>

          <g transform="translate(370, 80)">
            <!-- Particle 5: Spin Up -->
            <circle cx="0" cy="0" r="18" fill="#2980b9" stroke="#1a5276" stroke-width="2"/>
            <text x="0" y="6" text-anchor="middle" font-size="20" fill="white" font-weight="bold">+1/2</text>
            <text x="0" y="-28" text-anchor="middle" font-size="16" fill="#1a5276" font-weight="bold">s=+1/2</text>
            <text x="0" y="45" text-anchor="middle" font-size="14" fill="#555">n=5</text>
          </g>

          <!-- Caption -->
          <text x="210" y="165" text-anchor="middle" font-size="15" fill="#555" font-style="italic">Each particle has a definite quantum state (complete microscopic specification)</text>
        </svg>
      </div>
    </div>
    
    <div class="concept-card macrostate">
      <h4><i class="fas fa-temperature-high"></i> Macrostate</h4>
      <p>Specification of macroscopic variables (T, P, V, N, E)</p>
      <div class="visual-example">
        <svg viewBox="0 0 420 200" style="max-width: 500px; width: 100%;">
          <!-- Background -->
          <rect x="10" y="10" width="400" height="180" rx="8" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>

          <!-- Title -->
          <text x="210" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Thermodynamic State Variables</text>

          <!-- Container box representing the system -->
          <rect x="60" y="55" width="180" height="100" rx="5" fill="#e8f4f8" stroke="#2c3e50" stroke-width="3"/>

          <!-- Random particles inside (suggesting many particles without specifying states) -->
          <circle cx="90" cy="85" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="120" cy="100" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="150" cy="80" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="180" cy="110" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="100" cy="130" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="160" cy="125" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="200" cy="90" r="4" fill="#7f8c8d" opacity="0.5"/>
          <circle cx="140" cy="140" r="4" fill="#7f8c8d" opacity="0.5"/>

          <!-- Macroscopic properties panel -->
          <rect x="260" y="55" width="140" height="100" rx="5" fill="#2c3e50" stroke="#1a252f" stroke-width="2"/>
          <text x="330" y="78" text-anchor="middle" font-size="15" fill="white" font-weight="bold">Macroscopic</text>
          <text x="330" y="95" text-anchor="middle" font-size="15" fill="white" font-weight="bold">Properties</text>
          <line x1="275" y1="102" x2="385" y2="102" stroke="#5d6d7e" stroke-width="1"/>
          <text x="330" y="120" text-anchor="middle" font-size="16" fill="#3498db" font-weight="bold">T = 300 K</text>
          <text x="330" y="140" text-anchor="middle" font-size="16" fill="#e74c3c" font-weight="bold">P = 1 atm</text>

          <!-- Arrow connecting system to properties -->
          <path d="M 240 105 L 255 105" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowMacro)"/>
          <defs>
            <marker id="arrowMacro" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
            </marker>
          </defs>

          <!-- Caption -->
          <text x="210" y="175" text-anchor="middle" font-size="14" fill="#555" font-style="italic">Only bulk properties matter - individual particle states unknown</text>
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
      <svg viewBox="0 0 420 220" style="max-width: 500px; width: 100%;">
        <!-- Background -->
        <rect x="5" y="5" width="410" height="210" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

        <!-- Title -->
        <text x="210" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Microcanonical Ensemble: Isolated System</text>

        <!-- Outer insulating walls (thick barriers) -->
        <rect x="80" y="50" width="260" height="120" rx="5" fill="none" stroke="#c0392b" stroke-width="12"/>

        <!-- Inner system container -->
        <rect x="100" y="70" width="220" height="80" rx="3" fill="#ecf0f1" stroke="#34495e" stroke-width="2"/>

        <!-- Particles inside -->
        <circle cx="140" cy="100" r="8" fill="#3498db" opacity="0.8"/>
        <circle cx="180" cy="120" r="8" fill="#3498db" opacity="0.8"/>
        <circle cx="220" cy="95" r="8" fill="#3498db" opacity="0.8"/>
        <circle cx="260" cy="115" r="8" fill="#3498db" opacity="0.8"/>
        <circle cx="160" cy="130" r="8" fill="#3498db" opacity="0.8"/>
        <circle cx="280" cy="100" r="8" fill="#3498db" opacity="0.8"/>

        <!-- Fixed quantities labels -->
        <text x="210" y="92" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">E = constant</text>
        <text x="210" y="115" text-anchor="middle" font-size="16" fill="#555">V = fixed, N = fixed</text>

        <!-- Wall labels -->
        <text x="50" y="115" text-anchor="middle" font-size="14" fill="#c0392b" font-weight="bold" transform="rotate(-90, 50, 115)">Insulated Wall</text>
        <text x="370" y="115" text-anchor="middle" font-size="14" fill="#c0392b" font-weight="bold" transform="rotate(90, 370, 115)">Insulated Wall</text>

        <!-- No exchange indicators -->
        <g transform="translate(45, 60)">
          <line x1="0" y1="0" x2="20" y2="20" stroke="#c0392b" stroke-width="3"/>
          <line x1="20" y1="0" x2="0" y2="20" stroke="#c0392b" stroke-width="3"/>
        </g>
        <g transform="translate(355, 60)">
          <line x1="0" y1="0" x2="20" y2="20" stroke="#c0392b" stroke-width="3"/>
          <line x1="20" y1="0" x2="0" y2="20" stroke="#c0392b" stroke-width="3"/>
        </g>

        <!-- Caption -->
        <text x="210" y="195" text-anchor="middle" font-size="15" fill="#555">No energy or particle exchange with surroundings</text>
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
      <svg viewBox="0 0 420 260" style="max-width: 500px; width: 100%;">
        <!-- Define arrow marker -->
        <defs>
          <marker id="arrowCanon" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <polygon points="0 0, 10 5, 0 10" fill="#e67e22"/>
          </marker>
        </defs>

        <!-- Background -->
        <rect x="5" y="5" width="410" height="250" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

        <!-- Title -->
        <text x="210" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Canonical Ensemble: Thermal Contact</text>

        <!-- Heat bath (outer reservoir) -->
        <rect x="40" y="50" width="340" height="160" rx="8" fill="#fadbd8" stroke="#e74c3c" stroke-width="3"/>
        <text x="210" y="75" text-anchor="middle" font-size="15" fill="#c0392b" font-weight="bold">Heat Bath at Temperature T</text>

        <!-- System (inner) -->
        <rect x="120" y="95" width="180" height="80" rx="5" fill="#3498db" stroke="#2980b9" stroke-width="3"/>
        <text x="210" y="125" text-anchor="middle" font-size="18" font-weight="bold" fill="white">System</text>
        <text x="210" y="148" text-anchor="middle" font-size="15" fill="#d6eaf8">N, V fixed</text>
        <text x="210" y="165" text-anchor="middle" font-size="14" fill="#d6eaf8">E fluctuates</text>

        <!-- Energy exchange arrows (bidirectional) -->
        <g transform="translate(140, 175)">
          <!-- Arrow down (heat out) -->
          <path d="M 0 0 L 0 25" stroke="#e67e22" stroke-width="4" marker-end="url(#arrowCanon)"/>
          <text x="-5" y="40" text-anchor="middle" font-size="14" fill="#e67e22" font-weight="bold">Q</text>
        </g>
        <g transform="translate(280, 200)">
          <!-- Arrow up (heat in) -->
          <path d="M 0 0 L 0 -25" stroke="#e67e22" stroke-width="4" marker-end="url(#arrowCanon)"/>
          <text x="5" y="15" text-anchor="middle" font-size="14" fill="#e67e22" font-weight="bold">Q</text>
        </g>

        <!-- Diathermal wall label -->
        <text x="210" y="88" text-anchor="middle" font-size="12" fill="#555" font-style="italic">(diathermal wall allows heat exchange)</text>

        <!-- Caption -->
        <text x="210" y="235" text-anchor="middle" font-size="15" fill="#555">Energy can be exchanged; temperature is fixed by the bath</text>
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
      <svg viewBox="0 0 420 280" style="max-width: 500px; width: 100%;">
        <!-- Define arrow markers -->
        <defs>
          <marker id="arrowGrand" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <polygon points="0 0, 10 5, 0 10" fill="#27ae60"/>
          </marker>
          <marker id="arrowHeat" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <polygon points="0 0, 10 5, 0 10" fill="#e67e22"/>
          </marker>
        </defs>

        <!-- Background -->
        <rect x="5" y="5" width="410" height="270" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

        <!-- Title -->
        <text x="210" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Grand Canonical Ensemble: Open System</text>

        <!-- Reservoir (outer) -->
        <rect x="40" y="50" width="340" height="175" rx="8" fill="#fef9e7" stroke="#f39c12" stroke-width="3"/>
        <text x="210" y="75" text-anchor="middle" font-size="15" fill="#d68910" font-weight="bold">Reservoir at Temperature T, Chemical Potential mu</text>

        <!-- System (inner) -->
        <rect x="120" y="95" width="180" height="90" rx="5" fill="#9b59b6" stroke="#7d3c98" stroke-width="3"/>
        <text x="210" y="125" text-anchor="middle" font-size="18" font-weight="bold" fill="white">System</text>
        <text x="210" y="148" text-anchor="middle" font-size="15" fill="#e8daef">V fixed</text>
        <text x="210" y="168" text-anchor="middle" font-size="14" fill="#e8daef">E, N fluctuate</text>

        <!-- Particle exchange (left side) -->
        <g transform="translate(85, 130)">
          <circle cx="0" cy="0" r="8" fill="#27ae60" stroke="#1e8449" stroke-width="2"/>
          <circle cx="0" cy="25" r="8" fill="#27ae60" stroke="#1e8449" stroke-width="2"/>
          <path d="M 15 12 L 35 12" stroke="#27ae60" stroke-width="3" marker-end="url(#arrowGrand)"/>
          <text x="25" y="-8" text-anchor="middle" font-size="12" fill="#27ae60" font-weight="bold">particles</text>
        </g>

        <!-- Energy exchange (right side) -->
        <g transform="translate(300, 130)">
          <path d="M 0 12 L 25 12" stroke="#e67e22" stroke-width="4" marker-end="url(#arrowHeat)"/>
          <text x="12" y="-8" text-anchor="middle" font-size="12" fill="#e67e22" font-weight="bold">heat Q</text>
          <!-- Wavy line for heat -->
          <path d="M 5 30 Q 10 25, 15 30 T 25 30" stroke="#e67e22" stroke-width="2" fill="none"/>
        </g>

        <!-- Semipermeable membrane label -->
        <text x="210" y="200" text-anchor="middle" font-size="12" fill="#555" font-style="italic">(permeable boundary: particles and energy can cross)</text>

        <!-- Caption -->
        <text x="210" y="250" text-anchor="middle" font-size="15" fill="#555">Both energy and particles exchanged; T and mu fixed</text>
      </svg>
    </div>
    
    <div class="ensemble-equations">
      <p><strong>Grand partition function:</strong></p>
      <div class="equation-box">$$\mathcal{Z} = \sum_{N=0}^{\infty} \sum_i e^{-\beta(E_i - \mu N)}$$</div>
      <p><strong>Grand potential:</strong> $\Omega = -k_B T \ln \mathcal{Z}$</p>
    </div>
  </div>
</div>

## Classical Statistical Mechanics

<div class="classical-mechanics-section">
  <h3><i class="fas fa-chart-line"></i> Phase Space</h3>
  
  <div class="phase-space-visual">
    <p>6N-dimensional space of positions and momenta for N particles</p>
    
    <svg viewBox="0 0 500 280" class="phase-diagram" style="max-width: 500px; width: 100%;">
      <!-- Background -->
      <rect x="5" y="5" width="490" height="270" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

      <!-- Title -->
      <text x="250" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Phase Space Trajectory</text>

      <!-- Define arrow markers -->
      <defs>
        <marker id="arrowPhase" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
        </marker>
      </defs>

      <!-- Phase space axes -->
      <line x1="70" y1="220" x2="450" y2="220" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowPhase)"/>
      <line x1="70" y1="220" x2="70" y2="55" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowPhase)"/>

      <!-- Axis labels -->
      <text x="260" y="250" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Position q</text>
      <text x="35" y="140" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50" transform="rotate(-90, 35, 140)">Momentum p</text>

      <!-- Axis tick marks and values -->
      <line x1="150" y1="215" x2="150" y2="225" stroke="#2c3e50" stroke-width="2"/>
      <text x="150" y="238" text-anchor="middle" font-size="14" fill="#555">q1</text>
      <line x1="260" y1="215" x2="260" y2="225" stroke="#2c3e50" stroke-width="2"/>
      <text x="260" y="238" text-anchor="middle" font-size="14" fill="#555">q2</text>
      <line x1="370" y1="215" x2="370" y2="225" stroke="#2c3e50" stroke-width="2"/>
      <text x="370" y="238" text-anchor="middle" font-size="14" fill="#555">q3</text>

      <line x1="65" y1="170" x2="75" y2="170" stroke="#2c3e50" stroke-width="2"/>
      <text x="55" y="175" text-anchor="middle" font-size="14" fill="#555">p1</text>
      <line x1="65" y1="120" x2="75" y2="120" stroke="#2c3e50" stroke-width="2"/>
      <text x="55" y="125" text-anchor="middle" font-size="14" fill="#555">p2</text>
      <line x1="65" y1="80" x2="75" y2="80" stroke="#2c3e50" stroke-width="2"/>
      <text x="55" y="85" text-anchor="middle" font-size="14" fill="#555">p3</text>

      <!-- Phase space trajectory (Hamiltonian flow) -->
      <path d="M 120 180 Q 180 100, 260 130 T 400 100" fill="none" stroke="#2980b9" stroke-width="3"/>

      <!-- Start point -->
      <circle cx="120" cy="180" r="8" fill="#c0392b" stroke="#922b21" stroke-width="2"/>
      <text x="105" y="200" text-anchor="middle" font-size="14" fill="#c0392b" font-weight="bold">t = 0</text>

      <!-- End point -->
      <circle cx="400" cy="100" r="8" fill="#27ae60" stroke="#1e8449" stroke-width="2"/>
      <text x="420" y="90" text-anchor="middle" font-size="14" fill="#27ae60" font-weight="bold">t = T</text>

      <!-- Direction arrow on trajectory -->
      <polygon points="280,125 295,130 282,140" fill="#2980b9"/>

      <!-- Volume element (phase space cell) -->
      <rect x="220" y="115" width="60" height="45" fill="#f39c12" opacity="0.4" stroke="#d68910" stroke-width="2" stroke-dasharray="4,2"/>
      <text x="250" y="145" text-anchor="middle" font-size="16" font-weight="bold" fill="#b7950b">dGamma</text>

      <!-- Legend -->
      <rect x="350" y="180" width="120" height="50" rx="5" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="1"/>
      <text x="410" y="200" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Phase space cell:</text>
      <text x="410" y="218" text-anchor="middle" font-size="13" fill="#555">dGamma = dq dp</text>
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
      <svg viewBox="0 0 500 200" style="max-width: 500px; width: 100%;">
        <!-- Background -->
        <rect x="5" y="5" width="490" height="190" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

        <!-- Title -->
        <text x="250" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Phase Space Volume Conservation</text>

        <!-- Define arrow -->
        <defs>
          <marker id="arrowLiouville" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
          </marker>
        </defs>

        <!-- Initial distribution (t = 0) -->
        <ellipse cx="100" cy="100" rx="55" ry="40" fill="#3498db" opacity="0.5" stroke="#2980b9" stroke-width="2"/>
        <text x="100" y="105" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a5276">t = 0</text>
        <text x="100" y="155" text-anchor="middle" font-size="14" fill="#2980b9">Initial volume V</text>

        <!-- Time evolution arrow -->
        <path d="M 170 100 L 230 100" stroke="#2c3e50" stroke-width="4" marker-end="url(#arrowLiouville)"/>
        <text x="200" y="85" text-anchor="middle" font-size="14" fill="#555">Hamiltonian</text>
        <text x="200" y="125" text-anchor="middle" font-size="14" fill="#555">evolution</text>

        <!-- Final distribution (t = tau) - same area, different shape -->
        <ellipse cx="350" cy="100" rx="40" ry="55" fill="#3498db" opacity="0.5" stroke="#2980b9" stroke-width="2" transform="rotate(15, 350, 100)"/>
        <text x="350" y="105" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a5276">t = tau</text>
        <text x="350" y="170" text-anchor="middle" font-size="14" fill="#2980b9">Same volume V</text>

        <!-- Equals sign for volume -->
        <rect x="420" y="80" width="60" height="40" rx="5" fill="#27ae60" opacity="0.2" stroke="#27ae60" stroke-width="2"/>
        <text x="450" y="105" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e8449">V = V</text>
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
      <svg viewBox="0 0 450 220" style="max-width: 500px; width: 100%;">
        <!-- Background -->
        <rect x="5" y="5" width="440" height="210" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

        <!-- Title -->
        <text x="225" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Density Matrix Structure</text>

        <!-- Matrix bracket left -->
        <path d="M 80 50 L 65 50 L 65 175 L 80 175" stroke="#2c3e50" stroke-width="3" fill="none"/>
        <!-- Matrix bracket right -->
        <path d="M 230 50 L 245 50 L 245 175 L 230 175" stroke="#2c3e50" stroke-width="3" fill="none"/>

        <!-- Matrix label -->
        <text x="155" y="200" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">rho</text>

        <!-- 3x3 Matrix grid -->
        <!-- Row 1 -->
        <rect x="85" y="55" width="45" height="35" fill="#2980b9" stroke="#1a5276" stroke-width="1"/>
        <text x="107" y="78" text-anchor="middle" font-size="14" fill="white" font-weight="bold">rho_11</text>

        <rect x="135" y="55" width="45" height="35" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="157" y="78" text-anchor="middle" font-size="14" fill="#1a5276">rho_12</text>

        <rect x="185" y="55" width="45" height="35" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="207" y="78" text-anchor="middle" font-size="14" fill="#1a5276">rho_13</text>

        <!-- Row 2 -->
        <rect x="85" y="95" width="45" height="35" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="107" y="118" text-anchor="middle" font-size="14" fill="#1a5276">rho_21</text>

        <rect x="135" y="95" width="45" height="35" fill="#2980b9" stroke="#1a5276" stroke-width="1"/>
        <text x="157" y="118" text-anchor="middle" font-size="14" fill="white" font-weight="bold">rho_22</text>

        <rect x="185" y="95" width="45" height="35" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="207" y="118" text-anchor="middle" font-size="14" fill="#1a5276">rho_23</text>

        <!-- Row 3 -->
        <rect x="85" y="135" width="45" height="35" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="107" y="158" text-anchor="middle" font-size="14" fill="#1a5276">rho_31</text>

        <rect x="135" y="135" width="45" height="35" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="157" y="158" text-anchor="middle" font-size="14" fill="#1a5276">rho_32</text>

        <rect x="185" y="135" width="45" height="35" fill="#2980b9" stroke="#1a5276" stroke-width="1"/>
        <text x="207" y="158" text-anchor="middle" font-size="14" fill="white" font-weight="bold">rho_33</text>

        <!-- Legend -->
        <rect x="280" y="60" width="150" height="100" rx="5" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="1"/>
        <text x="355" y="82" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Legend</text>

        <rect x="295" y="95" width="20" height="15" fill="#2980b9" stroke="#1a5276" stroke-width="1"/>
        <text x="325" y="107" font-size="13" fill="#2c3e50">Diagonal: populations</text>

        <rect x="295" y="120" width="20" height="15" fill="#85c1e9" stroke="#5dade2" stroke-width="1"/>
        <text x="325" y="132" font-size="13" fill="#2c3e50">Off-diag: coherences</text>
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
        <svg viewBox="0 0 480 280" style="max-width: 500px; width: 100%;">
          <!-- Background -->
          <rect x="5" y="5" width="470" height="270" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

          <!-- Title -->
          <text x="240" y="30" text-anchor="middle" font-size="17" font-weight="bold" fill="#2c3e50">Fermi-Dirac Distribution Function</text>

          <!-- Define arrow markers -->
          <defs>
            <marker id="arrowFD" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
            </marker>
          </defs>

          <!-- Axes -->
          <line x1="60" y1="220" x2="420" y2="220" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowFD)"/>
          <line x1="60" y1="220" x2="60" y2="50" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowFD)"/>

          <!-- Axis labels -->
          <text x="240" y="255" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Energy epsilon</text>
          <text x="25" y="140" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50" transform="rotate(-90, 25, 140)">Occupation &lt;n&gt;</text>

          <!-- Y-axis tick marks -->
          <line x1="55" y1="80" x2="65" y2="80" stroke="#2c3e50" stroke-width="2"/>
          <text x="45" y="85" text-anchor="middle" font-size="14" fill="#555">1</text>
          <line x1="55" y1="150" x2="65" y2="150" stroke="#2c3e50" stroke-width="2"/>
          <text x="45" y="155" text-anchor="middle" font-size="14" fill="#555">0.5</text>
          <text x="45" y="225" text-anchor="middle" font-size="14" fill="#555">0</text>

          <!-- Chemical potential line -->
          <line x1="220" y1="50" x2="220" y2="220" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="8,4"/>
          <text x="220" y="45" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">mu (Fermi level)</text>

          <!-- T = 0 step function (dashed) -->
          <path d="M 60 80 L 220 80 L 220 210 L 400 210" fill="none" stroke="#2c3e50" stroke-width="3" stroke-dasharray="10,5"/>

          <!-- T > 0 smooth curve (solid red) -->
          <path d="M 60 82 Q 140 82, 180 95 Q 200 115, 220 150 Q 240 185, 280 200 Q 320 208, 400 210" fill="none" stroke="#c0392b" stroke-width="4"/>

          <!-- Legends -->
          <rect x="300" y="70" width="150" height="70" rx="5" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="1"/>
          <line x1="315" y1="95" x2="355" y2="95" stroke="#2c3e50" stroke-width="3" stroke-dasharray="8,4"/>
          <text x="365" y="100" font-size="14" fill="#2c3e50">T = 0 K</text>
          <line x1="315" y1="120" x2="355" y2="120" stroke="#c0392b" stroke-width="4"/>
          <text x="365" y="125" font-size="14" fill="#c0392b">T &gt; 0 K</text>

          <!-- Annotation: thermal broadening -->
          <path d="M 180 110 Q 160 90, 140 100" fill="none" stroke="#27ae60" stroke-width="2"/>
          <text x="120" y="95" font-size="13" fill="#27ae60" font-weight="bold">Thermal</text>
          <text x="120" y="110" font-size="13" fill="#27ae60" font-weight="bold">broadening</text>
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
        <svg viewBox="0 0 480 300" style="max-width: 500px; width: 100%;">
          <!-- Background -->
          <rect x="5" y="5" width="470" height="290" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>

          <!-- Title -->
          <text x="240" y="30" text-anchor="middle" font-size="17" font-weight="bold" fill="#2c3e50">Bose-Einstein Distribution Function</text>

          <!-- Define arrow markers -->
          <defs>
            <marker id="arrowBE" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <polygon points="0 0, 10 5, 0 10" fill="#2c3e50"/>
            </marker>
          </defs>

          <!-- Axes -->
          <line x1="80" y1="230" x2="430" y2="230" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowBE)"/>
          <line x1="80" y1="230" x2="80" y2="50" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrowBE)"/>

          <!-- Axis labels -->
          <text x="260" y="265" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Energy epsilon</text>
          <text x="35" y="140" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50" transform="rotate(-90, 35, 140)">Occupation &lt;n&gt;</text>

          <!-- Y-axis tick marks -->
          <line x1="75" y1="180" x2="85" y2="180" stroke="#2c3e50" stroke-width="2"/>
          <text x="65" y="185" text-anchor="middle" font-size="14" fill="#555">1</text>
          <line x1="75" y1="130" x2="85" y2="130" stroke="#2c3e50" stroke-width="2"/>
          <text x="65" y="135" text-anchor="middle" font-size="14" fill="#555">2</text>
          <text x="65" y="235" text-anchor="middle" font-size="14" fill="#555">0</text>

          <!-- Chemical potential line (mu = 0 for bosons) -->
          <line x1="160" y1="50" x2="160" y2="230" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="8,4"/>
          <text x="160" y="45" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">mu = 0</text>

          <!-- BEC condensation region (shaded) -->
          <rect x="80" y="55" width="80" height="175" fill="#3498db" opacity="0.15"/>

          <!-- Bose-Einstein distribution curves for different T -->
          <!-- Higher T (flatter curve) -->
          <path d="M 165 180 Q 200 190, 250 200 Q 320 208, 410 215" fill="none" stroke="#85c1e9" stroke-width="3"/>
          <text x="380" y="200" font-size="13" fill="#5dade2">High T</text>

          <!-- Medium T -->
          <path d="M 165 150 Q 200 170, 250 190 Q 320 205, 410 212" fill="none" stroke="#3498db" stroke-width="4"/>
          <text x="380" y="180" font-size="13" fill="#2980b9">Medium T</text>

          <!-- Low T (steep divergence near mu) -->
          <path d="M 165 60 Q 170 100, 180 140 Q 200 180, 250 200 Q 320 210, 410 215" fill="none" stroke="#1a5276" stroke-width="4"/>
          <text x="380" y="160" font-size="13" fill="#1a5276" font-weight="bold">Low T</text>

          <!-- Divergence arrow at mu -->
          <path d="M 163 55 L 163 75" stroke="#c0392b" stroke-width="3"/>
          <polygon points="163,50 158,60 168,60" fill="#c0392b"/>
          <text x="170" y="68" font-size="12" fill="#c0392b" font-weight="bold">diverges!</text>

          <!-- BEC label -->
          <text x="120" y="100" text-anchor="middle" font-size="16" fill="#2980b9" font-weight="bold">BEC</text>
          <text x="120" y="118" text-anchor="middle" font-size="14" fill="#2980b9">Region</text>

          <!-- Caption -->
          <text x="240" y="285" text-anchor="middle" font-size="14" fill="#555" font-style="italic">As T approaches Tc, occupation diverges at ground state (condensation)</text>
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