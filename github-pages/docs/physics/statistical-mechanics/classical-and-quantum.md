---
layout: docs
title: "Statistical Mechanics: Classical & Quantum Statistical Mechanics"
permalink: /docs/physics/statistical-mechanics/classical-and-quantum.html
toc: true
toc_sticky: true
---

<!-- Custom styles are now loaded via main.scss -->

[Statistical Mechanics](./)

Partition functions, quantum statistics, and ideal and interacting gases.

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
    <div class="equation-box" markdown="1">
$$\frac{d\rho}{dt} = \frac{\partial \rho}{\partial t} + \{\rho, H\} = 0$$
</div>
    
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
    <div class="equation-box" markdown="1">
$$Z = \frac{1}{N!h^{3N}} \int e^{-\beta H(\mathbf{r},\mathbf{p})} d\Gamma$$
</div>
    <p class="note">The factor $1/N!$ accounts for indistinguishability (Gibbs correction)</p>
  </div>
  
  <div class="equipartition-box">
    <h3><i class="fas fa-equals"></i> Equipartition Theorem</h3>
    <p>Each quadratic term in the energy contributes $\frac{1}{2}k_B T$ to the average energy. For example:</p>
    <ul>
      <li><strong>Harmonic oscillator:</strong> $\langle E \rangle = k_B T$ (kinetic + potential)</li>
      <li><strong>Ideal gas molecule:</strong> $\langle E_{\text{trans}} \rangle = \frac{3}{2}k_B T$ (3 translational DOF)</li>
    </ul>
  </div>
</div>

## Quantum Statistical Mechanics

The classical formalism above rests on a single object — the phase-space distribution $\rho(\mathbf{r}, \mathbf{p})$ — from which every observable follows by integrating against $d\Gamma$. Quantum mechanics forces us to generalize on two fronts. First, a quantum system in thermal contact with a reservoir is not in a definite pure state $|\psi\rangle$ but in a *statistical mixture* of energy eigenstates, so the distribution is promoted from a function on phase space to an operator on Hilbert space — the **density operator** $\rho$. Second, because position and momentum no longer commute, there is no joint $(\mathbf{r},\mathbf{p})$ distribution to integrate over; the phase-space integral $\frac{1}{h^{3N}}\int (\cdots)\, d\Gamma$ is replaced by the basis-independent **trace** $\text{Tr}(\cdots)$, which sums the diagonal matrix elements over any complete set of states. The dictionary is direct:

$$\rho(\mathbf{r},\mathbf{p}) \;\longrightarrow\; \hat{\rho}, \qquad \frac{1}{h^{3N}}\int d\Gamma \;\longrightarrow\; \text{Tr}, \qquad \langle A \rangle = \frac{1}{h^{3N}}\int A\,\rho\, d\Gamma \;\longrightarrow\; \langle A \rangle = \text{Tr}(\hat{\rho}\hat{A}).$$

The Gibbs $1/N!$ that we inserted by hand classically now appears automatically, encoded in the (anti)symmetry of the many-body Hilbert space. The two formalisms meet in the classical limit: when the thermal de Broglie wavelength is small compared to the inter-particle spacing, the trace over states reduces to the phase-space integral and $\hat{\rho}$ becomes diagonal in the classical sense, recovering the Boltzmann weight $e^{-\beta H}$ as an ordinary function.

<div class="quantum-stat-section">
  <div class="density-matrix-box">
    <h3><i class="fas fa-th"></i> Density Matrix</h3>
    <p>For a mixed state:</p>
    <div class="equation-box" markdown="1">
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$
</div>
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
    <div class="equation-box" markdown="1">
$$Z = \text{Tr}(e^{-\beta H}) = \sum_n e^{-\beta E_n}$$
</div>
  </div>
  
  <div class="statistics-comparison">
    <div class="stat-card fermi-dirac">
      <h3><i class="fas fa-minus-circle"></i> Fermi-Dirac Statistics</h3>
      <p class="particle-type">For fermions (half-integer spin)</p>
      
      <div class="occupation-formula">
        <p>Average occupation number:</p>
        <div class="equation-box" markdown="1">
$$\langle n_i \rangle = \frac{1}{e^{\beta(\epsilon_i - \mu)} + 1}$$
</div>
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
        <div class="equation-box" markdown="1">
$$\langle n_i \rangle = \frac{1}{e^{\beta(\epsilon_i - \mu)} - 1}$$
</div>
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

The ideal gas is the hydrogen atom of statistical mechanics — the one model simple enough to solve completely, yet rich enough to expose the deep difference between classical and quantum statistics. The single control parameter is how the thermal de Broglie wavelength $\lambda$ compares to the inter-particle spacing $n^{-1/3}$:

- When $\lambda \ll n^{-1/3}$ (hot or dilute), wavepackets don't overlap and the gas behaves *classically*.
- When $\lambda \gtrsim n^{-1/3}$ (cold or dense), wavepackets overlap and quantum statistics — Fermi or Bose — take over.

### Classical Ideal Gas
In the classical regime the partition function factorizes over particles, with the $1/N!$ Gibbs factor for indistinguishability:

$$Z = \frac{V^N}{N!\lambda^{3N}}, \qquad \lambda = \sqrt{\frac{2\pi\hbar^2}{mk_BT}}.$$

Differentiating $\ln Z$ recovers the familiar equation of state $PV = Nk_BT$ — a reassuring check that microstate-counting reproduces 19th-century gas laws.

### Quantum Ideal Gases

Once $\lambda \gtrsim n^{-1/3}$, the spin-statistics of the particles dominates, and fermions and bosons could hardly behave more differently.

#### Fermi Gas
The Pauli exclusion principle forbids two fermions from sharing a state, so even at $T = 0$ the particles stack up to the **Fermi energy**, filling a sphere in momentum space:

$$E_F = \frac{\hbar^2}{2m}(3\pi^2 n)^{2/3}.$$

Only the thin shell within $\sim k_B T$ of $E_F$ can be excited, giving the characteristic *linear* low-temperature heat capacity $C_V \propto T$. This degeneracy pressure is what holds up white dwarfs and neutron stars against gravity.

#### Bose Gas
Bosons have the opposite tendency — they *favor* sharing a state. Below a critical temperature a macroscopic fraction of them collapses into the single ground state, forming a **Bose-Einstein condensate**:

$$T_c = \frac{2\pi\hbar^2}{mk_B}\left(\frac{n}{2.612}\right)^{2/3}.$$

This is not ordinary condensation in real space but condensation in *momentum* space, first realized experimentally in dilute atomic gases in 1995.

## Interacting Systems

Interactions are where statistical mechanics gets hard — and interesting. Once particles influence each other, the partition function no longer factorizes, and exact solutions become rare. Two complementary strategies dominate: expand systematically in the *strength* of interactions (the virial expansion, good for dilute gases), or replace the many-body environment of each particle with a single *average* field (mean-field theory, good for capturing collective ordering).

### Virial Expansion
For a weakly interacting gas, corrections to the ideal-gas law come as a power series in density:

$$\frac{PV}{Nk_BT} = 1 + B_2(T)n + B_3(T)n^2 + \dots$$

The second virial coefficient $B_2$ measures the net effect of pairwise interactions — repulsive cores push it positive, attractive tails pull it negative:

$$B_2(T) = -\frac{1}{2V}\int \left(e^{-\beta u(r)} - 1\right)d^3r.$$

### Mean Field Theory
Rather than track every pairwise interaction, mean-field theory lets each particle feel the *average* effect of all the others — a single self-consistent field. For the Ising model, each spin sees an effective field set by the average magnetization $m$ of its $z$ neighbors, giving the self-consistency equation

$$m = \tanh\!\left(\frac{m z J}{k_B T}\right).$$

This has only the trivial solution $m = 0$ at high temperature, but a nonzero (spontaneously magnetized) solution appears below the **critical temperature** $T_c = zJ/k_B$ — mean-field theory's prediction of a phase transition. It gets the *existence* of the transition right but the critical *exponents* wrong, because it ignores the fluctuations that dominate near $T_c$ — exactly what the renormalization group was invented to handle.

### Correlation Functions
Two-point correlation:
$$G(r) = \langle s_i s_j \rangle - \langle s_i \rangle\langle s_j \rangle$$

Near critical point: $G(r) \sim \frac{e^{-r/\xi}}{r^{d-2+\eta}}$

---

## See Also

- [Statistical Mechanics Hub](./) — overview, microstates/macrostates, and ensembles.
- [Phase Transitions & Graduate Formalism](phase-transitions-and-advanced.html) — Next: critical phenomena, fluctuations, and the advanced reference block.
- [Thermodynamics](../thermodynamics.html) — the macroscopic laws these microscopic results reproduce.
- [Quantum Mechanics](../quantum-mechanics/) — the quantum foundation behind Fermi–Dirac and Bose–Einstein statistics.

**Next:** [Phase Transitions & Graduate Formalism](phase-transitions-and-advanced.html) →
