---
layout: docs
title: String Theory
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section">
  <div class="hero-content">
    
    <p class="hero-subtitle">The Quest for a Theory of Everything</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">String theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings. It attempts to describe all fundamental forces and forms of matter in a single, unified theory. String theory potentially provides a quantum theory of gravity and has profoundly influenced our understanding of spacetime, quantum mechanics, and cosmology.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-wave-square"></i>
      <h4>Vibrating Strings</h4>
      <p>Particles as vibrational modes</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-layer-group"></i>
      <h4>Extra Dimensions</h4>
      <p>Beyond our 4D spacetime</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-infinity"></i>
      <h4>Quantum Gravity</h4>
      <p>Unifying all forces</p>
    </div>
  </div>
</div>

## Fundamental Concepts

<div class="concepts-section">
  <div class="string-types">
    <h3><i class="fas fa-circle-notch"></i> From Points to Strings</h3>
    <p>In string theory, fundamental objects are not zero-dimensional points but one-dimensional strings:</p>
    
    <div class="string-comparison">
      <div class="string-card closed">
        <h4><i class="fas fa-ring"></i> Closed Strings</h4>
        <p>Form loops with no endpoints</p>
        <svg viewBox="0 0 150 150" class="string-visual">
          <circle cx="75" cy="75" r="40" fill="none" stroke="#3498db" stroke-width="3" />
          <!-- Vibrational pattern -->
          <path d="M 35 75 Q 55 55, 75 75 T 115 75" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.7" />
          <text x="75" y="130" text-anchor="middle" font-size="12">Closed String</text>
        </svg>
      </div>
      
      <div class="string-card open">
        <h4><i class="fas fa-wave-square"></i> Open Strings</h4>
        <p>Have two distinct endpoints</p>
        <svg viewBox="0 0 150 150" class="string-visual">
          <path d="M 30 75 Q 50 55, 75 75 T 120 75" fill="none" stroke="#27ae60" stroke-width="3" />
          <circle cx="30" cy="75" r="4" fill="#e74c3c" />
          <circle cx="120" cy="75" r="4" fill="#e74c3c" />
          <text x="75" y="130" text-anchor="middle" font-size="12">Open String</text>
        </svg>
      </div>
    </div>
    
    <div class="vibrational-modes">
      <h4>Vibrational Modes = Particles</h4>
      <div class="mode-spectrum">
        <div class="mode-item low-energy">
          <span class="energy-level">Low energy modes</span>
          <span class="arrow">→</span>
          <span class="particles">Known particles</span>
        </div>
        <div class="mode-item high-energy">
          <span class="energy-level">High energy modes</span>
          <span class="arrow">→</span>
          <span class="particles">New, heavy particles</span>
        </div>
      </div>
    </div>
  </div>
  
  <div class="string-scale">
    <h3><i class="fas fa-ruler"></i> String Scale</h3>
    <p>The fundamental length scale in string theory:</p>
    
    <div class="scale-equations">
      <div class="equation-box primary">
        $$\ell_s = \sqrt{\frac{\hbar}{T}} \approx 10^{-35} \text{ m}$$
      </div>
      <p>Where T is the string tension. This is near the Planck length:</p>
      <div class="equation-box">
        $$\ell_P = \sqrt{\frac{\hbar G}{c^3}} \approx 1.6 \times 10^{-35} \text{ m}$$
      </div>
    </div>
    
    <div class="scale-comparison">
      <svg viewBox="0 0 400 100">
        <!-- Scale bar -->
        <line x1="50" y1="50" x2="350" y2="50" stroke="#2c3e50" stroke-width="2" />
        <!-- Markers -->
        <line x1="50" y1="45" x2="50" y2="55" stroke="#2c3e50" stroke-width="2" />
        <line x1="150" y1="45" x2="150" y2="55" stroke="#2c3e50" stroke-width="2" />
        <line x1="250" y1="45" x2="250" y2="55" stroke="#2c3e50" stroke-width="2" />
        <line x1="350" y1="45" x2="350" y2="55" stroke="#2c3e50" stroke-width="2" />
        <!-- Labels -->
        <text x="50" y="70" text-anchor="middle" font-size="10">Planck</text>
        <text x="150" y="70" text-anchor="middle" font-size="10">String</text>
        <text x="250" y="70" text-anchor="middle" font-size="10">Proton</text>
        <text x="350" y="70" text-anchor="middle" font-size="10">Atom</text>
        <!-- Sizes -->
        <text x="50" y="35" text-anchor="middle" font-size="8">10⁻³⁵m</text>
        <text x="150" y="35" text-anchor="middle" font-size="8">10⁻³⁵m</text>
        <text x="250" y="35" text-anchor="middle" font-size="8">10⁻¹⁵m</text>
        <text x="350" y="35" text-anchor="middle" font-size="8">10⁻¹⁰m</text>
      </svg>
    </div>
  </div>
  
  <div class="worldsheet-concept">
    <h3><i class="fas fa-scroll"></i> Worldsheet</h3>
    <p>As a string moves through spacetime, it traces out a two-dimensional surface called a worldsheet:</p>
    
    <div class="worldsheet-comparison">
      <div class="trace-item">
        <svg viewBox="0 0 150 200">
          <!-- Point particle worldline -->
          <line x1="75" y1="180" x2="75" y2="20" stroke="#3498db" stroke-width="3" />
          <circle cx="75" cy="180" r="5" fill="#e74c3c" />
          <text x="75" y="195" text-anchor="middle" font-size="11">Point particle</text>
          <text x="120" y="100" font-size="10">Worldline (1D)</text>
          <!-- Time axis -->
          <path d="M 20 170 L 20 30" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrow)" />
          <text x="10" y="25" font-size="9">t</text>
        </svg>
      </div>
      
      <div class="trace-item">
        <svg viewBox="0 0 200 200">
          <!-- String worldsheet -->
          <path d="M 50 180 L 50 20 L 150 20 L 150 180 Z" fill="#3498db" opacity="0.3" stroke="#3498db" stroke-width="2" />
          <ellipse cx="100" cy="180" rx="50" ry="10" fill="none" stroke="#e74c3c" stroke-width="3" />
          <text x="100" y="195" text-anchor="middle" font-size="11">String</text>
          <text x="160" y="100" font-size="10">Worldsheet (2D)</text>
          <!-- Time axis -->
          <path d="M 20 170 L 20 30" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrow)" />
          <text x="10" y="25" font-size="9">t</text>
        </svg>
      </div>
    </div>
  </div>
</div>

## Classical String Theory

<div class="classical-string-section">
  <div class="action-formulations">
    <h3><i class="fas fa-integral"></i> String Actions</h3>
    
    <div class="action-cards">
      <div class="action-card nambu-goto">
        <h4>Nambu-Goto Action</h4>
        <p>The action for a relativistic string (area of worldsheet):</p>
        <div class="equation-box">
          $$S = -T \int dA = -T \int d\tau d\sigma \sqrt{-\det(h_{ab})}$$
        </div>
        <p class="note">Where $h_{ab}$ is the induced metric on the worldsheet</p>
        
        <div class="geometric-interpretation">
          <svg viewBox="0 0 200 150">
            <!-- Worldsheet area -->
            <path d="M 40 120 Q 60 80, 100 60 Q 140 80, 160 120" fill="#3498db" opacity="0.3" stroke="#3498db" stroke-width="2" />
            <text x="100" y="90" text-anchor="middle" font-size="10" fill="white">Area</text>
            <text x="100" y="140" text-anchor="middle" font-size="10">Minimize area</text>
          </svg>
        </div>
      </div>
      
      <div class="action-card polyakov">
        <h4>Polyakov Action</h4>
        <p>Equivalent formulation with manifest reparametrization invariance:</p>
        <div class="equation-box">
          $$S = -\frac{T}{2} \int d^2\sigma \sqrt{-h} h^{ab} \partial_a X^\mu \partial_b X_\mu$$
        </div>
        <p class="note">Independent worldsheet metric $h_{ab}$</p>
        
        <div class="advantages">
          <span class="advantage-tag">Easier quantization</span>
          <span class="advantage-tag">Manifest symmetries</span>
        </div>
      </div>
    </div>
  </div>
  
  <div class="equations-motion">
    <h3><i class="fas fa-wave-square"></i> Equations of Motion</h3>
    
    <div class="wave-equation">
      <p>The string satisfies the wave equation:</p>
      <div class="equation-box highlighted">
        $$\frac{\partial^2 X^\mu}{\partial \tau^2} - \frac{\partial^2 X^\mu}{\partial \sigma^2} = 0$$
      </div>
      
      <div class="wave-visualization">
        <svg viewBox="0 0 300 150">
          <!-- Wave on string -->
          <path d="M 30 75 Q 60 50, 90 75 T 150 75 T 210 75 T 270 75" 
                fill="none" stroke="#e74c3c" stroke-width="3" />
          <!-- Direction arrows -->
          <path d="M 150 60 L 170 60" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
          <path d="M 150 90 L 130 90" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
          <text x="150" y="120" text-anchor="middle" font-size="10">Wave propagation</text>
        </svg>
      </div>
    </div>
  </div>
  
  <div class="boundary-conditions">
    <h3><i class="fas fa-border-style"></i> Boundary Conditions</h3>
    
    <div class="bc-grid">
      <div class="bc-card closed-bc">
        <h4>Closed Strings</h4>
        <div class="equation-box">
          $$X^\mu(\tau, \sigma + 2\pi) = X^\mu(\tau, \sigma)$$
        </div>
        <p>Periodic boundary condition</p>
        
        <svg viewBox="0 0 150 150">
          <circle cx="75" cy="75" r="40" fill="none" stroke="#3498db" stroke-width="3" />
          <text x="75" y="130" text-anchor="middle" font-size="10">σ = 0 = 2π</text>
        </svg>
      </div>
      
      <div class="bc-card open-bc">
        <h4>Open Strings</h4>
        
        <div class="bc-types">
          <div class="bc-type neumann">
            <h5>Neumann BC</h5>
            <div class="equation-box small">
              $$\frac{\partial X^\mu}{\partial \sigma} = 0$$
            </div>
            <p>Free endpoints</p>
            <svg viewBox="0 0 120 80">
              <path d="M 20 40 Q 60 20, 100 40" fill="none" stroke="#27ae60" stroke-width="2" />
              <circle cx="20" cy="40" r="3" fill="#27ae60" />
              <circle cx="100" cy="40" r="3" fill="#27ae60" />
              <path d="M 10 40 L 30 40" stroke="#95a5a6" stroke-width="1" stroke-dasharray="2,2" />
              <path d="M 90 40 L 110 40" stroke="#95a5a6" stroke-width="1" stroke-dasharray="2,2" />
            </svg>
          </div>
          
          <div class="bc-type dirichlet">
            <h5>Dirichlet BC</h5>
            <div class="equation-box small">
              $$X^\mu = \text{const}$$
            </div>
            <p>Fixed endpoints (D-branes)</p>
            <svg viewBox="0 0 120 80">
              <rect x="15" y="35" width="10" height="10" fill="#e74c3c" />
              <rect x="95" y="35" width="10" height="10" fill="#e74c3c" />
              <path d="M 25 40 Q 60 20, 95 40" fill="none" stroke="#e74c3c" stroke-width="2" />
              <text x="60" y="65" text-anchor="middle" font-size="8">D-brane</text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

## Quantum String Theory

### Light-Cone Quantization

In light-cone gauge, the string oscillator modes satisfy:

**Commutation relations:**
```
[α^μ_m, α^ν_n] = m δ_{m+n,0} η^{μν}
```

### Virasoro Algebra

Constraints from reparametrization invariance:

```
[L_m, L_n] = (m-n)L_{m+n} + c/12 m(m²-1)δ_{m+n,0}
```

Where c is the central charge.

### Critical Dimension

Quantum consistency (no anomalies) requires:
- **Bosonic string:** D = 26
- **Superstring:** D = 10

This fixes the spacetime dimension!

### String Spectrum

**Bosonic string:**
- Tachyon: m² = -1/ℓ_s²
- Massless: graviton, dilaton, Kalb-Ramond field
- Massive tower: m² = (n-1)/ℓ_s²

**Superstring:**
- No tachyon
- Massless: supergravity multiplet
- Massive tower with supersymmetry

## Types of String Theories

<div class="string-theories-section">
  <div class="bosonic-theory">
    <h3><i class="fas fa-wave-square"></i> Bosonic String Theory</h3>
    
    <div class="theory-card bosonic">
      <div class="properties">
        <div class="property-item">
          <i class="fas fa-cube"></i>
          <span>26 dimensions required</span>
        </div>
        <div class="property-item warning">
          <i class="fas fa-exclamation-triangle"></i>
          <span>Contains tachyons (unstable)</span>
        </div>
        <div class="property-item">
          <i class="fas fa-times-circle"></i>
          <span>No fermions</span>
        </div>
        <div class="property-item info">
          <i class="fas fa-history"></i>
          <span>Mainly of historical interest</span>
        </div>
      </div>
    </div>
  </div>
  
  <div class="superstring-theories">
    <h3><i class="fas fa-atom"></i> Superstring Theories</h3>
    <p class="subtitle">Five consistent 10-dimensional theories:</p>
    
    <div class="theory-web">
      <svg viewBox="0 0 600 400" class="theory-diagram">
        <!-- Central node -->
        <circle cx="300" cy="200" r="50" fill="#3498db" opacity="0.3" />
        <text x="300" y="205" text-anchor="middle" font-size="12" font-weight="bold">10D SUSY</text>
        
        <!-- Type I -->
        <circle cx="150" cy="100" r="40" fill="#e74c3c" opacity="0.5" />
        <text x="150" y="105" text-anchor="middle" font-size="11" fill="white">Type I</text>
        <line x1="180" y1="120" x2="260" y2="170" stroke="#95a5a6" stroke-width="2" />
        
        <!-- Type IIA -->
        <circle cx="450" cy="100" r="40" fill="#27ae60" opacity="0.5" />
        <text x="450" y="105" text-anchor="middle" font-size="11" fill="white">Type IIA</text>
        <line x1="420" y1="120" x2="340" y2="170" stroke="#95a5a6" stroke-width="2" />
        
        <!-- Type IIB -->
        <circle cx="500" cy="250" r="40" fill="#f39c12" opacity="0.5" />
        <text x="500" y="255" text-anchor="middle" font-size="11" fill="white">Type IIB</text>
        <line x1="460" y1="240" x2="350" y2="210" stroke="#95a5a6" stroke-width="2" />
        
        <!-- Heterotic SO(32) -->
        <circle cx="100" cy="250" r="40" fill="#9b59b6" opacity="0.5" />
        <text x="100" y="250" text-anchor="middle" font-size="10" fill="white">Het</text>
        <text x="100" y="262" text-anchor="middle" font-size="10" fill="white">SO(32)</text>
        <line x1="140" y1="240" x2="250" y2="210" stroke="#95a5a6" stroke-width="2" />
        
        <!-- Heterotic E8xE8 -->
        <circle cx="300" cy="350" r="40" fill="#1abc9c" opacity="0.5" />
        <text x="300" y="350" text-anchor="middle" font-size="10" fill="white">Het</text>
        <text x="300" y="362" text-anchor="middle" font-size="10" fill="white">E₈×E₈</text>
        <line x1="300" y1="310" x2="300" y2="250" stroke="#95a5a6" stroke-width="2" />
      </svg>
    </div>
    
    <div class="theory-details">
      <div class="theory-card type-i">
        <h4><i class="fas fa-code-branch"></i> Type I</h4>
        <ul>
          <li>Open and closed strings</li>
          <li>N=1 supersymmetry</li>
          <li>Gauge group SO(32)</li>
          <li>Unoriented strings</li>
        </ul>
        <div class="visual-hint">
          <svg viewBox="0 0 100 50">
            <circle cx="50" cy="25" r="15" fill="none" stroke="#e74c3c" stroke-width="2" />
            <path d="M 20 25 L 80 25" stroke="#e74c3c" stroke-width="2" />
          </svg>
        </div>
      </div>
      
      <div class="theory-card type-iia">
        <h4><i class="fas fa-yin-yang"></i> Type IIA</h4>
        <ul>
          <li>Closed strings only</li>
          <li>N=2 supersymmetry (non-chiral)</li>
          <li>Massless fermions of both chiralities</li>
        </ul>
        <div class="visual-hint">
          <svg viewBox="0 0 100 50">
            <circle cx="35" cy="25" r="15" fill="none" stroke="#27ae60" stroke-width="2" />
            <circle cx="65" cy="25" r="15" fill="none" stroke="#27ae60" stroke-width="2" />
            <text x="35" y="30" text-anchor="middle" font-size="10">L</text>
            <text x="65" y="30" text-anchor="middle" font-size="10">R</text>
          </svg>
        </div>
      </div>
      
      <div class="theory-card type-iib">
        <h4><i class="fas fa-sync"></i> Type IIB</h4>
        <ul>
          <li>Closed strings only</li>
          <li>N=2 supersymmetry (chiral)</li>
          <li>Self-dual 4-form field</li>
        </ul>
        <div class="visual-hint">
          <svg viewBox="0 0 100 50">
            <circle cx="50" cy="25" r="15" fill="none" stroke="#f39c12" stroke-width="2" />
            <path d="M 50 10 Q 65 25, 50 40 Q 35 25, 50 10" fill="none" stroke="#f39c12" stroke-width="1" />
          </svg>
        </div>
      </div>
      
      <div class="theory-card heterotic-so">
        <h4><i class="fas fa-puzzle-piece"></i> Heterotic SO(32)</h4>
        <ul>
          <li>Closed strings only</li>
          <li>N=1 supersymmetry</li>
          <li>Left-moving: superstring</li>
          <li>Right-moving: bosonic string</li>
        </ul>
      </div>
      
      <div class="theory-card heterotic-e8">
        <h4><i class="fas fa-project-diagram"></i> Heterotic E₈×E₈</h4>
        <ul>
          <li>Closed strings only</li>
          <li>N=1 supersymmetry</li>
          <li>Exceptional gauge group</li>
        </ul>
        <div class="group-structure">
          <span class="group-tag">E₈</span>
          <span class="times">×</span>
          <span class="group-tag">E₈</span>
        </div>
      </div>
    </div>
  </div>
</div>

## D-Branes

### Definition

D-branes are extended objects where open strings can end:
- Dp-brane: p spatial dimensions
- Satisfy Dirichlet boundary conditions

### Dynamics

**DBI Action:**
```
S = -T_p ∫ d^{p+1}ξ e^{-φ} √(-det(G + B + 2πα'F))
```

Where:
- G = induced metric
- B = Kalb-Ramond field
- F = electromagnetic field strength

### D-Brane Charges

D-branes carry Ramond-Ramond charges:
```
μ_p = T_p/g_s
```

Where g_s is the string coupling.

## T-Duality

### Concept

Duality between small and large dimensions:
```
R ↔ α'/R
```

### Transformation Rules

Under T-duality in direction X^9:
- Type IIA ↔ Type IIB
- Heterotic SO(32) ↔ Heterotic E₈×E₈
- Dp-brane → D(p±1)-brane

### Winding Modes

T-duality exchanges momentum and winding:
```
p ↔ w
n/R ↔ mR/α'
```

## S-Duality

### Strong-Weak Duality

Relates strong and weak coupling:
```
g_s ↔ 1/g_s
```

### Type IIB Self-Duality

Type IIB is self-dual under S-duality:
```
τ → -1/τ
```

Where τ = C₀ + ie^{-φ} (axion-dilaton)

### F-Strings and D-Strings

S-duality relates:
- Fundamental strings (F-strings)
- D1-branes (D-strings)

## M-Theory

### Eleven Dimensions

Strong coupling limit of Type IIA:
- Extra dimension emerges
- 11D supergravity at low energy

### Relations

```
R₁₁ = g_s ℓ_s
```

Where R₁₁ is the radius of the 11th dimension.

### M2 and M5 Branes

Extended objects in M-theory:
- M2-brane: 2 spatial dimensions
- M5-brane: 5 spatial dimensions

### Web of Dualities

All five string theories and M-theory are connected:
```
Type IIA ←→ M-theory on S¹
Type IIB ←→ F-theory on T²
E₈×E₈ ←→ M-theory on S¹/Z₂
```

## Compactification

### Calabi-Yau Manifolds

To get 4D physics from 10D:
- Compactify 6 dimensions
- Calabi-Yau preserves N=1 supersymmetry

**Properties:**
- Ricci-flat (R_mn = 0)
- SU(3) holonomy
- Complex, Kähler

### Moduli

Parameters of compactification:
- **Kähler moduli:** Sizes of cycles
- **Complex structure moduli:** Shapes
- **Dilaton:** String coupling

### Flux Compactifications

Adding fluxes stabilizes moduli:
```
∫_Σ F = n ∈ Z
```

This leads to:
- Moduli stabilization
- de Sitter vacua
- Landscape of vacua

## AdS/CFT Correspondence

### Statement

Equivalence between:
- Type IIB string theory on AdS₅ × S⁵
- N=4 Super Yang-Mills in 4D

### Dictionary

```
g_YM² = g_s
λ = g_YM²N = R⁴/α'²
```

Where λ is the 't Hooft coupling.

### Applications

- Strong coupling physics
- Quantum gravity in AdS
- Condensed matter systems
- QCD-like theories

## Black Holes in String Theory

### Microscopic Entropy

String theory provides microscopic description:
```
S = A/4G = S_micro
```

Counting D-brane bound states reproduces Bekenstein-Hawking entropy.

### Fuzzballs

String theory resolution of singularities:
- Black holes as "fuzzballs"
- Smooth horizonless geometries
- Information paradox resolution

### Black Hole Correspondence

Small black holes ↔ Elementary strings at high temperature

## Cosmological Applications

### String Cosmology

**Pre-Big Bang scenario:**
- T-duality suggests pre-Bang phase
- Dilaton-driven inflation

**Brane World scenarios:**
- Our universe as a 3-brane
- Extra dimensions can be large

### String Landscape

Vast number of vacua: ~10⁵⁰⁰
- Different compactifications
- Different fluxes
- Anthropic principle debates

### Inflation in String Theory

Challenges and proposals:
- Moduli stabilization required
- DBI inflation
- Axion monodromy inflation

## Mathematical Structure

### Conformal Field Theory

2D CFT on worldsheet:
- Virasoro algebra
- Vertex operators
- BRST quantization

### Algebraic Geometry

- Calabi-Yau manifolds
- Mirror symmetry
- Derived categories

### Topological String Theory

Simplified versions:
- A-model: Kähler structure
- B-model: Complex structure
- Topological invariants

## Experimental Prospects

### Direct Tests

Challenging due to high energy scale:
- String scale ~10¹⁹ GeV
- Extra dimensions
- Supersymmetry

### Indirect Evidence

- Supersymmetric particles at LHC
- Cosmic strings
- Primordial gravitational waves
- Black hole physics

### Low-Energy Predictions

- Gauge coupling unification
- Yukawa couplings
- Neutrino masses
- Dark matter candidates

## Criticisms and Challenges

### Lack of Uniqueness

- Many consistent vacua
- No selection principle
- Landscape vs. Swampland

### Predictability

- Too many parameters
- Anthropic reasoning
- Post-dictions vs. predictions

### Mathematical Rigor

- Non-perturbative definition needed
- Background independence
- Off-shell formulation

## Current Research

### Swampland Program

Constraints on effective field theories:
- Distance conjecture
- Weak gravity conjecture
- de Sitter conjecture

### Holography

Extensions of AdS/CFT:
- dS/CFT correspondence
- Flat space holography
- Entanglement entropy

### Quantum Information

String theory meets quantum information:
- Error correcting codes
- Tensor networks
- Quantum complexity

### Amplitudes Program

Modern methods for scattering:
- Twistor strings
- Amplituhedron
- Double copy relations

## Graduate-Level Mathematical Formalism

### Worldsheet Conformal Field Theory

#### Polyakov Path Integral

**Gauge-fixed action:**
```
S = 1/(4πα') ∫ d²σ ∂X^μ∂̄X_μ
```

In conformal gauge: h_{ab} = e^φη_{ab}

**Mode expansion:**
```
X^μ(z,̄z) = x^μ - iα'/2 p^μ ln|z|² + i√(α'/2) Σ_{n≠0} (1/n)[α^μ_n z^{-n} + ̃α^μ_n ̄z^{-n}]
```

**Virasoro algebra:**
```
[L_m, L_n] = (m-n)L_{m+n} + c/12 m(m²-1)δ_{m+n,0}
```

For bosonic string: c = D (spacetime dimensions)

#### Vertex Operators

**Tachyon:** V_T = :e^{ik·X}:

**Graviton/Dilaton/B-field:**
```
V^{(1)} = ζ_{μν} :(∂X^μ + ik·ψψ^μ)e^{ik·X}:
```

**Integrated vertex operators:**
```
V^{(0)} = ∫ d²z V^{(1)}(z,̄z)
```

#### BRST Quantization

**BRST charge:**
```
Q_B = ∮ (cT + ½c∂c + ̃c̄T + ½̃c∂̄̃c)
```

**Physical states:** Q_B|φ⟩ = 0, |φ⟩ ≠ Q_B|χ⟩

**Cohomology:** H*(Q_B) gives physical spectrum

### Superstring Theory: RNS Formalism

#### Worldsheet Supersymmetry

**RNS action:**
```
S = 1/(4πα') ∫ d²σ [∂_αX^μ∂^αX_μ + ψ^μρ^α∂_αψ_μ]
```

**Superconformal algebra:**
```
{G_r, G_s} = 2L_{r+s} + c/2(r² - 1/4)δ_{r+s,0}
[L_m, G_r] = (m/2 - r)G_{m+r}
```

For superstring: c = 3D/2

#### GSO Projection

**Fermion number operator:**
```
F = (-1)^F 　with　F = Σ_{r>0} ψ^{-r}·ψ^r
```

**GSO projection:** Keep states with (-1)^F = ±(-1)^{̃F}

**Spin structures:**
- NS (Neveu-Schwarz): Half-integer modes
- R (Ramond): Integer modes

**Sectors:**
- NS-NS: Bosonic fields (graviton, dilaton, B-field)
- R-R: Form fields
- NS-R, R-NS: Fermions

### Green-Schwarz Formalism

#### Spacetime Supersymmetry

**GS action:**
```
S = -T/2 ∫ d²σ [√-h h^{ab}Π_a^μΠ_{bμ} + ε^{ab}Π_a^μ̄θ^AΓ_μ∂_bθ^A]
```

Where Π^μ = ∂X^μ - ̄θ^AΓ^μ∂θ^A

**Kappa symmetry:** Gauge symmetry ensuring spacetime SUSY

**Light-cone gauge:** Manifestly supersymmetric

### D-Brane Physics

#### Boundary Conditions

**Neumann:** ∂_n X^μ|_{∂Σ} = 0
**Dirichlet:** ∂_t X^μ|_{∂Σ} = 0

**T-duality:** N ↔ D boundary conditions

#### Effective Actions

**DBI action expanded:**
```
S = -T_p∫d^{p+1}ξ e^{-φ}[1 + (2πα')²/4 F_{μν}F^{μν} + O(F⁴)]
```

**Chern-Simons terms:**
```
S_{CS} = μ_p ∫ C ∧ e^{2πα'F}
```

#### D-Brane Interactions

**Open string spectrum:** Gauge fields on worldvolume

**Chan-Paton factors:** U(N) gauge theory for N coincident branes

**Tachyon condensation:** Brane annihilation, K-theory classification

### M-Theory and Dualities

#### M-Theory Basics

**11D supergravity low-energy limit:**
```
S = 1/(2κ²) ∫ d¹¹x √-g [R - ½|F_4|²] + 1/6 ∫ C_3 ∧ F_4 ∧ F_4
```

**M2-branes:** Membranes with worldvolume theory
**M5-branes:** 5-branes with self-dual 3-form

#### Web of Dualities

**S-duality:** Type IIB self-dual under g_s → 1/g_s

**Complete duality web:**
```
M-theory on S¹ → Type IIA
M-theory on T² → Type IIB
M-theory on S¹/Z₂ → E₈×E₈ heterotic
```

**U-duality:** Combines S and T dualities

### Compactification

#### Calabi-Yau Manifolds

**Definition:** Kähler manifold with SU(n) holonomy

**Properties:**
- Ricci-flat: R_{ij} = 0
- Admits covariantly constant spinor
- c₁ = 0

**Hodge numbers:** h^{p,q} characterize topology
- h^{1,1}: Kähler moduli
- h^{2,1}: Complex structure moduli

#### Moduli Stabilization

**Flux compactifications:**
```
W = ∫ Ω ∧ (F_3 - τH_3)
```

**KKLT scenario:** All moduli stabilized by fluxes and non-perturbative effects

**Large volume scenario:** Exponentially large extra dimensions

### AdS/CFT Correspondence

#### Precise Statement

**Type IIB on AdS₅×S⁵ ↔ N=4 SYM in 4D**

**Dictionary:**
```
⟨O(x)⟩_{CFT} = δS_{gravity}/δφ_0(x)|_{φ_0→O}
```

**Holographic renormalization:** Regulate divergences

#### Generalizations

**AdS₃/CFT₂:** M-theory on AdS₃×S⁸ ↔ ABJM theory

**AdS₂/CFT₁:** Near-horizon of extremal black holes

**Non-conformal:** Dp-branes for p≠3

### Black Holes and Entropy

#### Strominger-Vafa Calculation

**D-brane configuration:** D1-D5-P system

**Microscopic entropy:**
```
S_{micro} = 2π√(N₁N₅n)
```

**Bekenstein-Hawking:**
```
S_{BH} = A/4G = 2π√(N₁N₅n)
```

Perfect agreement!

#### Attractor Mechanism

**Near-horizon geometry:** AdS₂×S²

**Attractor equations:**
```
∂V/∂z^i|_{horizon} = 0
```

Moduli fixed by charges, independent of asymptotic values

### Topological String Theory

#### A-Model

**Action:** ∫ Σ φ*(ω) + {Q, V}

**Observables:** Gromov-Witten invariants

**Target space:** Kähler moduli

#### B-Model

**Holomorphic anomaly equation:**
```
∂F^{(g)}/∂̄t^i = ½C^{ijk}_{̄i}(D_jD_kF^{(g-1)} + Σ_{h} D_jF^{(h)}D_kF^{(g-h)})
```

**Mirror symmetry:** A-model(X) = B-model(Y)

### Amplitudes and Modern Methods

#### Scattering Equations

**CHY formulation:**
```
A_n = ∫ dμ_n I_L(σ)I_R(σ)
```

Where dμ_n = ∏_i dσ_i δ(Σ_j k_j·P_j/(σ_i-σ_j))

#### Ambitwistor Strings

**Action:** S = ∫ P_μ ∂̄X^μ

**Critical dimension:** None!

**Tree amplitudes:** Equivalent to CHY

### Swampland Program

#### Conjectures

**Distance conjecture:** Λ ~ M_P e^{-αd}

**Weak gravity conjecture:** m ≤ qM_P

**de Sitter conjecture:** |∇V| ≥ cV/M_P

#### Implications

- Constraints on inflation
- No stable dS vacua?
- Emergence of kinetic terms

### Quantum Information in String Theory

#### Holographic Entanglement Entropy

**Ryu-Takayanagi formula:**
```
S_A = Area(γ_A)/(4G_N)
```

**Quantum corrections:** S = ⟨Area/4G⟩ + S_{bulk}

#### Complexity

**CV conjecture:** C = V/GL

**CA conjecture:** C = Action/πℏ

**Applications:** Black hole interior, firewalls

### Modern Computational Tools

```python
import numpy as np
from sympy import symbols, Matrix, simplify

def calabi_yau_metric(z, z_bar, kahler_potential):
    """Compute CY metric from Kähler potential"""
    n = len(z)
    g = Matrix.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            g[i,j] = kahler_potential.diff(z[i]).diff(z_bar[j])
    
    return g

def yukawa_coupling(omega, A, B, C):
    """Compute Yukawa couplings from holomorphic 3-form"""
    # Y_ABC = ∫_X Ω ∧ ∂_A∂_B∂_C
    return omega.diff(A).diff(B).diff(C)

def gromov_witten_invariant(degree, genus, marked_points):
    """Placeholder for GW invariant calculation"""
    # In practice, use localization or mirror symmetry
    pass

def ads_cft_correlator(operators, positions):
    """Compute correlator using AdS/CFT"""
    # Solve classical equations in AdS
    # Extract boundary behavior
    pass
```

## Research Frontiers

### Non-perturbative String Theory

**Matrix models:** BFSS, IKKT proposals

**String field theory:** Covariant formulation

**Background independence:** Emergent spacetime

### Quantum Gravity Phenomenology

**String cosmology:** Trans-Planckian signatures

**Black hole information:** Fuzzballs vs firewalls

**Lorentz violation:** Stringy dispersion relations

### Mathematical Developments

**Topological modular forms:** tmf and string theory

**Derived categories:** D-branes and stability

**Moonshine:** Connections to sporadic groups

### Connections to Experiment

**Collider signatures:** Extra dimensions, SUSY

**Cosmological observations:** Primordial gravitational waves

**Condensed matter:** AdS/CMT applications

## References and Further Reading

### Classic Textbooks
1. **Polchinski** - *String Theory* (2 volumes)
2. **Green, Schwarz & Witten** - *Superstring Theory* (2 volumes)
3. **Becker, Becker & Schwarz** - *String Theory and M-Theory*
4. **Kiritsis** - *String Theory in a Nutshell*

### Advanced Monographs
1. **D'Hoker & Phong** - *Two-loop superstrings* (series)
2. **Hori et al.** - *Mirror Symmetry*
3. **Ammon & Erdmenger** - *Gauge/Gravity Duality*
4. **Vafa & Zaslow** - *Mirror Symmetry* (Clay monograph)

### Recent Reviews
1. **Aharony et al.** - *Large N field theories, string theory and gravity* (2000)
2. **Brennan, Carta & Vafa** - *The string landscape, the swampland, and the missing corner* (2017)
3. **Harlow** - *TASI lectures on the emergence of the bulk in AdS/CFT* (2018)
4. **Van Raamsdonk** - *Building up spacetime with quantum entanglement* (2010)

### Specialized Topics
1. **Sen** - *String field theory* reviews
2. **Douglas & Nekrasov** - *Noncommutative field theory* (2001)
3. **Berkovits** - *Pure spinor formalism*
4. **Gopakumar & Vafa** - *Topological strings and large N duality*

## Future Directions

1. **Non-perturbative formulation**
2. **Observable predictions**
3. **Quantum gravity phenomenology**
4. **Connection to real world physics**
5. **Mathematical foundations**

String theory remains one of the most active areas of theoretical physics, providing deep insights into quantum gravity, black holes, and the fundamental structure of spacetime. While experimental verification remains elusive, its mathematical richness and conceptual breakthroughs continue to influence many areas of physics and mathematics.