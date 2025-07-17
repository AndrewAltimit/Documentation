---
layout: docs
title: String Theory
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---

<!-- Custom styles for string theory visualizations -->
<link rel="stylesheet" href="{{ '/assets/css/physics-string-theory.css' | relative_url }}">

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
          <!-- Define gradients for vibrating string -->
          <defs>
            <linearGradient id="stringGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style="stop-color:#3498db;stop-opacity:0.8" />
              <stop offset="50%" style="stop-color:#e74c3c;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#3498db;stop-opacity:0.8" />
            </linearGradient>
          </defs>
          <!-- Fundamental mode (n=0) - the circle -->
          <circle cx="75" cy="75" r="40" fill="none" stroke="#2c3e50" stroke-width="1" stroke-dasharray="2,2" opacity="0.5" />
          <!-- First harmonic (n=1) -->
          <path d="M 35 75 Q 55 65, 75 75 Q 95 85, 115 75 Q 95 65, 75 75 Q 55 85, 35 75" 
                fill="none" stroke="#3498db" stroke-width="2.5" opacity="0.8" />
          <!-- Second harmonic (n=2) -->
          <path d="M 40 70 Q 50 65, 60 70 Q 70 75, 80 70 Q 90 65, 100 70 Q 110 75, 120 70 Q 110 80, 100 75 Q 90 70, 80 75 Q 70 80, 60 75 Q 50 70, 40 75" 
                fill="none" stroke="#e74c3c" stroke-width="1.5" opacity="0.6" />
          <!-- Third harmonic (n=3) - subtle -->
          <path d="M 38 72 Q 43 70, 48 72 Q 53 74, 58 72 Q 63 70, 68 72 Q 73 74, 78 72 Q 83 70, 88 72 Q 93 74, 98 72 Q 103 70, 108 72 Q 113 74, 118 72" 
                fill="none" stroke="#f39c12" stroke-width="1" opacity="0.4" />
          <!-- Arrows showing vibration direction -->
          <path d="M 55 60 L 55 55" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrowUp)" opacity="0.7" />
          <path d="M 95 90 L 95 95" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrowDown)" opacity="0.7" />
          <!-- Define arrow markers -->
          <defs>
            <marker id="arrowUp" markerWidth="10" markerHeight="10" refX="5" refY="0" orient="auto">
              <path d="M 0 5 L 5 0 L 10 5" stroke="#95a5a6" fill="none" />
            </marker>
            <marker id="arrowDown" markerWidth="10" markerHeight="10" refX="5" refY="10" orient="auto">
              <path d="M 0 5 L 5 10 L 10 5" stroke="#95a5a6" fill="none" />
            </marker>
          </defs>
          <text x="75" y="130" text-anchor="middle" font-size="11" fill="#2c3e50">Vibrating Closed String</text>
          <text x="75" y="142" text-anchor="middle" font-size="9" fill="#7f8c8d">Multiple modes (n=0,1,2,...)</text>
        </svg>
      </div>
      
      <div class="string-card open">
        <h4><i class="fas fa-wave-square"></i> Open Strings</h4>
        <p>Have two distinct endpoints</p>
        <svg viewBox="0 0 150 150" class="string-visual">
          <!-- Endpoints (enlarged and highlighted) -->
          <circle cx="30" cy="75" r="6" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
          <circle cx="120" cy="75" r="6" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
          <!-- Fundamental mode -->
          <path d="M 30 75 Q 75 45, 120 75" fill="none" stroke="#27ae60" stroke-width="3" opacity="0.9" />
          <!-- First overtone -->
          <path d="M 30 75 Q 52.5 60, 75 75 Q 97.5 90, 120 75" fill="none" stroke="#3498db" stroke-width="2" opacity="0.7" />
          <!-- Second overtone -->
          <path d="M 30 75 Q 45 68, 60 75 Q 75 82, 90 75 Q 105 68, 120 75" fill="none" stroke="#f39c12" stroke-width="1.5" opacity="0.5" />
          <!-- Boundary condition indicators -->
          <text x="30" y="60" text-anchor="middle" font-size="9" fill="#7f8c8d">Fixed</text>
          <text x="120" y="60" text-anchor="middle" font-size="9" fill="#7f8c8d">Fixed</text>
          <!-- Vibration arrows -->
          <path d="M 75 50 L 75 45" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrowUp2)" opacity="0.7" />
          <path d="M 52.5 85 L 52.5 90" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrowDown2)" opacity="0.7" />
          <path d="M 97.5 85 L 97.5 90" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrowDown2)" opacity="0.7" />
          <!-- Define arrow markers -->
          <defs>
            <marker id="arrowUp2" markerWidth="10" markerHeight="10" refX="5" refY="0" orient="auto">
              <path d="M 0 5 L 5 0 L 10 5" stroke="#95a5a6" fill="none" />
            </marker>
            <marker id="arrowDown2" markerWidth="10" markerHeight="10" refX="5" refY="10" orient="auto">
              <path d="M 0 5 L 5 10 L 10 5" stroke="#95a5a6" fill="none" />
            </marker>
          </defs>
          <text x="75" y="130" text-anchor="middle" font-size="11" fill="#2c3e50">Vibrating Open String</text>
          <text x="75" y="142" text-anchor="middle" font-size="9" fill="#7f8c8d">Standing wave modes</text>
        </svg>
      </div>
    </div>
    
    <div class="vibrational-modes">
      <h4>Vibrational Modes = Particles</h4>
      <div class="mode-spectrum">
        <svg viewBox="0 0 500 200">
          <!-- Energy level axis -->
          <line x1="50" y1="180" x2="50" y2="20" stroke="#2c3e50" stroke-width="2" marker-end="url(#energyArrow)" />
          <text x="40" y="15" font-size="10" text-anchor="end">E/M_s</text>
          
          <!-- Define arrow -->
          <defs>
            <marker id="energyArrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
              <path d="M 0 10 L 5 0 L 10 10" fill="none" stroke="#2c3e50" />
            </marker>
          </defs>
          
          <!-- Ground state (tachyon for bosonic string) -->
          <line x1="60" y1="160" x2="150" y2="160" stroke="#e74c3c" stroke-width="3" />
          <text x="155" y="163" font-size="9">n=0: Tachyon (m²<0)</text>
          <circle cx="55" cy="160" r="2" fill="#e74c3c" />
          
          <!-- First excited state -->
          <line x1="60" y1="130" x2="150" y2="130" stroke="#3498db" stroke-width="3" />
          <text x="155" y="133" font-size="9">n=1: Massless (graviton, photon)</text>
          <circle cx="55" cy="130" r="2" fill="#3498db" />
          <!-- Show mode shape -->
          <path d="M 70 125 Q 80 120, 90 125 Q 100 130, 110 125 Q 120 120, 130 125" 
                fill="none" stroke="#3498db" stroke-width="1" opacity="0.6" />
          
          <!-- Second excited state -->
          <line x1="60" y1="100" x2="150" y2="100" stroke="#27ae60" stroke-width="3" />
          <text x="155" y="103" font-size="9">n=2: Massive particles</text>
          <circle cx="55" cy="100" r="2" fill="#27ae60" />
          <!-- Show mode shape -->
          <path d="M 70 95 Q 75 92, 80 95 Q 85 98, 90 95 Q 95 92, 100 95 Q 105 98, 110 95 Q 115 92, 120 95 Q 125 98, 130 95" 
                fill="none" stroke="#27ae60" stroke-width="1" opacity="0.6" />
          
          <!-- Higher states -->
          <line x1="60" y1="70" x2="150" y2="70" stroke="#f39c12" stroke-width="2" opacity="0.7" />
          <line x1="60" y1="50" x2="150" y2="50" stroke="#9b59b6" stroke-width="2" opacity="0.5" />
          <line x1="60" y1="30" x2="150" y2="30" stroke="#95a5a6" stroke-width="2" opacity="0.3" />
          <text x="155" y="50" font-size="9" fill="#7f8c8d">n≥3: Heavy particles</text>
          
          <!-- Mass formula -->
          <rect x="280" y="40" width="200" height="120" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="1" rx="5" />
          <text x="380" y="60" text-anchor="middle" font-size="10" font-weight="bold">Mass Formula</text>
          <text x="380" y="85" text-anchor="middle" font-size="9">Bosonic: M² = (n-1)/ℓ_s²</text>
          <text x="380" y="105" text-anchor="middle" font-size="9">Superstring: M² = n/ℓ_s²</text>
          <text x="380" y="130" text-anchor="middle" font-size="8" fill="#7f8c8d">n = oscillator number</text>
          <text x="380" y="145" text-anchor="middle" font-size="8" fill="#7f8c8d">ℓ_s = string length</text>
        </svg>
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
          <!-- Define arrow marker -->
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M 0 0 L 8 3 L 0 6" fill="#95a5a6" />
            </marker>
          </defs>
          <!-- Point particle at different times -->
          <circle cx="75" cy="180" r="5" fill="#e74c3c" opacity="1" />
          <circle cx="75" cy="150" r="5" fill="#e74c3c" opacity="0.7" />
          <circle cx="75" cy="120" r="5" fill="#e74c3c" opacity="0.5" />
          <circle cx="75" cy="90" r="5" fill="#e74c3c" opacity="0.3" />
          <circle cx="75" cy="60" r="5" fill="#e74c3c" opacity="0.2" />
          <circle cx="75" cy="30" r="5" fill="#e74c3c" opacity="0.1" />
          <!-- Worldline -->
          <line x1="75" y1="180" x2="75" y2="30" stroke="#3498db" stroke-width="2" stroke-dasharray="none" />
          <text x="75" y="195" text-anchor="middle" font-size="11">Point particle</text>
          <text x="110" y="100" font-size="9">Worldline</text>
          <text x="110" y="112" font-size="8">(1D trajectory)</text>
          <!-- Time axis -->
          <path d="M 20 180 L 20 30" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrow)" />
          <text x="10" y="25" font-size="9">time</text>
          <!-- Space axis -->
          <path d="M 20 180 L 130 180" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrow)" />
          <text x="135" y="185" font-size="9">space</text>
        </svg>
      </div>
      
      <div class="trace-item">
        <svg viewBox="0 0 200 200">
          <!-- String at initial time -->
          <ellipse cx="100" cy="180" rx="40" ry="8" fill="none" stroke="#e74c3c" stroke-width="3" />
          <!-- String at intermediate times (showing evolution) -->
          <ellipse cx="100" cy="150" rx="35" ry="7" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.6" />
          <ellipse cx="100" cy="120" rx="30" ry="6" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.4" />
          <ellipse cx="100" cy="90" rx="25" ry="5" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.3" />
          <ellipse cx="100" cy="60" rx="20" ry="4" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.2" />
          <ellipse cx="100" cy="30" rx="15" ry="3" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.1" />
          <!-- Worldsheet surface (with gradient for depth) -->
          <defs>
            <linearGradient id="sheetGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#3498db;stop-opacity:0.1" />
              <stop offset="100%" style="stop-color:#3498db;stop-opacity:0.4" />
            </linearGradient>
          </defs>
          <path d="M 60 180 L 85 30 L 115 30 L 140 180 Z" fill="url(#sheetGradient)" stroke="#3498db" stroke-width="2" />
          <!-- Grid lines on worldsheet to show 2D nature -->
          <path d="M 70 150 L 130 150" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <path d="M 75 120 L 125 120" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <path d="M 80 90 L 120 90" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <path d="M 85 60 L 115 60" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <!-- Vertical lines -->
          <path d="M 80 180 L 95 30" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <path d="M 100 180 L 100 30" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <path d="M 120 180 L 105 30" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
          <text x="100" y="195" text-anchor="middle" font-size="11">String</text>
          <text x="155" y="100" font-size="9">Worldsheet</text>
          <text x="155" y="112" font-size="8">(2D surface)</text>
          <!-- Time axis -->
          <path d="M 20 180 L 20 30" stroke="#95a5a6" stroke-width="1" marker-end="url(#arrow)" />
          <text x="10" y="25" font-size="9">time</text>
          <!-- Space axes indicators -->
          <text x="30" y="190" font-size="8" fill="#7f8c8d">σ parameter</text>
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
            <!-- Initial string positions -->
            <ellipse cx="100" cy="120" rx="30" ry="6" fill="none" stroke="#e74c3c" stroke-width="2" />
            <ellipse cx="100" cy="40" rx="20" ry="4" fill="none" stroke="#e74c3c" stroke-width="2" />
            <!-- Minimal area worldsheet -->
            <path d="M 70 120 Q 80 80, 80 40 M 130 120 Q 120 80, 120 40" 
                  fill="none" stroke="#3498db" stroke-width="1" opacity="0.5" />
            <path d="M 70 120 Q 70 80, 80 40 L 120 40 Q 130 80, 130 120 Z" 
                  fill="#3498db" opacity="0.2" stroke="#3498db" stroke-width="2" />
            <!-- Grid lines to show surface -->
            <path d="M 75 100 L 125 100" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
            <path d="M 77 80 L 123 80" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
            <path d="M 79 60 L 121 60" stroke="#2980b9" stroke-width="0.5" opacity="0.5" />
            <!-- Comparison with non-minimal surface -->
            <path d="M 70 120 Q 50 80, 80 40" fill="none" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,2" opacity="0.5" />
            <path d="M 130 120 Q 150 80, 120 40" fill="none" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,2" opacity="0.5" />
            <text x="100" y="135" text-anchor="middle" font-size="9">Minimal area = extremal action</text>
            <text x="160" y="80" font-size="7" fill="#7f8c8d">Non-minimal</text>
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
          <!-- Define arrow marker -->
          <defs>
            <marker id="waveArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M 0 0 L 8 3 L 0 6" fill="#2c3e50" />
            </marker>
          </defs>
          <!-- Left-moving wave -->
          <path d="M 30 75 Q 45 60, 60 75 Q 75 90, 90 75 Q 105 60, 120 75 Q 135 90, 150 75" 
                fill="none" stroke="#3498db" stroke-width="2.5" opacity="0.8" />
          <!-- Right-moving wave -->
          <path d="M 150 75 Q 165 90, 180 75 Q 195 60, 210 75 Q 225 90, 240 75 Q 255 60, 270 75" 
                fill="none" stroke="#e74c3c" stroke-width="2.5" opacity="0.8" />
          <!-- Superposition in middle -->
          <path d="M 120 75 Q 135 55, 150 75 Q 165 95, 180 75" 
                fill="none" stroke="#9b59b6" stroke-width="3" />
          <!-- Direction arrows -->
          <path d="M 90 60 L 110 60" stroke="#3498db" stroke-width="2" marker-end="url(#waveArrow)" />
          <path d="M 210 90 L 190 90" stroke="#e74c3c" stroke-width="2" marker-end="url(#waveArrow)" />
          <!-- Labels -->
          <text x="75" y="50" text-anchor="middle" font-size="9" fill="#3498db">X(τ+σ)</text>
          <text x="225" y="50" text-anchor="middle" font-size="9" fill="#e74c3c">X(τ-σ)</text>
          <text x="150" y="130" text-anchor="middle" font-size="10">General solution: X = X_L(τ+σ) + X_R(τ-σ)</text>
          <text x="150" y="142" text-anchor="middle" font-size="8" fill="#7f8c8d">Left-moving + Right-moving waves</text>
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
          <!-- Parametrized closed string -->
          <circle cx="75" cy="75" r="40" fill="none" stroke="#2c3e50" stroke-width="1" stroke-dasharray="2,2" opacity="0.5" />
          <!-- Show parametrization -->
          <circle cx="115" cy="75" r="3" fill="#e74c3c" />
          <text x="125" y="70" font-size="8" fill="#e74c3c">σ=0</text>
          <text x="125" y="80" font-size="8" fill="#e74c3c">σ=2π</text>
          <!-- Vibrating modes -->
          <path d="M 35 75 Q 55 65, 75 75 Q 95 85, 115 75 Q 95 65, 75 75 Q 55 85, 35 75" 
                fill="none" stroke="#3498db" stroke-width="2" />
          <!-- Direction arrow showing parametrization -->
          <path d="M 110 70 Q 115 60, 120 70" fill="none" stroke="#7f8c8d" stroke-width="1" marker-end="url(#arrowParam)" />
          <defs>
            <marker id="arrowParam" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M 0 0 L 6 3 L 0 6" fill="#7f8c8d" />
            </marker>
          </defs>
          <text x="75" y="130" text-anchor="middle" font-size="10">Periodic: X(σ+2π) = X(σ)</text>
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
              <!-- String with free endpoints -->
              <path d="M 20 40 Q 60 20, 100 40" fill="none" stroke="#27ae60" stroke-width="2" />
              <!-- Second mode to show standing wave -->
              <path d="M 20 40 Q 40 50, 60 40 Q 80 30, 100 40" fill="none" stroke="#3498db" stroke-width="1.5" opacity="0.6" />
              <!-- Endpoints -->
              <circle cx="20" cy="40" r="4" fill="#27ae60" stroke="#229954" stroke-width="1" />
              <circle cx="100" cy="40" r="4" fill="#27ae60" stroke="#229954" stroke-width="1" />
              <!-- Tangent lines showing ∂X/∂σ = 0 -->
              <path d="M 10 40 L 30 40" stroke="#95a5a6" stroke-width="1" stroke-dasharray="2,2" />
              <path d="M 90 40 L 110 40" stroke="#95a5a6" stroke-width="1" stroke-dasharray="2,2" />
              <!-- Annotations -->
              <text x="20" y="30" text-anchor="middle" font-size="7" fill="#7f8c8d">∂X/∂σ=0</text>
              <text x="100" y="30" text-anchor="middle" font-size="7" fill="#7f8c8d">∂X/∂σ=0</text>
            </svg>
          </div>
          
          <div class="bc-type dirichlet">
            <h5>Dirichlet BC</h5>
            <div class="equation-box small">
              $$X^\mu = \text{const}$$
            </div>
            <p>Fixed endpoints (D-branes)</p>
            <svg viewBox="0 0 120 80">
              <!-- D-branes as surfaces -->
              <rect x="10" y="30" width="15" height="20" fill="#e74c3c" opacity="0.3" stroke="#c0392b" stroke-width="2" />
              <rect x="95" y="30" width="15" height="20" fill="#e74c3c" opacity="0.3" stroke="#c0392b" stroke-width="2" />
              <!-- String attached to D-branes -->
              <path d="M 25 40 Q 60 20, 95 40" fill="none" stroke="#e74c3c" stroke-width="2" />
              <!-- Show multiple modes -->
              <path d="M 25 40 Q 45 48, 60 40 Q 75 32, 95 40" fill="none" stroke="#f39c12" stroke-width="1.5" opacity="0.6" />
              <!-- Fixed points highlighted -->
              <circle cx="25" cy="40" r="3" fill="#fff" stroke="#e74c3c" stroke-width="2" />
              <circle cx="95" cy="40" r="3" fill="#fff" stroke="#e74c3c" stroke-width="2" />
              <!-- Labels -->
              <text x="17.5" y="25" text-anchor="middle" font-size="7" fill="#c0392b">D-brane</text>
              <text x="102.5" y="25" text-anchor="middle" font-size="7" fill="#c0392b">D-brane</text>
              <text x="60" y="65" text-anchor="middle" font-size="8">X=const at ends</text>
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
        <!-- M-theory at top -->
        <ellipse cx="300" cy="50" rx="60" ry="30" fill="#34495e" opacity="0.7" />
        <text x="300" y="55" text-anchor="middle" font-size="14" font-weight="bold" fill="white">M-Theory</text>
        <text x="300" y="70" text-anchor="middle" font-size="10" fill="white">(11D)</text>
        
        <!-- Type I -->
        <circle cx="150" cy="150" r="40" fill="#e74c3c" opacity="0.6" />
        <text x="150" y="155" text-anchor="middle" font-size="11" fill="white">Type I</text>
        
        <!-- Type IIA -->
        <circle cx="250" cy="200" r="40" fill="#27ae60" opacity="0.6" />
        <text x="250" y="205" text-anchor="middle" font-size="11" fill="white">Type IIA</text>
        
        <!-- Type IIB -->
        <circle cx="350" cy="200" r="40" fill="#f39c12" opacity="0.6" />
        <text x="350" y="205" text-anchor="middle" font-size="11" fill="white">Type IIB</text>
        
        <!-- Heterotic SO(32) -->
        <circle cx="450" cy="150" r="40" fill="#9b59b6" opacity="0.6" />
        <text x="450" y="150" text-anchor="middle" font-size="10" fill="white">Het</text>
        <text x="450" y="162" text-anchor="middle" font-size="10" fill="white">SO(32)</text>
        
        <!-- Heterotic E8xE8 -->
        <circle cx="300" cy="300" r="40" fill="#1abc9c" opacity="0.6" />
        <text x="300" y="300" text-anchor="middle" font-size="10" fill="white">Het</text>
        <text x="300" y="312" text-anchor="middle" font-size="10" fill="white">E₈×E₈</text>
        
        <!-- Duality connections with labels -->
        <!-- M-theory to Type IIA -->
        <path d="M 280 75 L 260 165" stroke="#2c3e50" stroke-width="2" stroke-dasharray="4,2" />
        <text x="240" y="120" font-size="8" fill="#2c3e50" transform="rotate(-80, 240, 120)">S¹ reduction</text>
        
        <!-- Type IIA to Type IIB (T-duality) -->
        <path d="M 290 200 L 310 200" stroke="#e67e22" stroke-width="3" />
        <text x="300" y="195" text-anchor="middle" font-size="8" fill="#d35400">T-duality</text>
        
        <!-- Type I to Het SO(32) (S-duality) -->
        <path d="M 190 150 L 410 150" stroke="#8e44ad" stroke-width="3" />
        <text x="300" y="145" text-anchor="middle" font-size="8" fill="#8e44ad">S-duality</text>
        
        <!-- M-theory to Het E8xE8 -->
        <path d="M 300 80 L 300 260" stroke="#2c3e50" stroke-width="2" stroke-dasharray="4,2" />
        <text x="305" y="170" font-size="8" fill="#2c3e50">S¹/Z₂</text>
        
        <!-- Type IIB S-duality (self-dual) -->
        <path d="M 380 180 Q 420 160, 420 200 Q 420 240, 380 220" 
              fill="none" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrowSelf)" />
        <text x="430" y="200" font-size="8" fill="#e74c3c">S-dual</text>
        
        <!-- Het SO(32) to Het E8xE8 (T-duality) -->
        <path d="M 425 180 Q 375 240, 325 270" stroke="#16a085" stroke-width="2" stroke-dasharray="3,3" />
        <text x="380" y="230" font-size="8" fill="#16a085" transform="rotate(-45, 380, 230)">T-duality</text>
        
        <!-- Define arrow marker for self-duality -->
        <defs>
          <marker id="arrowSelf" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M 0 0 L 8 3 L 0 6" fill="#e74c3c" />
          </marker>
        </defs>
        
        <!-- Legend -->
        <text x="20" y="380" font-size="10" fill="#2c3e50">Dualities:</text>
        <line x1="80" y1="378" x2="100" y2="378" stroke="#e67e22" stroke-width="3" />
        <text x="105" y="382" font-size="9" fill="#2c3e50">T-duality</text>
        <line x1="170" y1="378" x2="190" y2="378" stroke="#8e44ad" stroke-width="3" />
        <text x="195" y="382" font-size="9" fill="#2c3e50">S-duality</text>
        <line x1="260" y1="378" x2="280" y2="378" stroke="#2c3e50" stroke-width="2" stroke-dasharray="4,2" />
        <text x="285" y="382" font-size="9" fill="#2c3e50">Dimensional reduction</text>
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
            <!-- Closed string -->
            <circle cx="50" cy="25" r="12" fill="none" stroke="#e74c3c" stroke-width="2" />
            <!-- Open string with endpoints -->
            <path d="M 15 25 Q 32.5 15, 50 25 Q 67.5 35, 85 25" stroke="#e74c3c" stroke-width="2" />
            <circle cx="15" cy="25" r="3" fill="#e74c3c" />
            <circle cx="85" cy="25" r="3" fill="#e74c3c" />
            <text x="50" y="45" text-anchor="middle" font-size="8" fill="#7f8c8d">Open + Closed</text>
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
            <!-- Left-moving modes -->
            <circle cx="50" cy="25" r="15" fill="none" stroke="#27ae60" stroke-width="1" opacity="0.5" />
            <path d="M 35 25 Q 42.5 20, 50 25 Q 57.5 30, 65 25" 
                  fill="none" stroke="#27ae60" stroke-width="2" opacity="0.8" />
            <!-- Right-moving modes -->
            <path d="M 35 25 Q 42.5 30, 50 25 Q 57.5 20, 65 25" 
                  fill="none" stroke="#2ecc71" stroke-width="2" opacity="0.8" />
            <!-- Arrows showing chirality -->
            <path d="M 30 20 Q 35 15, 40 20" stroke="#27ae60" stroke-width="1" marker-end="url(#arrowL)" />
            <path d="M 70 30 Q 65 35, 60 30" stroke="#2ecc71" stroke-width="1" marker-end="url(#arrowR)" />
            <defs>
              <marker id="arrowL" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M 0 0 L 5 3 L 0 6" fill="#27ae60" />
              </marker>
              <marker id="arrowR" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M 0 0 L 5 3 L 0 6" fill="#2ecc71" />
              </marker>
            </defs>
            <text x="50" y="45" text-anchor="middle" font-size="8" fill="#7f8c8d">Non-chiral</text>
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
            <!-- Closed string with chiral modes -->
            <circle cx="50" cy="25" r="15" fill="none" stroke="#f39c12" stroke-width="1" opacity="0.5" />
            <!-- Same chirality for both left and right movers -->
            <path d="M 35 25 Q 42.5 18, 50 25 Q 57.5 32, 65 25" 
                  fill="none" stroke="#f39c12" stroke-width="2" />
            <path d="M 38 22 Q 45 15, 52 22" 
                  fill="none" stroke="#e67e22" stroke-width="1.5" opacity="0.7" />
            <!-- Chirality arrows (both same direction) -->
            <path d="M 30 20 Q 35 15, 40 20" stroke="#f39c12" stroke-width="1" marker-end="url(#arrowCh)" />
            <path d="M 60 20 Q 65 15, 70 20" stroke="#f39c12" stroke-width="1" marker-end="url(#arrowCh)" />
            <defs>
              <marker id="arrowCh" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M 0 0 L 5 3 L 0 6" fill="#f39c12" />
              </marker>
            </defs>
            <text x="50" y="45" text-anchor="middle" font-size="8" fill="#7f8c8d">Chiral</text>
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
        <div class="visual-hint">
          <svg viewBox="0 0 100 50">
            <!-- Heterotic = hybrid -->
            <circle cx="50" cy="25" r="15" fill="none" stroke="#9b59b6" stroke-width="1" opacity="0.5" />
            <!-- Left-moving: superstring (10D) -->
            <path d="M 35 25 Q 42.5 20, 50 25 Q 57.5 30, 65 25" 
                  fill="none" stroke="#9b59b6" stroke-width="2" />
            <!-- Right-moving: bosonic (26D compactified) -->
            <path d="M 35 25 Q 42.5 32, 50 25 Q 57.5 18, 65 25" 
                  fill="none" stroke="#8e44ad" stroke-width="2" stroke-dasharray="3,1" />
            <text x="20" y="15" font-size="7" fill="#9b59b6">10D</text>
            <text x="80" y="15" font-size="7" fill="#8e44ad">26D→10D</text>
            <text x="50" y="45" text-anchor="middle" font-size="8" fill="#7f8c8d">Hybrid</text>
          </svg>
        </div>
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