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
        <svg viewBox="0 0 220 200" class="string-visual" style="max-width: 500px; width: 100%;">
          <!-- Define gradients and markers -->
          <defs>
            <linearGradient id="stringGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style="stop-color:#2980b9;stop-opacity:0.9" />
              <stop offset="50%" style="stop-color:#c0392b;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#2980b9;stop-opacity:0.9" />
            </linearGradient>
            <marker id="arrowUp" markerWidth="10" markerHeight="10" refX="5" refY="0" orient="auto">
              <path d="M 0 5 L 5 0 L 10 5" stroke="#555" fill="none" stroke-width="1.5" />
            </marker>
            <marker id="arrowDown" markerWidth="10" markerHeight="10" refX="5" refY="10" orient="auto">
              <path d="M 0 5 L 5 10 L 10 5" stroke="#555" fill="none" stroke-width="1.5" />
            </marker>
          </defs>
          <!-- Background circle showing equilibrium position -->
          <circle cx="110" cy="85" r="50" fill="none" stroke="#34495e" stroke-width="2" stroke-dasharray="4,3" opacity="0.4" />
          <text x="170" y="50" font-size="12" fill="#555" font-style="italic">n=0 (ground)</text>
          <!-- First harmonic (n=1) - prominent -->
          <path d="M 60 85 Q 85 65, 110 85 Q 135 105, 160 85 Q 135 65, 110 85 Q 85 105, 60 85"
                fill="none" stroke="#2980b9" stroke-width="3.5" opacity="0.95" />
          <text x="185" y="75" font-size="12" fill="#2980b9" font-weight="bold">n=1</text>
          <!-- Second harmonic (n=2) -->
          <path d="M 65 80 Q 80 70, 95 80 Q 110 90, 125 80 Q 140 70, 155 80 Q 140 95, 125 85 Q 110 75, 95 85 Q 80 95, 65 85"
                fill="none" stroke="#c0392b" stroke-width="2.5" opacity="0.8" />
          <text x="185" y="95" font-size="12" fill="#c0392b" font-weight="bold">n=2</text>
          <!-- Third harmonic (n=3) -->
          <path d="M 63 82 Q 70 77, 77 82 Q 84 87, 91 82 Q 98 77, 105 82 Q 112 87, 119 82 Q 126 77, 133 82 Q 140 87, 147 82 Q 154 77, 161 82"
                fill="none" stroke="#d35400" stroke-width="2" opacity="0.65" />
          <text x="185" y="115" font-size="12" fill="#d35400" font-weight="bold">n=3</text>
          <!-- Vibration direction arrows -->
          <path d="M 85 55 L 85 45" stroke="#555" stroke-width="2" marker-end="url(#arrowUp)" />
          <path d="M 135 115 L 135 125" stroke="#555" stroke-width="2" marker-end="url(#arrowDown)" />
          <text x="75" y="42" font-size="11" fill="#555">vibration</text>
          <!-- Main labels -->
          <text x="110" y="160" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">Vibrating Closed String</text>
          <text x="110" y="178" text-anchor="middle" font-size="13" fill="#555">Harmonic modes n = 0, 1, 2, 3, ...</text>
          <text x="110" y="193" text-anchor="middle" font-size="11" fill="#777">Higher n = higher energy/mass</text>
        </svg>
      </div>
      
      <div class="string-card open">
        <h4><i class="fas fa-wave-square"></i> Open Strings</h4>
        <p>Have two distinct endpoints</p>
        <svg viewBox="0 0 240 200" class="string-visual" style="max-width: 500px; width: 100%;">
          <!-- Define arrow markers first -->
          <defs>
            <marker id="arrowUp2" markerWidth="10" markerHeight="10" refX="5" refY="0" orient="auto">
              <path d="M 0 5 L 5 0 L 10 5" stroke="#555" fill="none" stroke-width="1.5" />
            </marker>
            <marker id="arrowDown2" markerWidth="10" markerHeight="10" refX="5" refY="10" orient="auto">
              <path d="M 0 5 L 5 10 L 10 5" stroke="#555" fill="none" stroke-width="1.5" />
            </marker>
          </defs>
          <!-- Background equilibrium line -->
          <line x1="40" y1="85" x2="180" y2="85" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="5,3" opacity="0.5" />
          <!-- Endpoints (enlarged and highlighted) -->
          <circle cx="40" cy="85" r="8" fill="#c0392b" stroke="#922b21" stroke-width="2" />
          <circle cx="180" cy="85" r="8" fill="#c0392b" stroke="#922b21" stroke-width="2" />
          <!-- Fundamental mode (n=1) -->
          <path d="M 40 85 Q 110 40, 180 85" fill="none" stroke="#27ae60" stroke-width="3.5" opacity="0.95" />
          <text x="200" y="55" font-size="12" fill="#27ae60" font-weight="bold">n=1</text>
          <!-- First overtone (n=2) -->
          <path d="M 40 85 Q 75 60, 110 85 Q 145 110, 180 85" fill="none" stroke="#2980b9" stroke-width="2.8" opacity="0.85" />
          <text x="200" y="80" font-size="12" fill="#2980b9" font-weight="bold">n=2</text>
          <!-- Second overtone (n=3) -->
          <path d="M 40 85 Q 62 68, 85 85 Q 110 100, 135 85 Q 157 68, 180 85" fill="none" stroke="#d35400" stroke-width="2.2" opacity="0.7" />
          <text x="200" y="105" font-size="12" fill="#d35400" font-weight="bold">n=3</text>
          <!-- Boundary condition labels -->
          <text x="40" y="60" text-anchor="middle" font-size="12" fill="#555" font-weight="bold">Endpoint</text>
          <text x="180" y="60" text-anchor="middle" font-size="12" fill="#555" font-weight="bold">Endpoint</text>
          <!-- Vibration arrows -->
          <path d="M 110 45 L 110 32" stroke="#555" stroke-width="2" marker-end="url(#arrowUp2)" />
          <path d="M 75 108 L 75 120" stroke="#555" stroke-width="2" marker-end="url(#arrowDown2)" />
          <path d="M 145 108 L 145 120" stroke="#555" stroke-width="2" marker-end="url(#arrowDown2)" />
          <text x="125" y="30" font-size="11" fill="#555">vibration</text>
          <!-- Main labels -->
          <text x="110" y="150" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">Vibrating Open String</text>
          <text x="110" y="168" text-anchor="middle" font-size="13" fill="#555">Standing wave modes with fixed ends</text>
          <text x="110" y="185" text-anchor="middle" font-size="11" fill="#777">Endpoints can attach to D-branes</text>
        </svg>
      </div>
    </div>
    
    <div class="vibrational-modes">
      <h4>Vibrational Modes = Particles</h4>
      <div class="mode-spectrum">
        <svg viewBox="0 0 600 260" style="max-width: 500px; width: 100%;">
          <!-- Define arrow -->
          <defs>
            <marker id="energyArrow" markerWidth="12" markerHeight="12" refX="6" refY="6" orient="auto">
              <path d="M 0 12 L 6 0 L 12 12" fill="none" stroke="#2c3e50" stroke-width="2" />
            </marker>
          </defs>

          <!-- Energy level axis -->
          <line x1="60" y1="220" x2="60" y2="30" stroke="#2c3e50" stroke-width="2.5" marker-end="url(#energyArrow)" />
          <text x="30" y="25" font-size="14" fill="#2c3e50" font-weight="bold">Energy</text>
          <text x="30" y="42" font-size="12" fill="#555">(E/Ms)</text>

          <!-- Ground state (tachyon for bosonic string) -->
          <line x1="80" y1="195" x2="200" y2="195" stroke="#c0392b" stroke-width="4" />
          <circle cx="70" cy="195" r="4" fill="#c0392b" />
          <text x="210" y="200" font-size="14" fill="#c0392b" font-weight="bold">n=0: Tachyon</text>
          <text x="210" y="215" font-size="12" fill="#777">(m² &lt; 0, unstable in bosonic string)</text>

          <!-- First excited state - MASSLESS -->
          <line x1="80" y1="155" x2="200" y2="155" stroke="#2980b9" stroke-width="4" />
          <circle cx="70" cy="155" r="4" fill="#2980b9" />
          <!-- Mode shape visualization -->
          <path d="M 90 150 Q 110 140, 130 150 Q 150 160, 170 150 Q 190 140, 200 150"
                fill="none" stroke="#2980b9" stroke-width="2" opacity="0.7" />
          <text x="210" y="150" font-size="14" fill="#2980b9" font-weight="bold">n=1: Massless States</text>
          <text x="210" y="167" font-size="12" fill="#555">Graviton, Dilaton, B-field</text>

          <!-- Second excited state - MASSIVE -->
          <line x1="80" y1="110" x2="200" y2="110" stroke="#27ae60" stroke-width="4" />
          <circle cx="70" cy="110" r="4" fill="#27ae60" />
          <!-- Mode shape -->
          <path d="M 90 105 Q 100 98, 110 105 Q 120 112, 130 105 Q 140 98, 150 105 Q 160 112, 170 105 Q 180 98, 190 105"
                fill="none" stroke="#27ae60" stroke-width="2" opacity="0.7" />
          <text x="210" y="108" font-size="14" fill="#27ae60" font-weight="bold">n=2: Massive Particles</text>
          <text x="210" y="123" font-size="12" fill="#555">Mass proportional to 1/string length</text>

          <!-- Higher states (tower) -->
          <line x1="80" y1="75" x2="200" y2="75" stroke="#d35400" stroke-width="3" opacity="0.85" />
          <circle cx="70" cy="75" r="3" fill="#d35400" opacity="0.85" />
          <line x1="80" y1="50" x2="200" y2="50" stroke="#8e44ad" stroke-width="3" opacity="0.7" />
          <circle cx="70" cy="50" r="3" fill="#8e44ad" opacity="0.7" />
          <line x1="80" y1="35" x2="200" y2="35" stroke="#7f8c8d" stroke-width="2" opacity="0.5" />
          <circle cx="70" cy="35" r="2" fill="#7f8c8d" opacity="0.5" />
          <text x="210" y="60" font-size="14" fill="#555" font-weight="bold">n = 3, 4, 5, ...</text>
          <text x="210" y="77" font-size="12" fill="#777">Infinite tower of heavy particles</text>

          <!-- Mass formula box -->
          <rect x="400" y="85" width="185" height="130" fill="#f8f9fa" stroke="#34495e" stroke-width="2" rx="8" />
          <text x="492" y="110" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">Mass Formulas</text>
          <line x1="415" y1="120" x2="570" y2="120" stroke="#bdc3c7" stroke-width="1" />
          <text x="492" y="145" text-anchor="middle" font-size="13" fill="#c0392b">Bosonic:</text>
          <text x="492" y="162" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">M² = (n-1)/l_s²</text>
          <text x="492" y="185" text-anchor="middle" font-size="13" fill="#27ae60">Superstring:</text>
          <text x="492" y="202" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">M² = n/l_s²</text>

          <!-- Legend -->
          <text x="80" y="248" font-size="12" fill="#555">n = oscillator excitation number</text>
          <text x="320" y="248" font-size="12" fill="#555">l_s = string length scale</text>
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
      <svg viewBox="0 0 550 150" style="max-width: 500px; width: 100%;">
        <!-- Title -->
        <text x="275" y="22" text-anchor="middle" font-size="16" fill="#2c3e50" font-weight="bold">Length Scales in Physics</text>

        <!-- Scale bar with gradient -->
        <defs>
          <linearGradient id="scaleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#8e44ad;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#2980b9;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#27ae60;stop-opacity:1" />
          </linearGradient>
        </defs>
        <line x1="60" y1="70" x2="490" y2="70" stroke="url(#scaleGradient)" stroke-width="6" stroke-linecap="round" />

        <!-- Markers -->
        <line x1="60" y1="58" x2="60" y2="82" stroke="#8e44ad" stroke-width="3" />
        <line x1="170" y1="58" x2="170" y2="82" stroke="#9b59b6" stroke-width="3" />
        <line x1="330" y1="58" x2="330" y2="82" stroke="#2980b9" stroke-width="3" />
        <line x1="490" y1="58" x2="490" y2="82" stroke="#27ae60" stroke-width="3" />

        <!-- Scale labels (names) -->
        <text x="60" y="100" text-anchor="middle" font-size="14" fill="#8e44ad" font-weight="bold">Planck</text>
        <text x="170" y="100" text-anchor="middle" font-size="14" fill="#9b59b6" font-weight="bold">String</text>
        <text x="330" y="100" text-anchor="middle" font-size="14" fill="#2980b9" font-weight="bold">Proton</text>
        <text x="490" y="100" text-anchor="middle" font-size="14" fill="#27ae60" font-weight="bold">Atom</text>

        <!-- Scale values -->
        <text x="60" y="50" text-anchor="middle" font-size="13" fill="#555" font-weight="bold">10⁻³⁵ m</text>
        <text x="170" y="50" text-anchor="middle" font-size="13" fill="#555" font-weight="bold">~10⁻³⁵ m</text>
        <text x="330" y="50" text-anchor="middle" font-size="13" fill="#555" font-weight="bold">10⁻¹⁵ m</text>
        <text x="490" y="50" text-anchor="middle" font-size="13" fill="#555" font-weight="bold">10⁻¹⁰ m</text>

        <!-- Descriptive labels -->
        <text x="60" y="117" text-anchor="middle" font-size="11" fill="#777">Length</text>
        <text x="170" y="117" text-anchor="middle" font-size="11" fill="#777">Length</text>
        <text x="330" y="117" text-anchor="middle" font-size="11" fill="#777">Radius</text>
        <text x="490" y="117" text-anchor="middle" font-size="11" fill="#777">Radius</text>

        <!-- Scale factor annotations -->
        <text x="115" y="135" text-anchor="middle" font-size="11" fill="#aaa">~equal</text>
        <text x="250" y="135" text-anchor="middle" font-size="11" fill="#aaa">10²⁰ larger</text>
        <text x="410" y="135" text-anchor="middle" font-size="11" fill="#aaa">10⁵ larger</text>
      </svg>
    </div>
  </div>
  
  <div class="worldsheet-concept">
    <h3><i class="fas fa-scroll"></i> Worldsheet</h3>
    <p>As a string moves through spacetime, it traces out a two-dimensional surface called a worldsheet:</p>
    
    <div class="worldsheet-comparison">
      <div class="trace-item">
        <svg viewBox="0 0 200 260" style="max-width: 500px; width: 100%;">
          <!-- Define arrow marker -->
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M 0 0 L 8 3 L 0 6" fill="#555" />
            </marker>
          </defs>
          <!-- Title -->
          <text x="100" y="20" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">Point Particle</text>
          <!-- Point particle at different times -->
          <circle cx="100" cy="210" r="8" fill="#c0392b" opacity="1" />
          <circle cx="100" cy="175" r="7" fill="#c0392b" opacity="0.7" />
          <circle cx="100" cy="140" r="6" fill="#c0392b" opacity="0.5" />
          <circle cx="100" cy="105" r="5" fill="#c0392b" opacity="0.35" />
          <circle cx="100" cy="70" r="4" fill="#c0392b" opacity="0.2" />
          <circle cx="100" cy="45" r="3" fill="#c0392b" opacity="0.1" />
          <!-- Worldline -->
          <line x1="100" y1="210" x2="100" y2="45" stroke="#2980b9" stroke-width="3" />
          <!-- Labels -->
          <text x="100" y="235" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">0-Dimensional</text>
          <text x="145" y="125" font-size="13" fill="#2980b9" font-weight="bold">Worldline</text>
          <text x="145" y="142" font-size="12" fill="#555">(1D curve)</text>
          <!-- Time axis -->
          <path d="M 30 210 L 30 40" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />
          <text x="25" y="32" font-size="12" fill="#555" font-weight="bold">t</text>
          <text x="18" y="50" font-size="10" fill="#777">(time)</text>
          <!-- Space axis -->
          <path d="M 30 210 L 175 210" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />
          <text x="182" y="215" font-size="12" fill="#555" font-weight="bold">x</text>
        </svg>
      </div>

      <div class="trace-item">
        <svg viewBox="0 0 260 260" style="max-width: 500px; width: 100%;">
          <!-- Gradients and markers -->
          <defs>
            <linearGradient id="sheetGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#2980b9;stop-opacity:0.15" />
              <stop offset="100%" style="stop-color:#2980b9;stop-opacity:0.5" />
            </linearGradient>
            <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M 0 0 L 8 3 L 0 6" fill="#555" />
            </marker>
          </defs>
          <!-- Title -->
          <text x="130" y="20" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">Closed String</text>
          <!-- String at initial time (bottom) -->
          <ellipse cx="130" cy="205" rx="50" ry="12" fill="none" stroke="#c0392b" stroke-width="3.5" />
          <!-- String at intermediate times -->
          <ellipse cx="130" cy="170" rx="42" ry="10" fill="none" stroke="#c0392b" stroke-width="2.5" opacity="0.6" />
          <ellipse cx="130" cy="135" rx="34" ry="8" fill="none" stroke="#c0392b" stroke-width="2" opacity="0.4" />
          <ellipse cx="130" cy="100" rx="26" ry="6" fill="none" stroke="#c0392b" stroke-width="1.5" opacity="0.3" />
          <ellipse cx="130" cy="70" rx="18" ry="4" fill="none" stroke="#c0392b" stroke-width="1" opacity="0.2" />
          <ellipse cx="130" cy="45" rx="12" ry="3" fill="none" stroke="#c0392b" stroke-width="0.5" opacity="0.1" />
          <!-- Worldsheet surface -->
          <path d="M 80 205 L 118 45 L 142 45 L 180 205 Z" fill="url(#sheetGradient)" stroke="#2980b9" stroke-width="2.5" />
          <!-- Grid lines on worldsheet (tau = constant) -->
          <path d="M 88 170 L 172 170" stroke="#1a5276" stroke-width="1" opacity="0.5" />
          <path d="M 96 135 L 164 135" stroke="#1a5276" stroke-width="1" opacity="0.5" />
          <path d="M 104 100 L 156 100" stroke="#1a5276" stroke-width="1" opacity="0.5" />
          <path d="M 112 70 L 148 70" stroke="#1a5276" stroke-width="1" opacity="0.5" />
          <!-- Grid lines (sigma = constant) -->
          <path d="M 100 205 L 122 45" stroke="#1a5276" stroke-width="1" opacity="0.4" />
          <path d="M 130 205 L 130 45" stroke="#1a5276" stroke-width="1" opacity="0.4" />
          <path d="M 160 205 L 138 45" stroke="#1a5276" stroke-width="1" opacity="0.4" />
          <!-- Labels -->
          <text x="130" y="235" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">1-Dimensional</text>
          <text x="205" y="120" font-size="13" fill="#2980b9" font-weight="bold">Worldsheet</text>
          <text x="205" y="138" font-size="12" fill="#555">(2D surface)</text>
          <!-- Time axis -->
          <path d="M 35 205 L 35 40" stroke="#555" stroke-width="2" marker-end="url(#arrow2)" />
          <text x="30" y="32" font-size="12" fill="#555" font-weight="bold">t</text>
          <text x="23" y="50" font-size="10" fill="#777">(time)</text>
          <!-- Parameter labels -->
          <text x="60" y="250" font-size="11" fill="#777">Parameters: (tau, sigma)</text>
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
          <svg viewBox="0 0 280 200" style="max-width: 500px; width: 100%;">
            <defs>
              <linearGradient id="minAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#2980b9;stop-opacity:0.15" />
                <stop offset="100%" style="stop-color:#2980b9;stop-opacity:0.45" />
              </linearGradient>
            </defs>
            <!-- Title -->
            <text x="140" y="20" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">Minimal Area Principle</text>
            <!-- Initial string position (bottom) -->
            <ellipse cx="140" cy="160" rx="45" ry="10" fill="none" stroke="#c0392b" stroke-width="3" />
            <text x="200" y="165" font-size="12" fill="#c0392b" font-weight="bold">t = 0</text>
            <!-- Final string position (top) -->
            <ellipse cx="140" cy="55" rx="30" ry="7" fill="none" stroke="#c0392b" stroke-width="3" />
            <text x="185" y="60" font-size="12" fill="#c0392b" font-weight="bold">t = T</text>
            <!-- Minimal area worldsheet (highlighted) -->
            <path d="M 95 160 Q 95 107, 110 55 L 170 55 Q 185 107, 185 160 Z"
                  fill="url(#minAreaGrad)" stroke="#2980b9" stroke-width="2.5" />
            <!-- Grid lines on worldsheet -->
            <path d="M 100 140 L 180 140" stroke="#1a5276" stroke-width="1" opacity="0.5" />
            <path d="M 103 120 L 177 120" stroke="#1a5276" stroke-width="1" opacity="0.5" />
            <path d="M 106 100 L 174 100" stroke="#1a5276" stroke-width="1" opacity="0.5" />
            <path d="M 109 80 L 171 80" stroke="#1a5276" stroke-width="1" opacity="0.5" />
            <!-- Non-minimal surfaces (comparison) -->
            <path d="M 95 160 Q 60 107, 110 55" fill="none" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="5,3" opacity="0.6" />
            <path d="M 185 160 Q 220 107, 170 55" fill="none" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="5,3" opacity="0.6" />
            <text x="45" y="105" font-size="11" fill="#7f8c8d">Non-minimal</text>
            <text x="215" y="105" font-size="11" fill="#7f8c8d">Non-minimal</text>
            <!-- Caption -->
            <text x="140" y="185" text-anchor="middle" font-size="13" fill="#2c3e50" font-weight="bold">Classical path: Minimal worldsheet area</text>
            <text x="140" y="198" text-anchor="middle" font-size="11" fill="#555">S = -T x Area (Nambu-Goto action)</text>
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
        <svg viewBox="0 0 420 180" style="max-width: 500px; width: 100%;">
          <!-- Define arrow markers -->
          <defs>
            <marker id="waveArrowL" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M 0 0 L 8 3 L 0 6" fill="#2980b9" />
            </marker>
            <marker id="waveArrowR" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M 0 0 L 8 3 L 0 6" fill="#c0392b" />
            </marker>
          </defs>
          <!-- Title -->
          <text x="210" y="22" text-anchor="middle" font-size="15" fill="#2c3e50" font-weight="bold">Wave Equation Solutions</text>
          <!-- Left-moving wave (blue) -->
          <path d="M 40 85 Q 60 60, 80 85 Q 100 110, 120 85 Q 140 60, 160 85 Q 180 110, 200 85"
                fill="none" stroke="#2980b9" stroke-width="3.5" opacity="0.9" />
          <!-- Right-moving wave (red) -->
          <path d="M 220 85 Q 240 110, 260 85 Q 280 60, 300 85 Q 320 110, 340 85 Q 360 60, 380 85"
                fill="none" stroke="#c0392b" stroke-width="3.5" opacity="0.9" />
          <!-- Superposition region -->
          <rect x="185" y="55" width="50" height="60" fill="#9b59b6" opacity="0.15" rx="5" />
          <path d="M 180 85 Q 195 55, 210 85 Q 225 115, 240 85"
                fill="none" stroke="#8e44ad" stroke-width="4" />
          <!-- Direction arrows -->
          <path d="M 125 50 L 155 50" stroke="#2980b9" stroke-width="3" marker-end="url(#waveArrowL)" />
          <path d="M 295 120 L 265 120" stroke="#c0392b" stroke-width="3" marker-end="url(#waveArrowR)" />
          <!-- Wave labels -->
          <text x="90" y="42" font-size="14" fill="#2980b9" font-weight="bold">X_L(tau + sigma)</text>
          <text x="290" y="42" font-size="14" fill="#c0392b" font-weight="bold">X_R(tau - sigma)</text>
          <!-- Superposition label -->
          <text x="210" y="135" text-anchor="middle" font-size="12" fill="#8e44ad" font-weight="bold">Superposition</text>
          <!-- General solution formula -->
          <rect x="80" y="148" width="260" height="28" fill="#f8f9fa" stroke="#bdc3c7" stroke-width="1" rx="4" />
          <text x="210" y="167" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">X = X_L(tau+sigma) + X_R(tau-sigma)</text>
          <!-- Legend -->
          <line x1="360" y1="150" x2="380" y2="150" stroke="#2980b9" stroke-width="3" />
          <text x="385" y="154" font-size="11" fill="#2980b9">Left-moving</text>
          <line x1="360" y1="168" x2="380" y2="168" stroke="#c0392b" stroke-width="3" />
          <text x="385" y="172" font-size="11" fill="#c0392b">Right-moving</text>
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
        
        <svg viewBox="0 0 220 180" style="max-width: 500px; width: 100%;">
          <defs>
            <marker id="arrowParam" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M 0 0 L 6 3 L 0 6" fill="#555" />
            </marker>
          </defs>
          <!-- Title -->
          <text x="110" y="20" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">Periodic Boundary</text>
          <!-- Equilibrium circle -->
          <circle cx="110" cy="90" r="50" fill="none" stroke="#34495e" stroke-width="2" stroke-dasharray="4,3" opacity="0.4" />
          <!-- Vibrating string mode -->
          <path d="M 60 90 Q 85 70, 110 90 Q 135 110, 160 90 Q 135 70, 110 90 Q 85 110, 60 90"
                fill="none" stroke="#2980b9" stroke-width="3" />
          <!-- Parameter point sigma=0=2pi -->
          <circle cx="160" cy="90" r="6" fill="#c0392b" stroke="#922b21" stroke-width="2" />
          <!-- Direction arrow showing parametrization -->
          <path d="M 155 70 Q 165 55, 175 70" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrowParam)" />
          <!-- Parameter labels -->
          <text x="180" y="80" font-size="13" fill="#c0392b" font-weight="bold">sigma = 0</text>
          <text x="180" y="98" font-size="13" fill="#c0392b" font-weight="bold">sigma = 2pi</text>
          <text x="185" y="113" font-size="11" fill="#555">(same point!)</text>
          <!-- Caption -->
          <text x="110" y="160" text-anchor="middle" font-size="14" fill="#2c3e50" font-weight="bold">X(sigma + 2pi) = X(sigma)</text>
          <text x="110" y="175" text-anchor="middle" font-size="12" fill="#555">String forms closed loop</text>
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
            <svg viewBox="0 0 180 110" style="max-width: 500px; width: 100%;">
              <!-- Title -->
              <text x="90" y="15" text-anchor="middle" font-size="12" fill="#2c3e50" font-weight="bold">Free Endpoints</text>
              <!-- String vibration mode 1 -->
              <path d="M 30 55 Q 90 25, 150 55" fill="none" stroke="#27ae60" stroke-width="3" />
              <!-- String vibration mode 2 -->
              <path d="M 30 55 Q 60 70, 90 55 Q 120 40, 150 55" fill="none" stroke="#2980b9" stroke-width="2.5" opacity="0.7" />
              <!-- Endpoints (free to move) -->
              <circle cx="30" cy="55" r="6" fill="#27ae60" stroke="#1e8449" stroke-width="2" />
              <circle cx="150" cy="55" r="6" fill="#27ae60" stroke="#1e8449" stroke-width="2" />
              <!-- Tangent lines showing horizontal slope at endpoints -->
              <path d="M 15 55 L 45 55" stroke="#555" stroke-width="2" stroke-dasharray="4,2" />
              <path d="M 135 55 L 165 55" stroke="#555" stroke-width="2" stroke-dasharray="4,2" />
              <!-- Annotations -->
              <text x="30" y="85" text-anchor="middle" font-size="11" fill="#555">dX/d(sigma)=0</text>
              <text x="150" y="85" text-anchor="middle" font-size="11" fill="#555">dX/d(sigma)=0</text>
              <!-- Caption -->
              <text x="90" y="102" text-anchor="middle" font-size="10" fill="#777">Endpoints free to oscillate</text>
            </svg>
          </div>
          
          <div class="bc-type dirichlet">
            <h5>Dirichlet BC</h5>
            <div class="equation-box small">
              $$X^\mu = \text{const}$$
            </div>
            <p>Fixed endpoints (D-branes)</p>
            <svg viewBox="0 0 180 120" style="max-width: 500px; width: 100%;">
              <!-- Title -->
              <text x="90" y="15" text-anchor="middle" font-size="12" fill="#2c3e50" font-weight="bold">Fixed Endpoints</text>
              <!-- D-branes as surfaces -->
              <rect x="12" y="35" width="22" height="50" fill="#c0392b" opacity="0.25" stroke="#922b21" stroke-width="2" rx="3" />
              <rect x="146" y="35" width="22" height="50" fill="#c0392b" opacity="0.25" stroke="#922b21" stroke-width="2" rx="3" />
              <!-- String vibration mode 1 -->
              <path d="M 34 60 Q 90 25, 146 60" fill="none" stroke="#c0392b" stroke-width="3" />
              <!-- String vibration mode 2 -->
              <path d="M 34 60 Q 62 75, 90 60 Q 118 45, 146 60" fill="none" stroke="#d35400" stroke-width="2.5" opacity="0.7" />
              <!-- Fixed points on D-branes -->
              <circle cx="34" cy="60" r="5" fill="#fff" stroke="#c0392b" stroke-width="2.5" />
              <circle cx="146" cy="60" r="5" fill="#fff" stroke="#c0392b" stroke-width="2.5" />
              <!-- D-brane labels -->
              <text x="23" y="28" text-anchor="middle" font-size="11" fill="#922b21" font-weight="bold">D-brane</text>
              <text x="157" y="28" text-anchor="middle" font-size="11" fill="#922b21" font-weight="bold">D-brane</text>
              <!-- Caption -->
              <text x="90" y="100" text-anchor="middle" font-size="12" fill="#555" font-weight="bold">X = constant at ends</text>
              <text x="90" y="115" text-anchor="middle" font-size="10" fill="#777">Endpoints fixed on D-branes</text>
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

$$[\alpha^{\mu}_m, \alpha^{\nu}_n] = m \delta_{m+n,0} \eta^{\mu\nu}$$

### Virasoro Algebra

Constraints from reparametrization invariance:

$$[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12} m(m^2-1)\delta_{m+n,0}$$

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
      <svg viewBox="0 0 700 480" class="theory-diagram" style="max-width: 500px; width: 100%;">
        <!-- Define markers and gradients -->
        <defs>
          <marker id="arrowSelf" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M 0 0 L 8 3 L 0 6" fill="#c0392b" />
          </marker>
          <linearGradient id="mTheoryGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#2c3e50;stop-opacity:0.9" />
            <stop offset="100%" style="stop-color:#1a252f;stop-opacity:0.95" />
          </linearGradient>
        </defs>

        <!-- Title -->
        <text x="350" y="28" text-anchor="middle" font-size="18" fill="#2c3e50" font-weight="bold">Web of String Theory Dualities</text>

        <!-- M-theory at top (central, unified) -->
        <ellipse cx="350" cy="80" rx="80" ry="38" fill="url(#mTheoryGrad)" stroke="#1a252f" stroke-width="2" />
        <text x="350" y="75" text-anchor="middle" font-size="18" font-weight="bold" fill="white">M-Theory</text>
        <text x="350" y="95" text-anchor="middle" font-size="13" fill="#bdc3c7">(11 Dimensions)</text>

        <!-- Type I (left) -->
        <circle cx="130" cy="200" r="55" fill="#c0392b" opacity="0.85" stroke="#922b21" stroke-width="2" />
        <text x="130" y="195" text-anchor="middle" font-size="16" font-weight="bold" fill="white">Type I</text>
        <text x="130" y="213" text-anchor="middle" font-size="11" fill="#fce4ec">SO(32) gauge</text>

        <!-- Type IIA (center-left) -->
        <circle cx="280" cy="270" r="55" fill="#27ae60" opacity="0.85" stroke="#1e8449" stroke-width="2" />
        <text x="280" y="265" text-anchor="middle" font-size="16" font-weight="bold" fill="white">Type IIA</text>
        <text x="280" y="283" text-anchor="middle" font-size="11" fill="#e8f8f5">Non-chiral</text>

        <!-- Type IIB (center-right) -->
        <circle cx="420" cy="270" r="55" fill="#d35400" opacity="0.85" stroke="#a04000" stroke-width="2" />
        <text x="420" y="265" text-anchor="middle" font-size="16" font-weight="bold" fill="white">Type IIB</text>
        <text x="420" y="283" text-anchor="middle" font-size="11" fill="#fef5e7">Chiral</text>

        <!-- Heterotic SO(32) (right) -->
        <circle cx="570" cy="200" r="55" fill="#8e44ad" opacity="0.85" stroke="#6c3483" stroke-width="2" />
        <text x="570" y="190" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Heterotic</text>
        <text x="570" y="208" text-anchor="middle" font-size="14" font-weight="bold" fill="white">SO(32)</text>

        <!-- Heterotic E8xE8 (bottom) -->
        <circle cx="350" cy="390" r="55" fill="#16a085" opacity="0.85" stroke="#0e6655" stroke-width="2" />
        <text x="350" y="382" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Heterotic</text>
        <text x="350" y="402" text-anchor="middle" font-size="14" font-weight="bold" fill="white">E8 x E8</text>

        <!-- DUALITY CONNECTIONS -->

        <!-- M-theory to Type IIA (compactify on circle) -->
        <path d="M 310 115 L 295 220" stroke="#34495e" stroke-width="3" stroke-dasharray="8,4" />
        <rect x="255" y="155" width="70" height="22" fill="white" rx="3" />
        <text x="290" y="170" text-anchor="middle" font-size="11" fill="#34495e" font-weight="bold">S1 circle</text>

        <!-- M-theory to Heterotic E8xE8 (compactify on interval) -->
        <path d="M 350 118 L 350 335" stroke="#34495e" stroke-width="3" stroke-dasharray="8,4" />
        <rect x="355" y="220" width="75" height="22" fill="white" rx="3" />
        <text x="392" y="235" text-anchor="middle" font-size="11" fill="#34495e" font-weight="bold">S1/Z2 orbifold</text>

        <!-- Type IIA to Type IIB (T-duality - strongest connection) -->
        <line x1="335" y1="270" x2="365" y2="270" stroke="#e67e22" stroke-width="5" />
        <rect x="322" y="245" width="55" height="18" fill="white" rx="3" />
        <text x="350" y="258" text-anchor="middle" font-size="12" fill="#d35400" font-weight="bold">T-duality</text>

        <!-- Type I to Heterotic SO(32) (S-duality) -->
        <line x1="185" y1="200" x2="515" y2="200" stroke="#9b59b6" stroke-width="4" />
        <rect x="320" y="180" width="60" height="18" fill="white" rx="3" />
        <text x="350" y="193" text-anchor="middle" font-size="12" fill="#8e44ad" font-weight="bold">S-duality</text>

        <!-- Type IIB self-duality (S-duality loop) -->
        <path d="M 465 245 Q 520 220, 520 270 Q 520 320, 465 295"
              fill="none" stroke="#c0392b" stroke-width="3" marker-end="url(#arrowSelf)" />
        <text x="540" y="270" font-size="12" fill="#c0392b" font-weight="bold">S-dual</text>
        <text x="540" y="285" font-size="10" fill="#777">(self)</text>

        <!-- Heterotic SO(32) to Heterotic E8xE8 (T-duality) -->
        <path d="M 535 245 Q 460 320, 400 365" stroke="#1abc9c" stroke-width="3" stroke-dasharray="6,3" />
        <rect x="465" y="295" width="55" height="18" fill="white" rx="3" />
        <text x="492" y="308" text-anchor="middle" font-size="11" fill="#16a085" font-weight="bold">T-duality</text>

        <!-- LEGEND -->
        <rect x="30" y="420" width="640" height="50" fill="#f8f9fa" stroke="#bdc3c7" stroke-width="1" rx="5" />
        <text x="50" y="442" font-size="13" fill="#2c3e50" font-weight="bold">Dualities:</text>

        <!-- T-duality legend -->
        <line x1="130" y1="440" x2="160" y2="440" stroke="#e67e22" stroke-width="4" />
        <text x="170" y="444" font-size="12" fill="#2c3e50">T-duality (R to 1/R)</text>

        <!-- S-duality legend -->
        <line x1="310" y1="440" x2="340" y2="440" stroke="#9b59b6" stroke-width="4" />
        <text x="350" y="444" font-size="12" fill="#2c3e50">S-duality (g to 1/g)</text>

        <!-- Dimensional reduction legend -->
        <line x1="500" y1="440" x2="530" y2="440" stroke="#34495e" stroke-width="3" stroke-dasharray="8,4" />
        <text x="540" y="444" font-size="12" fill="#2c3e50">Compactification</text>
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
          <svg viewBox="0 0 160 80" style="max-width: 500px; width: 100%;">
            <!-- Title -->
            <text x="80" y="12" text-anchor="middle" font-size="11" fill="#2c3e50" font-weight="bold">Open + Closed Strings</text>
            <!-- Closed string (loop) -->
            <circle cx="50" cy="42" r="18" fill="none" stroke="#c0392b" stroke-width="2.5" />
            <text x="50" y="70" text-anchor="middle" font-size="10" fill="#555">Closed</text>
            <!-- Open string with endpoints -->
            <path d="M 90 42 Q 115 25, 140 42" fill="none" stroke="#c0392b" stroke-width="2.5" />
            <circle cx="90" cy="42" r="5" fill="#c0392b" />
            <circle cx="140" cy="42" r="5" fill="#c0392b" />
            <text x="115" y="70" text-anchor="middle" font-size="10" fill="#555">Open</text>
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
          <svg viewBox="0 0 160 90" style="max-width: 500px; width: 100%;">
            <defs>
              <marker id="arrowL" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M 0 0 L 6 3 L 0 6" fill="#1e8449" />
              </marker>
              <marker id="arrowR" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M 0 0 L 6 3 L 0 6" fill="#27ae60" />
              </marker>
            </defs>
            <!-- Title -->
            <text x="80" y="12" text-anchor="middle" font-size="11" fill="#2c3e50" font-weight="bold">Non-Chiral Fermions</text>
            <!-- Closed string background -->
            <circle cx="80" cy="45" r="22" fill="none" stroke="#27ae60" stroke-width="1.5" opacity="0.4" />
            <!-- Left-moving mode (one chirality) -->
            <path d="M 58 45 Q 70 32, 80 45 Q 90 58, 102 45" fill="none" stroke="#1e8449" stroke-width="2.5" />
            <!-- Right-moving mode (opposite chirality) -->
            <path d="M 58 45 Q 70 58, 80 45 Q 90 32, 102 45" fill="none" stroke="#27ae60" stroke-width="2.5" />
            <!-- Chirality arrows (opposite directions) -->
            <path d="M 50 35 L 62 35" stroke="#1e8449" stroke-width="2" marker-end="url(#arrowL)" />
            <path d="M 110 55 L 98 55" stroke="#27ae60" stroke-width="2" marker-end="url(#arrowR)" />
            <!-- Labels -->
            <text x="45" y="30" font-size="9" fill="#1e8449" font-weight="bold">L</text>
            <text x="112" y="62" font-size="9" fill="#27ae60" font-weight="bold">R</text>
            <text x="80" y="80" text-anchor="middle" font-size="11" fill="#555">Both chiralities present</text>
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
          <svg viewBox="0 0 160 90" style="max-width: 500px; width: 100%;">
            <defs>
              <marker id="arrowCh" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M 0 0 L 6 3 L 0 6" fill="#d35400" />
              </marker>
            </defs>
            <!-- Title -->
            <text x="80" y="12" text-anchor="middle" font-size="11" fill="#2c3e50" font-weight="bold">Chiral Fermions</text>
            <!-- Closed string background -->
            <circle cx="80" cy="45" r="22" fill="none" stroke="#d35400" stroke-width="1.5" opacity="0.4" />
            <!-- Both modes same chirality -->
            <path d="M 58 45 Q 70 32, 80 45 Q 90 58, 102 45" fill="none" stroke="#d35400" stroke-width="2.5" />
            <path d="M 60 43 Q 72 30, 82 43 Q 92 56, 100 43" fill="none" stroke="#e67e22" stroke-width="2" opacity="0.7" />
            <!-- Chirality arrows (same direction) -->
            <path d="M 50 35 L 62 35" stroke="#d35400" stroke-width="2" marker-end="url(#arrowCh)" />
            <path d="M 98 35 L 110 35" stroke="#d35400" stroke-width="2" marker-end="url(#arrowCh)" />
            <!-- Labels -->
            <text x="45" y="30" font-size="9" fill="#d35400" font-weight="bold">L</text>
            <text x="112" y="30" font-size="9" fill="#d35400" font-weight="bold">R</text>
            <text x="80" y="80" text-anchor="middle" font-size="11" fill="#555">Same chirality (both left-handed)</text>
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
          <svg viewBox="0 0 180 90" style="max-width: 500px; width: 100%;">
            <!-- Title -->
            <text x="90" y="12" text-anchor="middle" font-size="11" fill="#2c3e50" font-weight="bold">Hybrid String</text>
            <!-- Closed string background -->
            <circle cx="90" cy="45" r="25" fill="none" stroke="#8e44ad" stroke-width="1.5" opacity="0.4" />
            <!-- Left-moving: superstring (10D) - solid line -->
            <path d="M 65 45 Q 78 28, 90 45 Q 103 62, 115 45" fill="none" stroke="#9b59b6" stroke-width="3" />
            <!-- Right-moving: bosonic (26D compactified) - dashed line -->
            <path d="M 65 45 Q 78 62, 90 45 Q 103 28, 115 45" fill="none" stroke="#6c3483" stroke-width="3" stroke-dasharray="5,2" />
            <!-- Labels -->
            <text x="35" y="32" font-size="10" fill="#9b59b6" font-weight="bold">10D</text>
            <text x="30" y="44" font-size="9" fill="#9b59b6">Superstring</text>
            <text x="135" y="32" font-size="10" fill="#6c3483" font-weight="bold">26D</text>
            <text x="130" y="44" font-size="9" fill="#6c3483">Bosonic</text>
            <text x="90" y="82" text-anchor="middle" font-size="11" fill="#555">Left and right movers different</text>
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

$$S = -T_p \int d^{p+1}\xi \, e^{-\phi} \sqrt{-\det(G + B + 2\pi\alpha' F)}$$

Where:
- G = induced metric
- B = Kalb-Ramond field
- F = electromagnetic field strength

### D-Brane Charges

D-branes carry Ramond-Ramond charges:

$$\mu_p = \frac{T_p}{g_s}$$

Where g_s is the string coupling.

## T-Duality

### Concept

Duality between small and large dimensions:

$$R \leftrightarrow \frac{\alpha'}{R}$$

### Transformation Rules

Under T-duality in direction X^9:
- Type IIA ↔ Type IIB
- Heterotic SO(32) ↔ Heterotic E₈×E₈
- Dp-brane → D(p±1)-brane

### Winding Modes

T-duality exchanges momentum and winding:

$$p \leftrightarrow w$$

$$\frac{n}{R} \leftrightarrow \frac{mR}{\alpha'}$$

## S-Duality

### Strong-Weak Duality

Relates strong and weak coupling:

$$g_s \leftrightarrow \frac{1}{g_s}$$

### Type IIB Self-Duality

Type IIB is self-dual under S-duality:

$$\tau \rightarrow -\frac{1}{\tau}$$

Where $\tau = C_0 + ie^{-\phi}$ (axion-dilaton)

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

$$R_{11} = g_s \ell_s$$

Where $R_{11}$ is the radius of the 11th dimension.

### M2 and M5 Branes

Extended objects in M-theory:
- M2-brane: 2 spatial dimensions
- M5-brane: 5 spatial dimensions

### Web of Dualities

All five string theories and M-theory are connected:

$$\text{Type IIA} \leftrightarrow \text{M-theory on } S^1$$

$$\text{Type IIB} \leftrightarrow \text{F-theory on } T^2$$

$$E_8 \times E_8 \leftrightarrow \text{M-theory on } S^1/\mathbb{Z}_2$$

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

$$\int_{\Sigma} F = n \in \mathbb{Z}$$

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

$$g_{\text{YM}}^2 = g_s$$

$$\lambda = g_{\text{YM}}^2 N = \frac{R^4}{\alpha'^2}$$

Where $\lambda$ is the 't Hooft coupling.

### Applications

- Strong coupling physics
- Quantum gravity in AdS
- Condensed matter systems
- QCD-like theories

## Black Holes in String Theory

### Microscopic Entropy

String theory provides microscopic description:

$$S = \frac{A}{4G} = S_{\text{micro}}$$

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

$$S = \frac{1}{4\pi\alpha'} \int d^2\sigma \, \partial X^{\mu}\bar{\partial}X_{\mu}$$

In conformal gauge: $h_{ab} = e^{\phi}\eta_{ab}$

**Mode expansion:**

$$X^{\mu}(z,\bar{z}) = x^{\mu} - \frac{i\alpha'}{2} p^{\mu} \ln|z|^2 + i\sqrt{\frac{\alpha'}{2}} \sum_{n\neq 0} \frac{1}{n}\left[\alpha^{\mu}_n z^{-n} + \tilde{\alpha}^{\mu}_n \bar{z}^{-n}\right]$$

**Virasoro algebra:**

$$[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12} m(m^2-1)\delta_{m+n,0}$$

For bosonic string: $c = D$ (spacetime dimensions)

#### Vertex Operators

**Tachyon:** $V_T = :e^{ik\cdot X}:$

**Graviton/Dilaton/B-field:**

$$V^{(1)} = \zeta_{\mu\nu} :(\partial X^{\mu} + ik\cdot\psi\psi^{\mu})e^{ik\cdot X}:$$

**Integrated vertex operators:**

$$V^{(0)} = \int d^2z \, V^{(1)}(z,\bar{z})$$

#### BRST Quantization

**BRST charge:**

$$Q_B = \oint \left(cT + \frac{1}{2}c\partial c + \tilde{c}\bar{T} + \frac{1}{2}\tilde{c}\bar{\partial}\tilde{c}\right)$$

**Physical states:** $Q_B|\phi\rangle = 0$, $|\phi\rangle \neq Q_B|\chi\rangle$

**Cohomology:** $H^*(Q_B)$ gives physical spectrum

### Superstring Theory: RNS Formalism

#### Worldsheet Supersymmetry

**RNS action:**

$$S = \frac{1}{4\pi\alpha'} \int d^2\sigma \left[\partial_{\alpha}X^{\mu}\partial^{\alpha}X_{\mu} + \psi^{\mu}\rho^{\alpha}\partial_{\alpha}\psi_{\mu}\right]$$

**Superconformal algebra:**

$$\{G_r, G_s\} = 2L_{r+s} + \frac{c}{2}\left(r^2 - \frac{1}{4}\right)\delta_{r+s,0}$$

$$[L_m, G_r] = \left(\frac{m}{2} - r\right)G_{m+r}$$

For superstring: $c = \frac{3D}{2}$

#### GSO Projection

**Fermion number operator:**

$$F = (-1)^F \quad \text{with} \quad F = \sum_{r>0} \psi^{-r}\cdot\psi^r$$

**GSO projection:** Keep states with $(-1)^F = \pm(-1)^{\tilde{F}}$

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

$$S = -\frac{T}{2} \int d^2\sigma \left[\sqrt{-h} \, h^{ab}\Pi_a^{\mu}\Pi_{b\mu} + \varepsilon^{ab}\Pi_a^{\mu}\bar{\theta}^A\Gamma_{\mu}\partial_b\theta^A\right]$$

Where $\Pi^{\mu} = \partial X^{\mu} - \bar{\theta}^A\Gamma^{\mu}\partial\theta^A$

**Kappa symmetry:** Gauge symmetry ensuring spacetime SUSY

**Light-cone gauge:** Manifestly supersymmetric

### D-Brane Physics

#### Boundary Conditions

**Neumann:** $\partial_n X^{\mu}|_{\partial\Sigma} = 0$

**Dirichlet:** $\partial_t X^{\mu}|_{\partial\Sigma} = 0$

**T-duality:** N $\leftrightarrow$ D boundary conditions

#### Effective Actions

**DBI action expanded:**

$$S = -T_p\int d^{p+1}\xi \, e^{-\phi}\left[1 + \frac{(2\pi\alpha')^2}{4} F_{\mu\nu}F^{\mu\nu} + O(F^4)\right]$$

**Chern-Simons terms:**

$$S_{CS} = \mu_p \int C \wedge e^{2\pi\alpha'F}$$

#### D-Brane Interactions

**Open string spectrum:** Gauge fields on worldvolume

**Chan-Paton factors:** U(N) gauge theory for N coincident branes

**Tachyon condensation:** Brane annihilation, K-theory classification

### M-Theory and Dualities

#### M-Theory Basics

**11D supergravity low-energy limit:**

$$S = \frac{1}{2\kappa^2} \int d^{11}x \sqrt{-g} \left[R - \frac{1}{2}|F_4|^2\right] + \frac{1}{6} \int C_3 \wedge F_4 \wedge F_4$$

**M2-branes:** Membranes with worldvolume theory

**M5-branes:** 5-branes with self-dual 3-form

#### Web of Dualities

**S-duality:** Type IIB self-dual under $g_s \rightarrow 1/g_s$

**Complete duality web:**

$$\text{M-theory on } S^1 \rightarrow \text{Type IIA}$$

$$\text{M-theory on } T^2 \rightarrow \text{Type IIB}$$

$$\text{M-theory on } S^1/\mathbb{Z}_2 \rightarrow E_8\times E_8 \text{ heterotic}$$

**U-duality:** Combines S and T dualities

### Compactification

#### Calabi-Yau Manifolds

**Definition:** Kähler manifold with SU(n) holonomy

**Properties:**
- Ricci-flat: $R_{ij} = 0$
- Admits covariantly constant spinor
- $c_1 = 0$

**Hodge numbers:** $h^{p,q}$ characterize topology
- $h^{1,1}$: Kähler moduli
- $h^{2,1}$: Complex structure moduli

#### Moduli Stabilization

**Flux compactifications:**

$$W = \int \Omega \wedge (F_3 - \tau H_3)$$

**KKLT scenario:** All moduli stabilized by fluxes and non-perturbative effects

**Large volume scenario:** Exponentially large extra dimensions

### AdS/CFT Correspondence

#### Precise Statement

**Type IIB on AdS₅×S⁵ ↔ N=4 SYM in 4D**

**Dictionary:**

$$\langle O(x)\rangle_{\text{CFT}} = \frac{\delta S_{\text{gravity}}}{\delta\phi_0(x)}\bigg|_{\phi_0\rightarrow O}$$

**Holographic renormalization:** Regulate divergences

#### Generalizations

**AdS₃/CFT₂:** M-theory on AdS₃×S⁸ ↔ ABJM theory

**AdS₂/CFT₁:** Near-horizon of extremal black holes

**Non-conformal:** Dp-branes for p≠3

### Black Holes and Entropy

#### Strominger-Vafa Calculation

**D-brane configuration:** D1-D5-P system

**Microscopic entropy:**

$$S_{\text{micro}} = 2\pi\sqrt{N_1 N_5 n}$$

**Bekenstein-Hawking:**

$$S_{\text{BH}} = \frac{A}{4G} = 2\pi\sqrt{N_1 N_5 n}$$

Perfect agreement!

#### Attractor Mechanism

**Near-horizon geometry:** AdS₂×S²

**Attractor equations:**

$$\frac{\partial V}{\partial z^i}\bigg|_{\text{horizon}} = 0$$

Moduli fixed by charges, independent of asymptotic values

### Topological String Theory

#### A-Model

**Action:** $\int_{\Sigma} \phi^*(\omega) + \{Q, V\}$

**Observables:** Gromov-Witten invariants

**Target space:** Kähler moduli

#### B-Model

**Holomorphic anomaly equation:**

$$\frac{\partial F^{(g)}}{\partial\bar{t}^i} = \frac{1}{2}C^{ijk}_{\bar{i}}\left(D_j D_k F^{(g-1)} + \sum_{h} D_j F^{(h)} D_k F^{(g-h)}\right)$$

**Mirror symmetry:** A-model(X) = B-model(Y)

### Amplitudes and Modern Methods

#### Scattering Equations

**CHY formulation:**

$$A_n = \int d\mu_n \, I_L(\sigma)I_R(\sigma)$$

Where $d\mu_n = \prod_i d\sigma_i \, \delta\left(\sum_j \frac{k_j\cdot P_j}{\sigma_i-\sigma_j}\right)$

#### Ambitwistor Strings

**Action:** $S = \int P_{\mu} \bar{\partial}X^{\mu}$

**Critical dimension:** None!

**Tree amplitudes:** Equivalent to CHY

### Swampland Program

#### Conjectures

**Distance conjecture:** $\Lambda \sim M_P e^{-\alpha d}$

**Weak gravity conjecture:** $m \leq qM_P$

**de Sitter conjecture:** $|\nabla V| \geq \frac{cV}{M_P}$

#### Implications

- Constraints on inflation
- No stable dS vacua?
- Emergence of kinetic terms

### Quantum Information in String Theory

#### Holographic Entanglement Entropy

**Ryu-Takayanagi formula:**

$$S_A = \frac{\text{Area}(\gamma_A)}{4G_N}$$

**Quantum corrections:** $S = \langle\text{Area}/4G\rangle + S_{\text{bulk}}$

#### Complexity

**CV conjecture:** $C = \frac{V}{GL}$

**CA conjecture:** $C = \frac{\text{Action}}{\pi\hbar}$

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

## See Also

### Foundational Topics:
- [Quantum Mechanics](quantum-mechanics.html) - Quantum foundations essential for string theory
- [Quantum Field Theory](quantum-field-theory.html) - The starting point for string interactions
- [Relativity](relativity.html) - General relativity and spacetime geometry

### Related Topics:
- [Statistical Mechanics](statistical-mechanics.html) - Black hole thermodynamics and entropy
- [Condensed Matter Physics](condensed-matter.html) - AdS/CMT correspondence applications
- [Computational Physics](computational-physics.html) - Numerical methods in string theory