---
layout: docs
title: "Relativity: Special Relativity"
permalink: /docs/physics/relativity/special-relativity.html
toc: true
toc_sticky: true
hide_title: true
---

[Relativity](./)

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Special Relativity</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Space and Time Unite at High Speeds</p>
</div>

## Special Relativity

<div class="section-intro">
  <p>Special relativity, published in 1905, deals with objects moving at constant velocities and introduces revolutionary concepts about space and time.</p>
</div>

<div class="principle-card">
  <h4>The puzzle that forced the revolution</h4>
  <p>By the late 1800s, Maxwell's equations predicted a definite speed of light, $c$. But a speed relative to <em>what</em>? Every other wave — sound, water ripples — travels relative to a medium, and adding velocities was obvious: throw a ball forward on a moving train and the ground sees it go faster. Yet the Michelson–Morley experiment found light <em>always</em> travels at $c$, no matter how fast you chase it. Einstein's leap was to take this literally: if everyone measures the same light speed, then the things we assumed were absolute — the rate clocks tick and the length of rulers — must instead bend. Space and time give way so that $c$ can stay fixed. Everything that follows (time dilation, length contraction, $E=mc^2$) is the logical price of that one stubborn fact.</p>
</div>

<div class="postulates-section">
  <h3><i class="fas fa-gavel"></i> Postulates of Special Relativity</h3>
  
  <div class="postulate-cards">
    <div class="postulate-card">
      <div class="postulate-number">1</div>
      <h4>Principle of Relativity</h4>
      <p>The laws of physics are the same in all inertial reference frames</p>
      <div class="visual-demo">
        <svg viewBox="0 0 450 200" style="max-width: 500px; width: 100%;">
          <!-- Define arrow marker -->
          <defs>
            <marker id="arrow-rel" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#2c3e50" />
            </marker>
          </defs>

          <!-- Frame A - Stationary observer -->
          <rect x="30" y="40" width="160" height="100" fill="#e3f2fd" stroke="#1976d2" stroke-width="3" rx="5" />
          <text x="110" y="160" text-anchor="middle" font-size="16" font-weight="bold" fill="#1976d2">Frame A (Stationary)</text>
          <!-- Observer in Frame A -->
          <circle cx="110" cy="85" r="12" fill="#1976d2" />
          <text x="110" y="90" text-anchor="middle" font-size="11" fill="white">A</text>
          <!-- Physics symbol in Frame A -->
          <text x="70" y="115" font-size="14" fill="#333">F = ma</text>

          <!-- Frame B - Moving observer -->
          <rect x="260" y="40" width="160" height="100" fill="#ffebee" stroke="#c62828" stroke-width="3" rx="5" />
          <text x="340" y="160" text-anchor="middle" font-size="16" font-weight="bold" fill="#c62828">Frame B (Moving)</text>
          <!-- Observer in Frame B -->
          <circle cx="340" cy="85" r="12" fill="#c62828" />
          <text x="340" y="90" text-anchor="middle" font-size="11" fill="white">B</text>
          <!-- Physics symbol in Frame B -->
          <text x="300" y="115" font-size="14" fill="#333">F = ma</text>

          <!-- Velocity arrow between frames -->
          <line x1="195" y1="90" x2="250" y2="90" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrow-rel)" />
          <text x="222" y="78" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">v</text>

          <!-- Caption -->
          <text x="225" y="185" text-anchor="middle" font-size="14" fill="#555" font-style="italic">Same laws of physics in both frames</text>
        </svg>
      </div>
    </div>
    
    <div class="postulate-card">
      <div class="postulate-number">2</div>
      <h4>Constancy of Light Speed</h4>
      <p>The speed of light in vacuum is the same for all observers, regardless of motion</p>
      <div class="visual-demo">
        <svg viewBox="0 0 480 220" style="max-width: 500px; width: 100%;">
          <!-- Define arrow markers -->
          <defs>
            <marker id="arrow-light" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#e74c3c" />
            </marker>
            <!-- Light glow effect -->
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feFlood flood-color="#f39c12" flood-opacity="0.5" />
              <feComposite in2="blur" operator="in" />
              <feMerge>
                <feMergeNode />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <!-- Title -->
          <text x="240" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Speed of Light is Constant for All Observers</text>

          <!-- Light source -->
          <circle cx="50" cy="100" r="15" fill="#f39c12" filter="url(#glow)" />
          <text x="50" y="105" text-anchor="middle" font-size="12" fill="#333">Light</text>

          <!-- Light ray -->
          <line x1="70" y1="100" x2="430" y2="100" stroke="#f39c12" stroke-width="4" stroke-dasharray="10,5" />

          <!-- Speed label -->
          <rect x="160" y="55" width="180" height="30" fill="#fff3e0" stroke="#f39c12" stroke-width="2" rx="5" />
          <text x="250" y="76" text-anchor="middle" font-size="15" font-weight="bold" fill="#e65100">c = 299,792,458 m/s</text>

          <!-- Observer 1 - Stationary -->
          <circle cx="140" cy="160" r="15" fill="#1976d2" />
          <text x="140" y="165" text-anchor="middle" font-size="12" fill="white">1</text>
          <text x="140" y="195" text-anchor="middle" font-size="14" font-weight="bold" fill="#1976d2">Observer 1</text>
          <text x="140" y="210" text-anchor="middle" font-size="12" fill="#555">(stationary)</text>
          <!-- Speech bubble -->
          <rect x="90" y="115" width="100" height="25" fill="#e3f2fd" stroke="#1976d2" stroke-width="1" rx="3" />
          <text x="140" y="132" text-anchor="middle" font-size="12" fill="#1976d2">Measures: c</text>

          <!-- Observer 2 - Moving toward light -->
          <circle cx="340" cy="160" r="15" fill="#c62828" />
          <text x="340" y="165" text-anchor="middle" font-size="12" fill="white">2</text>
          <text x="340" y="195" text-anchor="middle" font-size="14" font-weight="bold" fill="#c62828">Observer 2</text>
          <text x="340" y="210" text-anchor="middle" font-size="12" fill="#555">(moving at 0.5c)</text>
          <!-- Motion arrow -->
          <line x1="375" y1="160" x2="415" y2="160" stroke="#c62828" stroke-width="3" marker-end="url(#arrow-light)" />
          <text x="395" y="150" text-anchor="middle" font-size="14" font-weight="bold" fill="#c62828">v</text>
          <!-- Speech bubble -->
          <rect x="290" y="115" width="100" height="25" fill="#ffebee" stroke="#c62828" stroke-width="1" rx="3" />
          <text x="340" y="132" text-anchor="middle" font-size="12" fill="#c62828">Measures: c</text>

          <!-- Connecting lines to light ray -->
          <line x1="140" y1="145" x2="140" y2="105" stroke="#1976d2" stroke-width="1" stroke-dasharray="3,3" />
          <line x1="340" y1="145" x2="340" y2="105" stroke="#c62828" stroke-width="1" stroke-dasharray="3,3" />
        </svg>
      </div>
    </div>
  </div>
</div>

### Spacetime and the Lorentz Transformation

<div class="spacetime-section">
  <h4><i class="fas fa-cube"></i> Spacetime Interval</h4>
  <p>The spacetime interval between two events is invariant:</p>

  <p><em>Convention note:</em> two metric-signature conventions are in common use. This section writes the interval with the <strong>(+,−,−,−)</strong> ("mostly-minus") convention in its primary algebraic form, then gives the differential form in the <strong>(−,+,+,+)</strong> ("mostly-plus") convention to match the Minkowski metric $\eta_{\mu\nu}$ below. The two differ only by an overall sign and describe identical physics.</p>

  <div class="equation-showcase">
    <div class="equation-box primary">
      $$(\Delta s)^2 = c^2(\Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$$
    </div>
    
    <p>In differential form (using the (−,+,+,+) convention):</p>
    <div class="equation-box">
      $$ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2 = \eta_{\mu\nu}\, dx^\mu dx^\nu$$
    </div>
    
    <div class="metric-display">
      <p>Where $\eta_{\mu\nu}$ is the Minkowski metric:</p>
      <div class="matrix-visual">
        $$\eta_{\mu\nu} = \begin{pmatrix}
        -1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
        \end{pmatrix}$$
      </div>
    </div>
  </div>
  
  <div class="spacetime-diagram">
    <svg viewBox="0 0 500 380" style="max-width: 500px; width: 100%;">
      <!-- Define arrow markers -->
      <defs>
        <marker id="arrow-st" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,8 L12,4 z" fill="#2c3e50" />
        </marker>
        <marker id="arrow-st-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#e65100" />
        </marker>
      </defs>

      <!-- Title -->
      <text x="250" y="25" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Spacetime Diagram</text>

      <!-- Background grid -->
      <g stroke="#e0e0e0" stroke-width="1">
        <line x1="100" y1="60" x2="100" y2="340" />
        <line x1="150" y1="60" x2="150" y2="340" />
        <line x1="200" y1="60" x2="200" y2="340" />
        <line x1="300" y1="60" x2="300" y2="340" />
        <line x1="350" y1="60" x2="350" y2="340" />
        <line x1="400" y1="60" x2="400" y2="340" />
        <line x1="50" y1="100" x2="450" y2="100" />
        <line x1="50" y1="150" x2="450" y2="150" />
        <line x1="50" y1="250" x2="450" y2="250" />
        <line x1="50" y1="300" x2="450" y2="300" />
      </g>

      <!-- Spacetime axes -->
      <line x1="250" y1="340" x2="250" y2="50" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrow-st)" />
      <line x1="50" y1="200" x2="450" y2="200" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrow-st)" />
      <text x="265" y="55" text-anchor="start" font-size="18" font-weight="bold" fill="#2c3e50">ct (time)</text>
      <text x="455" y="205" text-anchor="start" font-size="18" font-weight="bold" fill="#2c3e50">x (space)</text>

      <!-- Light cone lines -->
      <line x1="250" y1="200" x2="100" y2="50" stroke="#e65100" stroke-width="3" stroke-dasharray="8,4" />
      <line x1="250" y1="200" x2="400" y2="50" stroke="#e65100" stroke-width="3" stroke-dasharray="8,4" />
      <line x1="250" y1="200" x2="100" y2="350" stroke="#e65100" stroke-width="2" stroke-dasharray="8,4" opacity="0.5" />
      <line x1="250" y1="200" x2="400" y2="350" stroke="#e65100" stroke-width="2" stroke-dasharray="8,4" opacity="0.5" />

      <!-- Light cone labels -->
      <text x="115" y="90" font-size="14" font-weight="bold" fill="#e65100">Light (45 degrees)</text>
      <text x="355" y="90" font-size="14" font-weight="bold" fill="#e65100">v = c</text>

      <!-- Sample worldline (massive particle) -->
      <path d="M 180 340 Q 210 270, 230 200 Q 245 140, 260 70" stroke="#1976d2" stroke-width="4" fill="none" />
      <circle cx="180" cy="340" r="6" fill="#1976d2" />
      <circle cx="230" cy="200" r="6" fill="#1976d2" />
      <circle cx="260" cy="70" r="6" fill="#1976d2" />
      <text x="145" y="355" font-size="14" font-weight="bold" fill="#1976d2">Worldline</text>
      <text x="145" y="370" font-size="12" fill="#1976d2">(massive particle)</text>

      <!-- Event at origin -->
      <circle cx="250" cy="200" r="8" fill="#c62828" />
      <text x="265" y="215" font-size="15" font-weight="bold" fill="#c62828">Event P</text>
      <text x="265" y="232" font-size="12" fill="#555">(here, now)</text>

      <!-- Future region label -->
      <text x="250" y="120" text-anchor="middle" font-size="14" fill="#388e3c" font-weight="bold">FUTURE</text>

      <!-- Past region label -->
      <text x="250" y="290" text-anchor="middle" font-size="14" fill="#7b1fa2" font-weight="bold">PAST</text>

      <!-- Spacelike region labels -->
      <text x="100" y="205" text-anchor="middle" font-size="12" fill="#555">Elsewhere</text>
      <text x="400" y="205" text-anchor="middle" font-size="12" fill="#555">Elsewhere</text>

      <!-- Axis tick marks and labels -->
      <line x1="300" y1="195" x2="300" y2="205" stroke="#2c3e50" stroke-width="2" />
      <text x="300" y="220" text-anchor="middle" font-size="12" fill="#333">x</text>
      <line x1="350" y1="195" x2="350" y2="205" stroke="#2c3e50" stroke-width="2" />
      <text x="350" y="220" text-anchor="middle" font-size="12" fill="#333">2x</text>
      <line x1="245" y1="150" x2="255" y2="150" stroke="#2c3e50" stroke-width="2" />
      <text x="235" y="155" text-anchor="end" font-size="12" fill="#333">ct</text>
      <line x1="245" y1="100" x2="255" y2="100" stroke="#2c3e50" stroke-width="2" />
      <text x="235" y="105" text-anchor="end" font-size="12" fill="#333">2ct</text>
    </svg>
  </div>
  
  <div class="light-cone-diagram">
    <h4><i class="fas fa-hourglass-half"></i> Light Cone Structure</h4>
    <svg viewBox="0 0 550 480" style="max-width: 500px; width: 100%;">
      <!-- Title -->
      <text x="275" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Light Cone and Causal Structure</text>

      <!-- Define gradient for cones -->
      <defs>
        <linearGradient id="futureCone" x1="0%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stop-color="#e65100" stop-opacity="0.4" />
          <stop offset="100%" stop-color="#ff9800" stop-opacity="0.1" />
        </linearGradient>
        <linearGradient id="pastCone" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#e65100" stop-opacity="0.3" />
          <stop offset="100%" stop-color="#ff9800" stop-opacity="0.05" />
        </linearGradient>
        <marker id="arrow-lc" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#2c3e50" />
        </marker>
      </defs>

      <!-- 3D coordinate axes -->
      <line x1="275" y1="240" x2="450" y2="400" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="4,4" />
      <text x="460" y="410" font-size="14" fill="#666" font-weight="bold">x</text>

      <line x1="275" y1="240" x2="100" y2="400" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="4,4" />
      <text x="85" y="410" font-size="14" fill="#666" font-weight="bold">y</text>

      <!-- Time axis -->
      <line x1="275" y1="450" x2="275" y2="50" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrow-lc)" />
      <text x="295" y="55" font-size="18" font-weight="bold" fill="#2c3e50">ct (time)</text>

      <!-- Future light cone -->
      <path d="M 275 240 L 140 100 Q 275 50, 410 100 Z" fill="url(#futureCone)" stroke="#e65100" stroke-width="3" />
      <text x="275" y="85" text-anchor="middle" font-size="16" font-weight="bold" fill="#e65100">Future Light Cone</text>

      <!-- Past light cone -->
      <path d="M 275 240 L 140 380 Q 275 430, 410 380 Z" fill="url(#pastCone)" stroke="#e65100" stroke-width="2" stroke-dasharray="5,3" />
      <text x="275" y="410" text-anchor="middle" font-size="16" font-weight="bold" fill="#bf360c">Past Light Cone</text>

      <!-- Event at origin -->
      <circle cx="275" cy="240" r="10" fill="#c62828" stroke="#b71c1c" stroke-width="2" />
      <text x="295" y="235" font-size="16" font-weight="bold" fill="#c62828">Event P</text>
      <text x="295" y="255" font-size="13" fill="#555">(Here and Now)</text>

      <!-- Timelike future region -->
      <path d="M 240 180 L 310 180 L 295 120 L 255 120 Z" fill="#1976d2" opacity="0.25" />
      <text x="370" y="135" font-size="15" font-weight="bold" fill="#1976d2">Timelike Future</text>
      <text x="370" y="155" font-size="13" fill="#1565c0">(Causally connected)</text>
      <text x="370" y="173" font-size="13" fill="#1565c0">(v &lt; c reachable)</text>

      <!-- Timelike past region -->
      <path d="M 240 300 L 310 300 L 295 360 L 255 360 Z" fill="#7b1fa2" opacity="0.2" />
      <text x="370" y="340" font-size="15" font-weight="bold" fill="#7b1fa2">Timelike Past</text>
      <text x="370" y="360" font-size="13" fill="#6a1b9a">(Could have caused P)</text>

      <!-- Spacelike region -->
      <ellipse cx="275" cy="240" rx="110" ry="30" fill="#388e3c" opacity="0.2" />
      <text x="60" y="235" font-size="15" font-weight="bold" fill="#388e3c">Spacelike</text>
      <text x="60" y="255" font-size="13" fill="#2e7d32">(No causal</text>
      <text x="60" y="273" font-size="13" fill="#2e7d32">connection)</text>

      <!-- Sample worldlines -->
      <!-- Massive particle worldline -->
      <path d="M 275 240 Q 290 180, 300 120" stroke="#7b1fa2" stroke-width="4" fill="none" />
      <circle cx="300" cy="120" r="5" fill="#7b1fa2" />
      <text x="315" y="105" font-size="14" font-weight="bold" fill="#7b1fa2">Massive particle</text>
      <text x="315" y="120" font-size="12" fill="#7b1fa2">(v &lt; c)</text>

      <!-- Light ray -->
      <line x1="275" y1="240" x2="355" y2="160" stroke="#e65100" stroke-width="4" stroke-dasharray="6,3" />
      <circle cx="355" cy="160" r="4" fill="#e65100" />
      <text x="365" y="175" font-size="14" font-weight="bold" fill="#e65100">Light ray</text>
      <text x="365" y="190" font-size="12" fill="#e65100">(v = c)</text>

      <!-- Legend box -->
      <rect x="20" y="430" width="510" height="40" fill="#fafafa" stroke="#e0e0e0" stroke-width="1" rx="5" />
      <text x="275" y="458" text-anchor="middle" font-size="14" fill="#333">
        <tspan font-weight="bold" fill="#1976d2">ds^2 &gt; 0</tspan> (timelike)
        <tspan dx="20" font-weight="bold" fill="#e65100">ds^2 = 0</tspan> (null/lightlike)
        <tspan dx="20" font-weight="bold" fill="#388e3c">ds^2 &lt; 0</tspan> (spacelike)
      </text>
    </svg>
  </div>
</div>

#### Derivation of Lorentz Transformations
Starting from the invariance of the spacetime interval and the principle of relativity:

For two reference frames S and S', where S' moves with velocity v along the x-axis:

$$c^2t'^2 - x'^2 = c^2t^2 - x^2$$

Assuming linear transformation:

$$x' = Ax + Bt$$
$$t' = Cx + Dt$$

From the origin of S' (x' = 0) moving at x = vt:

$$0 = Avt + Bt \rightarrow B = -Av$$

From the invariance of light speed (x = ct implies x' = ct'):

$$ct' = Act + Bt = Act - Avt = A(c - v)t$$
$$x' = Act + Bt = Act - Avt = A(c - v)t$$

Therefore: A = γ = 1/√(1 - v²/c²)

Complete Lorentz transformations:

$$x' = \gamma(x - vt)$$
$$y' = y$$
$$z' = z$$
$$t' = \gamma(t - vx/c^2)$$

Inverse transformations:

$$x = \gamma(x' + vt')$$
$$y = y'$$
$$z = z'$$
$$t = \gamma(t' + vx'/c^2)$$

Matrix form:

$$\begin{pmatrix}
ct' \\
x' \\
y' \\
z'
\end{pmatrix} = \begin{pmatrix}
\gamma & -\beta\gamma & 0 & 0 \\
-\beta\gamma & \gamma & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix} \begin{pmatrix}
ct \\
x \\
y \\
z
\end{pmatrix}$$

Where β = v/c.

### Time Dilation

<div class="time-dilation-section">
  <div class="concept-header">
    <i class="fas fa-clock"></i>
    <h4>Moving clocks run slower relative to stationary observers</h4>
  </div>
  
  <div class="equation-display">
    <div class="equation-box highlighted">
      $$\Delta t = \gamma \Delta t_0$$
    </div>
    <p>Where $\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}$ is the Lorentz factor</p>
  </div>
  
  <div class="variable-definitions">
    <div class="var-item">
      <span class="var-symbol">Δt₀</span>
      <span class="var-desc">Proper time (time measured in the rest frame)</span>
    </div>
    <div class="var-item">
      <span class="var-symbol">Δt</span>
      <span class="var-desc">Dilated time (time measured in the moving frame)</span>
    </div>
  </div>
  
  <div class="interactive-demo">
    <h5>Time Dilation Calculator</h5>
    <div class="demo-controls">
      <label>Velocity (as fraction of c): <span id="velocity-value">0.5</span></label>
      <input type="range" id="velocity-slider" min="0" max="0.99" step="0.01" value="0.5" />
      <div class="results">
        <p>Lorentz factor γ = <span id="gamma-value">1.155</span></p>
        <p>1 hour proper time = <span id="dilated-time">1.155</span> hours observed</p>
      </div>
    </div>
  </div>
  
  <div class="real-world-example">
    <i class="fas fa-satellite"></i>
    <h5>GPS Example</h5>
    <p>GPS satellites must account for <em>both</em> special- and general-relativistic effects, which act in opposite directions. Their orbital velocity (~14,000 km/h) causes the special-relativistic slowing of about <strong>−7 µs/day</strong>. But the satellites also sit higher in Earth's gravitational well, where clocks run faster — a general-relativistic gain of about <strong>+45 µs/day</strong>. The two do not cancel: the gravitational term dominates, so the <strong>net effect makes GPS clocks run fast by roughly +38 µs/day</strong>. Left uncorrected, this would introduce navigation errors of about 10 km per day.</p>
    <div class="calculation-breakdown">
      <p>v ≈ 3,900 m/s</p>
      <p>γ - 1 ≈ 8.4 × 10⁻¹¹</p>
      <p>Special-relativistic (velocity) effect: ~−7.2 μs/day</p>
      <p>General-relativistic (gravitational) effect: ~+45.9 μs/day</p>
      <p>Net effect: ~+38 μs/day (clocks run fast)</p>
    </div>
  </div>
</div>

<script>
  // Time dilation interactive
  const slider = document.getElementById('velocity-slider');
  const velocityValue = document.getElementById('velocity-value');
  const gammaValue = document.getElementById('gamma-value');
  const dilatedTime = document.getElementById('dilated-time');
  
  slider?.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    const gamma = 1 / Math.sqrt(1 - v*v);
    velocityValue.textContent = v.toFixed(2);
    gammaValue.textContent = gamma.toFixed(3);
    dilatedTime.textContent = gamma.toFixed(3);
  });
</script>

### Length Contraction

<div class="length-contraction-section">
  <div class="concept-header">
    <i class="fas fa-compress-alt"></i>
    <h4>Objects appear shorter in the direction of motion</h4>
  </div>
  
  <div class="equation-display">
    <div class="equation-box highlighted">
      $$L = \frac{L_0}{\gamma}$$
    </div>
  </div>
  
  <div class="variable-definitions">
    <div class="var-item">
      <span class="var-symbol">L₀</span>
      <span class="var-desc">Proper length (length in the rest frame)</span>
    </div>
    <div class="var-item">
      <span class="var-symbol">L</span>
      <span class="var-desc">Contracted length (length in the moving frame)</span>
    </div>
  </div>
  
  <div class="visual-demonstration">
    <svg viewBox="0 0 520 280" style="max-width: 500px; width: 100%;">
      <!-- Title -->
      <text x="260" y="25" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Length Contraction Demonstration</text>

      <!-- Define arrow marker -->
      <defs>
        <marker id="arrow-lc2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#2c3e50" />
        </marker>
        <pattern id="ruler-pattern" x="0" y="0" width="20" height="10" patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="10" stroke="#555" stroke-width="1" />
        </pattern>
      </defs>

      <!-- Rest Frame Section -->
      <rect x="30" y="45" width="460" height="95" fill="#e3f2fd" stroke="#1976d2" stroke-width="2" rx="5" />
      <text x="260" y="65" text-anchor="middle" font-size="16" font-weight="bold" fill="#1565c0">Rest Frame (Object at rest)</text>

      <!-- Object at rest (full length) -->
      <rect x="80" y="85" width="300" height="40" fill="#1976d2" stroke="#0d47a1" stroke-width="3" rx="5" />
      <text x="230" y="112" text-anchor="middle" font-size="18" font-weight="bold" fill="white">L&#x2080; = Proper Length</text>

      <!-- Ruler for rest frame -->
      <line x1="80" y1="135" x2="380" y2="135" stroke="#333" stroke-width="2" />
      <line x1="80" y1="130" x2="80" y2="140" stroke="#333" stroke-width="2" />
      <line x1="380" y1="130" x2="380" y2="140" stroke="#333" stroke-width="2" />
      <text x="80" y="150" text-anchor="middle" font-size="12" fill="#333">0</text>
      <text x="380" y="150" text-anchor="middle" font-size="12" fill="#333">L&#x2080;</text>

      <!-- Moving Frame Section -->
      <rect x="30" y="160" width="460" height="110" fill="#ffebee" stroke="#c62828" stroke-width="2" rx="5" />
      <text x="260" y="180" text-anchor="middle" font-size="16" font-weight="bold" fill="#b71c1c">Moving Frame (v = 0.8c, gamma = 1.67)</text>

      <!-- Object moving (contracted) -->
      <rect x="140" y="200" width="180" height="40" fill="#c62828" stroke="#b71c1c" stroke-width="3" rx="5" />
      <text x="230" y="227" text-anchor="middle" font-size="16" font-weight="bold" fill="white">L = L&#x2080;/gamma</text>

      <!-- Ruler for moving frame -->
      <line x1="140" y1="250" x2="320" y2="250" stroke="#333" stroke-width="2" />
      <line x1="140" y1="245" x2="140" y2="255" stroke="#333" stroke-width="2" />
      <line x1="320" y1="245" x2="320" y2="255" stroke="#333" stroke-width="2" />
      <text x="140" y="265" text-anchor="middle" font-size="12" fill="#333">0</text>
      <text x="320" y="265" text-anchor="middle" font-size="12" fill="#333">0.6L&#x2080;</text>

      <!-- Motion arrow -->
      <line x1="340" y1="220" x2="410" y2="220" stroke="#2c3e50" stroke-width="4" marker-end="url(#arrow-lc2)" />
      <text x="375" y="210" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">v = 0.8c</text>

      <!-- Contraction percentage -->
      <rect x="420" y="195" width="60" height="50" fill="#fff3e0" stroke="#e65100" stroke-width="2" rx="5" />
      <text x="450" y="218" text-anchor="middle" font-size="14" font-weight="bold" fill="#e65100">60%</text>
      <text x="450" y="235" text-anchor="middle" font-size="11" fill="#e65100">original</text>

      <!-- Comparison arrows showing contraction -->
      <line x1="80" y1="127" x2="80" y2="195" stroke="#9e9e9e" stroke-width="1" stroke-dasharray="4,2" />
      <line x1="380" y1="127" x2="380" y2="195" stroke="#9e9e9e" stroke-width="1" stroke-dasharray="4,2" />
      <line x1="140" y1="195" x2="140" y2="127" stroke="#9e9e9e" stroke-width="1" stroke-dasharray="4,2" />
      <line x1="320" y1="195" x2="320" y2="127" stroke="#9e9e9e" stroke-width="1" stroke-dasharray="4,2" />
    </svg>
  </div>
</div>

### Relativistic Velocity Addition

Velocities don't simply add in special relativity:

$$u = \frac{v + w}{1 + vw/c^2}$$

This ensures that no velocity exceeds the speed of light.

<div class="example-card">
  <h4>Worked Example: chasing a light beam</h4>
  <p>Suppose a spaceship moves at $v = 0.9c$ relative to Earth and fires a probe forward at $w = 0.9c$ relative to the ship. Classically you would expect $1.8c$ — faster than light. Relativity gives instead:</p>
  $$u = \frac{0.9c + 0.9c}{1 + (0.9)(0.9)} = \frac{1.8c}{1.81} \approx 0.994c$$
  <p>The probe still travels below $c$. And if the ship instead fired a <em>light</em> beam ($w = c$), the formula returns exactly $c$ no matter the ship's speed — the second postulate, falling out of the algebra. Speeds combine so that $c$ is an unreachable ceiling, not a wall you can edge past by stacking velocities.</p>
</div>

<div class="tip-card">
  <h4>What time dilation and length contraction actually mean</h4>
  <p>These are not optical illusions or measurement errors — the effects are <em>real and symmetric</em>. Each observer genuinely sees the <strong>other's</strong> clock running slow and ruler shrunk. There is no contradiction because "now" itself is frame-dependent (the relativity of simultaneity): two observers don't agree on which distant events happen at the same time, so they slice spacetime differently. The single quantity everyone <em>does</em> agree on is the invariant interval $ds^2$ — distances and durations are just its shadows cast at different angles.</p>
</div>

### Mass-Energy Equivalence

Einstein's most famous equation:

$$E = mc^2$$

Total energy of a particle:

$$E^2 = (pc)^2 + (mc^2)^2$$

Where p is the relativistic momentum:

$$p = \gamma mv$$

### Relativistic Dynamics

#### Relativistic Momentum

$$p = \gamma mv$$

#### Relativistic Force

$$F = \frac{dp}{dt} = \frac{d(\gamma mv)}{dt}$$

#### Relativistic Kinetic Energy

$$KE = (\gamma - 1)mc^2$$

<div class="tip-card">
  <h4>Reading the energy-momentum relation</h4>
  <p>The relation $E^2 = (pc)^2 + (mc^2)^2$ is one of the most useful equations in all of physics, and it reads like a Pythagorean theorem for energy. Two limits make it intuitive. For a slow particle ($p \to 0$) it reduces to $E = mc^2$ plus, on Taylor-expanding, the Newtonian $\tfrac{1}{2}mv^2$ — classical kinetic energy is just the first correction to the rest energy. For a <em>massless</em> particle like the photon ($m = 0$) it collapses to $E = pc$, which is why light carries momentum despite having no mass. The rest mass $m$ is the invariant "length" of the energy-momentum four-vector: every observer disagrees on $E$ and $p$ separately, but all agree on $m$.</p>
</div>

<div class="example-card">
  <h4>Worked Example: how much energy is locked in one gram?</h4>
  <p>Mass-energy equivalence says even a stationary object stores energy $E = mc^2$. For $m = 1\ \text{gram} = 10^{-3}\ \text{kg}$:</p>
  $$E = (10^{-3}\ \text{kg})(3.0\times10^{8}\ \text{m/s})^2 = 9\times10^{13}\ \text{J}.$$
  <p>That is roughly the energy released by 20 kilotons of TNT — comparable to the Hiroshima bomb — from a single gram of matter. The reason chemistry never reveals this is that chemical bonds release a billionth of the rest energy; only nuclear and particle processes tap a meaningful fraction. The mass of a charged battery, a compressed spring, or a hot object is genuinely (if immeasurably) larger than its de-energized state.</p>
</div>

### Four-Vectors and Tensor Notation

In special relativity, we use four-vectors to unify space and time:

**Position four-vector:**

$$x^\mu = (ct, x, y, z)$$

**Four-momentum:**

$$p^\mu = (E/c, p_x, p_y, p_z)$$

**Four-velocity:**

$$u^\mu = \gamma(c, v_x, v_y, v_z)$$

**Invariants:**
- Spacetime interval: $s^2 = -c^2t^2 + x^2 + y^2 + z^2$
- Rest mass: $m^2c^2 = -p^\mu p_\mu / c^2$

<div class="tip-card">
  <h4>Tensor notation conventions</h4>
  <ul>
    <li><strong>Contravariant:</strong> Upper indices ($x^\mu$)</li>
    <li><strong>Covariant:</strong> Lower indices ($x_\mu$)</li>
    <li><strong>Einstein summation:</strong> Repeated indices are summed</li>
  </ul>
  <p>The fully developed tensor machinery — covariant derivatives, the Lorentz algebra, spinors — is collected in the <a href="advanced.html">Graduate Formalism &amp; Frontiers</a> page.</p>
</div>

---

## Continue

<div class="see-also-card">
  <h4>Previous / Next</h4>
  <ul>
    <li><strong>Up:</strong> <a href="./">Relativity</a> — overview and navigation hub.</li>
    <li><strong>Next:</strong> <a href="general-relativity.html">General Relativity</a> — gravity as the curvature of spacetime.</li>
  </ul>
</div>

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="general-relativity.html">General Relativity</a> — the equivalence principle and the Einstein field equations.</li>
    <li><a href="advanced.html">Graduate Formalism &amp; Frontiers</a> — the Lorentz group, spinors, and the full tensor formalism.</li>
    <li><a href="../classical-mechanics/">Classical Mechanics</a> — the low-speed limit special relativity reduces to.</li>
    <li><a href="../quantum-field-theory.html">Quantum Field Theory</a> — special relativity combined with quantum mechanics.</li>
  </ul>
</div>
