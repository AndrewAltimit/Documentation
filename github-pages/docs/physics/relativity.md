---
layout: docs
title: Relativity
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section">
  <div class="hero-content">
    
    <p class="hero-subtitle">The Unity of Space, Time, and Gravity</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Relativity encompasses two interrelated theories by Albert Einstein: special relativity and general relativity. These theories revolutionized our understanding of space, time, gravity, and the universe. They describe how measurements of various quantities are relative to the velocities of observers and how massive objects warp spacetime.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-rocket"></i>
      <h4>Special Relativity</h4>
      <p>Space and time unite at high speeds</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-globe"></i>
      <h4>General Relativity</h4>
      <p>Gravity as curved spacetime</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-atom"></i>
      <h4>E = mc²</h4>
      <p>Mass and energy are equivalent</p>
    </div>
  </div>
</div>

## Special Relativity

<div class="section-intro">
  <p>Special relativity, published in 1905, deals with objects moving at constant velocities and introduces revolutionary concepts about space and time.</p>
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
          <text x="70" cy="85" font-size="14" fill="#333">F = ma</text>

          <!-- Frame B - Moving observer -->
          <rect x="260" y="40" width="160" height="100" fill="#ffebee" stroke="#c62828" stroke-width="3" rx="5" />
          <text x="340" y="160" text-anchor="middle" font-size="16" font-weight="bold" fill="#c62828">Frame B (Moving)</text>
          <!-- Observer in Frame B -->
          <circle cx="340" cy="85" r="12" fill="#c62828" />
          <text x="340" y="90" text-anchor="middle" font-size="11" fill="white">B</text>
          <!-- Physics symbol in Frame B -->
          <text x="300" cy="85" font-size="14" fill="#333">F = ma</text>

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
  
  <div class="equation-showcase">
    <div class="equation-box primary">
      $$(\Delta s)² = c²(\Delta t)² - (\Delta x)² - (\Delta y)² - (\Delta z)²$$
    </div>
    
    <p>In differential form:</p>
    <div class="equation-box">
      $$ds² = -c²dt² + dx² + dy² + dz² = \eta_{\mu\nu} dx^\mu dx^\nu$$
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
    <p>GPS satellites must account for time dilation due to their orbital velocity (~14,000 km/h), causing their clocks to run slower by about 7 microseconds per day.</p>
    <div class="calculation-breakdown">
      <p>v ≈ 3,900 m/s</p>
      <p>γ - 1 ≈ 8.4 × 10⁻¹¹</p>
      <p>Daily effect: ~7.2 μs slower</p>
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

### Four-Vectors and Tensor Notation

In special relativity, we use four-vectors to unify space and time:

**Position four-vector:**

$$x^\mu = (ct, x, y, z)$$

**Four-momentum:**

$$p^\mu = (E/c, p_x, p_y, p_z)$$

**Four-velocity:**

$$u^\mu = \gamma(c, v_x, v_y, v_z)$$

**Invariants:**
- Spacetime interval: s² = -c²t² + x² + y² + z²
- Rest mass: m²c² = -(p^μp_μ)/c²

## General Relativity

<div class="section-intro gr-intro">
  <p>General relativity, published in 1915, extends special relativity to include gravity and accelerated reference frames. It describes gravity not as a force, but as the curvature of spacetime caused by mass and energy.</p>
</div>

<div class="core-principles">
  <h3><i class="fas fa-star"></i> Core Principles</h3>
  
  <div class="principle-cards">
    <div class="principle-card equivalence">
      <div class="principle-icon"><i class="fas fa-equals"></i></div>
      <h4>Equivalence Principle</h4>
      <p>The effects of gravity are locally indistinguishable from acceleration</p>
      <div class="principle-visual">
        <svg viewBox="0 0 420 240" style="max-width: 500px; width: 100%;">
          <!-- Define arrow markers -->
          <defs>
            <marker id="arrow-eq" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#c62828" />
            </marker>
            <marker id="arrow-eq-green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#2e7d32" />
            </marker>
          </defs>

          <!-- Title -->
          <text x="210" y="20" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Equivalence Principle</text>

          <!-- Scenario A: Accelerating in space -->
          <rect x="20" y="40" width="160" height="150" fill="#e3f2fd" stroke="#1976d2" stroke-width="3" rx="8" />
          <text x="100" y="60" text-anchor="middle" font-size="14" font-weight="bold" fill="#1565c0">In Space</text>
          <text x="100" y="78" text-anchor="middle" font-size="12" fill="#1976d2">(Accelerating rocket)</text>

          <!-- Elevator box in space -->
          <rect x="50" y="90" width="100" height="80" fill="#bbdefb" stroke="#1976d2" stroke-width="2" rx="3" />

          <!-- Person in space elevator -->
          <circle cx="100" cy="120" r="12" fill="#1976d2" />
          <line x1="100" y1="132" x2="100" y2="155" stroke="#1976d2" stroke-width="3" />
          <line x1="100" y1="140" x2="85" y2="150" stroke="#1976d2" stroke-width="3" />
          <line x1="100" y1="140" x2="115" y2="150" stroke="#1976d2" stroke-width="3" />

          <!-- Acceleration arrow (upward) -->
          <line x1="100" y1="40" x2="100" y2="15" stroke="#2e7d32" stroke-width="4" marker-end="url(#arrow-eq-green)" />
          <text x="125" y="25" font-size="14" font-weight="bold" fill="#2e7d32">a = g</text>

          <!-- Felt force (downward on person) -->
          <line x1="100" y1="158" x2="100" y2="185" stroke="#c62828" stroke-width="3" marker-end="url(#arrow-eq)" />
          <text x="125" y="175" font-size="12" fill="#c62828">Feels weight</text>

          <!-- Stars background indicator -->
          <text x="35" y="105" font-size="16" fill="#555">*</text>
          <text x="140" y="120" font-size="14" fill="#555">*</text>
          <text x="55" y="165" font-size="12" fill="#555">*</text>

          <!-- Equals sign -->
          <text x="200" y="130" text-anchor="middle" font-size="36" font-weight="bold" fill="#333">=</text>

          <!-- Scenario B: On Earth -->
          <rect x="240" y="40" width="160" height="150" fill="#fff3e0" stroke="#e65100" stroke-width="3" rx="8" />
          <text x="320" y="60" text-anchor="middle" font-size="14" font-weight="bold" fill="#e65100">On Earth</text>
          <text x="320" y="78" text-anchor="middle" font-size="12" fill="#e65100">(Stationary in gravity)</text>

          <!-- Elevator box on Earth -->
          <rect x="270" y="90" width="100" height="80" fill="#ffe0b2" stroke="#e65100" stroke-width="2" rx="3" />

          <!-- Person in Earth elevator -->
          <circle cx="320" cy="120" r="12" fill="#e65100" />
          <line x1="320" y1="132" x2="320" y2="155" stroke="#e65100" stroke-width="3" />
          <line x1="320" y1="140" x2="305" y2="150" stroke="#e65100" stroke-width="3" />
          <line x1="320" y1="140" x2="335" y2="150" stroke="#e65100" stroke-width="3" />

          <!-- Gravity arrow -->
          <line x1="320" y1="175" x2="320" y2="205" stroke="#c62828" stroke-width="4" marker-end="url(#arrow-eq)" />
          <text x="350" y="195" font-size="14" font-weight="bold" fill="#c62828">g</text>

          <!-- Felt force (downward on person) -->
          <line x1="320" y1="158" x2="320" y2="185" stroke="#c62828" stroke-width="3" />
          <text x="285" y="175" font-size="12" fill="#c62828">Feels weight</text>

          <!-- Ground indicator -->
          <rect x="250" y="195" width="140" height="10" fill="#8d6e63" />
          <text x="320" y="225" text-anchor="middle" font-size="12" fill="#5d4037">Ground</text>

          <!-- Caption -->
          <text x="210" y="235" text-anchor="middle" font-size="13" fill="#555" font-style="italic">Locally indistinguishable experiences</text>
        </svg>
      </div>
    </div>
    
    <div class="principle-card covariance">
      <div class="principle-icon"><i class="fas fa-sync-alt"></i></div>
      <h4>General Covariance</h4>
      <p>The laws of physics take the same form in all coordinate systems</p>
    </div>
    
    <div class="principle-card curvature">
      <div class="principle-icon"><i class="fas fa-globe"></i></div>
      <h4>Spacetime Curvature</h4>
      <p>Matter and energy curve spacetime, and this curvature guides motion</p>
      <div class="principle-visual">
        <svg viewBox="0 0 420 280" style="max-width: 500px; width: 100%;">
          <!-- Title -->
          <text x="210" y="25" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Spacetime Curvature by Mass</text>

          <!-- Define gradient for mass -->
          <defs>
            <radialGradient id="massGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#ef5350" />
              <stop offset="100%" stop-color="#b71c1c" />
            </radialGradient>
            <marker id="arrow-curve" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L8,3 z" fill="#1976d2" />
            </marker>
          </defs>

          <!-- Curved spacetime grid - horizontal lines -->
          <path d="M 30 60 Q 210 75, 390 60" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 30 90 Q 210 115, 390 90" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 30 120 Q 210 160, 390 120" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 30 150 Q 210 200, 390 150" stroke="#546e7a" stroke-width="2.5" fill="none" />
          <path d="M 30 180 Q 210 220, 390 180" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 30 210 Q 210 235, 390 210" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 30 240 Q 210 250, 390 240" stroke="#78909c" stroke-width="2" fill="none" />

          <!-- Curved spacetime grid - vertical lines -->
          <path d="M 50 50 Q 55 150, 50 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 90 50 Q 100 150, 90 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 130 50 Q 150 150, 130 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 170 50 Q 195 155, 170 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 210 50 Q 210 160, 210 255" stroke="#546e7a" stroke-width="2.5" fill="none" />
          <path d="M 250 50 Q 225 155, 250 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 290 50 Q 270 150, 290 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 330 50 Q 320 150, 330 255" stroke="#78909c" stroke-width="2" fill="none" />
          <path d="M 370 50 Q 365 150, 370 255" stroke="#78909c" stroke-width="2" fill="none" />

          <!-- Central mass -->
          <circle cx="210" cy="155" r="30" fill="url(#massGradient)" stroke="#b71c1c" stroke-width="3" />
          <text x="210" y="162" text-anchor="middle" font-size="20" font-weight="bold" fill="white">M</text>

          <!-- Object following geodesic -->
          <circle cx="90" cy="90" r="8" fill="#1976d2" />
          <path d="M 100 95 Q 150 130, 180 140" stroke="#1976d2" stroke-width="3" fill="none" stroke-dasharray="5,3" marker-end="url(#arrow-curve)" />
          <text x="60" y="80" font-size="14" font-weight="bold" fill="#1976d2">Object</text>
          <text x="60" y="95" font-size="12" fill="#1565c0">follows curved</text>
          <text x="60" y="110" font-size="12" fill="#1565c0">geodesic</text>

          <!-- Annotations -->
          <text x="340" y="85" font-size="13" fill="#455a64">Flat spacetime</text>
          <text x="340" y="100" font-size="13" fill="#455a64">(far from mass)</text>

          <text x="340" y="190" font-size="13" fill="#bf360c">Curved spacetime</text>
          <text x="340" y="205" font-size="13" fill="#bf360c">(near mass)</text>

          <!-- Caption -->
          <text x="210" y="275" text-anchor="middle" font-size="14" fill="#555" font-style="italic">"Matter tells spacetime how to curve"</text>
        </svg>
      </div>
    </div>
  </div>
</div>

### Einstein Field Equations

<div class="einstein-equations-section">
  <div class="equation-header">
    <i class="fas fa-equals"></i>
    <h4>The fundamental equation of general relativity</h4>
  </div>
  
  <div class="main-equation">
    <div class="equation-box einstein">
      $$R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}$$
    </div>
  </div>
  
  <div class="equation-components">
    <div class="component-grid">
      <div class="component">
        <div class="symbol">$R_{\mu\nu}$</div>
        <div class="name">Ricci curvature tensor</div>
        <div class="description">Describes spacetime curvature</div>
      </div>
      <div class="component">
        <div class="symbol">$g_{\mu\nu}$</div>
        <div class="name">Metric tensor</div>
        <div class="description">Describes spacetime geometry</div>
      </div>
      <div class="component">
        <div class="symbol">$R$</div>
        <div class="name">Scalar curvature</div>
        <div class="description">Trace of Ricci tensor</div>
      </div>
      <div class="component">
        <div class="symbol">$\Lambda$</div>
        <div class="name">Cosmological constant</div>
        <div class="description">Dark energy term</div>
      </div>
      <div class="component">
        <div class="symbol">$G$</div>
        <div class="name">Gravitational constant</div>
        <div class="description">$6.674 \times 10^{-11} \text{ m}^3\text{kg}^{-1}\text{s}^{-2}$</div>
      </div>
      <div class="component">
        <div class="symbol">$T_{\mu\nu}$</div>
        <div class="name">Stress-energy tensor</div>
        <div class="description">Matter and energy content</div>
      </div>
    </div>
  </div>
  
  <div class="equation-interpretation">
    <div class="interpretation-visual">
      <div class="side geometry">
        <h5>Geometry</h5>
        <p>Curvature of spacetime</p>
        <i class="fas fa-globe fa-3x"></i>
      </div>
      <div class="equals">=</div>
      <div class="side matter">
        <h5>Matter/Energy</h5>
        <p>Content of spacetime</p>
        <i class="fas fa-atom fa-3x"></i>
      </div>
    </div>
  </div>
</div>

#### Derivation from Action Principle
The Einstein-Hilbert action:

$$S = \int d^4x \sqrt{-g} \left[\frac{R}{16\pi G} + \mathcal{L}_m\right]$$

Where g = det(g_μν) and ℒ_m is the matter Lagrangian density.

Varying with respect to the metric:

$$\frac{\delta S}{\delta g^{\mu\nu}} = 0$$

Leads to:

$$R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R = \frac{8\pi G}{c^4}T_{\mu\nu}$$

Where the stress-energy tensor is:

$$T_{\mu\nu} = -\frac{2}{\sqrt{-g}} \frac{\delta(\sqrt{-g} \mathcal{L}_m)}{\delta g^{\mu\nu}}$$

#### Curvature Tensors
The Riemann curvature tensor:

$$R^\rho_{\sigma\mu\nu} = \partial_\mu\Gamma^\rho_{\nu\sigma} - \partial_\nu\Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$$

The Ricci tensor (contraction of Riemann):

$$R_{\mu\nu} = R^\rho_{\mu\rho\nu}$$

The scalar curvature:

$$R = g^{\mu\nu} R_{\mu\nu}$$

Bianchi identity ensures conservation:

$$\nabla_\mu G^{\mu\nu} = 0$$

Where G^μν = R^μν - ½g^μν R is the Einstein tensor.

### The Metric Tensor

The metric tensor describes the geometry of spacetime:

$$ds^2 = g_{\mu\nu} dx^\mu dx^\nu$$

For flat spacetime (Minkowski metric):

$$ds^2 = -c^2dt^2 + dx^2 + dy^2 + dz^2$$

### Schwarzschild Solution

For a non-rotating, spherically symmetric mass:

$$ds^2 = -\left(1 - \frac{2GM}{rc^2}\right)c^2dt^2 + \left(1 - \frac{2GM}{rc^2}\right)^{-1}dr^2 + r^2(d\theta^2 + \sin^2\theta d\phi^2)$$

This describes spacetime around stars, planets, and non-rotating black holes.

#### Schwarzschild Radius
The event horizon of a black hole:

$$r_s = \frac{2GM}{c^2}$$

### Gravitational Time Dilation

Clocks run slower in stronger gravitational fields:

$$\Delta t = \frac{\Delta\tau}{\sqrt{1 - 2GM/rc^2}}$$

Where Δτ is the proper time at radius r.

### Gravitational Redshift

Light climbing out of a gravitational field is redshifted:

$$z = \frac{\sqrt{1 - 2GM/r_1c^2}}{\sqrt{1 - 2GM/r_2c^2}} - 1$$

### Geodesics

Objects in free fall follow geodesics (shortest paths in curved spacetime):

$$\frac{d^2x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau}\frac{dx^\beta}{d\tau} = 0$$

Where Γ^μ_αβ are the Christoffel symbols describing the connection.

## Predictions and Confirmations

### Special Relativity Predictions

1. **Time Dilation:** Confirmed in particle accelerators and cosmic ray muons
2. **Length Contraction:** Indirectly confirmed through particle physics
3. **Mass-Energy Equivalence:** Confirmed in nuclear reactions
4. **Relativistic Doppler Effect:** Observed in astronomy

### General Relativity Predictions

1. **Perihelion Precession of Mercury:** 43 arcseconds per century
2. **Gravitational Lensing:** Light bending around massive objects
3. **Gravitational Waves:** Detected by LIGO in 2015
4. **Black Holes:** First imaged by Event Horizon Telescope in 2019
5. **Frame Dragging:** Confirmed by Gravity Probe B
6. **Cosmological Expansion:** Foundation of modern cosmology

## Applications

### Technology
- **GPS Navigation:** Requires both special and general relativistic corrections
- **Particle Accelerators:** Design based on relativistic mechanics
- **Electron Microscopes:** Relativistic corrections for high-energy electrons

### Astrophysics
- **Black Hole Physics:** Understanding accretion disks and jets
- **Neutron Stars:** Modeling extreme gravity environments
- **Cosmology:** Big Bang theory and universe evolution
- **Gravitational Wave Astronomy:** New window to observe the universe

### Fundamental Physics
- **Quantum Field Theory:** Combines special relativity with quantum mechanics
- **String Theory:** Attempts to unify general relativity with quantum mechanics
- **Tests of Fundamental Symmetries:** Lorentz invariance tests

## Paradoxes and Resolutions

### Twin Paradox
One twin travels at high speed and returns younger than the stationary twin. Resolution: The traveling twin experiences acceleration, breaking the symmetry.

### Ladder Paradox
A ladder moving at high speed appears contracted and fits in a smaller garage. Resolution: Relativity of simultaneity - the front and back of the ladder don't enter simultaneously in all frames.

### Grandfather Paradox
Time travel could allow changing the past. Resolution: Various theoretical solutions including self-consistent timelines or parallel universes.

## Mathematical Tools

### Four-Vectors
Quantities that transform like spacetime coordinates:

**Four-Position:**

$$x_\mu = (ct, x, y, z)$$

**Four-Velocity:**

$$u_\mu = \gamma(c, v_x, v_y, v_z)$$

**Four-Momentum:**

$$p_\mu = (E/c, p_x, p_y, p_z)$$

### Tensor Notation
- **Contravariant:** Upper indices (xμ)
- **Covariant:** Lower indices (xμ)
- **Einstein Summation:** Repeated indices are summed

### Christoffel Symbols
Connection coefficients:

$$\Gamma^\mu_{\alpha\beta} = \frac{1}{2}g^{\mu\nu}\left(\frac{\partial g_{\nu\alpha}}{\partial x^\beta} + \frac{\partial g_{\nu\beta}}{\partial x^\alpha} - \frac{\partial g_{\alpha\beta}}{\partial x^\nu}\right)$$

## Modern Developments

### Gravitational Wave Astronomy
LIGO and Virgo detectors have opened a new era of astronomy:
- Binary black hole mergers
- Neutron star collisions
- Tests of general relativity in strong field regime

### Cosmological Observations
- Dark energy and accelerating expansion
- Cosmic microwave background measurements
- Large-scale structure formation

### Quantum Gravity
Attempts to unify general relativity with quantum mechanics:
- String theory
- Loop quantum gravity
- Emergent gravity theories

## Experimental Tests

### Classic Tests
1. **Michelson-Morley Experiment:** Null result led to special relativity
2. **Eddington's 1919 Eclipse:** Confirmed light bending
3. **Pound-Rebka Experiment:** Gravitational redshift in Earth's field
4. **Hafele-Keating Experiment:** Time dilation with atomic clocks on planes

### Modern Precision Tests
1. **Lunar Laser Ranging:** Tests equivalence principle
2. **Gravity Probe A/B:** Tests frame dragging and geodetic effect
3. **Pulsar Timing:** Tests general relativity in strong fields
4. **LIGO/Virgo:** Direct detection of spacetime ripples

## Limitations and Open Questions

1. **Singularities:** General relativity predicts its own breakdown
2. **Quantum Gravity:** No complete theory unifying GR with quantum mechanics
3. **Dark Matter/Energy:** Unexplained observations requiring new physics
4. **Information Paradox:** Black hole information loss problem
5. **Cosmological Constant Problem:** Huge discrepancy with quantum predictions

## Graduate-Level Mathematical Formalism

### Special Relativity in Four-Vector Notation

**Minkowski Spacetime:** (M, η) with metric signature (-,+,+,+)

**Four-vector transformation:**

$$x'^\mu = \Lambda^\mu_\nu x^\nu$$

Where Λ is a Lorentz transformation satisfying:

$$\Lambda^\mu_\alpha \eta_{\mu\nu} \Lambda^\nu_\beta = \eta_{\alpha\beta}$$

**Proper Lorentz Group:** SO(3,1) - preserves orientation and time direction

**Generators of Lorentz transformations:**
- Rotations: J_i = ε_{ijk}x_j∂_k
- Boosts: K_i = x^0∂_i + x_i∂_0

**Lorentz algebra:**

$$[J_i, J_j] = i\varepsilon_{ijk}J_k$$
$$[K_i, K_j] = -i\varepsilon_{ijk}J_k$$
$$[J_i, K_j] = i\varepsilon_{ijk}K_k$$

### Relativistic Field Theory

**Action principle:**

$$S = \int d^4x \mathcal{L}(\phi, \partial_\mu\phi)$$

**Noether's theorem:** Symmetry → Conservation law
- Translation invariance → Energy-momentum conservation
- Lorentz invariance → Angular momentum conservation
- U(1) gauge invariance → Charge conservation

**Energy-momentum tensor:**

$$T^{\mu\nu} = \frac{\partial\mathcal{L}}{\partial(\partial_\mu\phi)} \partial^\nu\phi - g^{\mu\nu} \mathcal{L}$$

Conservation: ∂_μT^μν = 0

### Spinors and the Dirac Equation

**Clifford algebra:**

$$\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}$$

**Dirac equation:**

$$(i\gamma^\mu\partial_\mu - m)\psi = 0$$

**Spinor representation of Lorentz group:** SL(2,C) double covers SO(3,1)

## Differential Geometry for General Relativity

### Manifolds and Tensors

**Tangent space:** T_pM - vector space of directional derivatives at p

**Cotangent space:** T*_pM - dual space of linear functionals

**Tensor:** T^{μ₁...μₙ}_{ν₁...νₘ} - multilinear map

**Metric tensor properties:**
- Symmetric: g_{μν} = g_{νμ}
- Non-degenerate: det(g) ≠ 0
- Signature: (-,+,+,+) for spacetime

### Covariant Derivative and Connection

**Covariant derivative:**

$$\nabla_\mu V^\nu = \partial_\mu V^\nu + \Gamma^\nu_{\mu\lambda}V^\lambda$$
$$\nabla_\mu \omega_\nu = \partial_\mu \omega_\nu - \Gamma^\lambda_{\mu\nu}\omega_\lambda$$

**Metric compatibility:** ∇_λg_{μν} = 0

**Torsion-free:** Γ^λ_{μν} = Γ^λ_{νμ}

**Christoffel symbols:**

$$\Gamma^\lambda_{\mu\nu} = \frac{1}{2}g^{\lambda\sigma}(\partial_\mu g_{\sigma\nu} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu})$$

### Curvature

**Riemann tensor:**

$$R^\rho_{\sigma\mu\nu} = \partial_\mu\Gamma^\rho_{\nu\sigma} - \partial_\nu\Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$$

**Properties:**
- Antisymmetry: R_{ρσμν} = -R_{σρμν} = -R_{ρσνμ}
- First Bianchi identity: R_{ρ[σμν]} = 0
- Second Bianchi identity: ∇_{[λ}R_{ρσ]μν} = 0

**Ricci tensor:** R_{μν} = R^λ_{μλν}

**Scalar curvature:** R = g^{μν}R_{μν}

**Weyl tensor (conformal curvature):**

$$C_{\rho\sigma\mu\nu} = R_{\rho\sigma\mu\nu} - \frac{1}{2}(g_{\rho\mu}R_{\sigma\nu} - g_{\rho\nu}R_{\sigma\mu} + g_{\sigma\nu}R_{\rho\mu} - g_{\sigma\mu}R_{\rho\nu}) + \frac{R}{6}(g_{\rho\mu}g_{\sigma\nu} - g_{\rho\nu}g_{\sigma\mu})$$

## Einstein Field Equations: Detailed Analysis

### Variational Derivation

**Einstein-Hilbert action:**

$$S = S_{EH} + S_m = \frac{1}{16\pi G} \int d^4x \sqrt{-g} R + \int d^4x \sqrt{-g} \mathcal{L}_m$$

**Metric variation:**

$$\delta\sqrt{-g} = -\frac{1}{2}\sqrt{-g} g_{\mu\nu}\delta g^{\mu\nu}$$
$$\delta R = R_{\mu\nu}\delta g^{\mu\nu} + g_{\mu\nu}\nabla_\lambda\nabla^\lambda\delta g^{\mu\nu} - \nabla_\mu\nabla_\nu\delta g^{\mu\nu}$$

**Gibbons-Hawking-York boundary term:** Required for well-posed variational problem

$$S_{GHY} = \frac{1}{8\pi G} \int_{\partial M} d^3x \sqrt{h} K$$

Where K is the trace of extrinsic curvature.

### Solutions and Their Properties

#### Schwarzschild Solution

**Line element:**

$$ds^2 = -\left(1-\frac{2M}{r}\right)dt^2 + \left(1-\frac{2M}{r}\right)^{-1}dr^2 + r^2d\Omega^2$$

**Kruskal-Szekeres coordinates:** Maximal analytic extension

$$T^2 - X^2 = \left(\frac{r}{2M} - 1\right)e^{r/2M}$$

- TX > 0: exterior regions
- TX < 0: black/white hole regions

**Penrose diagram:** Conformal compactification
- i⁺: future timelike infinity
- i⁻: past timelike infinity
- i⁰: spatial infinity
- ℐ⁺: future null infinity
- ℐ⁻: past null infinity

#### Kerr Solution

**Rotating black hole metric (Boyer-Lindquist):**

$$ds^2 = -\left(1-\frac{2Mr}{\rho^2}\right)dt^2 - \frac{4Mar \sin^2\theta}{\rho^2} dtd\phi + \frac{\rho^2}{\Delta} dr^2 + \rho^2d\theta^2 + \sin^2\theta\left(r^2 + a^2 + \frac{2Ma^2r \sin^2\theta}{\rho^2}\right)d\phi^2$$

Where:
- ρ^2 = r^2 + a^2cos^2θ
- Δ = r^2 - 2Mr + a^2
- a = J/M (specific angular momentum)

**Ergosphere:** Region where frame-dragging prevents static observers
- Inner boundary: event horizon r₊ = M + √(M² - a²)
- Outer boundary: static limit r_s = M + √(M² - a²cos²θ)

**Penrose process:** Energy extraction from ergosphere

#### Reissner-Nordström Solution

**Charged black hole:**

$$ds^2 = -\left(1-\frac{2M}{r}+\frac{Q^2}{r^2}\right)dt^2 + \left(1-\frac{2M}{r}+\frac{Q^2}{r^2}\right)^{-1}dr^2 + r^2d\Omega^2$$

**Horizons:** r_± = M ± √(M² - Q²)
- Extremal case: Q = M (single degenerate horizon)
- Naked singularity: Q > M (cosmic censorship conjecture)

### Cosmological Solutions

#### FLRW Metric

**Friedmann-Lemaître-Robertson-Walker:**

$$ds^2 = -dt^2 + a(t)^2\left[\frac{dr^2}{1-kr^2} + r^2d\Omega^2\right]$$

Where k = {-1, 0, +1} for {open, flat, closed} universe.

**Friedmann equations:**

$$\left(\frac{\dot{a}}{a}\right)^2 = \frac{8\pi G\rho}{3} - \frac{k}{a^2} + \frac{\Lambda}{3}$$
$$\frac{\ddot{a}}{a} = -\frac{4\pi G(\rho + 3p)}{3} + \frac{\Lambda}{3}$$

**Equation of state:** p = wρ
- Radiation: w = 1/3
- Matter: w = 0
- Dark energy: w = -1

#### de Sitter and Anti-de Sitter

**de Sitter (Λ > 0):**

$$ds^2 = -\left(1-\frac{r^2}{\alpha^2}\right)dt^2 + \left(1-\frac{r^2}{\alpha^2}\right)^{-1}dr^2 + r^2d\Omega^2$$

Where α = √(3/Λ)

**Anti-de Sitter (Λ < 0):**

$$ds^2 = -\left(1+\frac{r^2}{\alpha^2}\right)dt^2 + \left(1+\frac{r^2}{\alpha^2}\right)^{-1}dr^2 + r^2d\Omega^2$$

## Black Hole Thermodynamics

### The Four Laws

**Zeroth Law:** Surface gravity κ is constant on horizon

**First Law:**

$$dM = \frac{\kappa}{8\pi G} dA + \Omega dJ + \Phi dQ$$

**Second Law:** Hawking area theorem

$$\delta A \geq 0$$

**Third Law:** Cannot achieve κ = 0 in finite operations

### Hawking Radiation

**Temperature:**

$$T_H = \frac{\hbar\kappa}{2\pi ck_B} = \frac{\hbar c^3}{8\pi GMk_B}$$

**Bekenstein-Hawking entropy:**

$$S = \frac{k_B A}{4l_P^2} = \frac{k_B c^3A}{4G\hbar}$$

**Unruh effect:** Accelerating observers see thermal radiation

$$T_U = \frac{\hbar a}{2\pi ck_B}$$

### Information Paradox

**Problem:** Unitarity violation in black hole evaporation

**Proposed solutions:**
- Complementarity
- Firewalls
- ER=EPR
- Soft hair
- Islands and replica wormholes

## Gravitational Waves

### Linearized Gravity

**Weak field approximation:**

$$g_{\mu\nu} = \eta_{\mu\nu} + h_{\mu\nu}, \quad |h_{\mu\nu}| \ll 1$$

**Gauge freedom:** Coordinate transformations

$$h'_{\mu\nu} = h_{\mu\nu} - \partial_\mu\xi_\nu - \partial_\nu\xi_\mu$$

**Transverse-traceless gauge:**

$$h^{\mu 0} = 0, \quad h^\mu_\mu = 0, \quad \partial^ih_{ij} = 0$$

**Wave equation:**

$$\Box h_{\mu\nu} = -16\pi G T_{\mu\nu}$$

### Quadrupole Formula

**Energy flux:**

$$\frac{dE}{dt} = -\frac{G}{5} \left\langle\frac{d^3Q_{ij}}{dt^3} \frac{d^3Q^{ij}}{dt^3}\right\rangle$$

Where Q_{ij} is the quadrupole moment.

**Gravitational wave strain:**

$$h_{ij}^{TT} = \frac{2G}{rc^4} \frac{d^2Q_{ij}^{TT}}{dt^2}$$

### Binary Systems

**Orbital decay (Peters-Mathews):**

$$\frac{da}{dt} = -\frac{64G^3}{5c^5} \frac{\mu M^2}{a^3}$$

**Chirp mass:**

$$\mathcal{M} = \frac{(m_1m_2)^{3/5}}{(m_1+m_2)^{1/5}}$$

**Waveform phases:**
1. Inspiral: Post-Newtonian expansion
2. Merger: Numerical relativity
3. Ringdown: Quasinormal modes

<div class="gw-waveform-diagram">
  <svg viewBox="0 0 600 380" style="max-width: 500px; width: 100%;">
    <!-- Title -->
    <text x="300" y="25" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Gravitational Wave from Binary Black Hole Merger</text>

    <!-- Define arrow markers -->
    <defs>
      <marker id="arrow-gw" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e50" />
      </marker>
    </defs>

    <!-- Axes -->
    <line x1="50" y1="200" x2="570" y2="200" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrow-gw)" />
    <line x1="50" y1="320" x2="50" y2="80" stroke="#2c3e50" stroke-width="3" marker-end="url(#arrow-gw)" />
    <text x="575" y="205" font-size="16" font-weight="bold" fill="#2c3e50">Time</text>
    <text x="55" y="75" font-size="16" font-weight="bold" fill="#2c3e50">Strain h</text>

    <!-- Zero line reference -->
    <line x1="50" y1="200" x2="550" y2="200" stroke="#bdbdbd" stroke-width="1" stroke-dasharray="4,4" />

    <!-- Waveform - Inspiral phase (increasing frequency and amplitude) -->
    <path d="M 70 200
             Q 85 190, 100 200 T 130 200
             Q 145 185, 160 200 T 190 200
             Q 210 175, 230 200 T 260 200
             Q 285 160, 310 200"
          stroke="#1976d2" stroke-width="4" fill="none" />
    <rect x="70" y="245" width="240" height="30" fill="#e3f2fd" stroke="#1976d2" stroke-width="2" rx="5" />
    <text x="190" y="267" text-anchor="middle" font-size="16" font-weight="bold" fill="#1565c0">INSPIRAL</text>

    <!-- Waveform - Merger phase (peak amplitude) -->
    <path d="M 310 200
             Q 330 130, 350 200
             Q 370 280, 390 200
             Q 405 110, 420 200"
          stroke="#c62828" stroke-width="4" fill="none" />
    <rect x="310" y="245" width="110" height="30" fill="#ffebee" stroke="#c62828" stroke-width="2" rx="5" />
    <text x="365" y="267" text-anchor="middle" font-size="16" font-weight="bold" fill="#b71c1c">MERGER</text>

    <!-- Waveform - Ringdown phase (decaying oscillation) -->
    <path d="M 420 200
             Q 440 230, 460 200
             Q 475 220, 490 200
             Q 500 210, 510 200
             Q 515 205, 520 200
             L 550 200"
          stroke="#2e7d32" stroke-width="4" fill="none" />
    <rect x="420" y="245" width="130" height="30" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2" rx="5" />
    <text x="485" y="267" text-anchor="middle" font-size="16" font-weight="bold" fill="#1b5e20">RINGDOWN</text>

    <!-- Phase boundaries -->
    <line x1="310" y1="90" x2="310" y2="240" stroke="#757575" stroke-width="2" stroke-dasharray="6,4" />
    <line x1="420" y1="90" x2="420" y2="240" stroke="#757575" stroke-width="2" stroke-dasharray="6,4" />

    <!-- Binary system illustrations -->
    <!-- Inspiral - two orbiting black holes -->
    <g transform="translate(190, 55)">
      <circle cx="-20" cy="0" r="10" fill="#1976d2" stroke="#0d47a1" stroke-width="2" />
      <circle cx="20" cy="0" r="10" fill="#1976d2" stroke="#0d47a1" stroke-width="2" />
      <ellipse cx="0" cy="0" rx="30" ry="8" fill="none" stroke="#1976d2" stroke-width="2" stroke-dasharray="4,3" />
      <text x="0" y="35" text-anchor="middle" font-size="13" fill="#1565c0">Orbiting BHs</text>
    </g>

    <!-- Merger - coalescing -->
    <g transform="translate(365, 55)">
      <circle cx="0" cy="0" r="18" fill="#c62828" stroke="#b71c1c" stroke-width="2" />
      <text x="0" y="35" text-anchor="middle" font-size="13" fill="#b71c1c">Coalescing</text>
    </g>

    <!-- Ringdown - final black hole -->
    <g transform="translate(485, 55)">
      <circle cx="0" cy="0" r="16" fill="#2e7d32" stroke="#1b5e20" stroke-width="2" />
      <circle cx="0" cy="0" r="24" fill="none" stroke="#2e7d32" stroke-width="2" opacity="0.5" />
      <circle cx="0" cy="0" r="32" fill="none" stroke="#2e7d32" stroke-width="1" opacity="0.3" />
      <text x="0" y="50" text-anchor="middle" font-size="13" fill="#1b5e20">Final BH</text>
    </g>

    <!-- Annotations -->
    <text x="190" y="310" text-anchor="middle" font-size="13" fill="#1565c0">Frequency increases</text>
    <text x="365" y="310" text-anchor="middle" font-size="13" fill="#b71c1c">Peak amplitude</text>
    <text x="485" y="310" text-anchor="middle" font-size="13" fill="#1b5e20">Damped oscillation</text>

    <!-- Frequency evolution formula -->
    <rect x="150" y="335" width="300" height="35" fill="#fafafa" stroke="#e0e0e0" stroke-width="1" rx="5" />
    <text x="300" y="360" text-anchor="middle" font-size="15" fill="#333">Chirp: f proportional to (time to merger)^(-3/8)</text>
  </svg>
</div>

## ADM Formalism

### 3+1 Decomposition

**Foliation:** M = ℝ × Σ

**ADM metric:**

$$ds^2 = -N^2dt^2 + \gamma_{ij}(dx^i + N^idt)(dx^j + N^jdt)$$

Where:
- N: lapse function
- N^i: shift vector
- γ_{ij}: induced 3-metric

**Extrinsic curvature:**

$$K_{ij} = \frac{1}{2N}(\partial_t\gamma_{ij} - D_iN_j - D_jN_i)$$

### Hamiltonian Formulation

**Canonical variables:** (γ_{ij}, π^{ij})

**Constraints:**
- Hamiltonian constraint: ℋ = 0
- Momentum constraints: ℋ_i = 0

**Evolution equations:**

$$\partial_t\gamma_{ij} = \{\gamma_{ij}, H\}$$
$$\partial_t\pi^{ij} = \{\pi^{ij}, H\}$$

## Modern Research Frontiers

### Quantum Gravity Approaches

#### String Theory

**Fundamental idea:** Point particles → 1D strings

**Critical dimensions:** D = 26 (bosonic), D = 10 (superstring)

**Dualities:**
- T-duality: R ↔ α'/R
- S-duality: Strong ↔ weak coupling
- AdS/CFT: Gauge/gravity duality

#### Loop Quantum Gravity

**Canonical quantization of GR:**
- Ashtekar variables
- Spin networks
- Discrete spacetime at Planck scale

**Area spectrum:**

$$A = 8\pi\gamma l_P^2 \sum_i\sqrt{j_i(j_i+1)}$$

#### Causal Sets

**Fundamental hypothesis:** Spacetime is discrete

**Hauptvermutung:** Manifold recoverable from causal structure

#### Asymptotic Safety

**UV fixed point:** Gravity non-perturbatively renormalizable

**Running couplings:** G(k), Λ(k) approach fixed point as k→∞

### Gravitational Wave Astronomy

**Sources:**
- Compact binary coalescence
- Core-collapse supernovae
- Neutron star mountains
- Cosmic strings
- Primordial GWs

**Detectors:**
- Ground-based: LIGO, Virgo, KAGRA
- Space-based: LISA (planned)
- Pulsar timing: NANOGrav

**Multi-messenger astronomy:** GW + EM + neutrinos

### Recent Discoveries (2023-2024)

**Gravitational Wave Breakthroughs:**
- **NANOGrav 15-year data**: Evidence for nanohertz gravitational wave background
- **LIGO-Virgo-KAGRA O4**: Detection of intermediate-mass black hole mergers
- **GW230529**: First neutron star-black hole merger with mass gap object
- **Continuous waves**: New limits on spinning neutron star deformations

**Tests of General Relativity:**
- **Event Horizon Telescope**: Sagittarius A* black hole image (2022)
- **Gravity Probe B**: Frame-dragging confirmed to 0.2% precision
- **Binary pulsar timing**: Tests of strong-field gravity
- **Cosmological tensions**: H₀ and σ₈ discrepancies challenging ΛCDM

### Tests of General Relativity

**Strong field tests:**
- Binary pulsars
- Black hole shadows
- Gravitational wave polarizations

**Parameterized post-Newtonian formalism:**

$$g_{00} = -1 + \frac{2U}{c^2} - \frac{2\beta U^2}{c^4} + \ldots$$
$$g_{0i} = -\frac{4\gamma U_i}{c^3} + \ldots$$
$$g_{ij} = \delta_{ij}\left(1 + \frac{2\gamma U}{c^2}\right) + \ldots$$

GR: β = γ = 1

### Cosmological Puzzles

**Dark energy:**
- Cosmological constant problem: 120 orders of magnitude
- Quintessence models
- Modified gravity (f(R), scalar-tensor)

**Dark matter:**
- Particle candidates (WIMPs, axions)
- Modified dynamics (MOND)
- Emergent gravity

**Inflation:**
- Scalar field dynamics
- Initial conditions
- Trans-Planckian problem

## Advanced Mathematical Methods

### Spinor Methods

**Newman-Penrose formalism:**
- Null tetrad: {l^μ, n^μ, m^μ, m̄^μ}
- Spin coefficients
- Weyl scalars: Ψ₀, ..., Ψ₄

**Petrov classification:**
- Type I: General
- Type II: One double principal null direction
- Type III: One triple PND
- Type N: One quadruple PND
- Type D: Two double PNDs (Schwarzschild, Kerr)

### Conformal Methods

**Conformal transformation:**

$$\tilde{g}_{\mu\nu} = \Omega^2 g_{\mu\nu}$$

**Conformal invariance of null geodesics**

**Penrose diagrams:** Conformal compactification

<div class="minkowski-penrose-diagram">
  <svg viewBox="0 0 520 500" style="max-width: 500px; width: 100%;">
    <!-- Title -->
    <text x="260" y="25" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Penrose Diagram (Minkowski Spacetime)</text>

    <!-- Background for diagram area -->
    <rect x="85" y="65" width="350" height="350" fill="#fafafa" />

    <!-- Diamond boundary (conformal boundary) -->
    <path d="M 260 70 L 430 240 L 260 410 L 90 240 Z" fill="#fff" stroke="#2c3e50" stroke-width="3" />

    <!-- Shaded regions -->
    <!-- Future region -->
    <path d="M 260 70 L 340 150 L 260 230 L 180 150 Z" fill="#e3f2fd" opacity="0.4" />
    <!-- Past region -->
    <path d="M 260 410 L 340 330 L 260 250 L 180 330 Z" fill="#f3e5f5" opacity="0.4" />

    <!-- Light rays (45-degree lines) - multiple rays -->
    <g stroke="#e65100" stroke-width="2" stroke-dasharray="5,4">
      <!-- Outgoing light rays from origin -->
      <line x1="260" y1="240" x2="345" y2="155" />
      <line x1="260" y1="240" x2="175" y2="155" />
      <!-- Incoming light rays to origin -->
      <line x1="175" y1="325" x2="260" y2="240" />
      <line x1="345" y1="325" x2="260" y2="240" />
      <!-- Additional light rays -->
      <line x1="130" y1="240" x2="260" y2="110" />
      <line x1="390" y1="240" x2="260" y2="110" />
    </g>

    <!-- Timelike worldlines -->
    <path d="M 175 360 Q 220 300, 260 240 Q 260 160, 260 70" stroke="#1976d2" stroke-width="3" fill="none" />
    <path d="M 345 360 Q 300 300, 260 240 Q 260 160, 260 70" stroke="#1976d2" stroke-width="3" fill="none" />

    <!-- Spacelike hypersurfaces (constant time slices) -->
    <g stroke="#388e3c" stroke-width="2">
      <line x1="140" y1="190" x2="380" y2="190" />
      <line x1="165" y1="165" x2="355" y2="165" />
      <line x1="190" y1="140" x2="330" y2="140" opacity="0.6" />
      <line x1="215" y1="115" x2="305" y2="115" opacity="0.4" />
    </g>
    <text x="395" y="192" font-size="14" fill="#388e3c" font-weight="bold">t = const</text>

    <!-- Center point (origin) -->
    <circle cx="260" cy="240" r="8" fill="#c62828" stroke="#b71c1c" stroke-width="2" />
    <text x="275" y="235" font-size="14" font-weight="bold" fill="#c62828">Origin</text>
    <text x="275" y="252" font-size="12" fill="#c62828">(r=0, t=0)</text>

    <!-- Infinity labels with better styling -->
    <!-- Future timelike infinity i+ -->
    <circle cx="260" cy="70" r="6" fill="#1976d2" />
    <text x="260" y="55" text-anchor="middle" font-size="18" font-weight="bold" fill="#1976d2">i+</text>
    <text x="260" y="45" text-anchor="middle" font-size="11" fill="#555">(future timelike infinity)</text>

    <!-- Past timelike infinity i- -->
    <circle cx="260" cy="410" r="6" fill="#7b1fa2" />
    <text x="260" y="435" text-anchor="middle" font-size="18" font-weight="bold" fill="#7b1fa2">i-</text>
    <text x="260" y="450" text-anchor="middle" font-size="11" fill="#555">(past timelike infinity)</text>

    <!-- Spatial infinity i0 (right) -->
    <circle cx="430" cy="240" r="6" fill="#388e3c" />
    <text x="455" y="245" text-anchor="start" font-size="18" font-weight="bold" fill="#388e3c">i0</text>

    <!-- Spatial infinity i0 (left) -->
    <circle cx="90" cy="240" r="6" fill="#388e3c" />
    <text x="65" y="245" text-anchor="end" font-size="18" font-weight="bold" fill="#388e3c">i0</text>
    <text x="50" y="262" text-anchor="middle" font-size="10" fill="#555">(spatial</text>
    <text x="50" y="275" text-anchor="middle" font-size="10" fill="#555">infinity)</text>

    <!-- Null infinity labels (Script I) -->
    <!-- Future null infinity (upper right) -->
    <text x="370" y="130" font-size="20" font-weight="bold" fill="#e65100" transform="rotate(45 370 130)">I+</text>
    <!-- Future null infinity (upper left) -->
    <text x="150" y="130" font-size="20" font-weight="bold" fill="#e65100" transform="rotate(-45 150 130)">I+</text>
    <!-- Past null infinity (lower right) -->
    <text x="370" y="350" font-size="20" font-weight="bold" fill="#bf360c" transform="rotate(-45 370 350)">I-</text>
    <!-- Past null infinity (lower left) -->
    <text x="150" y="350" font-size="20" font-weight="bold" fill="#bf360c" transform="rotate(45 150 350)">I-</text>

    <!-- Legend -->
    <rect x="15" y="65" width="70" height="100" fill="#fafafa" stroke="#e0e0e0" stroke-width="1" rx="5" />
    <text x="50" y="82" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Legend</text>
    <line x1="22" y1="95" x2="45" y2="95" stroke="#e65100" stroke-width="2" stroke-dasharray="4,3" />
    <text x="50" y="99" font-size="10" fill="#333">Light ray</text>
    <line x1="22" y1="115" x2="45" y2="115" stroke="#1976d2" stroke-width="2" />
    <text x="50" y="119" font-size="10" fill="#333">Worldline</text>
    <line x1="22" y1="135" x2="45" y2="135" stroke="#388e3c" stroke-width="2" />
    <text x="50" y="139" font-size="10" fill="#333">t = const</text>
    <circle cx="30" cy="152" r="4" fill="#c62828" />
    <text x="50" y="156" font-size="10" fill="#333">Event</text>

    <!-- Caption -->
    <rect x="100" y="460" width="320" height="30" fill="#fff3e0" stroke="#e65100" stroke-width="1" rx="5" />
    <text x="260" y="482" text-anchor="middle" font-size="14" fill="#e65100" font-style="italic">All of infinite Minkowski spacetime fits in this finite diamond</text>
  </svg>
</div>

### Killing Vectors and Symmetries

**Killing equation:**

$$\nabla_{(\mu}\xi_{\nu)} = 0$$

**Conserved quantities:**

$$E = -\xi^\mu_{(t)}p_\mu$$
$$L = \xi^\mu_{(\phi)}p_\mu$$

**Maximum symmetry:**
- Flat: 10 Killing vectors (Poincaré)
- (Anti-)de Sitter: 10 Killing vectors
- FLRW: 6 Killing vectors

## Computational General Relativity

### Numerical Relativity

**BSSN formulation:** Stable evolution system

**Constraint damping:** Γ-driver gauge

**Mesh refinement:** Adaptive for binary mergers

### Symbolic Computation

```python
import sympy as sp
from sympy.tensor.tensor import TensorIndexType, TensorHead, tensor_indices

# Define spacetime
Lorentz = TensorIndexType('Lorentz', dummy_name='L')
mu, nu, rho, sigma = tensor_indices('mu nu rho sigma', Lorentz)

# Metric tensor
g = TensorHead('g', [Lorentz, Lorentz], TensorSymmetry.fully_symmetric(2))

# Christoffel symbols
def christoffel(g_inv, g, coords):
    """Compute Christoffel symbols from metric"""
    n = len(coords)
    Gamma = sp.MutableDenseNDimArray.zeros(n, n, n)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Gamma[i,j,k] += sp.Rational(1,2) * g_inv[i,l] * (
                        sp.diff(g[l,j], coords[k]) +
                        sp.diff(g[l,k], coords[j]) -
                        sp.diff(g[j,k], coords[l])
                    )
    return Gamma

# Riemann tensor
def riemann(Gamma, coords):
    """Compute Riemann tensor from Christoffel symbols"""
    n = len(coords)
    R = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    R[i,j,k,l] = (sp.diff(Gamma[i,j,l], coords[k]) -
                                  sp.diff(Gamma[i,j,k], coords[l]))
                    for m in range(n):
                        R[i,j,k,l] += (Gamma[i,m,k]*Gamma[m,j,l] -
                                       Gamma[i,m,l]*Gamma[m,j,k])
    return R
```

## References and Further Reading

### Classic Textbooks
1. **Weinberg** - *Gravitation and Cosmology*
2. **Misner, Thorne & Wheeler** - *Gravitation*
3. **Wald** - *General Relativity*
4. **Carroll** - *Spacetime and Geometry*

### Advanced Monographs
1. **Hawking & Ellis** - *The Large Scale Structure of Space-Time*
2. **Penrose & Rindler** - *Spinors and Space-Time* (2 volumes)
3. **Chandrasekhar** - *The Mathematical Theory of Black Holes*
4. **Baumgarte & Shapiro** - *Numerical Relativity*

### Research Reviews
1. **Living Reviews in Relativity** - Online journal with comprehensive reviews
2. **Padmanabhan** - *Gravitation: Foundations and Frontiers*
3. **Maggiore** - *Gravitational Waves* (2 volumes)
4. **Rovelli** - *Quantum Gravity*

### Recent Developments
1. **LIGO/Virgo Collaboration** - Gravitational wave detections
2. **Event Horizon Telescope** - Black hole imaging
3. **Quantum gravity approaches** - Various review articles
4. **Cosmological observations** - Planck, WMAP results

## References and Resources

<div class="resources-section">
  <div class="prerequisites">
    <h3><i class="fas fa-calculator"></i> Mathematical Requirements</h3>
    <div class="prereq-grid">
      <div class="prereq-item">
        <i class="fas fa-th"></i>
        <span>Linear algebra and matrix operations</span>
      </div>
      <div class="prereq-item">
        <i class="fas fa-shapes"></i>
        <span>Differential geometry</span>
      </div>
      <div class="prereq-item">
        <i class="fas fa-superscript"></i>
        <span>Tensor calculus</span>
      </div>
      <div class="prereq-item">
        <i class="fas fa-wave-square"></i>
        <span>Partial differential equations</span>
      </div>
    </div>
  </div>
  
  <div class="study-tips">
    <h3><i class="fas fa-lightbulb"></i> Conceptual Understanding</h3>
    <div class="tip-cards">
      <div class="tip-card">
        <div class="tip-number">1</div>
        <p>Start with special relativity before general relativity</p>
      </div>
      <div class="tip-card">
        <div class="tip-number">2</div>
        <p>Use spacetime diagrams for visualization</p>
      </div>
      <div class="tip-card">
        <div class="tip-number">3</div>
        <p>Work through thought experiments</p>
      </div>
      <div class="tip-card">
        <div class="tip-number">4</div>
        <p>Practice with four-vector notation</p>
      </div>
    </div>
  </div>
</div>

<div class="conclusion-box">
  <p>The theory of relativity fundamentally changed our understanding of the universe, revealing that space and time are interwoven and dynamic, shaped by matter and energy. Its predictions continue to be confirmed with ever-increasing precision, while also pointing toward new physics yet to be discovered.</p>
</div>

## See Also

### Core Physics Topics:
- [Classical Mechanics](classical-mechanics.html) - Newtonian mechanics and the classical limit
- [Quantum Mechanics](quantum-mechanics.html) - Quantum theory and relativistic quantum mechanics
- [Quantum Field Theory](quantum-field-theory.html) - Unifying quantum mechanics and special relativity
- [String Theory](string-theory.html) - Quantum gravity and extra dimensions

### Related Topics:
- [Computational Physics](computational-physics.html) - Numerical relativity simulations
- [Condensed Matter Physics](condensed-matter.html) - Relativistic effects in graphene
- [Thermodynamics](thermodynamics.html) - Relativistic thermodynamics
- [Statistical Mechanics](statistical-mechanics.html) - Relativistic statistical mechanics