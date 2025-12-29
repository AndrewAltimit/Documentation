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
      <svg viewBox="0 0 720 280" class="lattice-diagram" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Simple Cubic -->
        <g transform="translate(20,30)">
          <text x="90" y="0" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Simple Cubic (SC)</text>
          <!-- Unit cell - complete 3D cube -->
          <!-- Front face -->
          <line x1="30" y1="60" x2="150" y2="60" stroke="#3498db" stroke-width="2.5" />
          <line x1="30" y1="60" x2="30" y2="180" stroke="#3498db" stroke-width="2.5" />
          <line x1="150" y1="60" x2="150" y2="180" stroke="#3498db" stroke-width="2.5" />
          <line x1="30" y1="180" x2="150" y2="180" stroke="#3498db" stroke-width="2.5" />
          <!-- Back face (dashed for depth) -->
          <line x1="70" y1="30" x2="190" y2="30" stroke="#3498db" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="70" y1="30" x2="70" y2="150" stroke="#3498db" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="190" y1="30" x2="190" y2="150" stroke="#3498db" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="70" y1="150" x2="190" y2="150" stroke="#3498db" stroke-width="2" stroke-dasharray="4,2" />
          <!-- Connecting edges -->
          <line x1="30" y1="60" x2="70" y2="30" stroke="#3498db" stroke-width="2" />
          <line x1="150" y1="60" x2="190" y2="30" stroke="#3498db" stroke-width="2" />
          <line x1="30" y1="180" x2="70" y2="150" stroke="#3498db" stroke-width="2" />
          <line x1="150" y1="180" x2="190" y2="150" stroke="#3498db" stroke-width="2" />
          <!-- Atoms at corners -->
          <circle cx="30" cy="60" r="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
          <circle cx="150" cy="60" r="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
          <circle cx="30" cy="180" r="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
          <circle cx="150" cy="180" r="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
          <circle cx="70" cy="30" r="8" fill="#e74c3c" stroke="#c0392b" stroke-width="1.5" opacity="0.8" />
          <circle cx="190" cy="30" r="8" fill="#e74c3c" stroke="#c0392b" stroke-width="1.5" opacity="0.8" />
          <circle cx="70" cy="150" r="8" fill="#e74c3c" stroke="#c0392b" stroke-width="1.5" opacity="0.8" />
          <circle cx="190" cy="150" r="8" fill="#e74c3c" stroke="#c0392b" stroke-width="1.5" opacity="0.8" />
          <!-- Label -->
          <text x="90" y="220" text-anchor="middle" font-size="14" fill="#555">1 atom/cell</text>
          <!-- Lattice parameter label -->
          <text x="90" y="200" text-anchor="middle" font-size="12" fill="#777" font-style="italic">a</text>
          <line x1="30" y1="190" x2="150" y2="190" stroke="#777" stroke-width="1" marker-start="url(#arrow-start)" marker-end="url(#arrow-end)" />
        </g>

        <!-- FCC -->
        <g transform="translate(250,30)">
          <text x="90" y="0" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Face-Centered Cubic (FCC)</text>
          <!-- Unit cell edges -->
          <!-- Front face -->
          <line x1="30" y1="60" x2="150" y2="60" stroke="#27ae60" stroke-width="2.5" />
          <line x1="30" y1="60" x2="30" y2="180" stroke="#27ae60" stroke-width="2.5" />
          <line x1="150" y1="60" x2="150" y2="180" stroke="#27ae60" stroke-width="2.5" />
          <line x1="30" y1="180" x2="150" y2="180" stroke="#27ae60" stroke-width="2.5" />
          <!-- Back face (dashed) -->
          <line x1="70" y1="30" x2="190" y2="30" stroke="#27ae60" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="70" y1="30" x2="70" y2="150" stroke="#27ae60" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="190" y1="30" x2="190" y2="150" stroke="#27ae60" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="70" y1="150" x2="190" y2="150" stroke="#27ae60" stroke-width="2" stroke-dasharray="4,2" />
          <!-- Connecting edges -->
          <line x1="30" y1="60" x2="70" y2="30" stroke="#27ae60" stroke-width="2" />
          <line x1="150" y1="60" x2="190" y2="30" stroke="#27ae60" stroke-width="2" />
          <line x1="30" y1="180" x2="70" y2="150" stroke="#27ae60" stroke-width="2" />
          <line x1="150" y1="180" x2="190" y2="150" stroke="#27ae60" stroke-width="2" />
          <!-- Corner atoms (smaller, semi-transparent) -->
          <circle cx="30" cy="60" r="8" fill="#f39c12" stroke="#d68910" stroke-width="1.5" opacity="0.7" />
          <circle cx="150" cy="60" r="8" fill="#f39c12" stroke="#d68910" stroke-width="1.5" opacity="0.7" />
          <circle cx="30" cy="180" r="8" fill="#f39c12" stroke="#d68910" stroke-width="1.5" opacity="0.7" />
          <circle cx="150" cy="180" r="8" fill="#f39c12" stroke="#d68910" stroke-width="1.5" opacity="0.7" />
          <circle cx="70" cy="30" r="6" fill="#f39c12" stroke="#d68910" stroke-width="1" opacity="0.5" />
          <circle cx="190" cy="30" r="6" fill="#f39c12" stroke="#d68910" stroke-width="1" opacity="0.5" />
          <circle cx="70" cy="150" r="6" fill="#f39c12" stroke="#d68910" stroke-width="1" opacity="0.5" />
          <circle cx="190" cy="150" r="6" fill="#f39c12" stroke="#d68910" stroke-width="1" opacity="0.5" />
          <!-- Face center atoms (larger, prominent) -->
          <circle cx="90" cy="60" r="10" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
          <circle cx="90" cy="180" r="10" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
          <circle cx="30" cy="120" r="10" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
          <circle cx="150" cy="120" r="10" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
          <circle cx="90" cy="105" r="9" fill="#9b59b6" stroke="#7d3c98" stroke-width="1.5" opacity="0.85" />
          <circle cx="130" cy="90" r="8" fill="#9b59b6" stroke="#7d3c98" stroke-width="1" opacity="0.7" />
          <!-- Legend -->
          <circle cx="40" cy="220" r="6" fill="#f39c12" stroke="#d68910" stroke-width="1" />
          <text x="52" y="224" font-size="11" fill="#555">Corner</text>
          <circle cx="100" cy="220" r="6" fill="#9b59b6" stroke="#7d3c98" stroke-width="1" />
          <text x="112" y="224" font-size="11" fill="#555">Face</text>
          <text x="90" y="250" text-anchor="middle" font-size="14" fill="#555">4 atoms/cell</text>
        </g>

        <!-- BCC -->
        <g transform="translate(480,30)">
          <text x="90" y="0" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Body-Centered Cubic (BCC)</text>
          <!-- Unit cell edges -->
          <!-- Front face -->
          <line x1="30" y1="60" x2="150" y2="60" stroke="#e67e22" stroke-width="2.5" />
          <line x1="30" y1="60" x2="30" y2="180" stroke="#e67e22" stroke-width="2.5" />
          <line x1="150" y1="60" x2="150" y2="180" stroke="#e67e22" stroke-width="2.5" />
          <line x1="30" y1="180" x2="150" y2="180" stroke="#e67e22" stroke-width="2.5" />
          <!-- Back face (dashed) -->
          <line x1="70" y1="30" x2="190" y2="30" stroke="#e67e22" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="70" y1="30" x2="70" y2="150" stroke="#e67e22" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="190" y1="30" x2="190" y2="150" stroke="#e67e22" stroke-width="2" stroke-dasharray="4,2" />
          <line x1="70" y1="150" x2="190" y2="150" stroke="#e67e22" stroke-width="2" stroke-dasharray="4,2" />
          <!-- Connecting edges -->
          <line x1="30" y1="60" x2="70" y2="30" stroke="#e67e22" stroke-width="2" />
          <line x1="150" y1="60" x2="190" y2="30" stroke="#e67e22" stroke-width="2" />
          <line x1="30" y1="180" x2="70" y2="150" stroke="#e67e22" stroke-width="2" />
          <line x1="150" y1="180" x2="190" y2="150" stroke="#e67e22" stroke-width="2" />
          <!-- Body diagonal lines (to show body center) -->
          <line x1="30" y1="60" x2="110" y2="105" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.6" />
          <line x1="150" y1="180" x2="110" y2="105" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.6" />
          <!-- Corner atoms -->
          <circle cx="30" cy="60" r="8" fill="#2c3e50" stroke="#1a252f" stroke-width="1.5" opacity="0.7" />
          <circle cx="150" cy="60" r="8" fill="#2c3e50" stroke="#1a252f" stroke-width="1.5" opacity="0.7" />
          <circle cx="30" cy="180" r="8" fill="#2c3e50" stroke="#1a252f" stroke-width="1.5" opacity="0.7" />
          <circle cx="150" cy="180" r="8" fill="#2c3e50" stroke="#1a252f" stroke-width="1.5" opacity="0.7" />
          <circle cx="70" cy="30" r="6" fill="#2c3e50" stroke="#1a252f" stroke-width="1" opacity="0.5" />
          <circle cx="190" cy="30" r="6" fill="#2c3e50" stroke="#1a252f" stroke-width="1" opacity="0.5" />
          <circle cx="70" cy="150" r="6" fill="#2c3e50" stroke="#1a252f" stroke-width="1" opacity="0.5" />
          <circle cx="190" cy="150" r="6" fill="#2c3e50" stroke="#1a252f" stroke-width="1" opacity="0.5" />
          <!-- Body center atom (prominent) -->
          <circle cx="110" cy="105" r="12" fill="#c0392b" stroke="#922b21" stroke-width="2" />
          <!-- Legend -->
          <circle cx="40" cy="220" r="6" fill="#2c3e50" stroke="#1a252f" stroke-width="1" />
          <text x="52" y="224" font-size="11" fill="#555">Corner</text>
          <circle cx="110" cy="220" r="6" fill="#c0392b" stroke="#922b21" stroke-width="1" />
          <text x="122" y="224" font-size="11" fill="#555">Body</text>
          <text x="90" y="250" text-anchor="middle" font-size="14" fill="#555">2 atoms/cell</text>
        </g>

        <!-- Arrow markers for dimension labels -->
        <defs>
          <marker id="arrow-start" markerWidth="6" markerHeight="6" refX="0" refY="3" orient="auto">
            <path d="M6,0 L0,3 L6,6" fill="none" stroke="#777" stroke-width="1" />
          </marker>
          <marker id="arrow-end" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6" fill="none" stroke="#777" stroke-width="1" />
          </marker>
        </defs>
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
      <svg viewBox="0 0 450 280" class="brillouin-diagram" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Title -->
        <text x="225" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">2D Square Lattice Brillouin Zone</text>

        <!-- Coordinate axes -->
        <defs>
          <marker id="bz-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 L2,4 Z" fill="#555" />
          </marker>
        </defs>
        <line x1="50" y1="150" x2="400" y2="150" stroke="#555" stroke-width="1.5" marker-end="url(#bz-arrow)" />
        <line x1="225" y1="260" x2="225" y2="40" stroke="#555" stroke-width="1.5" marker-end="url(#bz-arrow)" />
        <text x="410" y="155" font-size="14" fill="#333" font-weight="bold">k_x</text>
        <text x="230" y="35" font-size="14" fill="#333" font-weight="bold">k_y</text>

        <!-- Extended zones (faded) -->
        <polygon points="225,60 315,150 225,240 135,150" fill="#e8f4fc" stroke="#b8d4e8" stroke-width="1.5" stroke-dasharray="4,2" />

        <!-- First Brillouin zone (main) -->
        <polygon points="225,80 295,150 225,220 155,150" fill="#3498db" opacity="0.35" stroke="#2980b9" stroke-width="3" />
        <text x="225" y="155" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a5276">1st BZ</text>

        <!-- High symmetry points with clear labels -->
        <!-- Gamma point (center) -->
        <circle cx="225" cy="150" r="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
        <text x="225" y="175" text-anchor="middle" font-size="16" font-weight="bold" fill="#c0392b">Gamma</text>

        <!-- X points (edge centers) -->
        <circle cx="260" cy="115" r="6" fill="#27ae60" stroke="#1e8449" stroke-width="2" />
        <text x="275" y="108" font-size="14" font-weight="bold" fill="#1e8449">X</text>

        <circle cx="260" cy="185" r="6" fill="#27ae60" stroke="#1e8449" stroke-width="2" />
        <text x="275" y="192" font-size="14" font-weight="bold" fill="#1e8449">X</text>

        <circle cx="190" cy="115" r="6" fill="#27ae60" stroke="#1e8449" stroke-width="2" />
        <text x="172" y="108" font-size="14" font-weight="bold" fill="#1e8449">X</text>

        <circle cx="190" cy="185" r="6" fill="#27ae60" stroke="#1e8449" stroke-width="2" />
        <text x="172" y="192" font-size="14" font-weight="bold" fill="#1e8449">X</text>

        <!-- M points (corners) -->
        <circle cx="225" cy="80" r="6" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
        <text x="225" y="68" text-anchor="middle" font-size="14" font-weight="bold" fill="#7d3c98">M</text>

        <circle cx="295" cy="150" r="6" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
        <text x="310" y="155" font-size="14" font-weight="bold" fill="#7d3c98">M</text>

        <circle cx="225" cy="220" r="6" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
        <text x="225" y="238" text-anchor="middle" font-size="14" font-weight="bold" fill="#7d3c98">M</text>

        <circle cx="155" cy="150" r="6" fill="#9b59b6" stroke="#7d3c98" stroke-width="2" />
        <text x="138" y="155" font-size="14" font-weight="bold" fill="#7d3c98">M</text>

        <!-- Legend -->
        <rect x="320" y="200" width="120" height="70" fill="white" stroke="#ddd" stroke-width="1" rx="4" />
        <text x="380" y="218" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">High Symmetry Points</text>
        <circle cx="335" cy="235" r="5" fill="#e74c3c" />
        <text x="350" y="239" font-size="12" fill="#333">Gamma: (0,0)</text>
        <circle cx="335" cy="252" r="5" fill="#27ae60" />
        <text x="350" y="256" font-size="12" fill="#333">X: edge center</text>
        <circle cx="335" cy="269" r="5" fill="#9b59b6" />
        <text x="350" y="273" font-size="12" fill="#333">M: corner</text>
      </svg>
    </div>
  </div>
  
  <div class="xray-diffraction">
    <h3><i class="fas fa-radiation"></i> X-ray Diffraction</h3>
    
    <div class="bragg-law">
      <p>Bragg's law:</p>
      <div class="equation-box highlighted">$$2d\sin\theta = n\lambda$$</div>

      <svg viewBox="0 0 500 300" class="bragg-diagram" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Title -->
        <text x="250" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Bragg X-ray Diffraction</text>

        <!-- Arrow marker definitions -->
        <defs>
          <marker id="bragg-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 L2,5 Z" fill="#e74c3c" />
          </marker>
          <marker id="bragg-arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 L2,5 Z" fill="#3498db" />
          </marker>
        </defs>

        <!-- Crystal planes with atoms -->
        <g id="crystal-planes">
          <!-- Plane 1 -->
          <line x1="60" y1="80" x2="380" y2="80" stroke="#7f8c8d" stroke-width="2.5" />
          <circle cx="100" cy="80" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="180" cy="80" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="260" cy="80" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="340" cy="80" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />

          <!-- Plane 2 -->
          <line x1="60" y1="150" x2="380" y2="150" stroke="#7f8c8d" stroke-width="2.5" />
          <circle cx="100" cy="150" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="180" cy="150" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="260" cy="150" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="340" cy="150" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />

          <!-- Plane 3 -->
          <line x1="60" y1="220" x2="380" y2="220" stroke="#7f8c8d" stroke-width="2.5" />
          <circle cx="100" cy="220" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="180" cy="220" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="260" cy="220" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
          <circle cx="340" cy="220" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
        </g>

        <!-- Interplanar spacing indicator -->
        <line x1="400" y1="80" x2="400" y2="150" stroke="#333" stroke-width="1.5" />
        <line x1="395" y1="80" x2="405" y2="80" stroke="#333" stroke-width="1.5" />
        <line x1="395" y1="150" x2="405" y2="150" stroke="#333" stroke-width="1.5" />
        <text x="420" y="120" font-size="16" font-weight="bold" fill="#333">d</text>

        <!-- Incident X-ray beam (ray 1) -->
        <line x1="30" y1="30" x2="180" y2="80" stroke="#e74c3c" stroke-width="3" marker-end="url(#bragg-arrow)" />
        <text x="70" y="45" font-size="14" font-weight="bold" fill="#c0392b">Incident</text>
        <text x="70" y="60" font-size="14" font-weight="bold" fill="#c0392b">X-rays</text>

        <!-- Incident X-ray beam (ray 2 - to second plane) -->
        <line x1="30" y1="100" x2="180" y2="150" stroke="#e74c3c" stroke-width="3" marker-end="url(#bragg-arrow)" />

        <!-- Reflected X-ray beam (ray 1) -->
        <line x1="180" y1="80" x2="330" y2="30" stroke="#3498db" stroke-width="3" marker-end="url(#bragg-arrow-blue)" />
        <text x="340" y="45" font-size="14" font-weight="bold" fill="#2980b9">Reflected</text>
        <text x="340" y="60" font-size="14" font-weight="bold" fill="#2980b9">X-rays</text>

        <!-- Reflected X-ray beam (ray 2) -->
        <line x1="180" y1="150" x2="330" y2="100" stroke="#3498db" stroke-width="3" marker-end="url(#bragg-arrow-blue)" />

        <!-- Path difference visualization -->
        <line x1="180" y1="80" x2="180" y2="150" stroke="#27ae60" stroke-width="2" stroke-dasharray="5,3" />
        <text x="165" y="120" font-size="12" fill="#27ae60" font-weight="bold">Extra</text>
        <text x="165" y="135" font-size="12" fill="#27ae60" font-weight="bold">path</text>

        <!-- Angle theta indicators -->
        <!-- Incident angle -->
        <path d="M 155 80 Q 165 65, 180 62" fill="none" stroke="#2c3e50" stroke-width="2" />
        <text x="160" y="55" font-size="16" font-weight="bold" fill="#2c3e50">theta</text>

        <!-- Reflected angle -->
        <path d="M 205 80 Q 195 65, 180 62" fill="none" stroke="#2c3e50" stroke-width="2" />
        <text x="200" y="55" font-size="16" font-weight="bold" fill="#2c3e50">theta</text>

        <!-- Normal to plane -->
        <line x1="180" y1="80" x2="180" y2="40" stroke="#555" stroke-width="1" stroke-dasharray="3,2" />
        <text x="185" y="38" font-size="11" fill="#555">normal</text>

        <!-- Legend box -->
        <rect x="60" y="245" width="280" height="45" fill="white" stroke="#ddd" stroke-width="1" rx="4" />
        <text x="200" y="262" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Constructive interference when path difference = n x wavelength</text>
        <text x="200" y="280" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a5276">2d sin(theta) = n x wavelength</text>
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
      <svg viewBox="0 0 550 280" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Title -->
        <text x="275" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Bloch Wavefunction in Periodic Potential</text>

        <!-- Axes -->
        <defs>
          <marker id="bloch-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 L2,4 Z" fill="#333" />
          </marker>
        </defs>
        <line x1="40" y1="200" x2="520" y2="200" stroke="#333" stroke-width="2" marker-end="url(#bloch-arrow)" />
        <text x="530" y="205" font-size="14" font-weight="bold" fill="#333">x</text>

        <!-- Periodic potential V(x) - deeper wells for clarity -->
        <path d="M 50 200
                 Q 70 170, 90 200 Q 110 230, 130 200
                 Q 150 170, 170 200 Q 190 230, 210 200
                 Q 230 170, 250 200 Q 270 230, 290 200
                 Q 310 170, 330 200 Q 350 230, 370 200
                 Q 390 170, 410 200 Q 430 230, 450 200
                 Q 470 170, 490 200"
              fill="none" stroke="#7f8c8d" stroke-width="2.5" />
        <!-- Potential label -->
        <text x="510" y="190" font-size="14" font-weight="bold" fill="#7f8c8d">V(x)</text>

        <!-- Atom positions (ion cores) -->
        <circle cx="90" cy="200" r="6" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
        <circle cx="170" cy="200" r="6" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
        <circle cx="250" cy="200" r="6" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
        <circle cx="330" cy="200" r="6" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />
        <circle cx="410" cy="200" r="6" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" />

        <!-- Lattice spacing indicator -->
        <line x1="90" y1="220" x2="170" y2="220" stroke="#555" stroke-width="1.5" />
        <line x1="90" y1="215" x2="90" y2="225" stroke="#555" stroke-width="1.5" />
        <line x1="170" y1="215" x2="170" y2="225" stroke="#555" stroke-width="1.5" />
        <text x="130" y="238" text-anchor="middle" font-size="14" font-weight="bold" fill="#555">a</text>

        <!-- Plane wave envelope e^ikx (dashed) -->
        <path d="M 50 100 Q 150 70, 275 100 Q 400 130, 490 100"
              fill="none" stroke="#e74c3c" stroke-width="2.5" stroke-dasharray="8,4" />
        <text x="65" y="75" font-size="14" font-weight="bold" fill="#c0392b">Envelope</text>
        <text x="65" y="92" font-size="14" font-weight="bold" fill="#c0392b">exp(ikx)</text>

        <!-- Bloch wavefunction psi(x) = u(x) * e^ikx -->
        <path d="M 50 100
                 Q 60 75, 70 100 Q 80 120, 90 100
                 Q 100 70, 110 95 Q 120 115, 130 95
                 Q 140 60, 150 90 Q 160 115, 170 90
                 Q 180 50, 190 85 Q 200 115, 210 85
                 Q 220 45, 230 80 Q 240 110, 250 80
                 Q 260 40, 270 75 Q 280 105, 290 75
                 Q 300 40, 310 70 Q 320 100, 330 70
                 Q 340 40, 350 70 Q 360 95, 370 70
                 Q 380 45, 390 70 Q 400 95, 410 70
                 Q 420 50, 430 75 Q 440 100, 450 80
                 Q 460 60, 470 85 Q 480 105, 490 90"
              fill="none" stroke="#3498db" stroke-width="3" />

        <!-- Psi label with arrow pointing to wave -->
        <text x="300" y="45" font-size="15" font-weight="bold" fill="#2980b9">Bloch wave</text>
        <text x="300" y="62" font-size="15" font-weight="bold" fill="#2980b9">psi(x) = u(x) exp(ikx)</text>
        <line x1="350" y1="65" x2="350" y2="75" stroke="#2980b9" stroke-width="1.5" marker-end="url(#bloch-arrow)" />

        <!-- Periodic function u(x) illustration region -->
        <rect x="85" y="85" width="85" height="40" fill="none" stroke="#27ae60" stroke-width="2" stroke-dasharray="4,2" rx="3" />
        <text x="128" y="140" text-anchor="middle" font-size="12" fill="#27ae60" font-weight="bold">u(x) has</text>
        <text x="128" y="155" text-anchor="middle" font-size="12" fill="#27ae60" font-weight="bold">period a</text>

        <!-- Legend -->
        <rect x="350" y="230" width="180" height="45" fill="white" stroke="#ddd" stroke-width="1" rx="4" />
        <line x1="360" y1="245" x2="390" y2="245" stroke="#e74c3c" stroke-width="2.5" stroke-dasharray="6,3" />
        <text x="400" y="249" font-size="12" fill="#333">Plane wave envelope</text>
        <line x1="360" y1="262" x2="390" y2="262" stroke="#3498db" stroke-width="2.5" />
        <text x="400" y="266" font-size="12" fill="#333">Bloch wavefunction</text>
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
      
      <svg viewBox="0 0 420 300" class="band-diagram" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Title -->
        <text x="210" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Nearly Free Electron Band Structure</text>

        <!-- Axes -->
        <defs>
          <marker id="nfe-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 L2,4 Z" fill="#333" />
          </marker>
        </defs>
        <!-- X-axis (k) -->
        <line x1="50" y1="240" x2="380" y2="240" stroke="#333" stroke-width="2" marker-end="url(#nfe-arrow)" />
        <text x="390" y="245" font-size="14" font-weight="bold" fill="#333">k</text>

        <!-- Y-axis (E) -->
        <line x1="210" y1="260" x2="210" y2="40" stroke="#333" stroke-width="2" marker-end="url(#nfe-arrow)" />
        <text x="205" y="35" font-size="14" font-weight="bold" fill="#333">E</text>

        <!-- BZ boundaries -->
        <line x1="100" y1="250" x2="100" y2="50" stroke="#95a5a6" stroke-width="2" stroke-dasharray="5,3" />
        <text x="100" y="268" text-anchor="middle" font-size="14" font-weight="bold" fill="#555">-pi/a</text>

        <line x1="320" y1="250" x2="320" y2="50" stroke="#95a5a6" stroke-width="2" stroke-dasharray="5,3" />
        <text x="320" y="268" text-anchor="middle" font-size="14" font-weight="bold" fill="#555">+pi/a</text>

        <!-- Free electron parabola (faded, for reference) -->
        <path d="M 60 200 Q 140 80, 210 50 Q 280 80, 360 200" fill="none" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="4,3" />
        <text x="370" y="190" font-size="12" fill="#95a5a6">Free</text>
        <text x="370" y="205" font-size="12" fill="#95a5a6">electron</text>

        <!-- Lower band (valence band) -->
        <path d="M 60 220 Q 100 180, 140 130 Q 180 90, 210 85" fill="none" stroke="#3498db" stroke-width="3" />
        <path d="M 210 85 Q 240 90, 280 130 Q 320 180, 360 220" fill="none" stroke="#3498db" stroke-width="3" />

        <!-- Upper band (conduction band) -->
        <path d="M 60 100 Q 100 75, 140 70 Q 180 68, 210 65" fill="none" stroke="#e74c3c" stroke-width="3" />
        <path d="M 210 65 Q 240 68, 280 70 Q 320 75, 360 100" fill="none" stroke="#e74c3c" stroke-width="3" />

        <!-- Band gap visualization at zone center -->
        <line x1="203" y1="85" x2="203" y2="65" stroke="#27ae60" stroke-width="3" />
        <line x1="195" y1="85" x2="211" y2="85" stroke="#27ae60" stroke-width="2" />
        <line x1="195" y1="65" x2="211" y2="65" stroke="#27ae60" stroke-width="2" />
        <text x="175" y="80" text-anchor="end" font-size="14" font-weight="bold" fill="#27ae60">Band</text>
        <text x="175" y="95" text-anchor="end" font-size="14" font-weight="bold" fill="#27ae60">Gap</text>

        <!-- Gap size annotation -->
        <text x="225" y="80" font-size="14" font-weight="bold" fill="#1e8449">2|V_G|</text>

        <!-- Band labels -->
        <text x="75" y="230" font-size="14" font-weight="bold" fill="#2980b9">Valence Band</text>
        <text x="280" y="55" font-size="14" font-weight="bold" fill="#c0392b">Conduction Band</text>

        <!-- Legend -->
        <rect x="50" y="50" width="90" height="50" fill="white" stroke="#ddd" stroke-width="1" rx="4" />
        <line x1="60" y1="65" x2="85" y2="65" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="4,2" />
        <text x="92" y="69" font-size="11" fill="#555">Free e-</text>
        <line x1="60" y1="85" x2="85" y2="85" stroke="#3498db" stroke-width="2.5" />
        <text x="92" y="89" font-size="11" fill="#555">NFE band</text>
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
      
      <svg viewBox="0 0 420 300" class="tb-diagram" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Title -->
        <text x="210" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Tight-Binding Band Structure (1D)</text>

        <!-- Axes -->
        <defs>
          <marker id="tb-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 L2,4 Z" fill="#333" />
          </marker>
        </defs>

        <!-- X-axis (k) -->
        <line x1="50" y1="200" x2="380" y2="200" stroke="#333" stroke-width="2" marker-end="url(#tb-arrow)" />
        <text x="390" y="205" font-size="14" font-weight="bold" fill="#333">k</text>

        <!-- Y-axis (E) -->
        <line x1="60" y1="260" x2="60" y2="40" stroke="#333" stroke-width="2" marker-end="url(#tb-arrow)" />
        <text x="55" y="35" font-size="14" font-weight="bold" fill="#333">E</text>

        <!-- BZ boundaries -->
        <line x1="100" y1="200" x2="100" y2="50" stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="4,2" />
        <text x="100" y="218" text-anchor="middle" font-size="13" fill="#555">-pi/a</text>

        <line x1="320" y1="200" x2="320" y2="50" stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="4,2" />
        <text x="320" y="218" text-anchor="middle" font-size="13" fill="#555">+pi/a</text>

        <!-- k=0 line -->
        <line x1="210" y1="200" x2="210" y2="50" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,2" />
        <text x="210" y="218" text-anchor="middle" font-size="13" fill="#555">0</text>

        <!-- Energy levels reference lines -->
        <line x1="55" y1="80" x2="380" y2="80" stroke="#ddd" stroke-width="1" stroke-dasharray="2,2" />
        <text x="50" y="84" text-anchor="end" font-size="12" fill="#777">epsilon_0 + 2t</text>

        <line x1="55" y1="130" x2="380" y2="130" stroke="#ddd" stroke-width="1" stroke-dasharray="2,2" />
        <text x="50" y="134" text-anchor="end" font-size="12" fill="#777">epsilon_0</text>

        <line x1="55" y1="180" x2="380" y2="180" stroke="#ddd" stroke-width="1" stroke-dasharray="2,2" />
        <text x="50" y="184" text-anchor="end" font-size="12" fill="#777">epsilon_0 - 2t</text>

        <!-- Cosine dispersion curve E(k) = epsilon_0 - 2t*cos(ka) -->
        <path d="M 100 80
                 Q 130 95, 155 115
                 Q 180 145, 210 180
                 Q 240 145, 265 115
                 Q 290 95, 320 80"
              fill="none" stroke="#27ae60" stroke-width="4" />

        <!-- Band width annotation -->
        <line x1="340" y1="80" x2="340" y2="180" stroke="#e74c3c" stroke-width="2" />
        <line x1="335" y1="80" x2="345" y2="80" stroke="#e74c3c" stroke-width="2" />
        <line x1="335" y1="180" x2="345" y2="180" stroke="#e74c3c" stroke-width="2" />
        <text x="355" y="125" font-size="14" font-weight="bold" fill="#c0392b">Bandwidth</text>
        <text x="355" y="142" font-size="14" font-weight="bold" fill="#c0392b">= 4t</text>

        <!-- Key points annotation -->
        <circle cx="210" cy="180" r="6" fill="#e74c3c" stroke="#c0392b" stroke-width="2" />
        <text x="215" y="195" font-size="12" font-weight="bold" fill="#c0392b">Band bottom</text>

        <circle cx="100" cy="80" r="5" fill="#3498db" stroke="#2980b9" stroke-width="2" />
        <circle cx="320" cy="80" r="5" fill="#3498db" stroke="#2980b9" stroke-width="2" />
        <text x="320" y="68" text-anchor="middle" font-size="12" font-weight="bold" fill="#2980b9">Band top</text>

        <!-- Dispersion formula -->
        <rect x="100" y="230" width="220" height="35" fill="white" stroke="#27ae60" stroke-width="2" rx="5" />
        <text x="210" y="253" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e8449">E(k) = epsilon_0 - 2t cos(ka)</text>
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
      <svg viewBox="0 0 650 280" style="max-width: 500px; width: 100%; background: #fafbfc; border-radius: 8px;">
        <!-- Main title -->
        <text x="325" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Density of States in Different Dimensions</text>

        <!-- Arrow marker -->
        <defs>
          <marker id="dos-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 L2,4 Z" fill="#333" />
          </marker>
        </defs>

        <!-- 1D DOS -->
        <g transform="translate(20,40)">
          <text x="85" y="10" text-anchor="middle" font-size="15" font-weight="bold" fill="#2c3e50">1D: g(E) ~ 1/sqrt(E)</text>
          <!-- Axes -->
          <line x1="30" y1="180" x2="160" y2="180" stroke="#333" stroke-width="2" marker-end="url(#dos-arrow)" />
          <line x1="30" y1="185" x2="30" y2="30" stroke="#333" stroke-width="2" marker-end="url(#dos-arrow)" />
          <text x="165" y="185" font-size="13" font-weight="bold" fill="#333">E</text>
          <text x="15" y="105" font-size="13" font-weight="bold" fill="#333" transform="rotate(-90 15 105)">g(E)</text>

          <!-- 1D DOS curve: 1/sqrt(E) shape with Van Hove singularities -->
          <path d="M 40 180
                   L 40 40
                   Q 50 60, 60 100
                   Q 70 130, 85 150
                   Q 100 165, 120 172
                   Q 140 178, 155 180"
                fill="#3498db" opacity="0.35" stroke="none" />
          <path d="M 40 40
                   Q 50 60, 60 100
                   Q 70 130, 85 150
                   Q 100 165, 120 172
                   Q 140 178, 155 180"
                fill="none" stroke="#3498db" stroke-width="3" />

          <!-- Van Hove singularity annotation -->
          <text x="50" y="60" font-size="11" fill="#c0392b" font-weight="bold">Van Hove</text>
          <text x="50" y="73" font-size="11" fill="#c0392b" font-weight="bold">singularity</text>
          <line x1="40" y1="40" x2="40" y2="75" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="3,2" />

          <!-- Band edge label -->
          <text x="40" y="198" text-anchor="middle" font-size="11" fill="#555">E_band</text>
        </g>

        <!-- 2D DOS -->
        <g transform="translate(220,40)">
          <text x="85" y="10" text-anchor="middle" font-size="15" font-weight="bold" fill="#2c3e50">2D: g(E) = const</text>
          <!-- Axes -->
          <line x1="30" y1="180" x2="160" y2="180" stroke="#333" stroke-width="2" marker-end="url(#dos-arrow)" />
          <line x1="30" y1="185" x2="30" y2="30" stroke="#333" stroke-width="2" marker-end="url(#dos-arrow)" />
          <text x="165" y="185" font-size="13" font-weight="bold" fill="#333">E</text>
          <text x="15" y="105" font-size="13" font-weight="bold" fill="#333" transform="rotate(-90 15 105)">g(E)</text>

          <!-- 2D DOS: step function (constant for each band) -->
          <path d="M 40 180
                   L 40 120
                   L 80 120
                   L 80 80
                   L 120 80
                   L 120 180"
                fill="#e74c3c" opacity="0.35" stroke="none" />
          <path d="M 40 120
                   L 80 120
                   L 80 80
                   L 120 80
                   L 120 180"
                fill="none" stroke="#e74c3c" stroke-width="3" />
          <line x1="40" y1="180" x2="40" y2="120" stroke="#e74c3c" stroke-width="3" />

          <!-- Step labels -->
          <text x="58" y="115" text-anchor="middle" font-size="11" fill="#333">Band 1</text>
          <text x="98" y="75" text-anchor="middle" font-size="11" fill="#333">Band 2</text>

          <!-- Constant annotation -->
          <line x1="145" y1="120" x2="145" y2="80" stroke="#27ae60" stroke-width="1.5" />
          <text x="150" y="105" font-size="10" fill="#27ae60" font-weight="bold">Steps</text>
        </g>

        <!-- 3D DOS -->
        <g transform="translate(420,40)">
          <text x="85" y="10" text-anchor="middle" font-size="15" font-weight="bold" fill="#2c3e50">3D: g(E) ~ sqrt(E)</text>
          <!-- Axes -->
          <line x1="30" y1="180" x2="160" y2="180" stroke="#333" stroke-width="2" marker-end="url(#dos-arrow)" />
          <line x1="30" y1="185" x2="30" y2="30" stroke="#333" stroke-width="2" marker-end="url(#dos-arrow)" />
          <text x="165" y="185" font-size="13" font-weight="bold" fill="#333">E</text>
          <text x="15" y="105" font-size="13" font-weight="bold" fill="#333" transform="rotate(-90 15 105)">g(E)</text>

          <!-- 3D DOS: sqrt(E) parabolic shape -->
          <path d="M 40 180
                   Q 50 178, 60 170
                   Q 80 150, 100 120
                   Q 120 85, 140 50
                   L 140 180 Z"
                fill="#27ae60" opacity="0.35" stroke="none" />
          <path d="M 40 180
                   Q 50 178, 60 170
                   Q 80 150, 100 120
                   Q 120 85, 140 50"
                fill="none" stroke="#27ae60" stroke-width="3" />

          <!-- sqrt(E) annotation -->
          <text x="115" y="70" font-size="12" fill="#1e8449" font-weight="bold">~sqrt(E)</text>

          <!-- Band edge -->
          <text x="40" y="198" text-anchor="middle" font-size="11" fill="#555">E=0</text>
        </g>

        <!-- Common legend/note -->
        <rect x="200" y="235" width="250" height="35" fill="white" stroke="#ddd" stroke-width="1" rx="4" />
        <text x="325" y="255" text-anchor="middle" font-size="12" font-weight="bold" fill="#555">Free electron model: g(E) ~ E^((d-2)/2)</text>
        <text x="325" y="268" text-anchor="middle" font-size="11" fill="#777">d = dimension (1D, 2D, 3D)</text>
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