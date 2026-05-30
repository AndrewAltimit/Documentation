---
layout: docs
title: Condensed Matter Physics
permalink: /docs/physics/condensed-matter/
toc: false
hide_title: true
---

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Condensed Matter Physics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Exploring the Quantum World of Materials</p>
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

<div class="principle-card">
  <h4>The big idea: more is different</h4>
  <p>You could in principle know everything about a single electron and a single proton and still have no way to predict that $10^{23}$ of them, packed into a crystal, will conduct electricity, turn magnetic, or carry current with zero resistance. Condensed matter physics is built on this insight — Philip Anderson's "more is different": when enormous numbers of simple parts interact, qualitatively new <strong>emergent</strong> phenomena appear that exist only collectively. A superconductor's resistanceless flow, a magnet's spontaneous alignment, the rigidity of a solid — none of these are properties of the constituent particles; they are properties of the <em>organization</em>. The recurring strategy on this page is to start from the periodic arrangement of atoms (the crystal lattice), see how it reshapes the allowed electron energies (band theory), and then watch genuinely new states of matter emerge when interactions and topology take over.</p>
</div>

## Explore Condensed Matter

<div class="command-grid">
  <a href="emergent-phases.html" class="nav-card">
    <h4><i class="fas fa-snowflake"></i> Superconductivity, Quantum Hall &amp; Topological Phases</h4>
    <p>Superconductivity (Ginzburg-Landau, BCS, Josephson), the integer and fractional quantum Hall effects, topological insulators and Chern insulators, strongly correlated systems, and soft condensed matter.</p>
  </a>
  <a href="advanced-formalism.html" class="nav-card">
    <h4><i class="fas fa-superscript"></i> Graduate-Level Formalism &amp; Experiment</h4>
    <p>Experimental techniques and research frontiers, plus the full mathematical formalism: second quantization, Green's functions, advanced band theory, Bogoliubov-de Gennes, Chern-Simons, DMFT, and tensor networks.</p>
  </a>
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

Band theory delivers a strikingly simple classification of solids: it is not *how many* electrons a material has that decides whether it conducts, but *where the Fermi level sits relative to the band gap*. A partly filled band conducts (a metal); a filled band separated from the next empty band by a large gap insulates; a filled band separated by a *small* gap is a semiconductor — an insulator that thermal energy or doping can switch on. This single distinction underpins the entire electronics industry.

| Class | Band filling | Gap $E_g$ | Conductivity vs. $T$ |
|-------|--------------|-----------|----------------------|
| Metal | Partly filled band | none (bands overlap) | decreases with $T$ |
| Semiconductor | Filled valence band | small ($\sim 0.1$–$2$ eV) | increases with $T$ |
| Insulator | Filled valence band | large ($\gtrsim 4$ eV) | negligible |

<div class="insight-card">
  <h4>Why semiconductors heat up into conductors</h4>
  <p>In a metal, conductivity falls as temperature rises because lattice vibrations scatter the already-mobile electrons. A semiconductor does the opposite: its valence band is full and the conduction band empty, so it can only conduct once electrons are thermally promoted across the gap. The number of carriers grows exponentially as $e^{-E_g/2k_BT}$, swamping the scattering effect. That exponential sensitivity is exactly what makes a semiconductor a controllable switch.</p>
</div>

### Band Structure
The two band edges that matter are the **valence band maximum (VBM)** and the **conduction band minimum (CBM)**. Their relative position in momentum space sets the optical behavior:

- **Direct gap** — VBM and CBM lie at the same $\mathbf{k}$. An electron can cross the gap by absorbing or emitting a single photon, so direct-gap materials (e.g. GaAs) make efficient LEDs and lasers.
- **Indirect gap** — VBM and CBM lie at different $\mathbf{k}$. A photon alone cannot conserve momentum, so a phonon must assist; this makes silicon a poor light emitter despite being the workhorse of electronics.

### Carrier Statistics
For an intrinsic (undoped) semiconductor, electrons and holes are created in pairs, and their equilibrium concentration is set by the Boltzmann factor for crossing the gap:

$$n_i = \sqrt{N_c N_v}\, e^{-E_g/2k_BT}$$

where $N_c$ and $N_v$ are the effective densities of states in the conduction and valence bands. The factor of $2$ in the exponent reflects that each promoted electron leaves a hole behind, so the carriers are shared between the two bands.

### Doping
Pure semiconductors carry too few intrinsic carriers to be useful. **Doping** — substituting a few-parts-per-million of a foreign atom — overwhelms the intrinsic population with carriers of one chosen sign:

- **n-type**: donor atoms (e.g. phosphorus in silicon) contribute extra electrons to the conduction band.
- **p-type**: acceptor atoms (e.g. boron) accept electrons, leaving mobile holes in the valence band.

Even when doped, the product of electron and hole concentrations is pinned by the **mass-action law**, $np = n_i^2$ — adding majority carriers necessarily suppresses minority carriers.

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

Our picture of a metal evolved in three stages, each fixing a failure of the last: a classical gas of electrons (Drude), then a quantum gas obeying the Pauli principle (Sommerfeld), then an interacting fluid of "dressed" electrons (Landau). Each step kept the successes of the previous one while resolving a glaring discrepancy with experiment.

### Drude Model
The earliest picture treats conduction electrons as a classical gas of point particles bouncing off the ions, with a mean time $\tau$ between collisions. It gets the form of Ohm's law right:

$$\sigma = \frac{ne^2\tau}{m}, \qquad R_H = -\frac{1}{ne}.$$

The Hall coefficient $R_H$ even gives the carrier density and sign correctly for simple metals. But Drude predicts a huge electronic contribution to the heat capacity that experiments flatly contradict — the model's fatal flaw.

### Sommerfeld Model
The fix is quantum statistics. Electrons are fermions, so they fill states up to the **Fermi energy** rather than sharing energy classically:

$$E_F = \frac{\hbar^2}{2m}(3\pi^2 n)^{2/3}.$$

Only the thin shell of electrons within $\sim k_B T$ of $E_F$ can be thermally excited — the rest are Pauli-blocked. This is why the electronic heat capacity is small and *linear* in temperature, exactly as measured:

$$C_V = \gamma T, \qquad \gamma = \frac{\pi^2 k_B^2 g(E_F)}{3}.$$

### Fermi Liquid Theory
Real electrons repel each other strongly, so why does the free-electron picture work at all? Landau's answer: interactions "dress" each electron into a **quasiparticle** — an electron carrying a cloud of disturbance in the surrounding sea — that behaves like a free particle with a renormalized effective mass $m^*$. The low-energy excitations remain in one-to-one correspondence with the free gas, which is why Sommerfeld's results survive. Residual interactions are encoded in the Landau parameters $f_{\mathbf{k}\mathbf{k}'}^{\sigma\sigma'}$:

$$\delta E = \sum_{\mathbf{k}\sigma} \epsilon_{\mathbf{k}} n_{\mathbf{k}\sigma} + \frac{1}{2V}\sum_{\mathbf{k}\mathbf{k}'\sigma\sigma'} f_{\mathbf{k}\mathbf{k}'}^{\sigma\sigma'} n_{\mathbf{k}\sigma} n_{\mathbf{k}'\sigma'}.$$

## Magnetism

Magnetism is a purely quantum, purely collective effect — a classical system in thermal equilibrium cannot be magnetic at all (the Bohr–van Leeuwen theorem). The phenomena below differ in *how the atomic moments respond*: independently and weakly (paramagnetism), or by locking into collective order through exchange interactions (ferro- and antiferromagnetism).

| Order | Moment arrangement | Net moment | Hallmark temperature |
|-------|--------------------|------------|----------------------|
| Paramagnet | random, align weakly with field | zero at $H=0$ | none |
| Ferromagnet | parallel | large, spontaneous | Curie $T_C$ |
| Antiferromagnet | alternating up/down | zero (cancels) | Néel $T_N$ |

### Paramagnetism
Independent atomic moments align only weakly with an applied field, and thermal agitation fights that alignment — so the susceptibility falls off as $1/T$ (the **Curie law**):

$$\chi = \frac{C}{T}, \qquad C = \frac{N\mu_0\mu_B^2 g^2 J(J+1)}{3k_B}.$$

In a metal the story is different: only electrons near the Fermi surface can flip, giving a temperature-independent **Pauli paramagnetism**, $\chi = \mu_0\mu_B^2 g(E_F)$.

### Ferromagnetism
When the exchange interaction is strong enough, moments align *spontaneously* even with no applied field. Mean-field theory replaces the exchange coupling with an effective internal field $\lambda M$ proportional to the magnetization itself, giving a self-consistent equation:

$$M = Ng\mu_B J\, B_J\!\left(\frac{g\mu_B J(H + \lambda M)}{k_B T}\right).$$

Below the **Curie temperature** $T_C = g\mu_B J(J+1)\lambda / 3k_B$ this equation has a nonzero solution at $H = 0$ — spontaneous magnetization. Above $T_C$ thermal disorder wins and the material reverts to a paramagnet.

### Antiferromagnetism
Here the exchange favors *anti*-alignment: neighboring moments point opposite ways, so the net magnetization cancels even though the system is ordered. Order sets in below the **Néel temperature** $T_N$, marking the onset of staggered magnetization. A two-sublattice mean-field model gives

$$\chi = \frac{2C}{T + T_N},$$

with the telltale $T + T_N$ in the denominator — the opposite sign to a ferromagnet's $T - T_C$.

### Spin Waves
Just as a crystal's lowest-energy excitations are quantized lattice vibrations (phonons), an ordered magnet's are quantized waves of tilting spins — **magnons**. For a ferromagnet the dispersion is

$$\omega_{\mathbf{k}} = \frac{2JS}{\hbar}(1 - \cos(ka)),$$

which is *quadratic* at long wavelength ($\omega \sim k^2$), in contrast to the linear dispersion of acoustic phonons — a difference that shows up directly in the low-temperature heat capacity.

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>More is different</h4>
    <p>Collective behavior of $\sim 10^{23}$ particles produces emergent phenomena absent at the single-particle level.</p>
  </div>
  <div class="takeaway-card">
    <h4>Band structure governs solids</h4>
    <p>Whether a material is a metal, insulator, or semiconductor follows from how electron bands fill.</p>
  </div>
  <div class="takeaway-card">
    <h4>Quasiparticles simplify the many-body problem</h4>
    <p>Phonons, holes, and Cooper pairs let us treat strongly interacting systems with effective single-particle pictures.</p>
  </div>
  <div class="takeaway-card">
    <h4>Symmetry breaking creates order</h4>
    <p>Magnetism and superconductivity arise when the ground state breaks a symmetry of the Hamiltonian.</p>
  </div>
  <div class="takeaway-card">
    <h4>Topology classifies new phases</h4>
    <p>Topological insulators and the quantum Hall effect are robust against disorder because they are protected by topology.</p>
  </div>
  <div class="takeaway-card">
    <h4>Where quantum tech is built</h4>
    <p>Superconducting qubits, spintronics, and 2D materials make condensed matter the engine room of quantum technology.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="emergent-phases.html">Superconductivity, Quantum Hall &amp; Topological Phases</a> — emergent and topological states of matter.</li>
    <li><a href="advanced-formalism.html">Graduate-Level Formalism &amp; Experiment</a> — many-body theory and experimental probes.</li>
    <li><a href="../quantum-mechanics/">Quantum Mechanics</a> — wave functions and band theory underpin every solid.</li>
    <li><a href="../statistical-mechanics/">Statistical Mechanics</a> — many-body theory and the physics of phase transitions.</li>
    <li><a href="../quantum-field-theory.html">Quantum Field Theory</a> — field-theoretic methods for collective excitations.</li>
    <li><a href="../thermodynamics.html">Thermodynamics</a> — macroscopic properties and phase diagrams of materials.</li>
    <li><a href="../computational-physics/">Computational Physics</a> — DFT, Monte Carlo, and molecular-dynamics simulations.</li>
    <li><a href="../">Physics Hub</a> — browse all physics topics.</li>
  </ul>
</div>
