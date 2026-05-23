---
layout: docs
title: Physics Documentation Hub
hide_title: true
toc: false  # Index pages typically don't need TOC
---

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Physics Documentation Hub</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The fundamental laws that govern matter, energy, space, and time — from falling apples to the quantum vacuum.</p>
</div>

<div class="intro-card">
  <p class="lead-text">A reference-wiki for physics: rigorous mathematics paired with physical intuition. Each page builds from the core idea (why it matters) through the formalism (the equations) to where it applies. Pick a topic below, or follow a guided path if you are just getting started.</p>
</div>

## Browse by Topic

### Core Physics

The classical pillars — the physics of everyday scales, energy, and the structure of spacetime.

<div class="command-grid">
  <a href="classical-mechanics.html" class="nav-card">
    <h4><i class="fas fa-atom"></i> Classical Mechanics</h4>
    <p>Newton's laws, the Lagrangian and Hamiltonian formulations, conservation laws, and chaos.</p>
  </a>
  <a href="thermodynamics.html" class="nav-card">
    <h4><i class="fas fa-fire"></i> Thermodynamics</h4>
    <p>Heat, work, entropy, and the four laws governing every engine and the fate of the universe.</p>
  </a>
  <a href="statistical-mechanics.html" class="nav-card">
    <h4><i class="fas fa-dice"></i> Statistical Mechanics</h4>
    <p>How microscopic randomness becomes macroscopic law — ensembles, partition functions, phase transitions.</p>
  </a>
  <a href="relativity.html" class="nav-card">
    <h4><i class="fas fa-clock"></i> Relativity</h4>
    <p>Special and general relativity: spacetime, $E=mc^2$, curved geometry, and gravity as geometry.</p>
  </a>
</div>

### Quantum & Advanced

The quantum world and the many-body, high-energy, and theoretical frontiers built on top of it.

<div class="command-grid">
  <a href="quantum-mechanics.html" class="nav-card">
    <h4><i class="fas fa-wave-square"></i> Quantum Mechanics</h4>
    <p>Wave functions, the uncertainty principle, measurement, and the strange logic of the quantum world.</p>
  </a>
  <a href="quantum-field-theory.html" class="nav-card">
    <h4><i class="fas fa-project-diagram"></i> Quantum Field Theory</h4>
    <p>Quantum mechanics + special relativity. Particles as field excitations; the Standard Model.</p>
  </a>
  <a href="condensed-matter.html" class="nav-card">
    <h4><i class="fas fa-cube"></i> Condensed Matter</h4>
    <p>Solids, superconductors, topological materials, and emergent collective phenomena.</p>
  </a>
  <a href="string-theory.html" class="nav-card">
    <h4><i class="fas fa-infinity"></i> String Theory</h4>
    <p>Extra dimensions, dualities, branes, and the quest for a quantum theory of gravity.</p>
  </a>
</div>

### Methods

<div class="command-grid">
  <a href="computational-physics.html" class="nav-card">
    <h4><i class="fas fa-laptop-code"></i> Computational Physics</h4>
    <p>Numerical integration, Monte Carlo, molecular dynamics, PDE solvers, and machine learning for physics.</p>
  </a>
</div>

## How These Topics Connect

Physics is not a list of separate subjects — each field grows out of and feeds back into the others. The map below traces the main lines of descent.

```mermaid
graph TD
    CM[Classical Mechanics]
    TH[Thermodynamics]
    SR[Special Relativity]
    GR[General Relativity]
    SM[Statistical Mechanics]
    QM[Quantum Mechanics]
    QFT[Quantum Field Theory]
    CMP[Condensed Matter]
    ST[String Theory]
    COMP[Computational Physics]

    CM --> SM
    TH --> SM
    CM --> SR
    SR --> GR
    CM --> QM
    QM --> QFT
    SR --> QFT
    SM --> CMP
    QM --> CMP
    QFT --> CMP
    QFT --> ST
    GR --> ST
    COMP -.-> CM
    COMP -.-> QM
    COMP -.-> SM

    style CM fill:#11998e,color:#fff
    style QM fill:#11998e,color:#fff
    style ST fill:#38ef7d,color:#222
    style COMP fill:#ccf,color:#222
```

<p style="text-align:center; font-style:italic; opacity:0.8;">Solid arrows: one theory is built on or reduces to another. Dashed arrows: computational methods support every field.</p>

## Guided Paths

<div class="command-grid">
  <div class="step-card">
    <h4>New to physics</h4>
    <p>Start with <a href="classical-mechanics.html">Classical Mechanics</a> to build intuition for force, energy, and motion before anything else.</p>
  </div>
  <div class="step-card">
    <h4>Undergraduate sequence</h4>
    <p>Classical Mechanics → <a href="quantum-mechanics.html">Quantum Mechanics</a> → <a href="thermodynamics.html">Thermodynamics</a> → <a href="statistical-mechanics.html">Statistical Mechanics</a> → <a href="relativity.html">Relativity</a>.</p>
  </div>
  <div class="step-card">
    <h4>Graduate / research</h4>
    <p>Dive into <a href="quantum-field-theory.html">QFT</a>, <a href="condensed-matter.html">Condensed Matter</a>, or <a href="string-theory.html">String Theory</a>, and use <a href="computational-physics.html">Computational Physics</a> as a toolbox.</p>
  </div>
</div>

## Related Resources

<div class="see-also-card">
  <h4>Cross-Disciplinary Links</h4>
  <ul>
    <li><a href="../technology/quantumcomputing.html">Quantum Computing</a> — where quantum mechanics meets information processing.</li>
    <li><a href="../advanced/quantum-algorithms-research/">Quantum Algorithms Research</a> — advanced quantum information and algorithms.</li>
    <li><a href="../advanced/ai-mathematics/">AI Mathematics</a> — statistical-mechanics connections to machine learning.</li>
    <li><a href="../advanced/">Advanced Research Topics</a> — graduate-level physics and mathematics.</li>
    <li><a href="../reference/#physics-formulas--constants">Physics Reference</a> — CODATA constants, key equations, and unit conversions.</li>
  </ul>
</div>

---

*This physics documentation combines rigorous mathematical treatment with intuitive explanations. For corrections or suggestions, please visit our [GitHub repository](https://github.com/AndrewAltimit/Documentation).*
