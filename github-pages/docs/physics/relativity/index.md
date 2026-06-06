---
layout: docs
title: Relativity
permalink: /docs/physics/relativity/
toc: false
hide_title: true
---

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Relativity</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The Unity of Space, Time, and Gravity</p>
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

## Explore Relativity

### Foundations

<div class="command-grid">
  <a href="special-relativity.html" class="nav-card">
    <h4><i class="fas fa-rocket"></i> Special Relativity</h4>
    <p>The two postulates, Lorentz transformations, time dilation, length contraction, $E=mc^2$, relativistic dynamics, and four-vectors.</p>
  </a>
  <a href="general-relativity.html" class="nav-card">
    <h4><i class="fas fa-globe"></i> General Relativity</h4>
    <p>The equivalence principle, the Einstein field equations, the Schwarzschild solution, gravitational time dilation, geodesics, and experimental tests.</p>
  </a>
</div>

### Graduate Topics &amp; Deep Dives

<div class="command-grid">
  <a href="advanced.html" class="nav-card">
    <h4><i class="fas fa-superscript"></i> Graduate Topics Hub</h4>
    <p>Sub-hub for the graduate formalism and frontiers: where each deep-dive below fits, the prerequisites, and the path through them.</p>
  </a>
  <a href="tensor-formalism.html" class="nav-card">
    <h4><i class="fas fa-square-root-alt"></i> Tensor Formalism</h4>
    <p>Tensor calculus, the metric, covariant derivatives, the Riemann and Ricci tensors, and a full derivation of the Einstein field equations.</p>
  </a>
  <a href="black-holes.html" class="nav-card">
    <h4><i class="fas fa-circle"></i> Black Holes</h4>
    <p>Schwarzschild and Kerr geometries, horizons, singularities, the Penrose process, and black-hole thermodynamics.</p>
  </a>
  <a href="cosmology.html" class="nav-card">
    <h4><i class="fas fa-globe-americas"></i> Cosmology</h4>
    <p>The FLRW metric, the Friedmann equations, the expanding universe, the cosmological constant, and the standard ΛCDM model.</p>
  </a>
  <a href="gravitational-waves.html" class="nav-card">
    <h4><i class="fas fa-wave-square"></i> Gravitational Waves</h4>
    <p>Linearized gravity, the transverse-traceless gauge, the quadrupole formula, binary inspirals, and detection with LIGO.</p>
  </a>
  <a href="quantum-gravity.html" class="nav-card">
    <h4><i class="fas fa-atom"></i> Quantum Gravity</h4>
    <p>Why general relativity and quantum theory clash, the information paradox, and the string-theory and loop approaches to a unified theory.</p>
  </a>
</div>

### The Logic of Relativity

Both theories unfold from a single stubborn fact and a single guiding principle, each forcing the next conclusion. The chain below shows how one experimental observation (the constancy of light speed) cascades into the entire structure of special relativity, and how one further principle (equivalence) extends it into general relativity. Read it as the skeleton of this topic.

```mermaid
graph TD
    MM["Light speed c is the same<br/>for every observer"] --> POST["Two postulates of<br/>special relativity"]
    POST --> LT["Lorentz transformations"]
    LT --> TD["Time dilation"]
    LT --> LC["Length contraction"]
    LT --> RS["Relativity of simultaneity"]
    LT --> EMC["Mass-energy equivalence<br/>E = mc^2"]
    TD --> ST["Spacetime as one<br/>four-dimensional arena"]
    LC --> ST
    RS --> ST
    ST --> EP["Equivalence principle:<br/>gravity = acceleration"]
    EMC --> EP
    EP --> CURV["Mass-energy curves spacetime"]
    CURV --> EFE["Einstein field equations"]
    EFE --> PRED["Black holes, lensing,<br/>gravitational waves, cosmology"]
    classDef sr fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef gr fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    class MM,POST,LT,TD,LC,RS,EMC,ST sr;
    class EP,CURV,EFE,PRED gr;
```

### What You'll Find

| Page | What it covers |
|------|----------------|
| [Special Relativity](special-relativity.html) | The two postulates, Lorentz transformations, time dilation, length contraction, $E=mc^2$, four-vectors |
| [General Relativity](general-relativity.html) | The equivalence principle, curvature, the Einstein field equations, Schwarzschild & black holes, predictions and tests |
| [Graduate Topics Hub](advanced.html) | Sub-hub: how the deep-dives below fit together, prerequisites, and a suggested reading path |
| [Tensor Formalism](tensor-formalism.html) | Tensor calculus, the metric, covariant derivatives, the Riemann/Ricci tensors, deriving the field equations |
| [Black Holes](black-holes.html) | Schwarzschild and Kerr geometries, horizons, singularities, the Penrose process, black-hole thermodynamics |
| [Cosmology](cosmology.html) | The FLRW metric, the Friedmann equations, cosmic expansion, the cosmological constant, ΛCDM |
| [Gravitational Waves](gravitational-waves.html) | Linearized gravity, the quadrupole formula, binary inspirals, detection with LIGO |
| [Quantum Gravity](quantum-gravity.html) | Why GR and quantum theory clash, the information paradox, string and loop approaches |

<div class="tip-card">
  <h4>Level and prerequisites</h4>
  <p>The conceptual core — postulates, time dilation, $E=mc^2$, gravity as curvature — needs only algebra and a willingness to abandon "common sense" about absolute time. The Lorentz transformations and four-vectors use a little linear algebra. The graduate formalism (tensor calculus, the Riemann tensor, exact black-hole solutions) is reference material and can be skipped on a first read. Read <a href="special-relativity.html">Special Relativity</a> first; <a href="general-relativity.html">General Relativity</a> assumes it.</p>
</div>

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>The speed of light is absolute</h4>
    <p>$c$ is the same in every inertial frame; simultaneity, length, and time become observer-dependent.</p>
  </div>
  <div class="takeaway-card">
    <h4>Space and time are one</h4>
    <p>Special relativity unifies them into spacetime, with the invariant interval $ds^2$ replacing separate distances and durations.</p>
  </div>
  <div class="takeaway-card">
    <h4>Mass is energy</h4>
    <p>$E = mc^2$ (more generally $E^2 = (pc)^2 + (mc^2)^2$) — rest mass is a reservoir of energy.</p>
  </div>
  <div class="takeaway-card">
    <h4>Gravity is geometry</h4>
    <p>General relativity recasts gravity as the curvature of spacetime: $G_{\mu\nu} = 8\pi G\, T_{\mu\nu}$.</p>
  </div>
  <div class="takeaway-card">
    <h4>Free fall follows geodesics</h4>
    <p>Objects in free fall move along the straightest possible paths through curved spacetime.</p>
  </div>
  <div class="takeaway-card">
    <h4>Confirmed across scales</h4>
    <p>From GPS clock corrections to gravitational waves and black-hole images, relativity passes every test.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="../classical-mechanics/">Classical Mechanics</a> — Newtonian mechanics, recovered in the low-speed, weak-gravity limit.</li>
    <li><a href="../quantum-field-theory.html">Quantum Field Theory</a> — unifying special relativity with quantum mechanics.</li>
    <li><a href="../string-theory/">String Theory</a> — a leading candidate for quantum gravity and extra dimensions.</li>
    <li><a href="../quantum-mechanics/">Quantum Mechanics</a> — the quantum theory that relativity is reconciled with in QFT.</li>
    <li><a href="../computational-physics/">Computational Physics</a> — numerical relativity and gravitational-wave simulations.</li>
    <li><a href="../">Physics Hub</a> — browse all physics topics.</li>
  </ul>
</div>
