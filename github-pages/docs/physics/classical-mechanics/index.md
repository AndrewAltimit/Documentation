---
layout: docs
title: Classical Mechanics
permalink: /docs/physics/classical-mechanics/
toc: false
hide_title: true
---

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Classical Mechanics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The foundation of physics describing motion and forces, from Newton's laws to the elegant formulations of Lagrange and Hamilton.</p>
</div>

<div class="intro-card">
  <p class="lead-text">Classical mechanics is the physics of everyday motion — how forces, energy, and momentum dictate the paths of everything from thrown balls to orbiting planets. This topic is built in three layers: first Newton's force-based picture, then the deeper energy-based formulations of Lagrange and Hamilton, and finally the modern, geometric view that connects mechanics to chaos, quantum theory, and computation. Use the cards below to jump to a layer, or read them in order for a guided arc — intuition before formalism.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-rocket"></i>
    <h4>Force changes motion</h4>
    <p>Newton's $F = ma$ — objects keep their velocity unless a net force acts.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-recycle"></i>
    <h4>Symmetry conserves quantities</h4>
    <p>Energy, momentum, and angular momentum are conserved because of underlying symmetries (Noether).</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-route"></i>
    <h4>Action is extremized</h4>
    <p>The Lagrangian view: a system follows the path that extremizes $S = \int L\,dt$.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-random"></i>
    <h4>Determinism has limits</h4>
    <p>Nonlinear systems can be chaotic — perfectly deterministic yet practically unpredictable.</p>
  </div>
</div>

## Explore Classical Mechanics

<div class="command-grid">
  <a href="newtonian.html" class="nav-card">
    <h4><i class="fas fa-rocket"></i> Newtonian Mechanics &amp; Conservation Laws</h4>
    <p>Newton's three laws, kinematics, dynamics, work and energy, conservation laws and Noether, rotational motion, gravitation, and oscillations and waves.</p>
  </a>
  <a href="lagrangian-hamiltonian.html" class="nav-card">
    <h4><i class="fas fa-route"></i> Lagrangian &amp; Hamiltonian Mechanics</h4>
    <p>The principle of least action, Euler-Lagrange equations, generalized coordinates, Hamilton's equations, phase space, Poisson brackets, canonical transformations, and Hamilton-Jacobi theory.</p>
  </a>
  <a href="chaos-and-computational.html" class="nav-card">
    <h4><i class="fas fa-random"></i> Chaos, Modern Topics &amp; Computation</h4>
    <p>Chaos and nonlinear dynamics, KAM theory, symplectic geometry, modern applications, symplectic and variational integrators, and the bridge to quantum and statistical mechanics.</p>
  </a>
</div>

## The Landscape of Classical Mechanics

Classical mechanics is not a single recipe but a family of equivalent viewpoints that grew more abstract and more powerful over three centuries. The map below shows how the three great formulations relate, what mathematical home each lives in, and where each one ultimately points — toward chaos, statistical mechanics, and quantum theory. Keep it in mind as a guide while reading: every section is a stop on this route.

```mermaid
graph TD
    N["Newtonian Mechanics<br/>F = ma<br/>(vectors, forces)"] --> L["Lagrangian Mechanics<br/>L = T - V<br/>(configuration space)"]
    L --> H["Hamiltonian Mechanics<br/>H = T + V<br/>(phase space)"]
    N -.->|same physics| L
    L -.->|same physics| H
    L --> NO["Noether's Theorem<br/>symmetry to conservation"]
    H --> HJ["Hamilton-Jacobi<br/>action as a field"]
    H --> CH["Chaos and<br/>nonlinear dynamics"]
    H --> SM["Statistical<br/>Mechanics"]
    HJ --> QM["Quantum Mechanics<br/>path integral"]
    classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef bridge fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    class N,L,H core;
    class NO,HJ,CH,SM,QM bridge;
```

## The Three Formulations at a Glance

All three describe the *same physics* — they predict identical motion — but each takes a different starting point and excels at different problems.

| Aspect | Newtonian | Lagrangian | Hamiltonian |
|--------|-----------|------------|-------------|
| Central quantity | Force $\vec{F}$ | Lagrangian $L = T - V$ | Hamiltonian $H = T + V$ |
| Variables | Positions, accelerations | Generalized coordinates $q_i, \dot{q}_i$ | Coordinates and momenta $q_i, p_i$ |
| Core equation | $\vec{F} = m\vec{a}$ | $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0$ | $\dot{q}_i = \frac{\partial H}{\partial p_i},\ \dot{p}_i = -\frac{\partial H}{\partial q_i}$ |
| Handles constraints | Awkwardly (constraint forces) | Naturally (pick smart coordinates) | Naturally |
| Best for | Direct force problems, intuition | Complex/constrained systems, symmetries | Phase-space geometry, chaos, the bridge to QM |
| Mathematical home | Vectors in space | Configuration space (tangent bundle) | Phase space (cotangent bundle) |

<div class="tip-card">
  <h4>Which one should you reach for?</h4>
  <p>Use <strong>Newton</strong> when forces are simple and you want physical intuition. Switch to <strong>Lagrange</strong> the moment constraints appear (a bead on a wire, a double pendulum) — choosing the right generalized coordinates makes constraint forces vanish. Move to <strong>Hamilton</strong> when you care about the <em>structure</em> of all possible motions, conserved quantities, statistical mechanics, or the route to quantum theory.</p>
</div>

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Three formulations, one physics</h4>
    <p>Newtonian, Lagrangian, and Hamiltonian mechanics are equivalent — but each makes different problems easy and reveals different structure.</p>
  </div>
  <div class="takeaway-card">
    <h4>Conservation laws come from symmetry</h4>
    <p>Noether's theorem ties time-translation to energy, space-translation to momentum, and rotation to angular momentum.</p>
  </div>
  <div class="takeaway-card">
    <h4>Phase space is the natural arena</h4>
    <p>Hamiltonian dynamics lives in $(q,p)$ phase space, where the geometry (symplectic structure) is preserved by the flow.</p>
  </div>
  <div class="takeaway-card">
    <h4>Action is fundamental</h4>
    <p>The principle of least action underlies all of physics and is the bridge to quantum mechanics via the path integral.</p>
  </div>
  <div class="takeaway-card">
    <h4>Determinism is not predictability</h4>
    <p>Chaotic systems obey exact laws yet diverge exponentially, limiting long-term prediction (the butterfly effect).</p>
  </div>
  <div class="takeaway-card">
    <h4>It is a limiting case</h4>
    <p>Classical mechanics emerges from quantum mechanics ($\hbar \to 0$) and relativity ($v \ll c$); know where it breaks down.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="../quantum-mechanics/">Quantum Mechanics</a> — where classical mechanics meets the microscopic world and emerges as the $\hbar \to 0$ limit.</li>
    <li><a href="../relativity/">Relativity</a> — what replaces Newtonian mechanics when speeds approach $c$ or gravity gets strong.</li>
    <li><a href="../statistical-mechanics/">Statistical Mechanics</a> — bridging Newton's laws for many particles to thermodynamics.</li>
    <li><a href="../thermodynamics.html">Thermodynamics</a> — energy, work, and heat in mechanical systems.</li>
    <li><a href="../computational-physics/">Computational Physics</a> — symplectic integrators and numerical methods for complex mechanical systems.</li>
    <li><a href="../">Physics Hub</a> — browse all physics topics.</li>
  </ul>
</div>
