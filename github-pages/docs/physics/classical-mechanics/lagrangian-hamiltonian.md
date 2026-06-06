---
layout: docs
title: "Classical Mechanics: Lagrangian & Hamiltonian Mechanics"
permalink: /docs/physics/classical-mechanics/lagrangian-hamiltonian.html
toc: true
toc_sticky: true
---

[Classical Mechanics](./)

The principle of least action, generalized coordinates, phase space, Poisson brackets, and Hamilton-Jacobi theory.

## Lagrangian Mechanics: A New Perspective

### Beyond Forces: Why We Need Lagrangian Mechanics

Analyzing a bead sliding on a curved wire with Newton's laws means constantly tracking the constraint force from the wire. Lagrange's reformulation sidesteps this entirely: instead of forces, it focuses on energy and the *path* a system takes through its possible configurations, so constraint forces never need to be computed.

### The Principle of Least Action

Among all possible paths between two points, a system follows the path that extremizes (usually minimizes) a quantity called the **action**:

**Action:**

$$
S = \int_{t_1}^{t_2} L(q, \dot{q}, t) \, dt
$$

Where $L = T - V$ is the Lagrangian (kinetic minus potential energy).

This single principle replaces F = ma and automatically handles constraints.

**Euler-Lagrange Equations:**
$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = 0
$$

These equations provide a powerful alternative to Newton's laws. No constraint forces appear—they're automatically accounted for by choosing appropriate coordinates.

**Worked example — the simple pendulum the easy way.** A mass $m$ swings on a rigid rod of length $\ell$. With Newton you would resolve the tension and gravity along the arc — bookkeeping that involves a force (the tension) you do not even care about. Lagrange asks only for the energies, written in the single natural coordinate, the angle $\theta$:

$$L = T - V = \tfrac{1}{2}m\ell^2\dot{\theta}^2 - mg\ell(1 - \cos\theta)$$

The two pieces of the Euler-Lagrange equation are

$$\frac{\partial L}{\partial \dot{\theta}} = m\ell^2\dot{\theta}, \qquad \frac{\partial L}{\partial \theta} = -mg\ell\sin\theta,$$

so $\frac{d}{dt}(m\ell^2\dot{\theta}) - (-mg\ell\sin\theta) = 0$, which collapses to the familiar pendulum equation:

$$\ddot{\theta} + \frac{g}{\ell}\sin\theta = 0.$$

The tension never appeared — the rigid-rod constraint was absorbed the instant we chose $\theta$ as the coordinate. For small swings, $\sin\theta \approx \theta$ gives simple harmonic motion with $\omega = \sqrt{g/\ell}$, recovering the earlier oscillations result from a completely different starting point.

### Generalized Coordinates: Freedom from Cartesian Tyranny

The formulation is not restricted to $x, y, z$ — any set of coordinates that pins down the configuration will do, and the natural choice usually makes the problem simple:
- Pendulum? Use the angle θ instead of (x, y)
- Particle on a sphere? Use spherical coordinates (θ, φ)
- Double pendulum? Use two angles (θ₁, θ₂)

For a system with n degrees of freedom:
- **Configuration space:** Q = {q₁, q₂, ..., qₙ} - all possible positions
- **Generalized momentum:** pᵢ = ∂L/∂q̇ᵢ - not always mass × velocity!
- **Phase space:** Γ = {q₁, ..., qₙ, p₁, ..., pₙ} - positions and momenta together

### Constraints and Lagrange Multipliers

**Holonomic constraints:** f(q₁, ..., qₙ, t) = 0

**Modified Lagrangian with constraints:**
$$
L' = L + \sum_i \lambda_i f_i(q, t)
$$

### Noether's Theorem: The Deep Connection

Emmy Noether's theorem states that every continuous symmetry of the action corresponds to a conservation law:

**The Symmetry-Conservation Dictionary:**

| Symmetry | Conservation Law |
|----------|------------------|
| Time translation | Energy |
| Space translation | Linear momentum |
| Rotation | Angular momentum |
| Gauge invariance | Charge |

**Mathematical Statement:**
If the action is invariant under $q_i \to q_i + \varepsilon\xi_i(q)$, then:
$$
Q = \sum_i \frac{\partial L}{\partial \dot{q}_i}\xi_i = \text{constant}
$$

### Example: Double Pendulum - Where Newton Struggles

The double pendulum illustrates the payoff. Newton's approach would require tracking the tension forces in both rods, which constantly change direction. Lagrange needs only the energy:

```python
def double_pendulum_lagrangian(theta1, theta2, theta1_dot, theta2_dot, 
                                m1, m2, l1, l2, g):
    """Lagrangian for double pendulum - no constraint forces needed!"""
    # Kinetic energy - just from velocities
    T = 0.5*m1*(l1*theta1_dot)**2 + 0.5*m2*((l1*theta1_dot)**2 + 
        (l2*theta2_dot)**2 + 2*l1*l2*theta1_dot*theta2_dot*
        np.cos(theta1-theta2))
    
    # Potential energy - just from positions
    V = -m1*g*l1*np.cos(theta1) - m2*g*(l1*np.cos(theta1) + 
        l2*np.cos(theta2))
    
    return T - V  # Equations of motion follow from the Euler-Lagrange equations
```

### The Limitation of Lagrangian Mechanics

Lagrangian mechanics still thinks in terms of trajectories through configuration space. To reason more abstractly about the *state* of a system, the geometry of all possible motions, or the bridge to quantum mechanics, Hamilton reformulated mechanics once more — into the framework that remains central to theoretical physics today.

### The Three Formulations at a Glance

All three describe the *same physics* — they predict identical motion — but each takes a different starting point and excels at different problems. (For the full comparison and a diagram of how they relate, see the [Classical Mechanics hub](./).)

| Aspect | Newtonian | Lagrangian | Hamiltonian |
|--------|-----------|------------|-------------|
| Central quantity | Force $\vec{F}$ | Lagrangian $L = T - V$ | Hamiltonian $H = T + V$ |
| Variables | Positions, accelerations | Generalized coordinates $q_i, \dot{q}_i$ | Coordinates and momenta $q_i, p_i$ |
| Core equation | $\vec{F} = m\vec{a}$ | $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0$ | $\dot{q}_i = \frac{\partial H}{\partial p_i},\ \dot{p}_i = -\frac{\partial H}{\partial q_i}$ |
| Handles constraints | Awkwardly (constraint forces) | Naturally (pick smart coordinates) | Naturally |
| Best for | Direct force problems, intuition | Complex/constrained systems, symmetries | Phase-space geometry, chaos, the bridge to QM |
| Mathematical home | Vectors in space | Configuration space (tangent bundle) | Phase space (cotangent bundle) |

## Hamiltonian Mechanics: The Ultimate Abstraction

### From Configuration Space to Phase Space

Hamilton's key step was to treat positions and *momenta* as the independent variables, rather than positions and velocities. This apparently minor change reshapes the whole perspective. In phase space:
- The evolution of a system becomes a flow
- Conservation laws become geometric properties
- The connection to quantum mechanics becomes clear

### Hamilton's Equations: Beautiful Symmetry

The Hamiltonian H typically represents total energy:
$$
H(q, p, t) = \sum_i p_i\dot{q}_i - L(q, \dot{q}, t)
$$

**Hamilton's Canonical Equations:**
$$
\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

### Phase Space: Where Physics Becomes Geometry

In phase space, a system's state is a single point. As time evolves, this point traces a path. But here's the beautiful part: phase space has structure.

**Liouville's Theorem:** Phase space volume is preserved by time evolution
$$
\frac{d\rho}{dt} + \{\rho, H\} = 0
$$

This means if you start with a blob of initial conditions, the blob might deform and stretch, but its volume never changes. It's like incompressible fluid flow—a deep connection between mechanics and fluid dynamics!

### Poisson Brackets: The Language of Classical Mechanics

Poisson brackets are to classical mechanics what commutators are to quantum mechanics. They encode how quantities change with time and relate to each other:

$$
\{f, g\} = \sum_i \left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)
$$

**Properties:**
- Antisymmetry: {f, g} = -{g, f}
- Linearity: {af + bg, h} = a{f, h} + b{g, h}
- Jacobi identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0

**Time evolution:** df/dt = {f, H} + ∂f/∂t

This last equation is the Hamiltonian counterpart of Newton's second law: it says *every* observable evolves by taking its Poisson bracket with the Hamiltonian. The fundamental brackets of the coordinates and momenta, $\{q_i, p_j\} = \delta_{ij}$, encode the entire structure of phase space. The parallel with quantum mechanics is exact and is the cleanest statement of the classical-quantum correspondence:

| Concept | Classical (Poisson) | Quantum (commutator) |
|---------|---------------------|----------------------|
| Fundamental relation | $\{q, p\} = 1$ | $[\hat{q}, \hat{p}] = i\hbar$ |
| Time evolution | $\dfrac{df}{dt} = \{f, H\}$ | $\dfrac{d\hat{A}}{dt} = \dfrac{i}{\hbar}[\hat{H}, \hat{A}]$ |
| Conserved quantity | $\{f, H\} = 0$ | $[\hat{f}, \hat{H}] = 0$ |
| Correspondence | $\{f, g\}$ | $\dfrac{1}{i\hbar}[\hat{f}, \hat{g}]$ |

Dirac noticed this structural identity in 1925 and used it to build canonical quantization: replace Poisson brackets by commutators divided by $i\hbar$, and classical mechanics becomes quantum mechanics. The bridge is built into the formalism itself.

### Canonical Transformations

Transformations preserving Hamilton's equations:
$$
Q = Q(q, p, t), \quad P = P(q, p, t)
$$

**Generating functions:**
- Type 1: $F_1(q, Q, t) \rightarrow p = \partial F_1/\partial q$, $P = -\partial F_1/\partial Q$
- Type 2: $F_2(q, P, t) \rightarrow p = \partial F_2/\partial q$, $Q = \partial F_2/\partial P$
- Type 3: $F_3(p, Q, t) \rightarrow q = -\partial F_3/\partial p$, $P = -\partial F_3/\partial Q$
- Type 4: $F_4(p, P, t) \rightarrow q = -\partial F_4/\partial p$, $Q = \partial F_4/\partial P$

**Symplectic structure:** Canonical transformations preserve the 2-form:
$$
\omega = \sum_i dp_i \wedge dq_i = \sum_i dP_i \wedge dQ_i
$$

### Action-Angle Variables

For integrable systems, transform to (I, θ) where:
- **Action variables:** Iᵢ = ∮ pᵢ dqᵢ/(2π)
- **Angle variables:** θᵢ with period 2π
- **Hamilton's equations:** İᵢ = 0, θ̇ᵢ = ωᵢ(I) = ∂H/∂Iᵢ

## Hamilton-Jacobi Theory: The Final Synthesis

### Unifying Particles and Waves

The Hamilton-Jacobi equation represents the culmination of classical mechanics. It reformulates mechanics in terms of a single function S (the action), which satisfies a partial differential equation. This might seem like unnecessary abstraction, but it reveals something profound: classical mechanics has a wave-like character!

### Hamilton-Jacobi Equation

The action S(q, t) satisfies:
$$
\frac{\partial S}{\partial t} + H\left(q, \frac{\partial S}{\partial q}, t\right) = 0
$$

**Complete solution:** S = S(q, α, t) + const

Where α are n constants of integration.

**New momenta:** Pᵢ = αᵢ = constant
**New coordinates:** Qᵢ = ∂S/∂αᵢ = βᵢ + ωᵢt

### Separation of Variables

For separable systems:
$$
S = W(q) - Et
$$

**Time-independent HJ equation:**
$$
H\left(q, \frac{\partial W}{\partial q}\right) = E
$$

### The Bridge to Quantum Mechanics

Here's where Hamilton's genius shines. The mathematical structure of classical mechanics—Poisson brackets, phase space, canonical transformations—carries over almost unchanged to quantum mechanics. Just replace:

- Poisson brackets {A, B} → Commutators [Â, B̂]/(iℏ)
- Phase space points → Wave functions
- Trajectories → Probability amplitudes

In the classical limit (ℏ → 0):
$$
\psi = \exp(iS/\hbar)
$$

The quantum wave function reduces to classical action! This deep connection shows that classical and quantum mechanics aren't separate theories—they're different aspects of a unified framework.

---

## Continue

| Previous | Next |
|----------|------|
| [← Newtonian Mechanics &amp; Conservation Laws](newtonian.html) | [Chaos, Modern Topics &amp; Computation →](chaos-and-computational.html) |

### See Also

- [Newtonian Mechanics &amp; Conservation Laws](newtonian.html) — the force-based foundation these formulations reorganize.
- [Chaos, Modern Topics &amp; Computation](chaos-and-computational.html) — phase-space geometry, chaos, and symplectic integrators built on the Hamiltonian view.
- [Quantum Mechanics](../quantum-mechanics/) — where the Poisson-bracket structure becomes the commutator algebra.
