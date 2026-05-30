---
layout: docs
title: "Classical Mechanics: Newtonian Mechanics & Conservation Laws"
permalink: /docs/physics/classical-mechanics/newtonian.html
toc: true
toc_sticky: true
hide_title: true
---

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Newtonian Mechanics &amp; Conservation Laws</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Newton's three laws, kinematics and dynamics, work and energy, conservation laws, rotation, gravitation, and oscillations.</p>
</div>

[Classical Mechanics](./)

## The Foundation: Newton's Revolution

Before Newton, motion was a mystery. Objects moved, but why? Newton's genius was recognizing that motion follows precise mathematical laws. This section builds the force-based picture from the ground up.

### Newton's Laws of Motion
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://www.gutenberg.org/files/33229/33229-pdf.pdf"> Paper: <b><i>Philosophiæ Naturalis Principia Mathematica</i></b> - Isaac Newton</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/play-btn-fill.svg" class="icon"><a href="https://www.youtube.com/watch?v=kKKM8Y-u7ds"> Video: <b><i>Newton's Laws of Motion Explained</i></b></a></p>

The foundation of classical mechanics is built upon Newton's three laws of motion:

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion"> Article: <b><i>Newton's Laws of Motion - Wikipedia</i></b></a></p>

#### First Law (Law of Inertia)
An object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.

**Mathematical Expression:**
$$\text{If } \sum \vec{F} = 0, \text{ then } \vec{v} = \text{constant}$$

#### Second Law (Law of Acceleration)


The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.

**Mathematical Expression:**
$$\vec{F} = m\vec{a}$$
Where:
- $\vec{F}$ = net force (N)
- $m$ = mass (kg)
- $\vec{a}$ = acceleration (m/s²)

#### Third Law (Action-Reaction)
For every action, there is an equal and opposite reaction.

**Mathematical Expression:**
$$\vec{F}_{12} = -\vec{F}_{21}$$

### Why Newton's Laws Matter

These three simple laws explain an incredible range of phenomena:
- Why you lurch forward when a car brakes suddenly (First Law)
- How rockets work in the vacuum of space (Third Law)
- Why heavier objects don't fall faster (Second Law with gravity)

But as powerful as Newton's laws are, they have a limitation: you need to know all the forces. What if the forces are complicated? What if you don't care about forces at all, just motion? This is where the analytical formulations become useful.

### Conservation Laws: Nature's Hidden Symmetries
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.feynmanlectures.caltech.edu/I_04.html"> Lecture: <b><i>Conservation of Energy - Feynman Lectures</i></b></a></p>

Here's where physics gets elegant. Instead of tracking forces at every instant, we can use conservation laws—quantities that remain constant throughout motion. It's like having a bank account that never changes its total balance, even as money moves between checking and savings.

These conservation laws aren't arbitrary; they reflect deep symmetries in nature:

<details>
<summary><b>Click to expand: Interactive Conservation Laws Demonstration</b></summary>
<br>
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 5px;">
<p>Conservation laws are fundamental principles that remain constant in isolated systems:</p>
<ul>
<li><b>Energy Conservation:</b> Total energy (kinetic + potential) remains constant</li>
<li><b>Momentum Conservation:</b> Total momentum is conserved in collisions</li>
<li><b>Angular Momentum Conservation:</b> Rotational momentum is preserved</li>
</ul>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/play-btn-fill.svg" class="icon"><a href="https://phet.colorado.edu/en/simulation/energy-skate-park"> Interactive: <b><i>Energy Skate Park Simulation</i></b></a></p>
</div>
</details>

#### Conservation of Energy
The total energy of an isolated system remains constant over time.

**Types of Energy:**
- **Kinetic Energy**: $KE = \tfrac{1}{2}mv^2$
- **Potential Energy**: $PE = mgh$ (gravitational)
- **Elastic Potential Energy**: $PE = \tfrac{1}{2}kx^2$

**Conservation Equation:**
$$\begin{aligned}
E_{\text{initial}} &= E_{\text{final}} \\
KE_i + PE_i &= KE_f + PE_f
\end{aligned}$$

#### Conservation of Momentum

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.physicsclassroom.com/class/momentum/Lesson-2/Momentum-Conservation-Principle"> Tutorial: <b><i>Momentum Conservation in Collisions</i></b></a></p>

The total momentum of an isolated system remains constant.

**Linear Momentum:**
$$\vec{p} = m\vec{v}$$

**Conservation Equation:**
$$\sum \vec{p}_{\text{initial}} = \sum \vec{p}_{\text{final}}$$

#### Conservation of Angular Momentum
The total angular momentum of an isolated system remains constant.

**Angular Momentum:**
$$\vec{L} = I\vec{\omega} = \vec{r} \times \vec{p}$$

Where:
- $I$ = moment of inertia
- $\vec{\omega}$ = angular velocity
- $\vec{r}$ = position vector
- $\vec{p}$ = linear momentum

These conservation laws are not coincidences; each reflects a symmetry of nature, a connection made precise by Noether's theorem in the [Lagrangian and Hamiltonian](lagrangian-hamiltonian.html) formulation.

### The Power of Conservation

Conservation laws often provide shortcuts to solving problems. Consider a figure skater spinning: when she pulls her arms in, she spins faster. Why? Angular momentum (L = Iω) must stay constant. Decreasing I (moment of inertia) means ω (angular velocity) must increase. No forces needed—just conservation!

But even conservation laws have limitations when dealing with constraints and complex systems. This motivates the analytical formulations of Lagrange and Hamilton.

## Kinematics: Describing Motion

### From Observation to Mathematics

Now that we understand forces through Newton's laws, let's step back and look at motion itself. How do we describe where things are and how fast they're moving? This is kinematics—the language of motion.

Kinematics describes motion without asking "why?"—that's dynamics. It's like studying the choreography of a dance without worrying about the muscles involved.

### One-Dimensional Motion

**Position:** $x(t)$

**Velocity:** $v = \dfrac{dx}{dt}$

**Acceleration:** $a = \dfrac{dv}{dt} = \dfrac{d^2x}{dt^2}$

### Equations of Motion (Constant Acceleration)
1. $v = v_0 + at$
2. $x = x_0 + v_0 t + \tfrac{1}{2}at^2$
3. $v^2 = v_0^2 + 2a(x - x_0)$
4. $x = x_0 + \tfrac{1}{2}(v + v_0)t$

### Projectile Motion

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://phet.colorado.edu/en/simulations/projectile-motion"> Interactive: <b><i>Projectile Motion Simulator</i></b> - PhET</a></p>

For projectile motion under constant gravitational acceleration:

**Horizontal Motion:**
$$\begin{aligned}
x &= v_{0x}\,t \\
v_x &= v_{0x} \quad (\text{constant})
\end{aligned}$$

**Vertical Motion:**
$$\begin{aligned}
y &= v_{0y}\,t - \tfrac{1}{2}gt^2 \\
v_y &= v_{0y} - gt
\end{aligned}$$

**Range Formula:**
$$R = \frac{v_0^2 \sin(2\theta)}{g}$$

**Maximum Height:**
$$H = \frac{v_0^2 \sin^2\theta}{2g}$$

## Dynamics: Forces in Action

### Connecting Force to Motion

While kinematics describes motion, dynamics explains it. This is where Newton's second law (F = ma) becomes our primary tool. But as we'll see, thinking in terms of energy often provides a more elegant approach.

Dynamics bridges the gap between abstract forces and observable motion:

### Work and Energy

**Work:** The energy transferred to or from an object via the application of force along a displacement.

$$W = \vec{F}\cdot\vec{d} = Fd\cos\theta$$

For variable force:
$$W = \int \vec{F}\cdot d\vec{r}$$

### Power

Power is the rate at which work is done:

$$P = \frac{dW}{dt} = \vec{F}\cdot\vec{v}$$

### Simple Harmonic Motion
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/matplotlib/matplotlib/blob/main/galleries/examples/animation/simple_anim.py"> Code: <b><i>SHM Animation with Matplotlib</i></b></a></p>


Objects that experience a restoring force proportional to displacement exhibit simple harmonic motion.

**Hooke's Law:**
$$F = -kx$$

**Equation of Motion:**
$$x(t) = A\cos(\omega t + \varphi)$$

Where:
- $A$ = amplitude
- $\omega$ = angular frequency = $\sqrt{k/m}$
- $\varphi$ = phase constant

**Period:**
$$T = \frac{2\pi}{\omega} = 2\pi\sqrt{\frac{m}{k}}$$

## Rotational Motion: Beyond Point Particles

### Why Rotation Matters

So far, we've treated objects as point particles. But real objects have size and shape. They can spin, roll, and tumble. This adds richness to mechanics—and complexity.

The beautiful discovery: rotational motion follows the same patterns as linear motion, just with different quantities:

<details>
<summary><b>Mermaid Diagram: Linear vs Rotational Motion Analogy</b></summary>
<br>

```mermaid
graph LR
    subgraph Linear Motion
        A[Position x] --> B[Velocity v = dx/dt]
        B --> C[Acceleration a = dv/dt]
        D[Mass m] --> E[Force F = ma]
        F[Momentum p = mv]
    end
    
    subgraph Rotational Motion
        G[Angular Position θ] --> H[Angular Velocity ω = dθ/dt]
        H --> I[Angular Acceleration α = dω/dt]
        J[Moment of Inertia I] --> K[Torque τ = Iα]
        L[Angular Momentum L = Iω]
    end
    
    A -.->|Analogy| G
    B -.->|Analogy| H
    C -.->|Analogy| I
    D -.->|Analogy| J
    E -.->|Analogy| K
    F -.->|Analogy| L
```

</details>

### Rotational Kinematics

Analogous to linear motion:
- Angular position: θ
- Angular velocity: ω = dθ/dt
- Angular acceleration: α = dω/dt

### Moment of Inertia

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://hyperphysics.phy-astr.gsu.edu/hbase/mi.html"> Reference: <b><i>Moment of Inertia Tables</i></b></a></p>

The rotational equivalent of mass:

**Point Mass:**
$$I = mr^2$$

**Common Shapes:**
- Solid sphere: $I = \tfrac{2}{5}MR^2$
- Solid cylinder: $I = \tfrac{1}{2}MR^2$
- Thin rod (about center): $I = \tfrac{1}{12}ML^2$

### Torque

The rotational equivalent of force:

$$
\tau = r \times F = rF \sin(\theta)
$$

**Rotational Newton's Second Law:**
$$
\tau = I\alpha
$$

## Gravitation: The First Unified Theory

### From Falling Apples to Orbiting Planets

Newton's greatest triumph wasn't just explaining how things move—it was recognizing that the force pulling an apple to Earth is the same force keeping the Moon in orbit. This was humanity's first "unified theory," connecting terrestrial and celestial mechanics.

### Newton's Law of Universal Gravitation
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://www.wilbourhall.org/pdfs/NewtonPrincipia.pdf"> Paper: <b><i>Principia - Book III: The System of the World</i></b></a></p>

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.physicsclassroom.com/class/circles/Lesson-3/Newton-s-Law-of-Universal-Gravitation"> Tutorial: <b><i>Understanding Universal Gravitation</i></b></a></p>

Every particle attracts every other particle with a force:

$$
F = \frac{Gm_1m_2}{r^2}
$$

Where:
- G = 6.674 × 10⁻¹¹ N·m²/kg² (gravitational constant)
- m₁, m₂ = masses of the objects
- r = distance between centers

### Orbital Motion

For circular orbits:

**Orbital Velocity:**
$$
v = \sqrt{\frac{GM}{r}}
$$

**Orbital Period:**
$$
T = 2\pi\sqrt{\frac{r^3}{GM}}
$$

This is Kepler's Third Law for circular orbits.

### The Effective Potential: Reducing Orbits to One Dimension

Real orbits are rarely circular. The deeper insight is that any motion in a central force $-\dfrac{GMm}{r^2}\hat{r}$ conserves angular momentum $L = mr^2\dot{\phi}$, and that single fact lets us collapse a two-dimensional orbit into an equivalent one-dimensional problem in the radius alone. Substituting $\dot{\phi} = L/(mr^2)$ into the energy gives:

$$E = \tfrac{1}{2}m\dot{r}^2 + V_{\text{eff}}(r), \qquad V_{\text{eff}}(r) = -\frac{GMm}{r} + \frac{L^2}{2mr^2}$$

The first term is the genuine gravitational attraction; the second is the **centrifugal barrier** — a repulsive contribution that arises purely from angular momentum and keeps the orbit from collapsing to the center. A particle now behaves like a bead rolling in the one-dimensional landscape $V_{\text{eff}}(r)$:

| Total energy $E$ | Shape of orbit |
|------------------|----------------|
| $E < 0$, at minimum of $V_{\text{eff}}$ | Circular |
| $E < 0$, above minimum | Ellipse (bound) |
| $E = 0$ | Parabola (escape, marginally unbound) |
| $E > 0$ | Hyperbola (unbound flyby) |

This single picture unifies planets, comets, and interstellar flybys, and it reproduces all of Kepler's laws. It is also the template for the radial Schrödinger equation of the hydrogen atom, where the same $-1/r$ attraction and $L^2/2mr^2$ barrier reappear — a preview of how classical structure carries into quantum mechanics.

## Oscillations and Waves: Patterns in Motion

### The Universality of Oscillations

Look around: pendulum clocks tick, guitar strings vibrate, atoms in solids jiggle. Oscillatory motion is everywhere because it emerges whenever there's a restoring force—something that pushes a system back toward equilibrium.

But real oscillations don't last forever...

### Damped Oscillations

When friction is present:

$$
m\ddot{x} + b\dot{x} + kx = 0
$$

Solutions depend on the damping coefficient b:
- **Underdamped:** Oscillates with decreasing amplitude
- **Critically damped:** Returns to equilibrium fastest
- **Overdamped:** Returns to equilibrium without oscillating

### Wave Motion

**Wave Equation:**
$$
\frac{\partial^2 y}{\partial t^2} = v^2 \frac{\partial^2 y}{\partial x^2}
$$

**Wave Speed:**
$$
v = f\lambda
$$

Where:
- f = frequency
- λ = wavelength

---

## Continue

| Previous | Next |
|----------|------|
| [Classical Mechanics Hub](./) | [Lagrangian &amp; Hamiltonian Mechanics →](lagrangian-hamiltonian.html) |

### See Also

- [Lagrangian &amp; Hamiltonian Mechanics](lagrangian-hamiltonian.html) — the energy-based reformulation that handles constraints and reveals symmetries.
- [Chaos, Modern Topics &amp; Computation](chaos-and-computational.html) — where deterministic Newtonian systems become unpredictable.
- [Computational Physics](../computational-physics/) — numerical integration of equations of motion.
