---
layout: docs
title: "Classical Mechanics: Newtonian Mechanics & Conservation Laws"
description: "Newton's laws, inertial and non-inertial frames, kinematics, work and energy, momentum and collisions, angular momentum, fixed-axis rotation, gravitation, and the Kepler problem."
permalink: /docs/physics/classical-mechanics/newtonian.html
toc: true
toc_sticky: true
---

[Classical Mechanics](./) &raquo; Newtonian Mechanics &amp; Conservation Laws

**Newtonian mechanics** describes motion in terms of forces acting on bodies in three-dimensional space with an absolute time. It covers everyday speeds and scales, from projectiles to planetary orbits, and is the starting point for the analytical formulations of [Lagrange and Hamilton](lagrangian-hamiltonian.html). This page covers Newton's laws and reference frames, kinematics, work and energy, the three conservation laws, fixed-axis rotation, and universal gravitation with the Kepler problem. Oscillations are introduced briefly and developed in [Oscillations &amp; Waves](waves.html); three-dimensional rotation is in [Rigid Body Dynamics](rigid-body-dynamics.html).

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://www.gutenberg.org/files/33229/33229-pdf.pdf"> Paper: <b><i>Philosophiæ Naturalis Principia Mathematica</i></b> - Isaac Newton</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion"> Article: <b><i>Newton's Laws of Motion - Wikipedia</i></b></a></p>

## Newton's Laws of Motion

Newton published the three laws in the *Principia* (1687). In modern notation, with $\vec{p} = m\vec{v}$ the linear momentum:

| Law | Statement | Equation |
|-----|-----------|----------|
| First (inertia) | A body not acted on by a net force moves with constant velocity. | $\sum\vec{F} = 0 \;\Rightarrow\; \vec{v} = \text{const}$ |
| Second | The rate of change of momentum equals the net force. | $\vec{F} = \dfrac{d\vec{p}}{dt} = m\vec{a}$ (constant $m$) |
| Third (action-reaction) | Forces between two bodies are equal in magnitude and opposite in direction. | $\vec{F}_{12} = -\vec{F}_{21}$ |

**First law.** The first law is not merely the special case $\vec{F} = 0$ of the second. It asserts that **inertial reference frames** exist: frames in which free bodies move uniformly. The second law holds only in such frames. Any frame moving at constant velocity relative to an inertial frame is also inertial (Galilean relativity), so there is no experiment that detects absolute uniform motion.

**Second law.** Newton's own formulation says the "change of motion" is proportional to the impressed force, where "motion" means what is now called momentum. The momentum form $\vec{F} = d\vec{p}/dt$ is the general one. The form $m\vec{a}$ assumes constant mass. For variable-mass systems such as rockets, the law must be applied to a closed system that includes the expelled mass (see [the rocket equation](#rocket-propulsion)).

**Third law.** The *weak* form (equal and opposite) is what makes total linear momentum conserved. The *strong* form additionally requires the forces to act along the line joining the two bodies, which is needed for conservation of angular momentum. Contact forces and gravity satisfy both. Magnetic forces between moving charges violate the third law as stated; momentum is still conserved once the momentum carried by the electromagnetic field is included.

### Non-Inertial Frames and Fictitious Forces

In a frame that rotates with angular velocity $\vec{\omega}$ and whose origin accelerates at $\vec{A}_0$, Newton's second law acquires extra terms. For position $\vec{r}$ and velocity $\vec{v}$ measured in the rotating frame:

$$
m\vec{a} = \vec{F} - m\vec{A}_0 - 2m\,\vec{\omega}\times\vec{v} - m\,\vec{\omega}\times(\vec{\omega}\times\vec{r}) - m\,\dot{\vec{\omega}}\times\vec{r}.
$$

| Term | Name | Examples |
|------|------|----------|
| $-m\vec{A}_0$ | Translational (inertial) force | Being pushed back into a seat when a car accelerates |
| $-2m\,\vec{\omega}\times\vec{v}$ | Coriolis force | Rotation of cyclones; deflection of long-range projectiles; Foucault pendulum |
| $-m\,\vec{\omega}\times(\vec{\omega}\times\vec{r})$ | Centrifugal force | Earth's equatorial bulge; apparent gravity is smaller at the equator |
| $-m\,\dot{\vec{\omega}}\times\vec{r}$ | Euler force | Felt on a merry-go-round that is speeding up |

These are not interactions; they are consequences of describing motion in an accelerating frame. On Earth the Coriolis term is small for everyday motion but controls large-scale weather. A Foucault pendulum's plane of swing rotates at $\Omega\sin\lambda$, where $\Omega$ is Earth's rotation rate and $\lambda$ the latitude: once per sidereal day at the poles, not at all at the equator.

## Kinematics

Kinematics describes motion without reference to its causes. For a particle at position $\vec{r}(t)$:

$$
\vec{v} = \frac{d\vec{r}}{dt}, \qquad \vec{a} = \frac{d\vec{v}}{dt} = \frac{d^2\vec{r}}{dt^2}.
$$

### Constant Acceleration

Integrating $\vec{a} = \text{const}$ gives the standard results (written here in one dimension):

| Equation | Missing variable |
|----------|------------------|
| $v = v_0 + at$ | $x$ |
| $x = x_0 + v_0 t + \tfrac{1}{2}at^2$ | $v$ |
| $v^2 = v_0^2 + 2a(x - x_0)$ | $t$ |
| $x = x_0 + \tfrac{1}{2}(v_0 + v)\,t$ | $a$ |

### Projectile Motion

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://phet.colorado.edu/en/simulations/projectile-motion"> Interactive: <b><i>Projectile Motion Simulator</i></b> - PhET</a></p>

Without air resistance, a projectile launched at speed $v_0$ and angle $\theta$ above the horizontal moves uniformly in $x$ and with constant acceleration $-g$ in $y$:

$$
x = v_0\cos\theta\; t, \qquad y = v_0\sin\theta\; t - \tfrac{1}{2}gt^2.
$$

Eliminating $t$ shows the trajectory is a parabola. Over level ground the range and maximum height are

$$
R = \frac{v_0^2\sin 2\theta}{g}, \qquad H = \frac{v_0^2\sin^2\theta}{2g}.
$$

The range is greatest at $\theta = 45^\circ$, and complementary angles give equal ranges.

### Air Resistance

Real projectiles experience drag opposing their velocity. Two limiting forms apply depending on the Reynolds number $Re = \rho v D/\eta$ (with $\eta$ the fluid's viscosity):

| Regime | Drag force | Terminal speed | Applies to |
|--------|------------|----------------|------------|
| Linear (Stokes), $Re \lesssim 1$ | $\vec{F} = -b\vec{v}$, with $b = 6\pi\eta R$ for a sphere | $v_t = mg/b$ | Dust, droplets, microorganisms |
| Quadratic, $10^3 \lesssim Re \lesssim 10^5$ | $\vec{F} = -\tfrac{1}{2}\rho C_d A\,v\,\vec{v}$ | $v_t = \sqrt{2mg/(\rho C_d A)}$ | Balls, skydivers, vehicles |

Linear drag gives exponential approach to terminal velocity and an analytic trajectory. Quadratic drag couples the $x$ and $y$ motion and generally requires numerical integration. With drag, the optimal launch angle is below $45^\circ$. See [Fluid Mechanics](../fluid-mechanics.html) for the origin of drag.

## Work and Energy

### Work and the Work-Energy Theorem

The work done by a force along a path $C$ is

$$
W = \int_C \vec{F}\cdot d\vec{r},
$$

which reduces to $W = Fd\cos\theta$ for a constant force and straight displacement. Integrating Newton's second law along the path gives the **work-energy theorem**: the net work on a particle equals its change in kinetic energy,

$$
W_{\text{net}} = \Delta K = \tfrac{1}{2}mv_f^2 - \tfrac{1}{2}mv_i^2.
$$

**Power** is the rate of doing work, $P = dW/dt = \vec{F}\cdot\vec{v}$.

### Conservative Forces and Potential Energy

A force is **conservative** if the work it does between two points is independent of the path. The following are equivalent (on a simply connected region):

- the work around every closed path is zero;
- $\nabla\times\vec{F} = 0$;
- $\vec{F} = -\nabla U$ for some potential energy $U(\vec{r})$.

| Force | Potential energy $U$ |
|-------|----------------------|
| Uniform gravity near the surface | $mgh$ |
| Ideal spring (Hooke's law) | $\tfrac{1}{2}kx^2$ |
| Newtonian gravity between point masses | $-Gm_1m_2/r$ |
| Coulomb force between charges | $q_1q_2/(4\pi\varepsilon_0 r)$ |

Friction and drag are non-conservative: the work they do depends on the path and is converted into heat.

For one-dimensional motion in a potential $U(x)$, energy conservation alone determines where motion is possible. The particle moves only where $U(x) \le E$; points where $U(x) = E$ are **turning points**. Minima of $U$ are stable equilibria and maxima are unstable. This energy-diagram method is used for the Kepler problem [below](#the-effective-potential).

## Conservation Laws

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.feynmanlectures.caltech.edu/I_04.html"> Lecture: <b><i>Conservation of Energy - Feynman Lectures</i></b></a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/play-btn-fill.svg" class="icon"><a href="https://phet.colorado.edu/en/simulation/energy-skate-park"> Interactive: <b><i>Energy Skate Park Simulation</i></b> - PhET</a></p>

Conservation laws give quantities that stay fixed throughout the motion, so they answer many questions without solving the equations of motion. Each corresponds to a symmetry of the laws of physics, a correspondence made precise by [Noether's theorem](lagrangian-hamiltonian.html#noethers-theorem).

| Conserved quantity | Condition | Underlying symmetry |
|--------------------|-----------|---------------------|
| Mechanical energy $K + U$ | Only conservative forces do work | Time translation |
| Linear momentum $\vec{P} = \sum_i m_i\vec{v}_i$ | Zero net external force | Spatial translation |
| Angular momentum $\vec{L} = \sum_i \vec{r}_i\times\vec{p}_i$ | Zero net external torque | Rotation |

### Energy

When only conservative forces do work, $E = K + U$ is constant:

$$
K_i + U_i = K_f + U_f.
$$

When non-conservative forces act, $\Delta(K + U) = W_{\text{nc}}$. Total energy, including thermal energy, is still conserved; mechanical energy is not.

### Linear Momentum and the Center of Mass

For a system of particles, internal forces cancel in pairs by the third law, so

$$
\frac{d\vec{P}}{dt} = \vec{F}_{\text{ext}}, \qquad M\ddot{\vec{R}}_{\text{cm}} = \vec{F}_{\text{ext}}, \qquad \vec{R}_{\text{cm}} = \frac{1}{M}\sum_i m_i\vec{r}_i.
$$

The center of mass moves as if all the mass were concentrated there and all external forces acted on it. With no external force, $\vec{P}$ is conserved.

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.physicsclassroom.com/class/momentum/Lesson-2/Momentum-Conservation-Principle"> Tutorial: <b><i>Momentum Conservation in Collisions</i></b></a></p>

### Collisions

Momentum is conserved in every collision between isolated bodies; kinetic energy may or may not be. The **coefficient of restitution** $e = -(v_2' - v_1')/(v_2 - v_1)$ measures the ratio of separation to approach speed.

| Type | Kinetic energy | $e$ |
|------|----------------|-----|
| Elastic | Conserved | 1 |
| Inelastic | Partly converted to heat, sound, deformation | $0 < e < 1$ |
| Perfectly inelastic (bodies stick) | Maximum loss consistent with momentum conservation | 0 |

For a one-dimensional elastic collision, solving the momentum and energy equations together gives

$$
v_1' = \frac{(m_1 - m_2)\,v_1 + 2m_2 v_2}{m_1 + m_2}, \qquad v_2' = \frac{(m_2 - m_1)\,v_2 + 2m_1 v_1}{m_1 + m_2}.
$$

Equal masses exchange velocities. A light body bouncing off a heavy stationary one reverses its velocity. In the center-of-mass frame, an elastic collision only rotates the velocities, which is why collision analysis in particle physics is done in that frame.

### Rocket Propulsion

A rocket ejecting exhaust at speed $v_e$ relative to itself, with no external force, gains velocity according to the **Tsiolkovsky rocket equation**:

$$
\Delta v = v_e \ln\frac{m_0}{m_f},
$$

where $m_0$ and $m_f$ are the initial and final masses. The logarithm is why reaching orbit (about $9.4\ \text{km/s}$ of $\Delta v$ including gravity and drag losses) with chemical exhaust speeds of $3$ to $4.5\ \text{km/s}$ requires most of the launch mass to be propellant, and why rockets are built in stages.

### Angular Momentum

The angular momentum of a particle about an origin is $\vec{L} = \vec{r}\times\vec{p}$, and the torque is $\vec{\tau} = \vec{r}\times\vec{F}$. Differentiating gives the rotational form of the second law:

$$
\frac{d\vec{L}}{dt} = \vec{\tau}.
$$

For a system of particles whose internal forces obey the strong third law, internal torques cancel and $d\vec{L}_{\text{total}}/dt = \vec{\tau}_{\text{ext}}$. A **central force**, directed along $\vec{r}$, exerts no torque about the force center, so $\vec{L}$ is conserved. The motion is therefore confined to a plane, and the radius vector sweeps equal areas in equal times (Kepler's second law), since $dA/dt = L/2m$.

A figure skater pulling in her arms is the standard illustration: with no external torque about the spin axis, $L = I\omega$ is constant, so decreasing the moment of inertia $I$ increases the spin rate $\omega$. Her kinetic energy $L^2/2I$ increases; the extra energy comes from the work her muscles do pulling her arms inward.

## Rotation About a Fixed Axis

Rotation about a fixed axis follows the same mathematical pattern as one-dimensional translation:

| Translation | Rotation about a fixed axis | Relation |
|-------------|-----------------------------|----------|
| Position $x$ | Angle $\theta$ | $s = r\theta$ (arc length) |
| Velocity $v = \dot{x}$ | Angular velocity $\omega = \dot\theta$ | $v = r\omega$ |
| Acceleration $a = \dot{v}$ | Angular acceleration $\alpha = \dot\omega$ | $a_t = r\alpha$ |
| Mass $m$ | Moment of inertia $I = \sum_i m_i r_{\perp,i}^2$ | |
| Force $F$ | Torque $\tau = rF\sin\phi$ | |
| $F = ma$ | $\tau = I\alpha$ | |
| Momentum $p = mv$ | Angular momentum $L = I\omega$ | |
| Kinetic energy $\tfrac{1}{2}mv^2$ | Rotational kinetic energy $\tfrac{1}{2}I\omega^2$ | |
| Power $Fv$ | Power $\tau\omega$ | |

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://hyperphysics.phy-astr.gsu.edu/hbase/mi.html"> Reference: <b><i>Moment of Inertia Tables</i></b> - HyperPhysics</a></p>

Moments of inertia of common uniform bodies of mass $M$, about an axis through the center of mass:

| Body | Axis | $I$ |
|------|------|-----|
| Thin ring or hoop, radius $R$ | Symmetry axis | $MR^2$ |
| Solid cylinder or disk, radius $R$ | Symmetry axis | $\tfrac{1}{2}MR^2$ |
| Solid sphere, radius $R$ | Any diameter | $\tfrac{2}{5}MR^2$ |
| Thin spherical shell, radius $R$ | Any diameter | $\tfrac{2}{3}MR^2$ |
| Thin rod, length $\ell$ | Perpendicular, through center | $\tfrac{1}{12}M\ell^2$ |

The **parallel-axis theorem** gives the moment about any parallel axis a distance $d$ away: $I = I_{\text{cm}} + Md^2$. A thin rod about one end therefore has $I = \tfrac{1}{12}M\ell^2 + M(\ell/2)^2 = \tfrac{1}{3}M\ell^2$.

The scalar relation $L = I\omega$ holds only for rotation about a fixed axis, or about a principal axis of a free body. In general $\vec{L}$ and $\vec{\omega}$ are not parallel and $I$ becomes a tensor; see [Rigid Body Dynamics](rigid-body-dynamics.html).

## Gravitation and the Kepler Problem

### Universal Gravitation

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://www.wilbourhall.org/pdfs/NewtonPrincipia.pdf"> Paper: <b><i>Principia - Book III: The System of the World</i></b></a></p>

Newton's law of universal gravitation states that every pair of point masses attracts along the line joining them with a force

$$
\vec{F}_{12} = -\frac{Gm_1m_2}{r^2}\,\hat{r}_{12},
$$

where $\hat{r}_{12}$ points from body 2 to body 1. The CODATA 2022 recommended value of the gravitational constant is $G = 6.674\,30(15)\times10^{-11}\ \text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$, a relative uncertainty of $2.2\times10^{-5}$. $G$ is the least precisely known of the fundamental constants, because gravity is extremely weak between laboratory masses and published measurements disagree by more than their stated uncertainties. In practice, orbital mechanics uses products such as Earth's $GM$, which are known far more precisely from spacecraft tracking.

**Shell theorem.** A spherically symmetric body attracts external masses as if all its mass were at its center, and a spherical shell exerts no net force on a mass inside it. This justifies treating planets and stars as point masses.

### Circular Orbits

For a small mass in a circular orbit of radius $r$ around a mass $M$, gravity supplies the centripetal force $mv^2/r = GMm/r^2$:

$$
v = \sqrt{\frac{GM}{r}}, \qquad T = 2\pi\sqrt{\frac{r^3}{GM}}.
$$

The escape speed from radius $r$ is $\sqrt{2}$ times the circular speed, $v_{\text{esc}} = \sqrt{2GM/r}$, about $11.2\ \text{km/s}$ from Earth's surface.

### The Effective Potential

A two-body problem with masses $m_1$ and $m_2$ reduces to the motion of a single particle of **reduced mass** $\mu = m_1m_2/(m_1 + m_2)$ in the potential $U(r) = -k/r$, where $k = Gm_1m_2$ and $r$ is the separation. Because the force is central, the angular momentum $L = \mu r^2\dot{\phi}$ is conserved. Substituting $\dot\phi = L/(\mu r^2)$ into the energy reduces the problem to one radial dimension:

$$
E = \tfrac{1}{2}\mu\dot{r}^2 + V_{\text{eff}}(r), \qquad V_{\text{eff}}(r) = -\frac{k}{r} + \frac{L^2}{2\mu r^2}.
$$

The second term is the **centrifugal barrier**. It dominates at small $r$ and prevents a body with nonzero angular momentum from reaching the center.

<figure style="text-align:center;margin:1.5em 0">
<svg viewBox="0 0 640 340" role="img" aria-label="Effective potential for the Kepler problem with energy levels for circular, elliptical, parabolic and hyperbolic orbits" style="max-width:640px;width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="13">
<defs><marker id="arr-veff" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="currentColor"/></marker></defs>
<line x1="60" y1="159.1" x2="610" y2="159.1" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-veff)"/>
<line x1="60" y1="300" x2="60" y2="20" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-veff)"/>
<text x="605" y="177.1" text-anchor="end" fill="currentColor">r</text>
<text x="68" y="30" fill="currentColor">V_eff(r)</text>
<path d="M127.5,37.5 L133.8,57.2 L140.1,72.6 L146.3,84.7 L152.6,94.4 L158.9,102.4 L165.1,108.9 L171.4,114.4 L177.7,119.0 L183.9,123.0 L190.2,126.4 L196.5,129.3 L202.7,131.9 L209.0,134.1 L215.3,136.1 L221.5,137.9 L227.8,139.4 L234.1,140.8 L240.3,142.1 L246.6,143.2 L252.8,144.2 L259.1,145.1 L265.4,146.0 L271.6,146.7 L277.9,147.4 L284.2,148.1 L290.4,148.7 L296.7,149.2 L303.0,149.7 L309.2,150.2 L315.5,150.6 L321.8,151.0 L328.0,151.4 L334.3,151.8 L340.6,152.1 L346.8,152.4 L353.1,152.7 L359.4,152.9 L365.6,153.2 L371.9,153.4 L378.2,153.6 L384.4,153.9 L390.7,154.1 L397.0,154.2 L403.2,154.4 L409.5,154.6 L415.8,154.7 L422.0,154.9 L428.3,155.0 L434.6,155.2 L440.8,155.3 L447.1,155.4 L453.4,155.5 L459.6,155.7 L465.9,155.8 L472.2,155.9 L478.4,156.0 L484.7,156.1 L490.9,156.1 L497.2,156.2 L503.5,156.3 L509.7,156.4 L516.0,156.5 L522.3,156.5 L528.5,156.6 L534.8,156.7 L541.1,156.7 L547.3,156.8 L553.6,156.9 L559.9,156.9 L566.1,157.0 L572.4,157.0 L578.7,157.1 L584.9,157.1 L591.2,157.2 L597.5,157.2 L603.7,157.3 L610.0,157.3" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.6"/>
<path d="M176.9,297.2 L184.2,289.1 L191.6,281.8 L198.9,275.3 L206.2,269.5 L213.6,264.2 L220.9,259.4 L228.3,255.1 L235.6,251.0 L242.9,247.4 L250.3,244.0 L257.6,240.8 L265.0,237.9 L272.3,235.2 L279.7,232.6 L287.0,230.2 L294.3,228.0 L301.7,225.9 L309.0,224.0 L316.4,222.1 L323.7,220.3 L331.0,218.7 L338.4,217.1 L345.7,215.6 L353.1,214.2 L360.4,212.9 L367.7,211.6 L375.1,210.4 L382.4,209.2 L389.8,208.1 L397.1,207.0 L404.4,206.0 L411.8,205.0 L419.1,204.1 L426.5,203.2 L433.8,202.3 L441.2,201.5 L448.5,200.7 L455.8,199.9 L463.2,199.2 L470.5,198.4 L477.9,197.8 L485.2,197.1 L492.5,196.4 L499.9,195.8 L507.2,195.2 L514.6,194.6 L521.9,194.1 L529.2,193.5 L536.6,193.0 L543.9,192.5 L551.3,192.0 L558.6,191.5 L566.0,191.0 L573.3,190.6 L580.6,190.1 L588.0,189.7 L595.3,189.3 L602.7,188.9 L610.0,188.5" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.6"/>
<text x="163.1" y="88.7" fill="currentColor" opacity="0.8">centrifugal L²/2μr²</text>
<text x="348.8" y="229.6" fill="currentColor" opacity="0.8">gravity −k/r</text>
<path d="M88.9,52.7 L92.2,124.4 L95.4,172.7 L98.7,205.8 L102.0,228.8 L105.3,244.9 L108.5,256.2 L111.8,264.0 L115.1,269.3 L118.4,272.8 L121.7,275.0 L124.9,276.1 L128.2,276.5 L131.5,276.4 L134.8,275.8 L138.0,274.9 L141.3,273.7 L144.6,272.4 L147.9,271.0 L151.1,269.4 L154.4,267.8 L157.7,266.2 L161.0,264.6 L164.3,262.9 L167.5,261.3 L170.8,259.6 L174.1,258.0 L177.4,256.4 L180.6,254.8 L183.9,253.3 L187.2,251.7 L190.5,250.2 L193.8,248.8 L197.0,247.4 L200.3,246.0 L203.6,244.6 L206.9,243.3 L210.1,242.0 L213.4,240.8 L216.7,239.5 L220.0,238.3 L223.3,237.2 L226.5,236.0 L229.8,234.9 L233.1,233.9 L236.4,232.8 L239.6,231.8 L242.9,230.8 L246.2,229.8 L249.5,228.9 L252.8,227.9 L256.0,227.0 L259.3,226.1 L262.6,225.3 L265.9,224.4 L269.1,223.6 L272.4,222.8 L275.7,222.0 L279.0,221.3 L282.2,220.5 L285.5,219.8 L288.8,219.1 L292.1,218.4 L295.4,217.7 L298.6,217.0 L301.9,216.4 L305.2,215.7 L308.5,215.1 L311.7,214.5 L315.0,213.9 L318.3,213.3 L321.6,212.7 L324.9,212.2 L328.1,211.6 L331.4,211.1 L334.7,210.5 L338.0,210.0 L341.2,209.5 L344.5,209.0 L347.8,208.5 L351.1,208.0 L354.4,207.6 L357.6,207.1 L360.9,206.6 L364.2,206.2 L367.5,205.8 L370.7,205.3 L374.0,204.9 L377.3,204.5 L380.6,204.1 L383.9,203.7 L387.1,203.3 L390.4,202.9 L393.7,202.5 L397.0,202.1 L400.2,201.8 L403.5,201.4 L406.8,201.1 L410.1,200.7 L413.3,200.4 L416.6,200.0 L419.9,199.7 L423.2,199.4 L426.5,199.0 L429.7,198.7 L433.0,198.4 L436.3,198.1 L439.6,197.8 L442.8,197.5 L446.1,197.2 L449.4,196.9 L452.7,196.6 L456.0,196.4 L459.2,196.1 L462.5,195.8 L465.8,195.5 L469.1,195.3 L472.3,195.0 L475.6,194.8 L478.9,194.5 L482.2,194.3 L485.5,194.0 L488.7,193.8 L492.0,193.5 L495.3,193.3 L498.6,193.1 L501.8,192.8 L505.1,192.6 L508.4,192.4 L511.7,192.1 L515.0,191.9 L518.2,191.7 L521.5,191.5 L524.8,191.3 L528.1,191.1 L531.3,190.9 L534.6,190.7 L537.9,190.5 L541.2,190.3 L544.4,190.1 L547.7,189.9 L551.0,189.7 L554.3,189.5 L557.6,189.3 L560.8,189.1 L564.1,189.0 L567.4,188.8 L570.7,188.6 L573.9,188.4 L577.2,188.3 L580.5,188.1 L583.8,187.9 L587.1,187.8 L590.3,187.6 L593.6,187.4 L596.9,187.3 L600.2,187.1 L603.4,187.0 L606.7,186.8 L610.0,186.6" fill="none" stroke="#2f7fd8" stroke-width="3"/>
<circle cx="128.8" cy="276.5" r="5" fill="#d9534f"/>
<text x="138.8" y="294.5" fill="currentColor">circular (E = V_min)</text>
<line x1="102.1" y1="229.6" x2="247.1" y2="229.6" stroke="#d9534f" stroke-width="2"/>
<text x="253.1" y="233.6" fill="currentColor">ellipse (E &lt; 0): r bounces between turning points</text>
<text x="98.1" y="245.6" text-anchor="end" fill="currentColor" font-size="11">r_min</text>
<text x="247.1" y="245.6" text-anchor="middle" fill="currentColor" font-size="11">r_max</text>
<line x1="90.9" y1="100.4" x2="596.2" y2="100.4" stroke="#d9534f" stroke-width="2" stroke-dasharray="8 3"/>
<text x="300.6" y="93.4" fill="currentColor">hyperbola (E &gt; 0): one turning point, escapes</text>
<text x="458.8" y="152.1" fill="currentColor">E = 0: parabola</text>
</svg>
<figcaption>The effective potential (solid) is the sum of the gravitational attraction and the centrifugal barrier (dashed). The total energy <i>E</i> fixes the type of orbit: horizontal lines mark the radial range the body can reach.</figcaption>
</figure>

| Energy | Radial motion | Orbit | Eccentricity |
|--------|---------------|-------|--------------|
| $E = V_{\text{eff,min}} = -\mu k^2/(2L^2)$ | $r$ constant | Circle | $e = 0$ |
| $V_{\text{eff,min}} < E < 0$ | Between two turning points | Ellipse | $0 < e < 1$ |
| $E = 0$ | One turning point; reaches infinity with zero speed | Parabola | $e = 1$ |
| $E > 0$ | One turning point; escapes | Hyperbola | $e > 1$ |

### Orbit Equation and Kepler's Laws

Solving the radial equation for $r(\phi)$ (most easily with the substitution $u = 1/r$) gives a conic section with the force center at one focus:

$$
r(\phi) = \frac{p}{1 + e\cos\phi}, \qquad p = \frac{L^2}{\mu k}, \qquad e = \sqrt{1 + \frac{2EL^2}{\mu k^2}}.
$$

This reproduces all three of Kepler's empirical laws:

1. Bound orbits are ellipses with the Sun at one focus.
2. The line from the Sun to a planet sweeps equal areas in equal times (angular momentum conservation).
3. The square of the period is proportional to the cube of the semi-major axis $a$: $T^2 = 4\pi^2 a^3 / [G(m_1 + m_2)]$.

For a small body orbiting a large mass $M$, the speed at any point on the orbit follows from the **vis-viva equation**, $v^2 = GM\left(2/r - 1/a\right)$, which is the basis of transfer-orbit calculations.

Bound Kepler orbits close on themselves because the inverse-square force has an extra conserved quantity, the **Laplace-Runge-Lenz vector** $\vec{A} = \vec{p}\times\vec{L} - \mu k\,\hat{r}$, which points to the perihelion and fixes its direction. By **Bertrand's theorem**, the inverse-square force and the linear (harmonic) force are the only central forces for which every bound orbit is closed. Any perturbation, whether from other planets, a non-spherical Sun, or general relativity, makes the perihelion precess. The unexplained $43$ arcseconds per century in Mercury's perihelion precession was the first success of [general relativity](../relativity/).

## Oscillations: A First Look

Near any stable equilibrium the potential is approximately quadratic, so small displacements produce a linear restoring force $F = -kx$ (Hooke's law). The result is **simple harmonic motion**:

$$
x(t) = A\cos(\omega t + \varphi), \qquad \omega = \sqrt{\frac{k}{m}}, \qquad T = \frac{2\pi}{\omega}.
$$

The amplitude $A$ and phase $\varphi$ are set by initial conditions; the frequency is set by the system alone. Damping, driving and resonance, coupled oscillators, and the wave equation are developed in [Oscillations &amp; Waves](waves.html).

## Domain of Validity

Newtonian mechanics is an approximation, accurate when all of the following hold:

| Condition | When it fails | Replacement theory |
|-----------|---------------|--------------------|
| Speeds much less than light, $v \ll c$ | Particle accelerators, GPS clock corrections | [Special relativity](../relativity/) |
| Weak gravity, $GM/(rc^2) \ll 1$ | Neutron stars, black holes, precision orbits | [General relativity](../relativity/) |
| Action much greater than $\hbar$ | Atoms, molecules, low-temperature solids | [Quantum mechanics](../quantum-mechanics/) |
| Few enough bodies to track individually | Gases and fluids | [Statistical mechanics](../statistical-mechanics/) (built on Newtonian dynamics) |

Even inside its domain, deterministic Newtonian dynamics can be unpredictable in practice: nonlinear systems with three or more degrees of freedom are often [chaotic](chaos-and-computational.html).

---

## Continue

| Previous | Next |
|----------|------|
| [&larr; Classical Mechanics Hub](./) | [Oscillations &amp; Waves &rarr;](waves.html) |

## See Also

- [Oscillations &amp; Waves](waves.html): harmonic motion, resonance, normal modes, and the wave equation.
- [Lagrangian &amp; Hamiltonian Mechanics](lagrangian-hamiltonian.html): the energy-based reformulation, constraints, and Noether's theorem.
- [Rigid Body Dynamics](rigid-body-dynamics.html): the inertia tensor, Euler's equations, and gyroscopic motion.
- [Chaos &amp; Nonlinear Dynamics](chaos-and-computational.html): where deterministic Newtonian systems become unpredictable.
- [Computational Methods](computational-classical-mechanics.html): numerical integration of Newton's equations, including N-body simulation.
