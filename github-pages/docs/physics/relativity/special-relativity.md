---
layout: docs
title: "Relativity: Special Relativity"
description: "Einstein's two postulates and their consequences: relativity of simultaneity, the Lorentz transformation, time dilation, length contraction, velocity addition, the Doppler effect, four-vectors, relativistic dynamics, E = mc^2, and the experimental evidence."
permalink: /docs/physics/relativity/special-relativity.html
toc: true
toc_sticky: true
---

[Relativity](./) &raquo; Special Relativity

## Special Relativity

Special relativity (Einstein, 1905) is the theory of space and time in the absence of gravity. It replaces Newton's absolute time and absolute space with a single four-dimensional **spacetime** whose geometry is the same for every inertial observer. Two postulates suffice to derive everything on this page: time dilation, length contraction, the relativity of simultaneity, the velocity-addition law, $E = mc^2$, and the four-vector formalism that underlies both [general relativity](general-relativity.html) and [quantum field theory](../quantum-field-theory.html).

**Conventions.** Coordinates are $x^\mu = (ct, x, y, z)$ with $\mu = 0,1,2,3$. The metric signature is **(−,+,+,+)** ("mostly plus"), matching the [tensor formalism](tensor-formalism.html) and general-relativity pages. Particle-physics texts often use (+,−,−,−); the two differ only by an overall sign of every inner product. We write $\beta = v/c$ and $\gamma = 1/\sqrt{1-\beta^2}$. "Mass" $m$ always means the invariant (rest) mass.

## Historical Background

Maxwell's equations (1865) predict electromagnetic waves travelling at a fixed speed $c = 1/\sqrt{\mu_0\varepsilon_0}$, but do not say relative to what. The natural assumption was a medium, the **luminiferous aether**, in which case Earth's motion should produce a measurable "aether wind." The Michelson–Morley experiment (1887) found none. Lorentz and FitzGerald proposed that bodies physically contract when moving through the aether, and Lorentz and Poincaré found the transformation that leaves Maxwell's equations invariant. Einstein's contribution was to drop the aether altogether and take the invariance of $c$ as a fact about space and time rather than about electromagnetism. Minkowski (1908) then recast the theory geometrically as the study of a four-dimensional spacetime.

Since 1983 the SI metre has been defined by fixing $c = 299\,792\,458$ m/s exactly, so the constancy of $c$ is now built into the unit system.

## The Postulates

1. **Principle of relativity.** The laws of physics take the same form in every inertial frame. No experiment done inside a closed laboratory can detect its uniform motion.
2. **Invariance of the speed of light.** Light in vacuum travels at the same speed $c$ in every inertial frame, independent of the motion of the source or the observer.

The second postulate conflicts with Galilean velocity addition ($u = v + w$). Resolving the conflict forces a new relation between the time and space coordinates of different frames: the Lorentz transformation.

## Relativity of Simultaneity

The key conceptual change is that **simultaneity is frame-dependent**: two spatially separated events that happen at the same time in one inertial frame generally happen at different times in another. Time dilation and length contraction are consequences of this.

**Train thought experiment.** A railway car moves at speed $v$ along a platform. A lamp at the car's midpoint flashes once.

- *Car frame.* The car is at rest and the walls are equidistant from the lamp. Light travels at $c$ in both directions, so it reaches the front and rear walls at the same moment.
- *Platform frame.* Light also travels at $c$ here, but during the flight the rear wall moves toward the emission point and the front wall moves away from it. The rear wall is struck first.

Both analyses are correct. The two wall strikes are **spacelike separated** (no signal could travel between them), and for such pairs the time order depends on the frame. Events that are **timelike** or **lightlike** separated, and so could be causally connected, have the same time order in every frame, so causality is preserved.

### Leading clocks lag

Suppose clocks are synchronized along the car in its rest frame. From the time component of the Lorentz transformation (derived below), $t' = \gamma(t - vx/c^2)$, the readings of two car clocks at a single platform instant ($\Delta t = 0$) differ by

$$\Delta t' = -\frac{\gamma v\, \Delta x}{c^2} = -\frac{v L_0}{c^2}$$

where $\Delta x = L_0/\gamma$ is their separation measured on the platform and $L_0$ is their proper (car-frame) separation. The clock further forward in the direction of motion reads *behind* by $vL_0/c^2$. This rule resolves most textbook "paradoxes": each frame regards the other's clocks as unsynchronized, which is why both can consistently see the other's clocks running slow.

**Example.** A car of proper length $L_0 = 100$ m passes at $v = 0.6c$. At any platform instant the front clock reads behind the rear clock by

$$\frac{v L_0}{c^2} = \frac{0.6 \times 100\ \text{m}}{3.00 \times 10^{8}\ \text{m/s}} = 2.0 \times 10^{-7}\ \text{s}$$

Over astronomical distances the same offset becomes large: two observers in relative motion at walking speed disagree about "now" in the Andromeda galaxy by days.

## Spacetime and the Invariant Interval

An **event** is a point in spacetime, labelled by $(ct, x, y, z)$ in some inertial frame. For two events the **interval**

$$\Delta s^2 = -c^2 \Delta t^2 + \Delta x^2 + \Delta y^2 + \Delta z^2 = \eta_{\mu\nu}\, \Delta x^\mu \Delta x^\nu$$

has the same value in every inertial frame, where the Minkowski metric is

$$\eta_{\mu\nu} = \begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

The interval plays the role that distance plays in Euclidean geometry, but it is not positive-definite, and its sign classifies every pair of events:

| Separation | Sign of $\Delta s^2$ (−,+,+,+) | Meaning | Frame-invariant quantity |
|---|---|---|---|
| Timelike | $\Delta s^2 < 0$ | A massive body can be present at both; time order is absolute | Proper time $\Delta\tau = \sqrt{-\Delta s^2}/c$ |
| Lightlike (null) | $\Delta s^2 = 0$ | Connected by a light signal | Both events lie on each other's light cone |
| Spacelike | $\Delta s^2 > 0$ | No causal connection; time order is frame-dependent | Proper distance $\sqrt{\Delta s^2}$ |

For a particle's worldline the **proper time**, the time shown by a clock carried along, is

$$d\tau^2 = -\frac{ds^2}{c^2} = dt^2\left(1 - \frac{v^2}{c^2}\right) \quad\Longrightarrow\quad d\tau = \frac{dt}{\gamma}$$

### Spacetime diagrams

A **Minkowski diagram** plots $ct$ vertically against $x$. Light rays are lines at 45°, forming the **light cone** of an event. The worldline of an inertial observer moving at velocity $v$ is a straight line of slope $c/v$ (steeper than 45°), and it serves as that observer's time axis $ct'$. That observer's space axis $x'$ (the set of events simultaneous with the origin in the moving frame) is tilted by the same angle toward the light cone. Lines parallel to $x'$ are lines of constant $t'$.

<figure style="margin: 1.5em auto; max-width: 480px;">
<svg viewBox="0 0 460 400" role="img" aria-label="Minkowski diagram showing the light cone, the axes of a frame moving at half the speed of light, and two events simultaneous in the moving frame but not in the rest frame" style="width: 100%; height: auto; color: inherit; font-family: inherit;">
  <defs>
    <marker id="sr-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L8,4 L0,8 z" fill="currentColor"/>
    </marker>
    <marker id="sr-arrow-b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L8,4 L0,8 z" fill="#1c7ed6"/>
    </marker>
  </defs>
  <!-- light cone -->
  <line x1="50" y1="380" x2="410" y2="20" stroke="#e8590c" stroke-width="2" stroke-dasharray="7,5"/>
  <line x1="50" y1="20" x2="410" y2="380" stroke="#e8590c" stroke-width="2" stroke-dasharray="7,5"/>
  <text x="404" y="40" font-size="13" fill="#e8590c" text-anchor="end">light, x = ct</text>
  <!-- rest-frame axes -->
  <line x1="230" y1="385" x2="230" y2="18" stroke="currentColor" stroke-width="1.8" marker-end="url(#sr-arrow)"/>
  <line x1="45" y1="200" x2="442" y2="200" stroke="currentColor" stroke-width="1.8" marker-end="url(#sr-arrow)"/>
  <text x="238" y="24" font-size="15" fill="currentColor">ct</text>
  <text x="436" y="192" font-size="15" fill="currentColor">x</text>
  <!-- boosted axes, beta = 0.5 -->
  <line x1="145" y1="370" x2="315" y2="30" stroke="#1c7ed6" stroke-width="2" marker-end="url(#sr-arrow-b)"/>
  <line x1="60" y1="285" x2="400" y2="115" stroke="#1c7ed6" stroke-width="2" marker-end="url(#sr-arrow-b)"/>
  <text x="320" y="36" font-size="15" fill="#1c7ed6">ct'</text>
  <text x="404" y="112" font-size="15" fill="#1c7ed6">x'</text>
  <!-- line of constant t' -->
  <line x1="122.5" y1="190" x2="412.5" y2="45" stroke="#1c7ed6" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="416" y="60" font-size="12" fill="#1c7ed6">t' = const</text>
  <!-- events A and B -->
  <circle cx="172.5" cy="165" r="5" fill="currentColor"/>
  <circle cx="372.5" cy="65" r="5" fill="currentColor"/>
  <text x="164" y="156" font-size="14" fill="currentColor" text-anchor="end">A</text>
  <text x="372" y="84" font-size="14" fill="currentColor" text-anchor="middle">B</text>
  <!-- projections to ct axis -->
  <line x1="172.5" y1="165" x2="230" y2="165" stroke="currentColor" stroke-width="1" stroke-dasharray="2,3" opacity="0.7"/>
  <line x1="230" y1="65" x2="372.5" y2="65" stroke="currentColor" stroke-width="1" stroke-dasharray="2,3" opacity="0.7"/>
  <text x="224" y="161" font-size="12" fill="currentColor" text-anchor="end">t_A</text>
  <text x="224" y="61" font-size="12" fill="currentColor" text-anchor="end">t_B</text>
  <!-- region labels -->
  <text x="175" y="88" font-size="13" fill="currentColor" text-anchor="middle" opacity="0.85">timelike future</text>
  <text x="230" y="345" font-size="13" fill="currentColor" text-anchor="middle" opacity="0.85">timelike past</text>
  <text x="72" y="222" font-size="13" fill="currentColor" text-anchor="middle" opacity="0.85">spacelike</text>
  <text x="395" y="250" font-size="13" fill="currentColor" text-anchor="middle" opacity="0.85">spacelike</text>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">Minkowski diagram for a frame S' moving at $\beta = 0.5$. The $ct'$ axis is the worldline of the S' origin; the $x'$ axis and the dashed line parallel to it are lines of constant $t'$. Events A and B are simultaneous in S' but not in S, where B occurs later ($t_B > t_A$). Events inside the light cone of the origin are timelike separated from it; events outside are spacelike.</figcaption>
</figure>

## The Lorentz Transformation

Consider a frame S' moving at velocity $v$ along the $x$ axis of frame S, with origins coinciding at $t = t' = 0$ ("standard configuration").

### Derivation

Homogeneity of space and time requires the transformation to be linear. Transverse coordinates are unchanged ($y' = y$, $z' = z$), since a length perpendicular to the motion can be compared directly by both frames and any change would single out a direction. The S' origin $x' = 0$ moves along $x = vt$, so

$$x' = \gamma\,(x - vt)$$

for some factor $\gamma(v)$. By the principle of relativity, S moves at $-v$ as seen from S', and the inverse must have the same form with the same factor:

$$x = \gamma\,(x' + vt')$$

Now apply the second postulate. A light pulse leaving the origin obeys $x = ct$ in S and $x' = ct'$ in S'. Substituting into both equations gives

$$ct' = \gamma\,(c - v)\,t, \qquad ct = \gamma\,(c + v)\,t'$$

Multiplying the two, $c^2 t t' = \gamma^2 (c^2 - v^2)\, t t'$, so

$$\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}$$

Eliminating $x'$ between the two equations for $x'$ and $x$ gives the time transformation. The result is the **Lorentz boost**:

$$ct' = \gamma\left(ct - \beta x\right), \qquad x' = \gamma\left(x - \beta\, ct\right), \qquad y' = y, \qquad z' = z$$

The inverse is obtained by $v \to -v$. In matrix form, $x'^\mu = \Lambda^\mu{}_\nu x^\nu$ with

$$\Lambda^\mu{}_\nu = \begin{pmatrix} \gamma & -\beta\gamma & 0 & 0 \\ -\beta\gamma & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Any matrix satisfying $\Lambda^T \eta\, \Lambda = \eta$ preserves the interval. These form the **Lorentz group** $O(1,3)$; combined with spacetime translations they form the **Poincaré group**, the symmetry group of special relativity. For $v \ll c$, $\gamma \to 1$ and the boost reduces to the Galilean transformation $x' = x - vt$, $t' = t$.

### Rapidity

Writing $\beta = \tanh\phi$ gives $\gamma = \cosh\phi$ and $\beta\gamma = \sinh\phi$, so a boost is a hyperbolic rotation:

$$\begin{pmatrix} ct' \\ x' \end{pmatrix} = \begin{pmatrix} \cosh\phi & -\sinh\phi \\ -\sinh\phi & \cosh\phi \end{pmatrix} \begin{pmatrix} ct \\ x \end{pmatrix}$$

The **rapidity** $\phi$ is additive for collinear boosts, $\phi_{\text{total}} = \phi_1 + \phi_2$, which is the cleanest way to derive the velocity-addition law. Collider physics uses the closely related longitudinal rapidity $y = \tfrac{1}{2}\ln\bigl[(E + p_z c)/(E - p_z c)\bigr]$ because rapidity differences are invariant under boosts along the beam.

## Time Dilation

A clock at rest in S' ticks at fixed $x'$. Its proper time interval $\Delta\tau$ corresponds to a coordinate interval in S of

$$\Delta t = \gamma\, \Delta\tau$$

A moving clock is measured to run slow, by the factor $\gamma$, by observers who see it move. The effect is symmetric: each of two inertial observers measures the other's clock as slow, and the relativity of simultaneity keeps this consistent.

<figure style="margin: 1.5em auto; max-width: 480px;">
<svg viewBox="0 0 460 280" role="img" aria-label="Plot of the Lorentz factor gamma against speed as a fraction of c, rising slowly and then diverging as beta approaches 1" style="width: 100%; height: auto; font-family: inherit;">
  <g stroke="currentColor" stroke-width="1" opacity="0.2">
    <line x1="50" y1="203.3" x2="430" y2="203.3"/>
    <line x1="50" y1="166.7" x2="430" y2="166.7"/>
    <line x1="50" y1="130" x2="430" y2="130"/>
    <line x1="50" y1="93.3" x2="430" y2="93.3"/>
    <line x1="50" y1="56.7" x2="430" y2="56.7"/>
    <line x1="50" y1="20" x2="430" y2="20"/>
  </g>
  <line x1="50" y1="240" x2="430" y2="240" stroke="currentColor" stroke-width="1.5"/>
  <line x1="50" y1="240" x2="50" y2="15" stroke="currentColor" stroke-width="1.5"/>
  <line x1="430" y1="240" x2="430" y2="15" stroke="#e8590c" stroke-width="1.5" stroke-dasharray="5,4"/>
  <g font-size="12" fill="currentColor" text-anchor="end">
    <text x="44" y="244">1</text><text x="44" y="207">2</text><text x="44" y="171">3</text>
    <text x="44" y="134">4</text><text x="44" y="97">5</text><text x="44" y="61">6</text><text x="44" y="24">7</text>
  </g>
  <g font-size="12" fill="currentColor" text-anchor="middle">
    <text x="50" y="258">0</text><text x="145" y="258">0.25</text><text x="240" y="258">0.5</text>
    <text x="335" y="258">0.75</text><text x="430" y="258">1</text>
  </g>
  <text x="240" y="276" font-size="13" fill="currentColor" text-anchor="middle">speed v/c</text>
  <text x="16" y="130" font-size="13" fill="currentColor" text-anchor="middle" transform="rotate(-90 16 130)">Lorentz factor γ</text>
  <polyline fill="none" stroke="#1c7ed6" stroke-width="2.5" points="50.0,240.0 69.0,240.0 88.0,239.8 107.0,239.6 126.0,239.2 145.0,238.8 164.0,238.2 183.0,237.5 202.0,236.7 221.0,235.6 240.0,234.3 259.0,232.8 278.0,230.8 297.0,228.4 316.0,225.3 335.0,221.2 354.0,215.6 373.0,207.1 392.0,192.5 393.9,190.5 395.8,188.2 397.7,185.8 399.6,183.1 401.5,180.2 403.4,176.9 405.3,173.3 407.2,169.2 409.1,164.6 411.0,159.2 412.9,153.0 414.8,145.7 416.7,136.9 418.6,125.8 420.5,111.7 422.4,92.4 424.3,64.2 426.2,16.7"/>
  <g fill="#1c7ed6">
    <circle cx="278" cy="230.8" r="3.5"/><circle cx="354" cy="215.6" r="3.5"/><circle cx="392" cy="192.5" r="3.5"/>
  </g>
  <g font-size="11" fill="currentColor">
    <text x="272" y="222" text-anchor="end">0.6c: γ = 1.25</text>
    <text x="348" y="207" text-anchor="end">0.8c: γ = 1.67</text>
    <text x="386" y="184" text-anchor="end">0.9c: γ = 2.29</text>
    <text x="420" y="16" text-anchor="end">0.99c: γ = 7.09</text>
  </g>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">The Lorentz factor $\gamma = 1/\sqrt{1-\beta^2}$. It stays within 1% of unity below about $0.14c$ and diverges as $v \to c$.</figcaption>
</figure>

<div class="interactive-demo" style="margin: 1em 0;">
  <label for="velocity-slider"><strong>Lorentz factor calculator.</strong> Speed $v/c$ = <span id="velocity-value">0.50</span></label><br/>
  <input type="range" id="velocity-slider" min="0" max="0.999" step="0.001" value="0.5" style="width: 100%; max-width: 420px;" />
  <p style="margin: 0.4em 0 0;">γ = <span id="gamma-value">1.155</span>. One hour of proper time on the moving clock spans <span id="dilated-time">1.155</span> hours in the observer's frame; a 1 m rod is measured as <span id="contracted-length">0.866</span> m long.</p>
</div>

<script>
  (function () {
    var slider = document.getElementById('velocity-slider');
    if (!slider) return;
    slider.addEventListener('input', function (e) {
      var v = parseFloat(e.target.value);
      var gamma = 1 / Math.sqrt(1 - v * v);
      document.getElementById('velocity-value').textContent = v.toFixed(3);
      document.getElementById('gamma-value').textContent = gamma.toFixed(3);
      document.getElementById('dilated-time').textContent = gamma.toFixed(3);
      document.getElementById('contracted-length').textContent = (1 / gamma).toFixed(3);
    });
  })();
</script>

### The twin paradox

One twin stays on Earth; the other travels at $0.8c$ ($\gamma = 5/3$) to a star 4 light-years away and returns. Earth time elapsed: $2 \times 4/0.8 = 10$ years. Traveller's proper time: $10/\gamma = 6$ years. The situation is not symmetric: the traveller changes inertial frame at turnaround, and in general the elapsed proper time along a worldline,

$$\tau = \int \sqrt{1 - \frac{v(t)^2}{c^2}}\; dt$$

is **maximized** by the inertial (straight) worldline between two events. The staying twin's worldline is the straight one. Acceleration is not the cause of the age difference (it can be made arbitrarily brief); the geometry of the two paths is. The same principle, "free fall maximizes proper time," becomes the geodesic principle of [general relativity](tensor-formalism.html#geodesics-as-extremal-proper-time).

### GPS: special and general relativity together

GPS satellites orbit at about 3.9 km/s at an altitude of 20,200 km. Special-relativistic time dilation slows their clocks by about **7 µs per day** ($\gamma - 1 \approx 8.3 \times 10^{-11}$). Their higher gravitational potential speeds them up by about **45 µs per day** (a general-relativistic effect). The net drift is about **+38 µs per day**. The satellite clocks are therefore set to tick slightly slow before launch (10.22999999543 MHz instead of 10.23 MHz); without the correction ranging errors would grow by roughly 10 km per day.

## Length Contraction

An object has its greatest length, the **proper length** $L_0$, in its rest frame. Measured in a frame where it moves along its length at speed $v$, with both ends located at the same time in that frame,

$$L = \frac{L_0}{\gamma}$$

Dimensions perpendicular to the motion are unchanged. Contraction is a consequence of the relativity of simultaneity: the two frames disagree about which pair of end-events are simultaneous, so they measure different spatial separations.

Two points often misunderstood:

- **Visual appearance.** A photograph of a fast-moving object does not simply show it contracted. Light from different parts leaves at different times, and a sphere still appears circular, rotated rather than flattened (Terrell–Penrose effect, 1959).
- **Ladder (pole-in-barn) paradox.** A ladder that "fits" in a barn in the barn frame does not fit in the ladder frame. Both are right, because "the ladder is entirely inside at one instant" refers to simultaneous events and is frame-dependent.

## Velocity Addition

If an object moves at velocity $w$ along $x'$ in S', and S' moves at $v$ relative to S, its velocity in S is

$$u = \frac{v + w}{1 + vw/c^2}$$

Equivalently, rapidities add: $\tanh^{-1}(u/c) = \tanh^{-1}(v/c) + \tanh^{-1}(w/c)$. For $w = c$, $u = c$ for every $v$, recovering the second postulate. For $v, w < c$, $u < c$.

**Example.** A spacecraft at $0.9c$ launches a probe forward at $0.9c$ relative to itself. The probe's speed relative to Earth is

$$u = \frac{0.9c + 0.9c}{1 + 0.81} = \frac{1.8c}{1.81} \approx 0.994c$$

For velocities with components perpendicular to the boost, $u_\perp = w_\perp / \bigl[\gamma\,(1 + v w_x/c^2)\bigr]$. Two non-collinear boosts do not compose to a pure boost; the extra rotation is the **Thomas–Wigner rotation**, responsible for the Thomas precession factor of 1/2 in atomic spin–orbit coupling.

## The Relativistic Doppler Effect

A source emitting frequency $f_s$ in its rest frame, moving directly toward an observer at speed $v$, is received at

$$f_{\text{obs}} = f_s \sqrt{\frac{1 + \beta}{1 - \beta}}$$

and at $f_s\sqrt{(1-\beta)/(1+\beta)}$ when receding. For motion at angle $\theta$ to the line of sight (measured in the observer's frame), $f_{\text{obs}} = f_s / \bigl[\gamma\,(1 - \beta\cos\theta)\bigr]$. At $\theta = 90°$ there is a pure **transverse Doppler shift** $f_{\text{obs}} = f_s/\gamma$, with no classical counterpart; it is time dilation seen directly. Ives and Stilwell first observed the second-order effect in 1938.

## Four-Vectors

A **four-vector** is a set of four components that transforms like $x^\mu$ under Lorentz transformations, $A'^\mu = \Lambda^\mu{}_\nu A^\nu$. The inner product $A \cdot B = \eta_{\mu\nu} A^\mu B^\nu = -A^0 B^0 + \mathbf{A}\cdot\mathbf{B}$ is invariant. Indices are raised and lowered with $\eta$: $A_\mu = \eta_{\mu\nu} A^\nu = (-A^0, \mathbf{A})$. Repeated upper and lower indices are summed.

| Four-vector | Components | Invariant square |
|---|---|---|
| Position | $x^\mu = (ct, \mathbf{x})$ | $x \cdot x = -c^2t^2 + \lvert\mathbf{x}\rvert^2$ |
| Four-velocity | $u^\mu = dx^\mu/d\tau = \gamma\,(c, \mathbf{v})$ | $u \cdot u = -c^2$ |
| Four-momentum | $p^\mu = m u^\mu = (E/c, \mathbf{p})$ | $p \cdot p = -m^2c^2$ |
| Four-acceleration | $a^\mu = du^\mu/d\tau$ | $a \cdot a = \alpha^2$ (proper acceleration squared); $a \cdot u = 0$ |
| Wave four-vector | $k^\mu = (\omega/c, \mathbf{k})$ | $k \cdot k = 0$ for light |
| Four-current | $J^\mu = (c\rho, \mathbf{J})$ | $\partial_\mu J^\mu = 0$ expresses charge conservation |

Because inner products are invariant, many problems are fastest in whichever frame makes them simplest. For example, the Doppler formula follows from evaluating the invariant $k \cdot u$ in the source frame and the observer frame.

## Relativistic Dynamics

### Momentum and energy

The spatial and time components of $p^\mu = m u^\mu$ are

$$\mathbf{p} = \gamma m \mathbf{v}, \qquad E = \gamma m c^2$$

The invariant $p \cdot p = -m^2c^2$ gives the **energy–momentum relation**

$$E^2 = (pc)^2 + (mc^2)^2$$

Special cases:

- **At rest** ($\mathbf{p} = 0$): $E = mc^2$, the rest energy.
- **Slow motion**: $E = \gamma mc^2 = mc^2 + \tfrac{1}{2}mv^2 + \tfrac{3}{8}mv^4/c^2 + \cdots$, so Newtonian kinetic energy is the first correction to rest energy.
- **Kinetic energy**: $K = (\gamma - 1)\,mc^2$.
- **Massless particles** ($m = 0$): $E = pc$ and $v = c$. Photons carry momentum $p = E/c = h/\lambda$.
- **Ultra-relativistic** ($E \gg mc^2$): $E \approx pc$; the velocity is $v/c = pc/E$.

Observers disagree on $E$ and $\mathbf{p}$ separately but agree on $m$, the invariant length of $p^\mu$. Modern usage reserves "mass" for this invariant; the older "relativistic mass" $\gamma m$ is just $E/c^2$ and is avoided because it does not correspond to the inertial response to force in all directions.

### Force

Newton's second law generalizes as $\mathbf{F} = d\mathbf{p}/dt$, or covariantly $f^\mu = dp^\mu/d\tau$ (the **four-force**). Because $\gamma$ depends on speed, force and acceleration are not parallel in general:

$$\mathbf{F}_\parallel = \gamma^3 m\, \mathbf{a}_\parallel, \qquad \mathbf{F}_\perp = \gamma\, m\, \mathbf{a}_\perp$$

It takes ever more force to increase speed near $c$, which is why particle accelerators add energy while the speed barely changes. A body with constant proper acceleration $\alpha$ follows **hyperbolic motion**, $x^2 - c^2t^2 = c^4/\alpha^2$, asymptotically approaching a light ray; its proper time grows only logarithmically with coordinate time.

### Mass–energy equivalence

The rest energy $mc^2$ is real energy. The mass of a composite system is the invariant mass of its total four-momentum and includes internal kinetic and binding energy:

$$M c^2 = \sqrt{E_{\text{tot}}^2 - (p_{\text{tot}}c)^2}$$

A hydrogen atom is lighter than a free proton plus electron by 13.6 eV/$c^2$; a helium-4 nucleus is lighter than two protons and two neutrons by about 28.3 MeV/$c^2$ (0.75% of its mass), which is the energy released in fusion. Most of the mass of ordinary matter is the energy of quarks and gluons confined in nucleons, not the rest mass of the quarks themselves.

**Example.** One gram of mass corresponds to

$$E = (10^{-3}\ \text{kg})\,(3.00 \times 10^{8}\ \text{m/s})^2 = 9.0 \times 10^{13}\ \text{J}$$

about 21 kilotons of TNT. Chemical reactions convert roughly $10^{-10}$ of the rest energy of the reactants, fission about $10^{-3}$, and hydrogen fusion about $7 \times 10^{-3}$.

### Collisions and thresholds

Four-momentum is conserved in every collision. The invariant $s = -(p_1 + p_2)^2 c^2$ (the squared centre-of-mass energy) determines what can be produced. A beam of energy $E$ striking a stationary target of mass $m$ has $\sqrt{s} \approx \sqrt{2 E\, mc^2}$ for $E \gg mc^2$, growing only as the square root of beam energy, whereas two colliding beams of energy $E$ give $\sqrt{s} = 2E$. This is why high-energy physics moved to colliders.

## Electromagnetism

Maxwell's equations are already Lorentz-covariant; special relativity reveals their structure. The electric and magnetic fields are components of a single antisymmetric **field-strength tensor** $F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu$ built from the four-potential $A^\mu = (\phi/c, \mathbf{A})$, and Maxwell's equations become

$$\partial_\mu F^{\mu\nu} = -\mu_0 J^\nu, \qquad \partial_{[\lambda} F_{\mu\nu]} = 0$$

(the sign of the first depends on the index convention for $F^{0i}$). Under a boost with velocity $\mathbf{v}$ the field components parallel to $\mathbf{v}$ are unchanged, and the perpendicular components mix:

$$\mathbf{E}'_\perp = \gamma\,(\mathbf{E} + \mathbf{v} \times \mathbf{B})_\perp, \qquad \mathbf{B}'_\perp = \gamma\left(\mathbf{B} - \frac{\mathbf{v} \times \mathbf{E}}{c^2}\right)_\perp$$

A purely electric field in one frame has a magnetic part in another; magnetism is, in this sense, the relativistic companion of electrostatics. The combinations $\lvert\mathbf{E}\rvert^2 - c^2\lvert\mathbf{B}\rvert^2$ and $\mathbf{E}\cdot\mathbf{B}$ are Lorentz invariants.

## Experimental Tests

Special relativity is among the most precisely tested theories in physics, and it is built into the design of particle accelerators, synchrotron light sources, and satellite navigation.

| Test | What it checks | Result |
|---|---|---|
| Michelson–Morley (1887) and modern cavity versions | Isotropy of $c$ | Modern optical-resonator experiments find no anisotropy at the $\sim 10^{-18}$ level |
| Kennedy–Thorndike (1932) and successors | Independence of $c$ from the lab's velocity | Null; improved by many orders of magnitude in modern cryogenic-resonator versions |
| Ives–Stilwell (1938); storage-ring spectroscopy (GSI, 2014) | Time-dilation factor via Doppler shifts | Agrees with $\gamma$ to about $2 \times 10^{-8}$ for ions at $0.34c$ |
| Cosmic-ray muons (Rossi–Hall, 1941) | Time dilation | Muons from ~15 km altitude reach the ground despite a 2.2 µs lifetime |
| CERN muon storage ring (1977) | Time dilation at $\gamma \approx 29.3$ | Lifetime dilated from 2.2 µs to about 64 µs, as predicted, to ~0.1% |
| Hafele–Keating (1971) | Combined kinematic and gravitational clock shifts | Agreement with predictions for flights east and west |
| Optical atomic clocks (NIST, 2010) | Time dilation at everyday speeds | Detected at relative speeds below 10 m/s |
| Particle accelerators | $E^2 = (pc)^2 + (mc^2)^2$, speed limit $c$ | Electrons at LEP reached $\gamma \sim 2 \times 10^5$ without exceeding $c$ |
| Gamma-ray bursts, GW170817 | Energy-independence of $c$; speed of gravity | No dispersion found at Planck-scale sensitivity; gravitational waves travel at $c$ to $\sim 10^{-15}$ |

Searches for small violations of Lorentz invariance, parametrized in the Standard-Model Extension, continue across atomic, nuclear, particle, and astrophysical systems; all results to date are null. Why such violations are expected in some quantum-gravity models is discussed in [Toward Quantum Gravity](quantum-gravity.html#experimental-situation).

## Summary of Key Formulas

| Quantity | Formula |
|---|---|
| Lorentz factor | $\gamma = 1/\sqrt{1 - v^2/c^2}$ |
| Interval | $\Delta s^2 = -c^2\Delta t^2 + \Delta x^2 + \Delta y^2 + \Delta z^2$ |
| Boost | $ct' = \gamma(ct - \beta x)$, $x' = \gamma(x - \beta ct)$ |
| Time dilation | $\Delta t = \gamma\,\Delta\tau$ |
| Length contraction | $L = L_0/\gamma$ |
| Simultaneity offset | $\Delta t' = vL_0/c^2$ |
| Velocity addition | $u = (v + w)/(1 + vw/c^2)$ |
| Longitudinal Doppler | $f_{\text{obs}} = f_s\sqrt{(1+\beta)/(1-\beta)}$ (approaching) |
| Energy, momentum | $E = \gamma mc^2$, $\mathbf{p} = \gamma m\mathbf{v}$, $E^2 = (pc)^2 + (mc^2)^2$ |
| Kinetic energy | $K = (\gamma - 1)mc^2$ |

## See Also

**Up:** [Relativity](./) — overview and navigation hub. **Next:** [General Relativity](general-relativity.html) — gravity as the curvature of spacetime.

- [Tensor Formalism & the Field Equations](tensor-formalism.html) — the index notation and geometry used above, extended to curved spacetime.
- [General Relativity](general-relativity.html) — the equivalence principle and Einstein's field equations.
- [Graduate Formalism & Frontiers](advanced.html) — overview of the advanced relativity pages.
- [Classical Mechanics](../classical-mechanics/) — the low-speed limit special relativity reduces to.
- [Quantum Field Theory](../quantum-field-theory.html) — special relativity combined with quantum mechanics.
