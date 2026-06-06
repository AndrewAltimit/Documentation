---
layout: docs
title: "Classical Mechanics: Computational Methods"
permalink: /docs/physics/classical-mechanics/computational-classical-mechanics.html
toc: true
toc_sticky: true
hide_title: true
---

[Classical Mechanics](./) &raquo; Computational Methods

Symplectic and variational integrators, molecular dynamics, N-body methods, and the numerical analysis that keeps long simulations physical.

## Why Numerics Need the Physics

Computers cannot solve differential equations exactly—they advance the state of a system in small discrete steps. A naive scheme such as forward Euler accumulates error every step, and for mechanical systems that error is not random: it systematically violates the conservation laws that define the dynamics. Energy drifts, orbits spiral in or out, and after enough steps the simulation describes a system that no longer resembles the one you started with.

The central lesson of computational classical mechanics is that the *structure* of the equations matters as much as their local accuracy. Hamilton's equations are not arbitrary ODEs: their flow preserves a symplectic 2-form (and hence phase-space volume, by Liouville's theorem). Integrators that preserve this structure—**symplectic integrators**—are often less accurate per step than a high-order Runge–Kutta method, yet vastly more faithful over millions of steps, because the errors they make are bounded rather than secular. This page develops those methods, the numerical-stability analysis that justifies them, and their use in molecular dynamics and gravitational N-body simulation.

## Symplectic Integrators: Preserving What Matters

### The geometry to be preserved

A Hamiltonian system with coordinates $q$ and conjugate momenta $p$ evolves under

$$
\dot{q} = \frac{\partial H}{\partial p}, \qquad \dot{p} = -\frac{\partial H}{\partial q}.
$$

The time-$t$ flow map $\phi_t$ is a **canonical (symplectic) transformation**: it preserves the 2-form $\omega = \sum_i dp_i \wedge dq_i$. A numerical one-step map $\Phi_h$ is called *symplectic* if it preserves the same form,

$$
\Phi_h^{*}\, \omega = \omega,
$$

equivalently if its Jacobian $M = \partial \Phi_h / \partial(q,p)$ satisfies $M^{\top} J M = J$ with

$$
J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}.
$$

A symplectic map automatically preserves phase-space volume, so it cannot produce the artificial attractors or repellers that plague non-symplectic schemes.

### First-order: the symplectic Euler method

The simplest symplectic scheme updates one variable before the other so that the composite map is canonical. For a separable Hamiltonian $H(q,p) = T(p) + V(q)$:

$$
p_{n+1} = p_n - h\, \frac{\partial V}{\partial q}(q_n), \qquad
q_{n+1} = q_n + h\, \frac{\partial T}{\partial p}(p_{n+1}).
$$

Note the asymmetry: $p$ is updated using the *old* position, then $q$ is updated using the *new* momentum. That ordering is exactly what makes the Jacobian symplectic.

```python
def symplectic_euler(q, p, H, dt):
    """First-order symplectic integrator"""
    p_new = p - dt * grad_q(H, q, p)
    q_new = q + dt * grad_p(H, q, p_new)
    return q_new, p_new
```

### Second-order: the Störmer–Verlet / leapfrog method

By symmetrizing the half-kick / drift / half-kick pattern we obtain a second-order, *time-reversible*, symplectic method:

$$
p_{n+1/2} = p_n - \tfrac{1}{2} h\, \frac{\partial V}{\partial q}(q_n),
$$

$$
q_{n+1} = q_n + h\, \frac{\partial T}{\partial p}(p_{n+1/2}),
$$

$$
p_{n+1} = p_{n+1/2} - \tfrac{1}{2} h\, \frac{\partial V}{\partial q}(q_{n+1}).
$$

```python
def stormer_verlet(q, p, H, dt):
    """Second-order symplectic integrator"""
    p_half = p - 0.5*dt * grad_q(H, q, p)
    q_new = q + dt * grad_p(H, q, p_half)
    p_new = p_half - 0.5*dt * grad_q(H, q_new, p_half)
    return q_new, p_new
```

For the common case $T(p) = p^2/2m$ this reduces to the familiar **velocity Verlet** update used throughout molecular dynamics (below): a position step, a force evaluation, and a velocity step that splits the kick into two halves.

### Why symplectic methods conserve energy so well

A symplectic integrator does not conserve the true Hamiltonian $H$ exactly. Instead, **backward error analysis** shows that $\Phi_h$ is the *exact* time-$h$ flow of a nearby **shadow Hamiltonian**

$$
\tilde{H} = H + h^2 H_2 + h^4 H_4 + \cdots
$$

The integrator conserves $\tilde{H}$ to all orders (the asymptotic series is not generally convergent, but is exponentially accurate up to a remainder $\sim e^{-c/h}$). Because $\tilde{H} = H + O(h^p)$, the measured energy $H$ oscillates within an $O(h^p)$ band but exhibits **no secular drift** over exponentially long times. This is the precise statement behind the empirical observation that leapfrog can integrate a Kepler orbit for billions of periods without losing it.

By contrast, a non-symplectic method of the same order has no shadow Hamiltonian; its energy error grows linearly (or worse) with the number of steps.

### Higher-order symplectic schemes

Higher order is obtained by *composition*. If $\Phi_h$ is a symmetric second-order method, the **Yoshida** composition

$$
\Psi_h = \Phi_{w_1 h} \circ \Phi_{w_0 h} \circ \Phi_{w_1 h}, \qquad
w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}, \quad
w_1 = \frac{1}{2 - 2^{1/3}},
$$

is symplectic and fourth-order. The negative central weight $w_0$ (a backward sub-step) is unavoidable: any symplectic method of order $> 2$ built by composition must take some steps with negative time. Recursing the construction yields sixth-, eighth-, and higher-order symplectic integrators, all reusing the same elementary kick/drift maps.

```python
import numpy as np

# Fourth-order Yoshida coefficients for kick-drift-kick base method
cbrt2 = 2.0 ** (1.0 / 3.0)
w1 = 1.0 / (2.0 - cbrt2)
w0 = -cbrt2 / (2.0 - cbrt2)
weights = [w1, w0, w1]

def yoshida4_step(q, p, force, dt, mass=1.0):
    """Fourth-order symplectic step by Yoshida composition of leapfrog."""
    for w in weights:
        h = w * dt
        p = p + 0.5 * h * force(q)          # half kick
        q = q + h * p / mass                # drift
        p = p + 0.5 * h * force(q)          # half kick
    return q, p
```

## Variational Integrators: Discrete Mechanics Done Right

Recall that Lagrangian mechanics arises from extremizing the action. **Variational integrators** apply this principle *directly to discrete time*, rather than discretizing the resulting differential equations. The payoff is excellent long-term conservation of a discrete momentum and energy—exactly what lets us simulate the solar system for millions of years.

We replace the continuous action by a sum over a **discrete Lagrangian** $L_d(q_k, q_{k+1}, h)$, which approximates the action over one step:

$$
L_d(q_k, q_{k+1}, h) \approx \int_{t_k}^{t_{k+1}} L\!\left(q(t), \dot q(t)\right)\, dt,
$$

so that the discrete action is

$$
S_d = \sum_k L_d(q_k, q_{k+1}, h).
$$

Extremizing $S_d$ over the interior points $q_k$ gives the **discrete Euler–Lagrange equations**:

$$
D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1}) = 0,
$$

where $D_1$ and $D_2$ denote partial derivatives with respect to the first and second slot. This is a two-step recurrence: given $(q_{k-1}, q_k)$ it determines $q_{k+1}$ implicitly.

The discrete momenta defined by

$$
p_k = -D_1 L_d(q_k, q_{k+1}) = D_2 L_d(q_{k-1}, q_k)
$$

turn this into the equivalent one-step **position–momentum form** $(q_k, p_k) \mapsto (q_{k+1}, p_{k+1})$. Two structural theorems make these methods attractive:

- **Symplecticity.** The resulting map is automatically symplectic, regardless of the quadrature used to build $L_d$—the discrete variational structure guarantees it.
- **Discrete Noether theorem.** Every symmetry of $L_d$ produces an *exactly* conserved discrete momentum. A rotation-invariant $L_d$ conserves discrete angular momentum to machine precision for all time.

A simple choice—the **midpoint discrete Lagrangian**

$$
L_d(q_k, q_{k+1}) = h\, L\!\left(\frac{q_k + q_{k+1}}{2},\; \frac{q_{k+1} - q_k}{h}\right)
$$

—reproduces the implicit midpoint rule, while the trapezoidal choice reproduces Störmer–Verlet. Variational integrators thus unify many familiar symplectic schemes under one principle and extend naturally to constrained systems (via discrete Lagrange multipliers) and to systems with forcing or dissipation (via a discrete Lagrange–d'Alembert principle).

## Molecular Dynamics

Molecular dynamics (MD) integrates Newton's equations for a collection of interacting particles, almost always with the **velocity Verlet** form of Störmer–Verlet because its symplecticity controls energy drift over the millions of steps a simulation requires.

```python
def verlet_integration(positions, velocities, forces, dt, mass):
    """Velocity Verlet algorithm for MD simulation"""
    # Update positions
    positions += velocities * dt + 0.5 * forces/mass * dt**2

    # Calculate new forces
    forces_new = calculate_forces(positions)

    # Update velocities
    velocities += 0.5 * (forces + forces_new)/mass * dt

    return positions, velocities, forces_new
```

### The interaction potential

The physics lives in `calculate_forces`. A canonical pairwise model is the **Lennard-Jones** potential, capturing short-range Pauli repulsion and long-range van der Waals attraction:

$$
U(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right],
\qquad
\mathbf{F}_{ij} = -\nabla U = \frac{24\varepsilon}{r}\left[2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]\hat{\mathbf{r}}_{ij}.
$$

```python
import numpy as np

def lennard_jones_forces(positions, epsilon=1.0, sigma=1.0, cutoff=2.5):
    """Total LJ force on each particle, with a distance cutoff."""
    n = len(positions)
    forces = np.zeros_like(positions)
    rc2 = cutoff * cutoff
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r2 = rij @ rij
            if r2 >= rc2 or r2 == 0.0:
                continue
            inv_r2 = 1.0 / r2
            sr2 = sigma * sigma * inv_r2
            sr6 = sr2 ** 3
            sr12 = sr6 * sr6
            # magnitude/r so we can multiply by the vector rij directly
            fmag = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2
            fij = fmag * rij
            forces[i] += fij
            forces[j] -= fij
    return forces
```

### Thermostats: sampling the right ensemble

Plain velocity Verlet samples the **microcanonical (NVE)** ensemble—constant energy. To model a system at fixed temperature $T$ (the **canonical, NVT** ensemble) we couple the particles to a heat bath via a *thermostat*. The instantaneous temperature follows from equipartition,

$$
\frac{3}{2} N k_B T_{\text{inst}} = \frac{1}{2}\sum_i m_i \lvert \mathbf{v}_i \rvert^2 ,
$$

and a thermostat nudges $T_{\text{inst}}$ toward the target. The Berendsen scheme rescales velocities by a factor

$$
\lambda = \sqrt{1 + \frac{\Delta t}{\tau}\left(\frac{T_{\text{target}}}{T_{\text{inst}}} - 1\right)},
$$

which is simple but does not generate the exact canonical distribution; the **Nosé–Hoover** thermostat (an extended-Lagrangian method) and **Langevin** dynamics (adding friction and a fluctuating force consistent with the fluctuation–dissipation theorem) do. The choice of thermostat is itself a numerical-physics decision: a poorly tuned coupling time $\tau$ distorts dynamical quantities like diffusion coefficients even when it reproduces the correct average temperature.

### Periodic boundaries and neighbor lists

Bulk matter is modeled with **periodic boundary conditions** and the **minimum-image convention**, so each particle interacts with the nearest periodic image of every other. The naive double loop above is $O(N^2)$; production codes restore near-linear scaling with **cell lists** (bin particles into a grid of boxes of side $\geq r_{\text{cut}}$ and only check neighboring boxes) or **Verlet neighbor lists** (cache the neighbors within $r_{\text{cut}} + r_{\text{skin}}$ and rebuild only when a particle has moved more than half the skin distance).

## N-body Methods (Celestial Mechanics)

The gravitational **N-body problem** has no general closed-form solution for $N \geq 3$. Each body feels

$$
\ddot{\mathbf{r}}_i = G \sum_{j \neq i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{\lvert \mathbf{r}_j - \mathbf{r}_i \rvert^{3}}.
$$

```python
import numpy as np

def gravitational_acceleration(positions, masses, G=1.0, softening=1e-3):
    """Pairwise gravitational acceleration with Plummer softening."""
    n = len(masses)
    acc = np.zeros_like(positions)
    eps2 = softening * softening
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = positions[j] - positions[i]
            r2 = d @ d + eps2          # softening avoids singular close encounters
            inv_r3 = r2 ** -1.5
            acc[i] += G * masses[j] * d * inv_r3
    return acc
```

### Softening and adaptive time-stepping

Two close bodies produce a $1/r^2$ force that blows up numerically. **Plummer softening** replaces $r^2$ by $r^2 + \varepsilon^2$, bounding the force at small separation—appropriate for collisionless systems (galaxies) where you are not resolving genuine two-body encounters. For collisional systems (star clusters, planetary systems) one instead uses an **individual / adaptive time-step**, shrinking $h$ during close encounters so the fast pair is resolved while distant bodies coast. Symplectic integrators with fixed $h$ lose their structure-preserving guarantee under naive step-size changes, motivating *symplectic* adaptive schemes and regularization techniques (e.g. Kustaanheimo–Stiefel) that remove the singularity by a change of variables.

### Fast force evaluation

The direct sum is $O(N^2)$ per step—prohibitive beyond $\sim 10^4$ bodies. Two approximations dominate large simulations:

- **Barnes–Hut tree code**, $O(N \log N)$: build an octree, and approximate the pull of a distant, sufficiently compact cell by its center of mass (controlled by an opening angle $\theta$).
- **Fast Multipole Method (FMM)**, $O(N)$: expand the potential of distant groups in multipoles *and* evaluate them efficiently for whole groups of targets, with rigorous error bounds.

For periodic gravitational or electrostatic systems, **Particle–Mesh (PM)** and **Particle–Particle/Particle–Mesh (P³M)** methods solve Poisson's equation on a grid with an FFT, combining a fast long-range mesh force with a direct short-range correction; **Ewald summation** plays the analogous role for Coulomb interactions in MD.

### The restricted three-body problem

A tractable special case fixes two massive bodies (the *primaries*) in circular orbit and studies a massless test particle. In the co-rotating frame the dynamics admit a conserved **Jacobi integral**, and five equilibria—the **Lagrange points** $L_1$–$L_5$—appear. $L_1$–$L_3$ are saddle-unstable; $L_4$ and $L_5$ are stable for mass ratios below the Routh value $\approx 0.0385$, which is why the Sun–Jupiter $L_4/L_5$ points trap the Trojan asteroids and why spacecraft (e.g. JWST at Sun–Earth $L_2$) are stationed near these points with modest station-keeping.

## Numerical Stability and Energy Drift

### Local vs. global error, and order

A one-step method of **order $p$** has local truncation error $O(h^{p+1})$ per step and global error $O(h^{p})$ over a fixed time interval. But for long mechanical integrations the relevant question is not the order—it is whether the error *accumulates*. The contrast between methods is sharpest here:

| Method | Order | Symplectic | Time-reversible | Energy behavior over many steps |
|--------|:-----:|:----------:|:---------------:|---------------------------------|
| Forward (explicit) Euler | 1 | No | No | Secular drift; energy grows, orbits spiral out |
| Symplectic Euler | 1 | Yes | No | Bounded oscillation, no drift |
| Störmer–Verlet / leapfrog | 2 | Yes | Yes | Bounded oscillation, no drift |
| RK4 | 4 | No | No | Small per-step error but slow secular drift |
| Yoshida (composed) | 4, 6, … | Yes | Yes | Bounded oscillation, no drift |

The takeaway: a non-symplectic high-order method can beat a symplectic low-order one on a *short* run, but on a long run the symplectic method wins because its energy error never accumulates.

### Linear stability and the harmonic oscillator

Stability is analyzed on the harmonic oscillator $H = \tfrac{1}{2}(p^2 + \omega^2 q^2)$, whose exact flow is a rotation. Each integrator becomes a linear map $(q,p) \mapsto M (q,p)$, and stability requires the eigenvalues of $M$ to lie on the unit circle. For leapfrog one finds

$$
\det M = 1 \quad\text{(symplectic, exactly)}, \qquad
\text{stable} \iff h\omega < 2 .
$$

So leapfrog is *conditionally stable*: the step must resolve the fastest oscillation, $h < 2/\omega_{\max}$. This is a genuine constraint in MD (stiff bond vibrations force $h \sim 1$ fs) and motivates **multiple-time-step** integrators such as **r-RESPA**, which evaluate cheap stiff forces often and expensive slow forces rarely while staying symplectic. Explicit Euler, by contrast, has $\det M = 1 + (h\omega)^2 > 1$: its phase-space map is *expanding* for every $h$, the geometric origin of its energy growth.

### Roundoff and the symmetry of the update

Beyond truncation error, finite floating-point precision injects roundoff each step. For very long integrations this is mitigated by **compensated (Kahan) summation** of the position update and by writing the update in *increment* form so that small displacements are not lost in the low bits of large coordinates. Time-reversible methods like Verlet also benefit from the fact that their errors are even in $h$, leaving no first-order drift to accumulate.

### Demonstration: comparing integrators

The following script integrates a pendulum with explicit Euler, symplectic Euler, and leapfrog, and plots the phase-space orbits, the energy error over time, and the long-run phase-space volume. It makes the table above concrete: Euler's orbit spirals outward and its energy diverges on a log scale, while both symplectic methods trace closed curves with bounded energy error.

```python
import numpy as np
import matplotlib.pyplot as plt

def hamiltonian_pendulum(q, p, m=1, l=1, g=9.81):
    """Hamiltonian for simple pendulum"""
    return p**2/(2*m*l**2) + m*g*l*(1 - np.cos(q))

def euler_step(q, p, H, dt):
    """Standard Euler method (not symplectic)"""
    dH_dq = (H(q + 1e-8, p) - H(q, p))/1e-8
    dH_dp = (H(q, p + 1e-8) - H(q, p))/1e-8

    q_new = q + dt * dH_dp
    p_new = p - dt * dH_dq
    return q_new, p_new

def symplectic_euler_step(q, p, H, dt):
    """Symplectic Euler method"""
    dH_dq = (H(q + 1e-8, p) - H(q, p))/1e-8
    p_new = p - dt * dH_dq

    dH_dp = (H(q, p_new + 1e-8) - H(q, p_new))/1e-8
    q_new = q + dt * dH_dp
    return q_new, p_new

def leapfrog_step(q, p, H, dt):
    """Leapfrog/Störmer-Verlet method"""
    dH_dq = (H(q + 1e-8, p) - H(q, p))/1e-8
    p_half = p - 0.5*dt * dH_dq

    dH_dp = (H(q, p_half + 1e-8) - H(q, p_half))/1e-8
    q_new = q + dt * dH_dp

    dH_dq_new = (H(q_new + 1e-8, p_half) - H(q_new, p_half))/1e-8
    p_new = p_half - 0.5*dt * dH_dq_new
    return q_new, p_new

# Compare integrators
q0, p0 = 3.0, 0.0  # Large amplitude
dt = 0.1
n_steps = 10000

# Storage for trajectories
trajectories = {
    'Euler': {'q': [q0], 'p': [p0], 'E': []},
    'Symplectic Euler': {'q': [q0], 'p': [p0], 'E': []},
    'Leapfrog': {'q': [q0], 'p': [p0], 'E': []}
}

# Run simulations
for method, integrator in [('Euler', euler_step),
                           ('Symplectic Euler', symplectic_euler_step),
                           ('Leapfrog', leapfrog_step)]:
    q, p = q0, p0
    for _ in range(n_steps):
        q, p = integrator(q, p, hamiltonian_pendulum, dt)
        trajectories[method]['q'].append(q)
        trajectories[method]['p'].append(p)
        trajectories[method]['E'].append(hamiltonian_pendulum(q, p))

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Phase space
ax = axes[0]
for method, style in [('Euler', 'r-'), ('Symplectic Euler', 'g-'),
                     ('Leapfrog', 'b-')]:
    traj = trajectories[method]
    ax.plot(traj['q'], traj['p'], style, alpha=0.7, label=method)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p_\theta$')
ax.set_title('Phase Space Trajectories')
ax.legend()
ax.grid(True, alpha=0.3)

# Energy conservation
ax = axes[1]
t = np.arange(n_steps + 1) * dt
E0 = hamiltonian_pendulum(q0, p0)
for method, style in [('Euler', 'r-'), ('Symplectic Euler', 'g-'),
                     ('Leapfrog', 'b-')]:
    E = np.array([E0] + trajectories[method]['E'])
    ax.semilogy(t, np.abs(E - E0) + 1e-16, style, label=method)
ax.set_xlabel('Time')
ax.set_ylabel('Energy Error')
ax.set_title('Energy Conservation')
ax.legend()
ax.grid(True, alpha=0.3)

# Phase space area preservation
ax = axes[2]
for method, color in [('Euler', 'red'), ('Symplectic Euler', 'green'),
                     ('Leapfrog', 'blue')]:
    traj = trajectories[method]
    # Sample points in phase space
    q_vals = np.array(traj['q'][::100])
    p_vals = np.array(traj['p'][::100])
    ax.scatter(q_vals[:50], p_vals[:50], c=color, alpha=0.6,
              label=f'{method} (early)', s=30)
    ax.scatter(q_vals[-50:], p_vals[-50:], c=color, alpha=0.6,
              marker='x', label=f'{method} (late)', s=30)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p_\theta$')
ax.set_title('Phase Space Volume Preservation')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

<details>
<summary><b>Expected Output</b></summary>
<br>
The figure has three panels:
<ol>
<li><b>Phase space</b> — the Euler orbit (red) spirals outward as energy is artificially injected, while symplectic Euler (green) and leapfrog (blue) trace nearly closed loops.</li>
<li><b>Energy error (log scale)</b> — Euler's error climbs steadily; the symplectic methods oscillate within a fixed band with no upward trend.</li>
<li><b>Phase-space volume</b> — early-vs-late sample points stay co-located for the symplectic methods (volume preserved) but disperse for Euler.</li>
</ol>
</details>

### A complete leapfrog N-body simulation

Putting the pieces together, the following self-contained example evolves a small gravitational system with the kick–drift–kick leapfrog scheme and tracks the conserved total energy as a stability diagnostic.

```python
import numpy as np

def total_energy(positions, velocities, masses, G=1.0, softening=1e-3):
    """Kinetic plus (softened) gravitational potential energy."""
    ke = 0.5 * np.sum(masses[:, None] * velocities**2)
    pe = 0.0
    eps2 = softening * softening
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[j] - positions[i]
            r = np.sqrt(d @ d + eps2)
            pe -= G * masses[i] * masses[j] / r
    return ke + pe

def leapfrog_nbody(positions, velocities, masses, dt, n_steps,
                   G=1.0, softening=1e-3):
    """Kick-drift-kick leapfrog (velocity Verlet) for an N-body system."""
    acc = gravitational_acceleration(positions, masses, G, softening)
    energies = [total_energy(positions, velocities, masses, G, softening)]
    for _ in range(n_steps):
        velocities += 0.5 * dt * acc                 # half kick
        positions += dt * velocities                 # drift
        acc = gravitational_acceleration(positions, masses, G, softening)
        velocities += 0.5 * dt * acc                 # half kick
        energies.append(
            total_energy(positions, velocities, masses, G, softening)
        )
    return positions, velocities, np.array(energies)

# A simple Sun-planet-moon hierarchy (units: G = 1)
masses = np.array([1.0, 1e-3, 1e-6])
positions = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [1.01, 0.0]])
velocities = np.array([[0.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 1.0 + 0.0316]])   # ~circular about the planet

pos, vel, E = leapfrog_nbody(positions, velocities, masses,
                             dt=1e-3, n_steps=20000)
rel_drift = (E - E[0]) / abs(E[0])
print(f"Max relative energy drift: {np.max(np.abs(rel_drift)):.2e}")
```

The relative energy drift stays small and oscillatory rather than growing—the hallmark of a symplectic integrator and the reason this scheme, not RK4, underlies long-term solar-system integrations.

## Modern Directions: Structure-Preserving Machine Learning

The newest frontier learns dynamics from data, but naive neural networks ignore the geometry above and conserve nothing. The fix is to *bake the structure into the model*:

- **Hamiltonian Neural Networks** parameterize $H_\theta(q,p)$ with a network and obtain the dynamics from $\dot q = \partial H_\theta/\partial p$, $\dot p = -\partial H_\theta/\partial q$, so energy conservation is exact by construction.
- **Lagrangian Neural Networks** learn $L_\theta(q,\dot q)$ and recover accelerations through the Euler–Lagrange equations, extending to systems where the momenta are not known a priori.
- **Symplectic / Neural ODE integrators** wrap a symplectic step around the learned vector field so the *trained* model still preserves phase-space volume.

```python
import torch
import torch.nn as nn

class LagrangianNN(nn.Module):
    """Neural network that learns the Lagrangian from data"""
    def __init__(self, q_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*q_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 1)
        )

    def forward(self, q, q_dot):
        """Returns learned Lagrangian L(q, q_dot)"""
        return self.net(torch.cat([q, q_dot], dim=-1))

    def get_accelerations(self, q, q_dot):
        """Derive accelerations using Euler-Lagrange equations"""
        L = self.forward(q, q_dot)

        # Compute dL/dq and dL/dq_dot
        dL_dq = torch.autograd.grad(L.sum(), q, create_graph=True)[0]
        dL_dq_dot = torch.autograd.grad(L.sum(), q_dot, create_graph=True)[0]

        # Time derivative of dL/dq_dot
        d_dt_dL_dq_dot = torch.autograd.grad(
            (dL_dq_dot * q_dot).sum(), q, create_graph=True
        )[0]

        # Euler-Lagrange: q'' = (dL/dq - d/dt(dL/dq_dot)) / (d^2 L / dq_dot^2)
        return dL_dq - d_dt_dL_dq_dot
```

## Practical Guidance

- **Default to symplectic.** For any conservative mechanical system simulated over many periods, prefer velocity Verlet (or a Yoshida composition for higher accuracy) over RK4. Reserve adaptive non-symplectic solvers for short, dissipative, or stiff non-Hamiltonian problems.
- **Choose $h$ from the fastest mode.** Stability needs $h \lesssim 2/\omega_{\max}$; accuracy usually wants several times smaller. Use multiple-time-step methods when one stiff mode would otherwise throttle the whole simulation.
- **Monitor a conserved quantity.** Track total energy (and angular momentum where applicable) as a running diagnostic. A bounded oscillation is healthy; a trend means the method, the step size, or the force evaluation is wrong.
- **Match the force solver to the regime.** Direct $O(N^2)$ sums are fine for small systems; switch to tree codes, FMM, or particle–mesh methods as $N$ grows, and use softening or regularization to tame close encounters.

## See Also

- [Lagrangian &amp; Hamiltonian Mechanics](lagrangian-hamiltonian.html) — the action principle, phase space, and Poisson brackets that these integrators discretize.
- [Newtonian Mechanics](newtonian.html) — the equations of motion being solved numerically.
- [Chaos, Modern Topics &amp; Computation](chaos-and-computational.html) — nonlinear dynamics, KAM theory, and symplectic geometry.
- [Computational Physics](../computational-physics/) — numerical methods across physics, including ODE/PDE solvers and Monte Carlo.
- [Statistical Mechanics](../statistical-mechanics/) — the ensemble theory behind molecular-dynamics thermostats.
- [Classical Mechanics Hub](./) — back to the overview.
