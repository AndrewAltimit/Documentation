---
layout: docs
title: Computational Physics
permalink: /docs/physics/computational-physics/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Computational Physics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Where physics meets computation, using numerical algorithms and simulations to solve problems beyond analytical reach.</p>
</div>

<div class="intro-card">
  <p class="lead-text">Computational physics is the third pillar of modern physics, sitting alongside theory and experiment. When equations are too hard to solve by hand — nonlinear dynamics, many interacting bodies, complex geometries — we turn the physics into algorithms and let computers do the work. This hub is a practical, code-first tour of the core numerical methods, from integrating differential equations to Monte Carlo sampling, molecular dynamics, fluid solvers, and GPU acceleration. Use the cards below to jump to a topic.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-square-root-variable"></i>
    <h4>Discretize the continuum</h4>
    <p>Derivatives become finite differences, integrals become sums — turning calculus into arithmetic a computer can run.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-dice"></i>
    <h4>Sample what you can't solve</h4>
    <p>Monte Carlo methods use randomness to evaluate high-dimensional integrals and explore huge state spaces.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-scale-balanced"></i>
    <h4>Respect the physics</h4>
    <p>Good algorithms preserve conservation laws and stability — a fast method that drifts in energy is worthless.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-microchip"></i>
    <h4>Scale with hardware</h4>
    <p>Parallelism (MPI, GPUs) and machine learning push simulations to sizes no single processor could reach.</p>
  </div>
</div>

## Explore the Topics

<div class="command-grid">
  <a href="monte-carlo-and-md.html" class="nav-card">
    <h4><i class="fas fa-dice"></i> Monte Carlo &amp; Molecular Dynamics</h4>
    <p>Random sampling, Markov chains, quantum Monte Carlo, and classical/advanced MD with thermostats and Ewald sums.</p>
  </a>
  <a href="fem-and-cfd.html" class="nav-card">
    <h4><i class="fas fa-vector-square"></i> Finite Elements &amp; Fluid Dynamics</h4>
    <p>1D/2D finite element methods, the Navier-Stokes projection method, and the Lattice Boltzmann scheme.</p>
  </a>
  <a href="quantum-methods.html" class="nav-card">
    <h4><i class="fas fa-wave-square"></i> Quantum Computational Methods</h4>
    <p>Split-operator and Crank-Nicolson time evolution, quantum tunneling, and density functional theory.</p>
  </a>
  <a href="hpc-and-ml.html" class="nav-card">
    <h4><i class="fas fa-microchip"></i> Parallel Computing &amp; Machine Learning</h4>
    <p>MPI domain decomposition, GPU kernels, sparse linear algebra, PINNs, neural potentials, and Fourier neural operators.</p>
  </a>
  <a href="tools-and-practices.html" class="nav-card">
    <h4><i class="fas fa-toolbox"></i> Visualization, Libraries &amp; Best Practices</h4>
    <p>Scientific visualization, data analysis, the core Python physics stack, and validation best practices.</p>
  </a>
</div>

## Introduction to Computational Physics

### What is Computational Physics?

Computational physics is the study and implementation of numerical analysis to solve physical problems. It forms the third pillar of modern physics, alongside experimental and theoretical physics:

- **Theoretical Physics**: Develops mathematical models and equations
- **Experimental Physics**: Tests predictions through observation
- **Computational Physics**: Bridges theory and experiment through simulation

### Why Do We Need It?

Many physical systems involve equations that cannot be solved analytically:

1. **Nonlinear Systems**: Most real-world physics is nonlinear
2. **Many-Body Problems**: Systems with more than two interacting particles
3. **Complex Geometries**: Real-world shapes rarely have simple mathematical forms
4. **Time Evolution**: Following systems through long time periods

### The Computational Approach

```python
# A simple example: projectile motion with air resistance
import numpy as np
import matplotlib.pyplot as plt

def projectile_with_drag(v0, angle, dt=0.01, drag=True):
    """Simulate projectile motion with optional quadratic air resistance"""
    # Constants
    g = 9.81  # gravity (m/s^2)
    rho = 1.225  # air density (kg/m^3)
    Cd = 0.47  # drag coefficient for sphere
    A = 0.045  # cross-sectional area (m^2)
    m = 0.145  # mass (kg)
    
    # Initial conditions
    vx = v0 * np.cos(np.radians(angle))
    vy = v0 * np.sin(np.radians(angle))
    x, y = 0, 0
    
    # Store trajectory
    trajectory = [(x, y)]
    
    while y >= 0:
        # Calculate drag force (zero if drag disabled)
        v = np.sqrt(vx**2 + vy**2)
        Fd = 0.5 * rho * Cd * A * v**2 if drag else 0.0
        
        # Update velocities
        ax = -(Fd/m) * (vx/v) if v > 0 else 0.0
        ay = -g - ((Fd/m) * (vy/v) if v > 0 else 0.0)
        
        vx += ax * dt
        vy += ay * dt
        
        # Update position
        x += vx * dt
        y += vy * dt
        
        trajectory.append((x, y))
    
    return np.array(trajectory)

# Compare with and without air resistance
trajectory_drag = projectile_with_drag(50, 45, drag=True)
trajectory_no_drag = projectile_with_drag(50, 45, drag=False)

plt.plot(trajectory_drag[:, 0], trajectory_drag[:, 1], label='With air resistance')
plt.plot(trajectory_no_drag[:, 0], trajectory_no_drag[:, 1], '--', label='Vacuum (no drag)')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Fundamental Numerical Methods

### Numerical Integration

Numerical integration (quadrature) replaces an integral with a weighted sum of function values. The methods differ in how fast the error shrinks as you add sample points — captured by the **order of convergence**. Higher order means fewer evaluations for the same accuracy, which matters enormously when each evaluation is expensive.

| Method | Idea | Error scaling | Best for |
|--------|------|---------------|----------|
| Midpoint / rectangle | flat-top strips | $O(h^2)$ | quick estimates |
| Trapezoidal | straight-line tops | $O(h^2)$ | smooth 1D functions |
| Simpson's rule | parabolic tops | $O(h^4)$ | smooth 1D functions, cheap accuracy |
| Gaussian quadrature | optimal node placement | spectral (very fast) | smooth integrands, few points |
| Monte Carlo | random sampling | $O(N^{-1/2})$ | high dimensions |

Note the punchline of the last row: Monte Carlo's error falls only as $N^{-1/2}$ regardless of dimension, while grid methods like Simpson's degrade as $O(h^4) = O(N^{-4/d})$ in $d$ dimensions. So in low dimensions deterministic rules win easily, but above $d \approx 4$–$8$ Monte Carlo becomes the only practical choice — the reason it dominates statistical and quantum many-body physics. The [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) page explores these sampling methods in depth.

#### Riemann Sums to Sophisticated Quadrature

The simplest integration methods approximate the area under a curve:

```python
def integrate_simpson(f, a, b, n):
    """Simpson's rule for numerical integration"""
    if n % 2 != 0:
        n += 1  # Simpson's rule needs even intervals
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Simpson's rule: (h/3) * [y0 + 4(y1+y3+...) + 2(y2+y4+...) + yn]
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])  # odd indices
    integral += 2 * np.sum(y[2:-1:2])  # even indices
    
    return integral * h / 3

# Example: Calculate the integral of sin(x) from 0 to π
result = integrate_simpson(np.sin, 0, np.pi, 100)
print(f"∫sin(x)dx from 0 to π = {result:.6f} (exact: 2.0)")
```

#### Advanced Integration Methods

For higher dimensions and complex domains:

```python
def monte_carlo_integrate(f, bounds, n_samples=10000):
    """Monte Carlo integration for arbitrary dimensions"""
    dim = len(bounds)
    volume = np.prod([b[1] - b[0] for b in bounds])
    
    # Generate random points
    points = np.random.uniform(0, 1, (n_samples, dim))
    for i, (low, high) in enumerate(bounds):
        points[:, i] = points[:, i] * (high - low) + low
    
    # Evaluate function at random points
    values = np.array([f(*point) for point in points])
    
    # Monte Carlo estimate
    integral = volume * np.mean(values)
    error = volume * np.std(values) / np.sqrt(n_samples)
    
    return integral, error
```

### Numerical Differentiation

#### Finite Difference Methods

```python
def derivative_schemes(f, x, h=1e-5):
    """Various finite difference schemes for derivatives"""
    # Forward difference: O(h)
    forward = (f(x + h) - f(x)) / h
    
    # Backward difference: O(h)
    backward = (f(x) - f(x - h)) / h
    
    # Central difference: O(h²)
    central = (f(x + h) - f(x - h)) / (2 * h)
    
    # Five-point stencil: O(h⁴)
    five_point = (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)
    
    return {
        'forward': forward,
        'backward': backward,
        'central': central,
        'five_point': five_point
    }
```

---

## Differential Equations

### Ordinary Differential Equations (ODEs)

Most of physics is differential equations: Newton's laws, Maxwell's equations, the Schrödinger equation. Solving them numerically means stepping the state forward in small increments and accepting some error per step. The art is choosing a scheme that is *accurate* (small error per step), *stable* (errors don't blow up over many steps), and *cheap* (few function evaluations).

#### The Runge-Kutta Family

The workhorse explicit method is **fourth-order Runge-Kutta (RK4)**. Rather than using the slope only at the start of a step (as the crude Euler method does), it samples the derivative at four points across the step and takes a weighted average, cancelling error terms up to fourth order. The result: error per step of $O(h^5)$ and global error $O(h^4)$ — halving the step size cuts the error roughly sixteen-fold, for only four derivative evaluations.

```python
def rk4_step(f, t, y, dt):
    """Fourth-order Runge-Kutta step"""
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt/2, y + k1/2)
    k3 = dt * f(t + dt/2, y + k2/2)
    k4 = dt * f(t + dt, y + k3)
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode(f, y0, t_span, dt=0.01):
    """Solve ODE using RK4"""
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(1, len(t)):
        y[i] = rk4_step(f, t[i-1], y[i-1], dt)
    
    return t, y

# Example: Damped harmonic oscillator
def harmonic_oscillator(t, state):
    """dx/dt = v, dv/dt = -k*x - c*v"""
    x, v = state
    k = 1.0  # spring constant
    c = 0.1  # damping coefficient
    return np.array([v, -k*x - c*v])

t, solution = solve_ode(harmonic_oscillator, [1.0, 0.0], [0, 20])
```

#### Adaptive Step Size Methods

```python
def rk45_adaptive(f, y0, t_span, tol=1e-6):
    """Adaptive Runge-Kutta-Fehlberg method"""
    # Butcher tableau coefficients
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    ])
    
    b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    t, t_end = t_span[0], t_span[1]
    y = y0.copy()
    h = 0.01  # initial step size
    
    t_values = [t]
    y_values = [y.copy()]
    
    while t < t_end:
        # Calculate k values
        k = np.zeros((6, len(y)))
        k[0] = f(t, y)
        
        for i in range(1, 6):
            y_temp = y + h * sum(a[i, j] * k[j] for j in range(i))
            k[i] = f(t + h * sum(a[i, :i]), y_temp)
        
        # Calculate two different approximations
        y4 = y + h * sum(b4[i] * k[i] for i in range(6))
        y5 = y + h * sum(b5[i] * k[i] for i in range(6))
        
        # Estimate error
        error = np.max(np.abs(y5 - y4))
        
        if error <= tol:
            # Accept step
            t += h
            y = y5
            t_values.append(t)
            y_values.append(y.copy())
        
        # Adjust step size
        h = h * min(2, max(0.1, 0.9 * (tol / error) ** 0.2))
        
        # Don't overshoot
        if t + h > t_end:
            h = t_end - t
    
    return np.array(t_values), np.array(y_values)
```

### Partial Differential Equations (PDEs)

#### Finite Difference Methods for PDEs

```python
def heat_equation_2d(nx=50, ny=50, nt=1000, alpha=0.01):
    """Solve 2D heat equation using finite differences"""
    # Grid setup
    dx = dy = 1.0 / (nx - 1)
    dt = 0.25 * dx**2 / alpha  # Stability condition
    
    # Initial condition: hot spot in center
    u = np.zeros((nx, ny))
    u[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = 100
    
    # Time evolution
    for n in range(nt):
        un = u.copy()
        
        # Update interior points
        u[1:-1, 1:-1] = un[1:-1, 1:-1] + alpha * dt * (
            (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]) / dx**2 +
            (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) / dy**2
        )
        
        # Boundary conditions (Dirichlet: u = 0 at boundaries)
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
        
        if n % 100 == 0:
            yield u.copy()

# Visualize heat diffusion
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
im = ax.imshow(next(heat_equation_2d()), cmap='hot', interpolation='bilinear')
ax.set_title('2D Heat Diffusion')

def animate(frame):
    im.set_array(frame)
    return [im]

# Create animation
heat_gen = heat_equation_2d()
ani = FuncAnimation(fig, animate, frames=heat_gen, interval=50, blit=True)
plt.show()
```

#### Spectral Methods

```python
def solve_poisson_spectral(f, L=2*np.pi, N=64):
    """Solve Poisson equation using spectral methods"""
    # Create grid
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # Right-hand side
    F = f(X, Y)
    
    # Take FFT
    F_hat = np.fft.fft2(F)
    
    # Wave numbers
    kx = np.fft.fftfreq(N, d=L/(2*np.pi*N)) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/(2*np.pi*N)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    
    # Solve in Fourier space (avoid division by zero)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1  # Set DC component
    U_hat = -F_hat / K2
    U_hat[0, 0] = 0  # Set mean to zero
    
    # Transform back
    U = np.real(np.fft.ifft2(U_hat))
    
    return X, Y, U
```

---

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Start simple</h4>
    <p>Master basic numerical methods (integration, ODE solvers, root finding) before reaching for complex algorithms.</p>
  </div>
  <div class="takeaway-card">
    <h4>Validate constantly</h4>
    <p>Always check results against known analytic solutions, conserved quantities, and convergence as step size shrinks.</p>
  </div>
  <div class="takeaway-card">
    <h4>Optimize wisely</h4>
    <p>Profile first, then optimize the actual bottlenecks; premature optimization wastes effort and obscures bugs.</p>
  </div>
  <div class="takeaway-card">
    <h4>Use the right tool</h4>
    <p>Match the algorithm to the problem: stiff ODEs need implicit solvers, high dimensions favor Monte Carlo.</p>
  </div>
  <div class="takeaway-card">
    <h4>Mind numerical error</h4>
    <p>Round-off, truncation, and stability (e.g. the CFL condition) determine whether a simulation is trustworthy.</p>
  </div>
  <div class="takeaway-card">
    <h4>Visualize for insight</h4>
    <p>Good visualization turns raw arrays into physical understanding and exposes errors a table of numbers hides.</p>
  </div>
</div>

---

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="../classical-mechanics/">Classical Mechanics</a> — symplectic integrators, $N$-body problems, and chaos.</li>
    <li><a href="../quantum-mechanics/">Quantum Mechanics</a> — numerical solutions of the Schrödinger equation.</li>
    <li><a href="../statistical-mechanics/">Statistical Mechanics</a> — Monte Carlo sampling and molecular dynamics.</li>
    <li><a href="../condensed-matter/">Condensed Matter Physics</a> — density-functional theory and band-structure calculations.</li>
    <li><a href="../relativity/">Relativity</a> — numerical relativity and gravitational-wave simulations.</li>
    <li><a href="../">Physics Hub</a> — browse all physics topics.</li>
  </ul>
</div>
