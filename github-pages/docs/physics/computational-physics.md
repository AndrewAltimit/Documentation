---
layout: docs
title: Computational Physics
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "laptop-code"
---

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Computational Physics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Where physics meets computation, using numerical algorithms and simulations to solve problems beyond analytical reach.</p>
</div>

<!-- Custom styles are now loaded via main.scss -->

## Journey Through Computational Physics

**Getting Started**
- [Why Computational Physics?](#introduction-to-computational-physics)
- [Essential Numerical Methods](#fundamental-numerical-methods)
- [Your First Simulations](#getting-started-with-simulations)

**Core Techniques**
- [Solving Differential Equations](#differential-equations)
- [Monte Carlo Methods](#monte-carlo-methods)
- [Molecular Dynamics](#molecular-dynamics)

**Advanced Methods**
- [Finite Element Analysis](#finite-element-methods)
- [Computational Fluid Dynamics](#computational-fluid-dynamics)
- [Quantum Simulations](#quantum-computational-methods)

**Modern Tools & Applications**
- [Parallel Computing](#parallel-computing-for-physics)
- [Machine Learning in Physics](#machine-learning-applications)
- [Visualization Techniques](#visualization-and-analysis)

---

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

def projectile_with_drag(v0, angle, dt=0.01):
    """Simulate projectile motion with quadratic air resistance"""
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
        # Calculate drag force
        v = np.sqrt(vx**2 + vy**2)
        Fd = 0.5 * rho * Cd * A * v**2
        
        # Update velocities
        ax = -(Fd/m) * (vx/v)
        ay = -g - (Fd/m) * (vy/v)
        
        vx += ax * dt
        vy += ay * dt
        
        # Update position
        x += vx * dt
        y += vy * dt
        
        trajectory.append((x, y))
    
    return np.array(trajectory)

# Compare with and without air resistance
trajectory_drag = projectile_with_drag(50, 45)
trajectory_no_drag = projectile_with_drag(50, 45)  # Modify to remove drag

plt.plot(trajectory_drag[:, 0], trajectory_drag[:, 1], label='With air resistance')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Fundamental Numerical Methods

### Numerical Integration

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

#### The Runge-Kutta Family

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

## Monte Carlo Methods

### Basic Monte Carlo Principles

Monte Carlo methods use random sampling to solve problems that might be deterministic in principle:

```python
class MonteCarloSampler:
    """Base class for Monte Carlo sampling"""
    
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
    
    def estimate_pi(self, n_samples=100000):
        """Classic example: estimate π using random sampling"""
        # Generate random points in unit square
        x = self.rng.uniform(-1, 1, n_samples)
        y = self.rng.uniform(-1, 1, n_samples)
        
        # Count points inside unit circle
        inside = np.sum(x**2 + y**2 <= 1)
        
        # Area ratio: circle/square = π/4
        pi_estimate = 4 * inside / n_samples
        error = np.abs(pi_estimate - np.pi)
        
        return pi_estimate, error
    
    def importance_sampling(self, f, p, q, n_samples=10000):
        """Importance sampling for variance reduction"""
        # Sample from importance distribution q
        samples = q.rvs(size=n_samples, random_state=self.rng)
        
        # Compute weights
        weights = p.pdf(samples) / q.pdf(samples)
        
        # Estimate expectation
        expectation = np.mean(weights * f(samples))
        variance = np.var(weights * f(samples)) / n_samples
        
        return expectation, np.sqrt(variance)
```

### Markov Chain Monte Carlo (MCMC)

```python
class MetropolisHastings:
    """Metropolis-Hastings algorithm for sampling from complex distributions"""
    
    def __init__(self, target_pdf, proposal_std=1.0):
        self.target_pdf = target_pdf
        self.proposal_std = proposal_std
    
    def sample(self, x0, n_samples, burn_in=1000):
        """Generate samples using Metropolis-Hastings"""
        samples = []
        x = x0
        n_accepted = 0
        
        for i in range(n_samples + burn_in):
            # Propose new state
            x_proposed = x + np.random.normal(0, self.proposal_std, size=x.shape)
            
            # Calculate acceptance ratio
            ratio = self.target_pdf(x_proposed) / self.target_pdf(x)
            
            # Accept or reject
            if np.random.rand() < ratio:
                x = x_proposed
                n_accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                samples.append(x.copy())
        
        acceptance_rate = n_accepted / (n_samples + burn_in)
        return np.array(samples), acceptance_rate

# Example: Sample from a bimodal distribution
def bimodal_pdf(x):
    """Mixture of two Gaussians"""
    return 0.3 * np.exp(-0.5 * (x - 2)**2) + 0.7 * np.exp(-0.5 * (x + 2)**2)

sampler = MetropolisHastings(bimodal_pdf)
samples, acc_rate = sampler.sample(x0=np.array([0.0]), n_samples=10000)
print(f"Acceptance rate: {acc_rate:.2%}")
```

### Quantum Monte Carlo

```python
def variational_monte_carlo(psi_trial, hamiltonian, params, n_samples=10000):
    """Variational Monte Carlo for quantum systems"""
    
    def local_energy(x, params):
        """Calculate local energy: H𝜓/𝜓"""
        psi = psi_trial(x, params)
        H_psi = hamiltonian(psi_trial, x, params)
        return H_psi / psi
    
    # Sample from |𝜓|²
    samples = metropolis_sample(lambda x: np.abs(psi_trial(x, params))**2, n_samples)
    
    # Calculate expectation value of energy
    E_local = np.array([local_energy(x, params) for x in samples])
    E_mean = np.mean(E_local)
    E_var = np.var(E_local) / n_samples
    
    return E_mean, np.sqrt(E_var)

# Example: Hydrogen atom ground state
def hydrogen_trial(r, alpha):
    """Trial wavefunction: exp(-alpha * r)"""
    return np.exp(-alpha * np.linalg.norm(r))

def hydrogen_hamiltonian(psi, r, alpha):
    """Hamiltonian for hydrogen atom"""
    r_norm = np.linalg.norm(r)
    
    # Kinetic energy (using analytical expression)
    T = -0.5 * alpha * (alpha - 2/r_norm)
    
    # Potential energy
    V = -1/r_norm
    
    return (T + V) * psi(r, alpha)
```

---

## Molecular Dynamics

### Classical Molecular Dynamics

```python
class MolecularDynamics:
    """Classical molecular dynamics simulation"""
    
    def __init__(self, n_particles, box_size, temperature=1.0):
        self.n_particles = n_particles
        self.box_size = box_size
        self.temperature = temperature
        
        # Initialize positions randomly
        self.positions = np.random.rand(n_particles, 3) * box_size
        
        # Initialize velocities from Maxwell-Boltzmann distribution
        self.velocities = np.random.normal(0, np.sqrt(temperature), 
                                         (n_particles, 3))
        
        # Remove center of mass motion
        self.velocities -= np.mean(self.velocities, axis=0)
    
    def lennard_jones_force(self, r, epsilon=1.0, sigma=1.0):
        """Lennard-Jones potential: 4ε[(σ/r)¹² - (σ/r)⁶]"""
        r_norm = np.linalg.norm(r)
        if r_norm < 0.01:  # Avoid singularity
            return np.zeros_like(r)
        
        r6 = (sigma / r_norm) ** 6
        force_magnitude = 24 * epsilon * (2 * r6**2 - r6) / r_norm**2
        return force_magnitude * r
    
    def calculate_forces(self):
        """Calculate all pairwise forces"""
        forces = np.zeros_like(self.positions)
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                # Minimum image convention
                r = self.positions[j] - self.positions[i]
                r = r - self.box_size * np.round(r / self.box_size)
                
                # Calculate force
                f = self.lennard_jones_force(r)
                forces[i] += f
                forces[j] -= f  # Newton's third law
        
        return forces
    
    def velocity_verlet_step(self, dt):
        """Velocity Verlet integration"""
        # Update positions
        forces = self.calculate_forces()
        self.positions += self.velocities * dt + 0.5 * forces * dt**2
        
        # Apply periodic boundary conditions
        self.positions = self.positions % self.box_size
        
        # Update velocities (half step)
        self.velocities += 0.5 * forces * dt
        
        # Calculate new forces
        forces_new = self.calculate_forces()
        
        # Complete velocity update
        self.velocities += 0.5 * forces_new * dt
    
    def run(self, n_steps, dt=0.001):
        """Run MD simulation"""
        trajectory = []
        energies = []
        
        for step in range(n_steps):
            self.velocity_verlet_step(dt)
            
            if step % 10 == 0:
                trajectory.append(self.positions.copy())
                
                # Calculate total energy
                ke = 0.5 * np.sum(self.velocities**2)
                pe = self.calculate_potential_energy()
                energies.append({'kinetic': ke, 'potential': pe, 
                               'total': ke + pe})
        
        return np.array(trajectory), energies
    
    def calculate_potential_energy(self):
        """Calculate total potential energy"""
        pe = 0
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r = self.positions[j] - self.positions[i]
                r = r - self.box_size * np.round(r / self.box_size)
                r_norm = np.linalg.norm(r)
                
                if r_norm < 2.5:  # Cutoff distance
                    r6 = (1.0 / r_norm) ** 6
                    pe += 4 * (r6**2 - r6)
        
        return pe

# Run a simple MD simulation
md = MolecularDynamics(n_particles=108, box_size=10.0, temperature=0.5)
trajectory, energies = md.run(n_steps=1000)
```

### Advanced MD Techniques

```python
class AdvancedMD(MolecularDynamics):
    """Advanced molecular dynamics techniques"""
    
    def nose_hoover_thermostat(self, dt, Q=1.0, target_temp=1.0):
        """Nosé-Hoover thermostat for constant temperature"""
        # Calculate current temperature
        ke = 0.5 * np.sum(self.velocities**2)
        current_temp = 2 * ke / (3 * self.n_particles)
        
        # Update thermostat variable
        if not hasattr(self, 'xi'):
            self.xi = 0.0
        
        xi_dot = (current_temp - target_temp) / Q
        self.xi += xi_dot * dt
        
        # Apply thermostat to velocities
        self.velocities *= np.exp(-self.xi * dt)
    
    def neighbor_list(self, cutoff=2.5):
        """Verlet neighbor list for efficiency"""
        neighbors = {i: [] for i in range(self.n_particles)}
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r = self.positions[j] - self.positions[i]
                r = r - self.box_size * np.round(r / self.box_size)
                
                if np.linalg.norm(r) < cutoff * 1.2:  # Include buffer
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        return neighbors
    
    def ewald_summation(self, charges, alpha=5.0/self.box_size):
        """Ewald summation for long-range electrostatics"""
        # Real space contribution
        energy_real = 0
        forces_real = np.zeros_like(self.positions)
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r = self.positions[j] - self.positions[i]
                r = r - self.box_size * np.round(r / self.box_size)
                r_norm = np.linalg.norm(r)
                
                # Real space interaction
                erfc_term = erfc(alpha * r_norm)
                energy_real += charges[i] * charges[j] * erfc_term / r_norm
                
                force_mag = charges[i] * charges[j] * (
                    erfc_term / r_norm**2 + 
                    2 * alpha / np.sqrt(np.pi) * np.exp(-alpha**2 * r_norm**2) / r_norm
                )
                forces_real[i] += force_mag * r / r_norm
                forces_real[j] -= force_mag * r / r_norm
        
        # Reciprocal space contribution (simplified)
        # ... (implementation of k-space sum)
        
        return energy_real, forces_real
```

---

## Finite Element Methods

### Basic FEM Implementation

```python
class FiniteElement1D:
    """1D finite element method for solving differential equations"""
    
    def __init__(self, n_elements, domain=(0, 1)):
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.domain = domain
        self.L = domain[1] - domain[0]
        self.h = self.L / n_elements
        
        # Node positions
        self.nodes = np.linspace(domain[0], domain[1], self.n_nodes)
        
    def shape_functions(self, xi):
        """Linear shape functions on reference element [-1, 1]"""
        N1 = 0.5 * (1 - xi)
        N2 = 0.5 * (1 + xi)
        return np.array([N1, N2])
    
    def shape_derivatives(self, xi):
        """Derivatives of shape functions"""
        dN1 = -0.5
        dN2 = 0.5
        return np.array([dN1, dN2])
    
    def element_stiffness_matrix(self):
        """Stiffness matrix for linear element"""
        # Gauss quadrature points and weights
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
        weights = [1, 1]
        
        K_e = np.zeros((2, 2))
        
        for gp, w in zip(gauss_points, weights):
            dN = self.shape_derivatives(gp)
            
            # Jacobian for transformation
            J = self.h / 2
            
            # Add contribution
            K_e += w * np.outer(dN, dN) / J
        
        return K_e
    
    def element_mass_matrix(self):
        """Mass matrix for linear element"""
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
        weights = [1, 1]
        
        M_e = np.zeros((2, 2))
        
        for gp, w in zip(gauss_points, weights):
            N = self.shape_functions(gp)
            J = self.h / 2
            M_e += w * np.outer(N, N) * J
        
        return M_e
    
    def assemble_global_matrices(self):
        """Assemble global stiffness and mass matrices"""
        K = np.zeros((self.n_nodes, self.n_nodes))
        M = np.zeros((self.n_nodes, self.n_nodes))
        
        K_e = self.element_stiffness_matrix()
        M_e = self.element_mass_matrix()
        
        for e in range(self.n_elements):
            # Global node numbers for element e
            nodes = [e, e + 1]
            
            # Add element contributions
            for i in range(2):
                for j in range(2):
                    K[nodes[i], nodes[j]] += K_e[i, j]
                    M[nodes[i], nodes[j]] += M_e[i, j]
        
        return K, M
    
    def solve_poisson(self, f, bc_left=0, bc_right=0):
        """Solve -u'' = f with Dirichlet boundary conditions"""
        K, _ = self.assemble_global_matrices()
        
        # Load vector
        F = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            F[i] = f(self.nodes[i]) * self.h
        
        # Apply boundary conditions
        K[0, :] = 0
        K[0, 0] = 1
        F[0] = bc_left
        
        K[-1, :] = 0
        K[-1, -1] = 1
        F[-1] = bc_right
        
        # Solve
        u = np.linalg.solve(K, F)
        
        return self.nodes, u

# Example: Solve -u'' = sin(πx) on [0, 1]
fem = FiniteElement1D(n_elements=20)
x, u = fem.solve_poisson(lambda x: np.sin(np.pi * x))

# Exact solution for comparison
u_exact = np.sin(np.pi * x) / np.pi**2

plt.plot(x, u, 'o-', label='FEM solution')
plt.plot(x, u_exact, '--', label='Exact solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()
```

### 2D Finite Elements

```python
class FiniteElement2D:
    """2D finite element method using triangular elements"""
    
    def __init__(self, vertices, elements):
        self.vertices = np.array(vertices)
        self.elements = np.array(elements)
        self.n_vertices = len(vertices)
        self.n_elements = len(elements)
    
    def shape_functions_2d(self, xi, eta):
        """Linear shape functions for triangular element"""
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        return np.array([N1, N2, N3])
    
    def shape_derivatives_2d(self):
        """Derivatives of shape functions"""
        # dN/dxi
        dN_dxi = np.array([-1, 1, 0])
        # dN/deta
        dN_deta = np.array([-1, 0, 1])
        return dN_dxi, dN_deta
    
    def element_stiffness_2d(self, element_idx):
        """Stiffness matrix for triangular element"""
        # Get vertex coordinates
        v_idx = self.elements[element_idx]
        coords = self.vertices[v_idx]
        
        # Jacobian matrix
        x = coords[:, 0]
        y = coords[:, 1]
        
        J = np.array([
            [x[1] - x[0], x[2] - x[0]],
            [y[1] - y[0], y[2] - y[0]]
        ])
        
        det_J = np.linalg.det(J)
        J_inv = np.linalg.inv(J)
        
        # Shape function derivatives in physical coordinates
        dN_dxi, dN_deta = self.shape_derivatives_2d()
        dN_local = np.array([dN_dxi, dN_deta])
        dN_physical = J_inv.T @ dN_local
        
        # Element stiffness matrix
        K_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_e[i, j] = 0.5 * det_J * (
                    dN_physical[0, i] * dN_physical[0, j] +
                    dN_physical[1, i] * dN_physical[1, j]
                )
        
        return K_e
    
    def create_mesh_grid(nx, ny, L=1.0):
        """Create a structured triangular mesh"""
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        
        vertices = []
        elements = []
        
        # Create vertices
        for j in range(ny):
            for i in range(nx):
                vertices.append([x[i], y[j]])
        
        # Create elements (two triangles per square)
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Bottom-left vertex of square
                v0 = j * nx + i
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v2 + 1
                
                # Lower triangle
                elements.append([v0, v1, v2])
                # Upper triangle
                elements.append([v1, v3, v2])
        
        return np.array(vertices), np.array(elements)
```

---

## Computational Fluid Dynamics

### Basic CFD: Navier-Stokes Solver

```python
class FluidSolver2D:
    """2D incompressible Navier-Stokes solver using finite differences"""
    
    def __init__(self, nx=64, ny=64, L=1.0, nu=0.01):
        self.nx, self.ny = nx, ny
        self.L = L
        self.dx = self.dy = L / (nx - 1)
        self.nu = nu  # kinematic viscosity
        
        # Grid
        self.x = np.linspace(0, L, nx)
        self.y = np.linspace(0, L, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Flow variables
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # pressure
        
    def set_lid_driven_cavity_bc(self):
        """Boundary conditions for lid-driven cavity"""
        # Top lid moves with velocity 1
        self.u[-1, :] = 1.0
        
        # All other boundaries: no-slip (u = v = 0)
        self.u[0, :] = self.u[:, 0] = self.u[:, -1] = 0
        self.v[0, :] = self.v[-1, :] = self.v[:, 0] = self.v[:, -1] = 0
    
    def solve_poisson_pressure(self, div_u, max_iter=1000, tol=1e-6):
        """Solve pressure Poisson equation using Jacobi iteration"""
        p = self.p.copy()
        
        for _ in range(max_iter):
            p_old = p.copy()
            
            # Jacobi iteration
            p[1:-1, 1:-1] = 0.25 * (
                p_old[2:, 1:-1] + p_old[:-2, 1:-1] +
                p_old[1:-1, 2:] + p_old[1:-1, :-2] -
                self.dx**2 * div_u[1:-1, 1:-1]
            )
            
            # Neumann BC: dp/dn = 0
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            
            # Check convergence
            if np.max(np.abs(p - p_old)) < tol:
                break
        
        return p
    
    def step(self, dt):
        """Single time step using projection method"""
        u, v, p = self.u, self.v, self.p
        dx, dy = self.dx, self.dy
        
        # Step 1: Compute intermediate velocity (ignore pressure)
        # Advection terms (upwind scheme)
        u_x = np.where(u > 0,
                      (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx,
                      (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx)
        u_y = np.where(v[1:-1, 1:-1] > 0,
                      (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dy,
                      (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy)
        
        v_x = np.where(u[1:-1, 1:-1] > 0,
                      (v[1:-1, 1:-1] - v[1:-1, :-2]) / dx,
                      (v[1:-1, 2:] - v[1:-1, 1:-1]) / dx)
        v_y = np.where(v > 0,
                      (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy,
                      (v[2:, 1:-1] - v[1:-1, 1:-1]) / dy)
        
        # Diffusion terms
        u_xx = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
        u_yy = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        
        v_xx = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
        v_yy = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
        
        # Update intermediate velocity
        u_star = u.copy()
        v_star = v.copy()
        
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            -u[1:-1, 1:-1] * u_x - v[1:-1, 1:-1] * u_y +
            self.nu * (u_xx + u_yy)
        )
        
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            -u[1:-1, 1:-1] * v_x - v[1:-1, 1:-1] * v_y +
            self.nu * (v_xx + v_yy)
        )
        
        # Apply boundary conditions
        self.set_lid_driven_cavity_bc()
        
        # Step 2: Solve pressure Poisson equation
        div_u_star = (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2*dx) +
            (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2*dy)
        )
        
        div_field = np.zeros_like(p)
        div_field[1:-1, 1:-1] = div_u_star / dt
        
        self.p = self.solve_poisson_pressure(div_field)
        
        # Step 3: Correct velocity with pressure gradient
        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * (
            (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2*dx)
        )
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * (
            (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2*dy)
        )
        
        # Re-apply boundary conditions
        self.set_lid_driven_cavity_bc()
    
    def run(self, t_end, dt=0.001):
        """Run simulation"""
        t = 0
        step_count = 0
        
        while t < t_end:
            self.step(dt)
            t += dt
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Time: {t:.3f}, Max velocity: {np.max(np.sqrt(self.u**2 + self.v**2)):.3f}")
        
        return self.u, self.v, self.p

# Run lid-driven cavity simulation
solver = FluidSolver2D(nx=64, ny=64, nu=0.01)
u, v, p = solver.run(t_end=10.0)

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.contourf(solver.X, solver.Y, np.sqrt(u**2 + v**2), levels=20)
plt.colorbar(label='Velocity magnitude')
plt.title('Velocity Magnitude')

plt.subplot(132)
plt.streamplot(solver.X, solver.Y, u, v, density=1.5)
plt.title('Streamlines')

plt.subplot(133)
plt.contourf(solver.X, solver.Y, p, levels=20)
plt.colorbar(label='Pressure')
plt.title('Pressure Field')

plt.tight_layout()
plt.show()
```

### Lattice Boltzmann Method

```python
class LatticeBoltzmann2D:
    """2D Lattice Boltzmann Method for fluid simulation"""
    
    def __init__(self, nx, ny, Re=100, U=0.1):
        self.nx, self.ny = nx, ny
        self.Re = Re  # Reynolds number
        self.U = U    # Characteristic velocity
        
        # D2Q9 lattice
        self.c = np.array([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ])
        
        self.w = np.array([
            4/9,
            1/9, 1/9, 1/9, 1/9,
            1/36, 1/36, 1/36, 1/36
        ])
        
        # Relaxation time (related to viscosity)
        self.nu = U * ny / Re
        self.tau = 3 * self.nu + 0.5
        
        # Initialize distribution functions
        self.f = np.zeros((9, ny, nx))
        self.feq = np.zeros((9, ny, nx))
        
        # Initialize to equilibrium
        rho = np.ones((ny, nx))
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        
        self.equilibrium(rho, u, v)
        self.f = self.feq.copy()
    
    def equilibrium(self, rho, u, v):
        """Calculate equilibrium distribution"""
        for i in range(9):
            cu = self.c[i, 0] * u + self.c[i, 1] * v
            self.feq[i] = rho * self.w[i] * (
                1 + 3*cu + 4.5*cu**2 - 1.5*(u**2 + v**2)
            )
    
    def macroscopic(self):
        """Calculate macroscopic variables from distribution"""
        rho = np.sum(self.f, axis=0)
        u = np.sum(self.f * self.c[:, 0, np.newaxis, np.newaxis], axis=0) / rho
        v = np.sum(self.f * self.c[:, 1, np.newaxis, np.newaxis], axis=0) / rho
        return rho, u, v
    
    def collision(self):
        """BGK collision operator"""
        rho, u, v = self.macroscopic()
        self.equilibrium(rho, u, v)
        self.f = self.f - (self.f - self.feq) / self.tau
    
    def streaming(self):
        """Stream distribution functions"""
        for i in range(9):
            self.f[i] = np.roll(self.f[i], self.c[i], axis=(0, 1))
    
    def boundary_conditions(self):
        """Apply boundary conditions"""
        # Lid-driven cavity: top wall moves with velocity U
        # Zou-He boundary conditions
        
        # Top wall (moving)
        rho_top = (self.f[0, -1, :] + self.f[1, -1, :] + self.f[3, -1, :] +
                  2 * (self.f[2, -1, :] + self.f[5, -1, :] + self.f[6, -1, :])) / (1 + self.U)
        
        self.f[4, -1, :] = self.f[2, -1, :]
        self.f[7, -1, :] = self.f[5, -1, :] - 0.5 * (self.f[1, -1, :] - self.f[3, -1, :]) + 0.5 * rho_top * self.U
        self.f[8, -1, :] = self.f[6, -1, :] + 0.5 * (self.f[1, -1, :] - self.f[3, -1, :]) - 0.5 * rho_top * self.U
        
        # Other walls (no-slip)
        # Bottom
        self.f[[2, 5, 6], 0, :] = self.f[[4, 7, 8], 0, :]
        
        # Left
        self.f[[1, 5, 8], :, 0] = self.f[[3, 7, 6], :, 0]
        
        # Right
        self.f[[3, 7, 6], :, -1] = self.f[[1, 5, 8], :, -1]
    
    def step(self):
        """Single LBM step"""
        self.collision()
        self.streaming()
        self.boundary_conditions()
    
    def run(self, n_steps):
        """Run simulation"""
        for step in range(n_steps):
            self.step()
            
            if step % 1000 == 0:
                rho, u, v = self.macroscopic()
                print(f"Step {step}: Max velocity = {np.max(np.sqrt(u**2 + v**2)):.4f}")
        
        return self.macroscopic()

# Run LBM simulation
lbm = LatticeBoltzmann2D(nx=100, ny=100, Re=1000)
rho, u, v = lbm.run(n_steps=10000)
```

---

## Quantum Computational Methods

### Time-Dependent Schrödinger Equation

```python
class QuantumEvolution:
    """Solve time-dependent Schrödinger equation"""
    
    def __init__(self, x_range, n_points=256):
        self.x_min, self.x_max = x_range
        self.n = n_points
        self.dx = (self.x_max - self.x_min) / (n_points - 1)
        self.x = np.linspace(self.x_min, self.x_max, n_points)
        
        # Momentum space
        self.k = 2 * np.pi * np.fft.fftfreq(n_points, self.dx)
        
    def gaussian_wavepacket(self, x0, k0, sigma):
        """Initialize Gaussian wavepacket"""
        psi = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        psi *= np.exp(1j * k0 * self.x)
        psi /= (2 * np.pi * sigma**2) ** 0.25
        return psi
    
    def split_operator_step(self, psi, V, dt):
        """Split-operator method for time evolution"""
        # Half step in position space
        psi *= np.exp(-1j * V * dt / 2)
        
        # Full step in momentum space
        psi_k = np.fft.fft(psi)
        psi_k *= np.exp(-1j * self.k**2 * dt / 2)
        psi = np.fft.ifft(psi_k)
        
        # Half step in position space
        psi *= np.exp(-1j * V * dt / 2)
        
        return psi
    
    def crank_nicolson_step(self, psi, H, dt):
        """Crank-Nicolson method for time evolution"""
        # (1 + i*dt*H/2)𝜓(t+dt) = (1 - i*dt*H/2)𝜓(t)
        I = np.eye(self.n)
        A = I + 1j * dt * H / 2
        B = I - 1j * dt * H / 2
        
        # Solve linear system
        b = B @ psi
        psi_new = np.linalg.solve(A, b)
        
        return psi_new
    
    def finite_difference_hamiltonian(self, V):
        """Construct Hamiltonian matrix using finite differences"""
        H = np.zeros((self.n, self.n), dtype=complex)
        
        # Kinetic energy (second derivative)
        for i in range(1, self.n - 1):
            H[i, i-1] = -0.5 / self.dx**2
            H[i, i] = 1.0 / self.dx**2 + V[i]
            H[i, i+1] = -0.5 / self.dx**2
        
        # Boundary conditions
        H[0, 0] = 1.0 / self.dx**2 + V[0]
        H[0, 1] = -0.5 / self.dx**2
        H[-1, -2] = -0.5 / self.dx**2
        H[-1, -1] = 1.0 / self.dx**2 + V[-1]
        
        return H
    
    def tunnel_barrier_simulation(self):
        """Quantum tunneling through a barrier"""
        # Potential barrier
        V = np.zeros_like(self.x)
        barrier_width = 2.0
        barrier_height = 5.0
        V[np.abs(self.x) < barrier_width/2] = barrier_height
        
        # Initial wavepacket
        psi = self.gaussian_wavepacket(x0=-5, k0=3, sigma=1)
        
        # Time evolution
        dt = 0.01
        n_steps = 1000
        
        results = []
        for step in range(n_steps):
            psi = self.split_operator_step(psi, V, dt)
            
            if step % 10 == 0:
                # Calculate transmission and reflection
                transmitted = np.sum(np.abs(psi[self.x > barrier_width/2])**2) * self.dx
                reflected = np.sum(np.abs(psi[self.x < -barrier_width/2])**2) * self.dx
                
                results.append({
                    'time': step * dt,
                    'psi': psi.copy(),
                    'transmitted': transmitted,
                    'reflected': reflected
                })
        
        return results, V

# Visualize quantum tunneling
qe = QuantumEvolution(x_range=(-10, 10), n_points=512)
results, V = qe.tunnel_barrier_simulation()

# Animation of wavefunction
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Wavefunction plot
line1, = ax1.plot(qe.x, np.abs(results[0]['psi'])**2, 'b-', label='|𝜓|²')
line2, = ax1.plot(qe.x, np.real(results[0]['psi']), 'r--', label='Re(𝜓)')
ax1.fill_between(qe.x, 0, V/10, alpha=0.3, label='Potential')
ax1.set_ylabel('Wavefunction')
ax1.legend()
ax1.grid(True)

# Transmission/Reflection plot
times = [r['time'] for r in results]
trans = [r['transmitted'] for r in results]
refl = [r['reflected'] for r in results]

ax2.plot(times, trans, 'g-', label='Transmitted')
ax2.plot(times, refl, 'r-', label='Reflected')
ax2.set_xlabel('Time')
ax2.set_ylabel('Probability')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Density Functional Theory (DFT)

```python
class SimpleDFT:
    """Simplified 1D DFT implementation"""
    
    def __init__(self, n_grid=100, box_length=10):
        self.n = n_grid
        self.L = box_length
        self.dx = self.L / n_grid
        self.x = np.linspace(-self.L/2, self.L/2, n_grid)
        
        # Kinetic energy operator in momentum space
        self.k = 2 * np.pi * np.fft.fftfreq(n_grid, self.dx)
        self.T_k = 0.5 * self.k**2
    
    def thomas_fermi_functional(self, n):
        """Thomas-Fermi kinetic energy functional"""
        C_TF = (3/10) * (3 * np.pi**2)**(2/3)
        return C_TF * np.sum(n**(5/3)) * self.dx
    
    def exchange_functional(self, n):
        """Local density approximation for exchange"""
        C_x = -(3/4) * (3/np.pi)**(1/3)
        return C_x * np.sum(n**(4/3)) * self.dx
    
    def hartree_potential(self, n):
        """Solve Poisson equation for Hartree potential"""
        # Fourier space solution
        n_k = np.fft.fft(n)
        V_H_k = np.zeros_like(n_k)
        
        # V_H = 4π n / k² (avoiding k=0)
        V_H_k[1:] = 4 * np.pi * n_k[1:] / self.k[1:]**2
        V_H_k[0] = 0  # Set DC component
        
        return np.real(np.fft.ifft(V_H_k))
    
    def kohn_sham_step(self, n, V_ext):
        """Single Kohn-Sham iteration"""
        # Calculate potentials
        V_H = self.hartree_potential(n)
        
        # Exchange-correlation potential (LDA)
        V_xc = -(3/np.pi)**(1/3) * n**(1/3)
        
        # Total effective potential
        V_eff = V_ext + V_H + V_xc
        
        # Solve Kohn-Sham equation
        # Convert to momentum space
        H_diag = self.T_k + np.fft.fft(V_eff)
        
        # Find eigenvalues and eigenvectors
        # (Simplified: assuming non-interacting electrons)
        energies = np.sort(np.real(H_diag))
        
        # Construct new density (simplified)
        # In real DFT, we'd solve for eigenfunctions
        n_new = self.initial_density(V_ext)  # Placeholder
        
        return n_new, energies
    
    def initial_density(self, V_ext):
        """Initial guess for electron density"""
        # Use Thomas-Fermi approximation
        mu = 1.0  # Chemical potential (adjust as needed)
        n = np.maximum(0, mu - V_ext)**(3/2)
        
        # Normalize to correct number of electrons
        N_electrons = 10  # Example
        n *= N_electrons / (np.sum(n) * self.dx)
        
        return n
    
    def self_consistent_field(self, V_ext, max_iter=50, tol=1e-6):
        """Self-consistent field iteration"""
        n = self.initial_density(V_ext)
        
        for i in range(max_iter):
            n_old = n.copy()
            
            # Kohn-Sham step
            n, energies = self.kohn_sham_step(n, V_ext)
            
            # Mix old and new density
            alpha = 0.3  # Mixing parameter
            n = alpha * n + (1 - alpha) * n_old
            
            # Check convergence
            error = np.max(np.abs(n - n_old))
            if error < tol:
                print(f"Converged in {i+1} iterations")
                break
        
        return n, energies
```

---

## Parallel Computing for Physics

### MPI for Distributed Computing

```python
# Example: Parallel Monte Carlo simulation
# Run with: mpirun -n 4 python script.py

from mpi4py import MPI
import numpy as np

class ParallelMonteCarlo:
    """Parallel Monte Carlo simulation using MPI"""
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def parallel_pi_estimation(self, n_samples_total):
        """Estimate π using parallel Monte Carlo"""
        # Divide work among processes
        n_samples_local = n_samples_total // self.size
        
        # Handle remainder
        if self.rank < n_samples_total % self.size:
            n_samples_local += 1
        
        # Local random number generation with different seeds
        np.random.seed(42 + self.rank)
        
        # Generate random points
        x = np.random.uniform(-1, 1, n_samples_local)
        y = np.random.uniform(-1, 1, n_samples_local)
        
        # Count points inside unit circle
        inside_local = np.sum(x**2 + y**2 <= 1)
        
        # Gather results from all processes
        inside_total = self.comm.reduce(inside_local, op=MPI.SUM, root=0)
        
        if self.rank == 0:
            pi_estimate = 4 * inside_total / n_samples_total
            print(f"π estimate: {pi_estimate:.6f}")
            print(f"Error: {abs(pi_estimate - np.pi):.6f}")
            return pi_estimate
        
        return None
    
    def parallel_domain_decomposition(self, global_shape, local_func):
        """Domain decomposition for PDEs"""
        nx, ny = global_shape
        
        # 2D processor grid
        px = int(np.sqrt(self.size))
        py = self.size // px
        
        # Local domain size
        nx_local = nx // px
        ny_local = ny // py
        
        # Processor coordinates
        px_coord = self.rank % px
        py_coord = self.rank // px
        
        # Local domain boundaries
        x_start = px_coord * nx_local
        x_end = (px_coord + 1) * nx_local
        y_start = py_coord * ny_local
        y_end = (py_coord + 1) * ny_local
        
        # Create local array with ghost cells
        local_array = np.zeros((nx_local + 2, ny_local + 2))
        
        # Apply local function
        local_array[1:-1, 1:-1] = local_func(x_start, x_end, y_start, y_end)
        
        return local_array, (px_coord, py_coord)
    
    def exchange_ghost_cells(self, local_array, proc_coords):
        """Exchange ghost cells with neighbors"""
        px_coord, py_coord = proc_coords
        
        # Define neighbor ranks
        north = self.comm.Get_rank() if py_coord == 0 else self.rank - 1
        south = self.comm.Get_rank() if py_coord == self.size - 1 else self.rank + 1
        west = self.comm.Get_rank() if px_coord == 0 else self.rank - self.size
        east = self.comm.Get_rank() if px_coord == self.size - 1 else self.rank + self.size
        
        # Exchange in y-direction
        self.comm.Sendrecv(local_array[-2, :], south, 
                          recvbuf=local_array[0, :], source=north)
        self.comm.Sendrecv(local_array[1, :], north,
                          recvbuf=local_array[-1, :], source=south)
        
        # Exchange in x-direction
        self.comm.Sendrecv(local_array[:, -2], east,
                          recvbuf=local_array[:, 0], source=west)
        self.comm.Sendrecv(local_array[:, 1], west,
                          recvbuf=local_array[:, -1], source=east)
        
        return local_array
```

### GPU Computing with CUDA/CuPy

```python
import cupy as cp

class GPUPhysics:
    """GPU-accelerated physics simulations using CuPy"""
    
    def __init__(self):
        self.device = cp.cuda.Device()
        print(f"Using GPU: {self.device}")
    
    def gpu_nbody_simulation(self, n_bodies=1000, n_steps=100):
        """N-body gravitational simulation on GPU"""
        # Initialize positions and velocities
        pos = cp.random.randn(n_bodies, 3).astype(cp.float32)
        vel = cp.random.randn(n_bodies, 3).astype(cp.float32) * 0.1
        mass = cp.ones(n_bodies, dtype=cp.float32)
        
        # Softening parameter to avoid singularities
        eps = 0.01
        dt = 0.01
        
        # Custom CUDA kernel for force calculation
        force_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void compute_forces(float* pos, float* mass, float* forces, 
                           int n_bodies, float eps) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_bodies) return;
            
            float3 f = make_float3(0.0f, 0.0f, 0.0f);
            float3 pi = make_float3(pos[3*i], pos[3*i+1], pos[3*i+2]);
            
            for (int j = 0; j < n_bodies; j++) {
                if (i == j) continue;
                
                float3 pj = make_float3(pos[3*j], pos[3*j+1], pos[3*j+2]);
                float3 r = make_float3(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
                
                float r2 = r.x*r.x + r.y*r.y + r.z*r.z + eps*eps;
                float r3 = r2 * sqrtf(r2);
                
                float f_mag = mass[j] / r3;
                f.x += f_mag * r.x;
                f.y += f_mag * r.y;
                f.z += f_mag * r.z;
            }
            
            forces[3*i] = f.x;
            forces[3*i+1] = f.y;
            forces[3*i+2] = f.z;
        }
        ''', 'compute_forces')
        
        # Simulation loop
        forces = cp.zeros_like(pos)
        
        for step in range(n_steps):
            # Compute forces
            threads_per_block = 256
            blocks = (n_bodies + threads_per_block - 1) // threads_per_block
            
            force_kernel((blocks,), (threads_per_block,), 
                        (pos.ravel(), mass, forces.ravel(), n_bodies, eps))
            
            # Update velocities and positions (Leapfrog integration)
            vel += forces * dt
            pos += vel * dt
            
            if step % 10 == 0:
                # Calculate total energy
                kinetic = 0.5 * cp.sum(mass[:, None] * vel**2)
                
                # Potential energy (simplified)
                print(f"Step {step}: KE = {float(kinetic):.3f}")
        
        return cp.asnumpy(pos), cp.asnumpy(vel)
    
    def gpu_fft_spectral_method(self, n=512):
        """Spectral method for PDEs using GPU FFT"""
        # Create grid
        x = cp.linspace(0, 2*np.pi, n, endpoint=False)
        y = cp.linspace(0, 2*np.pi, n, endpoint=False)
        X, Y = cp.meshgrid(x, y)
        
        # Initial condition
        u = cp.sin(X) * cp.cos(2*Y)
        
        # Wave numbers
        kx = cp.fft.fftfreq(n, d=2*np.pi/n) * 2 * cp.pi
        ky = cp.fft.fftfreq(n, d=2*np.pi/n) * 2 * cp.pi
        KX, KY = cp.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        
        # Time stepping
        dt = 0.01
        n_steps = 100
        
        for step in range(n_steps):
            # Transform to Fourier space
            u_hat = cp.fft.fft2(u)
            
            # Solve in Fourier space (heat equation example)
            u_hat *= cp.exp(-K2 * dt)
            
            # Transform back
            u = cp.real(cp.fft.ifft2(u_hat))
        
        return cp.asnumpy(u)
```

### Parallel Linear Algebra

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, eigsh
import multiprocessing as mp

class ParallelLinearAlgebra:
    """Parallel solutions for large linear systems in physics"""
    
    def __init__(self, n_cores=None):
        self.n_cores = n_cores or mp.cpu_count()
    
    def parallel_jacobi(self, A, b, x0=None, max_iter=1000, tol=1e-6):
        """Parallel Jacobi iteration for Ax = b"""
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Extract diagonal
        D = np.diag(A)
        R = A - np.diag(D)
        
        def update_chunk(args):
            """Update a chunk of the solution vector"""
            start, end, x_old, D_chunk, R_chunk, b_chunk = args
            x_new = (b_chunk - R_chunk @ x_old) / D_chunk
            return start, end, x_new
        
        # Create chunks for parallel processing
        chunk_size = n // self.n_cores
        chunks = []
        
        for i in range(self.n_cores):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_cores - 1 else n
            chunks.append((start, end))
        
        # Iteration
        with mp.Pool(self.n_cores) as pool:
            for iteration in range(max_iter):
                x_old = x.copy()
                
                # Prepare arguments for parallel execution
                args_list = []
                for start, end in chunks:
                    args_list.append((
                        start, end, x_old,
                        D[start:end],
                        R[start:end, :],
                        b[start:end]
                    ))
                
                # Parallel update
                results = pool.map(update_chunk, args_list)
                
                # Gather results
                for start, end, x_chunk in results:
                    x[start:end] = x_chunk
                
                # Check convergence
                if np.linalg.norm(x - x_old) < tol:
                    print(f"Converged in {iteration + 1} iterations")
                    break
        
        return x
    
    def lanczos_eigenvalues(self, H_func, n_eigs=10, n_lanczos=50):
        """Lanczos algorithm for sparse eigenvalue problems"""
        n = H_func.shape[0]
        
        # Random initial vector
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        
        # Lanczos vectors and tridiagonal matrix elements
        V = np.zeros((n, n_lanczos))
        alpha = np.zeros(n_lanczos)
        beta = np.zeros(n_lanczos - 1)
        
        V[:, 0] = v
        
        for j in range(n_lanczos - 1):
            # Apply Hamiltonian
            w = H_func @ V[:, j]
            
            # Orthogonalize
            alpha[j] = np.dot(w, V[:, j])
            w -= alpha[j] * V[:, j]
            
            if j > 0:
                w -= beta[j-1] * V[:, j-1]
            
            beta[j] = np.linalg.norm(w)
            
            if beta[j] < 1e-12:
                print(f"Lanczos breakdown at iteration {j}")
                break
            
            V[:, j+1] = w / beta[j]
        
        # Final alpha
        w = H_func @ V[:, j+1]
        alpha[j+1] = np.dot(w, V[:, j+1])
        
        # Construct tridiagonal matrix
        T = diags([beta[:-1], alpha[:j+2], beta[:-1]], [-1, 0, 1])
        
        # Solve eigenvalue problem for T
        eigs, eigvecs = eigsh(T, k=min(n_eigs, j+1), which='SA')
        
        return eigs, V[:, :j+2] @ eigvecs
```

---

## Machine Learning Applications

### Recent Advances in Physics-ML Integration (2023-2024)

The intersection of machine learning and physics has seen explosive growth:

**Major Breakthroughs:**
- **Neural Operators**: Learning solution operators for entire families of PDEs
- **Equivariant Neural Networks**: Networks that respect physical symmetries
- **Differentiable Physics Engines**: End-to-end learning through simulations
- **Foundation Models for Science**: Large models trained on diverse physics data

### Physics-Informed Neural Networks (PINNs)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PhysicsInformedNN(nn.Module):
    """Neural network for solving PDEs"""
    
    def __init__(self, layers):
        super().__init__()
        
        # Build network
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Activation
        self.activation = nn.Tanh()
    
    def forward(self, x):
        """Forward pass through network"""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def physics_loss(self, x, t):
        """Physics-informed loss for heat equation"""
        x.requires_grad = True
        t.requires_grad = True
        
        # Network output
        u = self(torch.cat([x, t], dim=1))
        
        # Compute derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                  create_graph=True)[0]
        
        # Heat equation: u_t - α*u_xx = 0
        alpha = 0.1
        f = u_t - alpha * u_xx
        
        return torch.mean(f**2)

def train_pinn(model, n_epochs=5000):
    """Train physics-informed neural network"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training points
    n_points = 1000
    x = torch.rand(n_points, 1) * 2 - 1  # x in [-1, 1]
    t = torch.rand(n_points, 1)  # t in [0, 1]
    
    # Boundary conditions
    n_bc = 100
    x_bc = torch.ones(n_bc, 1) * -1
    t_bc = torch.rand(n_bc, 1)
    u_bc = torch.zeros(n_bc, 1)  # u(-1, t) = 0
    
    # Initial condition
    x_ic = torch.rand(n_bc, 1) * 2 - 1
    t_ic = torch.zeros(n_bc, 1)
    u_ic = torch.sin(np.pi * x_ic)  # u(x, 0) = sin(πx)
    
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Physics loss
        loss_physics = model.physics_loss(x, t)
        
        # Boundary condition loss
        u_pred_bc = model(torch.cat([x_bc, t_bc], dim=1))
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Initial condition loss
        u_pred_ic = model(torch.cat([x_ic, t_ic], dim=1))
        loss_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        # Total loss
        loss = loss_physics + loss_bc + loss_ic
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            losses.append(loss.item())
    
    return losses

# Example usage
model = PhysicsInformedNN([2, 50, 50, 50, 1])  # 2 inputs (x, t), 1 output (u)
losses = train_pinn(model)
```

### Neural Network Potentials

```python
class NeuralPotential(nn.Module):
    """Neural network for learning interatomic potentials"""
    
    def __init__(self, n_features=10, hidden_layers=[64, 64]):
        super().__init__()
        
        layers = [n_features] + hidden_layers + [1]
        self.network = self._build_network(layers)
        
        # Symmetry functions for atomic environments
        self.symmetry_params = self._init_symmetry_functions()
    
    def _build_network(self, layers):
        """Build the neural network"""
        network = []
        for i in range(len(layers) - 1):
            network.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                network.append(nn.ReLU())
        return nn.Sequential(*network)
    
    def _init_symmetry_functions(self):
        """Initialize Behler-Parrinello symmetry functions"""
        # Radial symmetry function parameters
        eta_values = [0.05, 0.5, 1.0, 2.0]
        Rs_values = [0.0, 1.0, 2.0, 3.0]
        
        # Angular symmetry function parameters
        zeta_values = [1.0, 2.0, 4.0]
        lambda_values = [-1.0, 1.0]
        
        return {
            'eta': eta_values,
            'Rs': Rs_values,
            'zeta': zeta_values,
            'lambda': lambda_values
        }
    
    def compute_symmetry_functions(self, positions, types, cutoff=6.0):
        """Compute symmetry functions for atomic environments"""
        n_atoms = len(positions)
        n_features = len(self.symmetry_params['eta']) * len(self.symmetry_params['Rs'])
        features = torch.zeros(n_atoms, n_features)
        
        for i in range(n_atoms):
            feature_idx = 0
            
            # Radial symmetry functions
            for eta in self.symmetry_params['eta']:
                for Rs in self.symmetry_params['Rs']:
                    G_rad = 0
                    
                    for j in range(n_atoms):
                        if i == j:
                            continue
                        
                        r_ij = torch.norm(positions[j] - positions[i])
                        
                        if r_ij < cutoff:
                            fc = 0.5 * (torch.cos(np.pi * r_ij / cutoff) + 1)
                            G_rad += torch.exp(-eta * (r_ij - Rs)**2) * fc
                    
                    features[i, feature_idx] = G_rad
                    feature_idx += 1
        
        return features
    
    def forward(self, features):
        """Predict energy from symmetry functions"""
        return self.network(features)
    
    def calculate_forces(self, positions, types):
        """Calculate forces as negative gradient of energy"""
        positions.requires_grad = True
        
        # Compute features
        features = self.compute_symmetry_functions(positions, types)
        
        # Predict atomic energies
        atomic_energies = self(features)
        total_energy = torch.sum(atomic_energies)
        
        # Calculate forces
        forces = -torch.autograd.grad(total_energy, positions,
                                     create_graph=True)[0]
        
        return forces, total_energy
```

### Fourier Neural Operators (FNO)

```python
class SpectralConv2d(nn.Module):
    """2D Fourier layer for Neural Operators"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), 
                           x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
    def compl_mul2d(self, input, weights):
        # Complex multiplication
        return torch.einsum("bixy,ioxy->boxy", input, weights)

class FourierNeuralOperator2d(nn.Module):
    """Fourier Neural Operator for learning solution operators of PDEs"""
    def __init__(self, modes1, modes2, width=64, in_channels=3, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Input lifting
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Regular convolutions for local features
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: (batch, x, y, channels)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, x, y)
        
        # Fourier layers with residual connections
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.activation(x1 + x2)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.activation(x1 + x2)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.activation(x1 + x2)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        x = x.permute(0, 2, 3, 1)  # (batch, x, y, channels)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Example: Learning the solution operator for 2D Navier-Stokes
def train_fno_navier_stokes():
    """Train FNO to learn the solution operator for 2D turbulence"""
    model = FourierNeuralOperator2d(modes1=12, modes2=12, width=32)
    
    # Training would involve:
    # 1. Generate training data: initial conditions → solutions at time T
    # 2. Train model to map: u(x,y,0) → u(x,y,T)
    # 3. Model learns the solution operator, can generalize to new initial conditions
    
    print("FNO architecture created for learning Navier-Stokes solution operator")
```

---

## Visualization and Analysis

### Advanced Scientific Visualization

```python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class PhysicsVisualizer:
    """Advanced visualization for physics simulations"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_phase_space(self, trajectories, title="Phase Space"):
        """Plot phase space trajectories"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        for traj in trajectories:
            # Position vs velocity
            axes[0, 0].plot(traj[:, 0], traj[:, 1], alpha=0.7)
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Velocity')
            axes[0, 0].set_title('Phase Portrait')
            
            # Poincaré section
            # (simplified: when x crosses zero with positive velocity)
            crossings = []
            for i in range(1, len(traj)):
                if traj[i-1, 0] < 0 and traj[i, 0] >= 0:
                    # Linear interpolation
                    alpha = -traj[i-1, 0] / (traj[i, 0] - traj[i-1, 0])
                    v_crossing = traj[i-1, 1] + alpha * (traj[i, 1] - traj[i-1, 1])
                    crossings.append(v_crossing)
            
            if crossings:
                axes[0, 1].scatter(range(len(crossings)), crossings, s=10)
            axes[0, 1].set_xlabel('Crossing Number')
            axes[0, 1].set_ylabel('Velocity at x=0')
            axes[0, 1].set_title('Poincaré Section')
            
            # Energy over time
            E = 0.5 * traj[:, 1]**2 + 0.5 * traj[:, 0]**2  # Example: harmonic oscillator
            axes[1, 0].plot(E)
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Total Energy')
            axes[1, 0].set_title('Energy Conservation')
            
            # 3D trajectory (if available)
            if traj.shape[1] >= 3:
                ax3d = fig.add_subplot(224, projection='3d')
                ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2])
                ax3d.set_xlabel('X')
                ax3d.set_ylabel('Y')
                ax3d.set_zlabel('Z')
                ax3d.set_title('3D Trajectory')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def animate_field(self, field_data, times, title="Field Evolution"):
        """Animate 2D field evolution"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Initial plot
        im = ax.imshow(field_data[0], cmap='viridis', animated=True)
        ax.set_title(f'{title} - Time: {times[0]:.2f}')
        cbar = plt.colorbar(im)
        
        def update(frame):
            im.set_array(field_data[frame])
            ax.set_title(f'{title} - Time: {times[frame]:.2f}')
            return [im]
        
        ani = animation.FuncAnimation(fig, update, frames=len(field_data),
                                    interval=50, blit=True)
        
        return ani
    
    def plot_spectrum(self, frequencies, amplitudes, log_scale=True):
        """Plot frequency spectrum"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Amplitude spectrum
        if log_scale:
            ax1.semilogy(frequencies, np.abs(amplitudes))
        else:
            ax1.plot(frequencies, np.abs(amplitudes))
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Amplitude Spectrum')
        ax1.grid(True)
        
        # Phase spectrum
        phase = np.angle(amplitudes)
        ax2.plot(frequencies, phase)
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Phase Spectrum')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def vector_field_plot(self, X, Y, U, V, title="Vector Field"):
        """Plot 2D vector field with streamlines"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Quiver plot
        magnitude = np.sqrt(U**2 + V**2)
        ax1.quiver(X, Y, U, V, magnitude, cmap='plasma')
        ax1.set_title(f'{title} - Quiver Plot')
        ax1.set_aspect('equal')
        
        # Streamline plot
        ax2.streamplot(X, Y, U, V, density=1.5, color=magnitude, cmap='plasma')
        ax2.set_title(f'{title} - Streamlines')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def contour_analysis(self, X, Y, Z, levels=20):
        """Detailed contour analysis of 2D data"""
        fig = plt.figure(figsize=(15, 10))
        
        # 3D surface plot
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_title('3D Surface')
        
        # Filled contour
        ax2 = fig.add_subplot(222)
        cf = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis')
        ax2.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
        plt.colorbar(cf, ax=ax2)
        ax2.set_title('Filled Contour')
        
        # Gradient magnitude
        ax3 = fig.add_subplot(223)
        Zy, Zx = np.gradient(Z)
        grad_mag = np.sqrt(Zx**2 + Zy**2)
        im = ax3.imshow(grad_mag, extent=[X.min(), X.max(), Y.min(), Y.max()],
                       origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax3)
        ax3.set_title('Gradient Magnitude')
        
        # Critical points
        ax4 = fig.add_subplot(224)
        ax4.contour(X, Y, Z, levels=levels, cmap='viridis')
        
        # Find approximate critical points (where gradient is small)
        threshold = 0.1 * np.max(grad_mag)
        critical = grad_mag < threshold
        ax4.scatter(X[critical], Y[critical], c='red', s=10, label='Critical regions')
        ax4.legend()
        ax4.set_title('Critical Points')
        
        plt.tight_layout()
        plt.show()
```

### Data Analysis Tools

```python
class PhysicsDataAnalysis:
    """Tools for analyzing physics simulation data"""
    
    @staticmethod
    def autocorrelation(data, max_lag=None):
        """Calculate autocorrelation function"""
        n = len(data)
        if max_lag is None:
            max_lag = n // 4
        
        # Normalize data
        data = data - np.mean(data)
        c0 = np.dot(data, data) / n
        
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            c_lag = np.dot(data[:-lag-1], data[lag+1:]) / (n - lag - 1)
            acf[lag] = c_lag / c0
        
        return acf
    
    @staticmethod
    def power_spectrum(data, dt=1.0):
        """Calculate power spectrum using Welch's method"""
        from scipy import signal
        
        # Welch's method for smoother spectrum
        frequencies, psd = signal.welch(data, fs=1/dt, nperseg=len(data)//8)
        
        return frequencies, psd
    
    @staticmethod
    def lyapunov_exponent(trajectory, dt=0.01):
        """Estimate largest Lyapunov exponent"""
        n_steps = len(trajectory)
        n_dim = trajectory.shape[1]
        
        # Initialize nearby trajectory
        eps = 1e-8
        separation = eps * np.random.randn(n_dim)
        
        lyap_sum = 0
        
        for i in range(1, n_steps):
            # Evolution of separation vector (linearized dynamics)
            # This is simplified - real implementation needs Jacobian
            separation_new = separation * 1.1  # Placeholder
            
            # Renormalization
            d = np.linalg.norm(separation_new)
            lyap_sum += np.log(d / eps)
            
            separation = eps * separation_new / d
        
        lyapunov = lyap_sum / (n_steps * dt)
        return lyapunov
    
    @staticmethod
    def structure_factor(positions, k_vectors):
        """Calculate structure factor S(k) for particle system"""
        n_particles = len(positions)
        n_k = len(k_vectors)
        
        S_k = np.zeros(n_k)
        
        for i, k in enumerate(k_vectors):
            # Calculate density fluctuation
            rho_k = 0
            for r in positions:
                rho_k += np.exp(1j * np.dot(k, r))
            
            S_k[i] = np.abs(rho_k)**2 / n_particles
        
        return S_k
```

---

## Popular Physics Libraries

### Core Libraries

```python
# Essential imports for computational physics
import numpy as np
import scipy
from scipy import integrate, optimize, linalg
from scipy.sparse import csr_matrix, diags
from scipy.fft import fft, ifft, fft2, ifft2

# Specialized physics libraries
import sympy  # Symbolic mathematics
import h5py   # HDF5 for large datasets
import pandas as pd  # Data analysis

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Interactive plots

# Example: Using SciPy for physics problems
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import jv, yv  # Bessel functions
```

### QuTiP - Quantum Toolbox in Python

```python
import qutip as qt

# Quantum harmonic oscillator
N = 20  # Number of Fock states
a = qt.destroy(N)  # Annihilation operator
H = a.dag() * a  # Hamiltonian

# Initial state: coherent state
alpha = 2.0
psi0 = qt.coherent(N, alpha)

# Time evolution
times = np.linspace(0, 10, 100)
result = qt.mesolve(H, psi0, times)

# Expectation values
n_expect = qt.expect(a.dag() * a, result.states)
```

### MDAnalysis - Molecular Dynamics Analysis

```python
import MDAnalysis as mda

# Load trajectory
u = mda.Universe('topology.pdb', 'trajectory.dcd')

# Analysis example: Radial distribution function
from MDAnalysis.analysis import rdf

g = rdf.InterRDF(u.select_atoms('name O'),
                 u.select_atoms('name O'),
                 nbins=100)
g.run()
```

### FEniCS - Finite Element Library

```python
from fenics import *

# Create mesh and function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
bc = DirichletBC(V, u_D, 'on_boundary')

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Solve
u = Function(V)
solve(a == L, u, bc)
```

### PyCUDA/PyOpenCL - GPU Computing

```python
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel
mod = SourceModule("""
__global__ void add_vectors(float *a, float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}
""")

add_vectors = mod.get_function("add_vectors")
```

---

## Best Practices and Tips

### Performance Optimization

1. **Vectorization**: Always use NumPy operations instead of loops
2. **Memory Management**: Pre-allocate arrays, use in-place operations
3. **Profiling**: Use `cProfile` and `line_profiler` to find bottlenecks
4. **Numba**: JIT compilation for numerical functions

```python
from numba import jit, njit, prange

@njit(parallel=True)
def fast_matrix_multiply(A, B):
    """Numba-accelerated matrix multiplication"""
    m, n = A.shape
    n2, p = B.shape
    C = np.zeros((m, p))
    
    for i in prange(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C
```

### Debugging and Validation

1. **Conservation Laws**: Always check energy, momentum conservation
2. **Dimensional Analysis**: Verify units are consistent
3. **Limiting Cases**: Test known analytical solutions
4. **Convergence Studies**: Vary discretization parameters

```python
def validate_simulation(results):
    """Validation checks for physics simulations"""
    # Energy conservation
    energy = results['kinetic'] + results['potential']
    energy_drift = (energy[-1] - energy[0]) / energy[0]
    assert abs(energy_drift) < 1e-6, f"Energy drift: {energy_drift}"
    
    # Momentum conservation
    momentum = np.sum(results['momenta'], axis=1)
    momentum_change = np.max(np.abs(momentum - momentum[0]))
    assert momentum_change < 1e-10, f"Momentum not conserved: {momentum_change}"
    
    print("✓ Validation passed")
```

---

## Summary

Computational physics has transformed our ability to understand and predict physical phenomena. From solving complex differential equations to simulating quantum systems, the combination of physics knowledge and computational techniques opens up endless possibilities for discovery.

### Key Takeaways

1. **Start Simple**: Master basic numerical methods before tackling complex algorithms
2. **Validate Constantly**: Always check your results against known solutions
3. **Optimize Wisely**: Profile first, optimize the bottlenecks
4. **Use the Right Tool**: Choose algorithms based on problem characteristics
5. **Visualize Results**: Good visualization leads to physical insight

### Next Steps

- Implement a complete simulation for a physical system of interest
- Explore parallel computing for large-scale problems
- Learn specialized libraries for your domain
- Contribute to open-source physics software
- Apply machine learning to enhance traditional methods

Remember: computational physics is not just about writing code—it's about understanding the physics deeply enough to translate it into efficient, accurate algorithms. The computer is a tool for exploration, but physical intuition guides the journey.

---

## Essential Resources

### Software Libraries
- **NumPy/SciPy**: Foundation for scientific computing in Python
- **LAMMPS**: Large-scale molecular dynamics
- **Quantum ESPRESSO**: Electronic structure calculations
- **FEniCS**: Automated finite element methods
- **PETSc**: Scalable solution of PDEs
- **JAX**: Differentiable physics and machine learning

### References
- **Books**: "Computational Physics" by Newman, "Numerical Recipes" series
- **Courses**: MIT OCW Computational Physics, Coursera Scientific Computing
- **Communities**: Stack Exchange Physics, GitHub Physics repositories

---

## See Also

### Core Physics Topics:
- [Classical Mechanics](classical-mechanics.html) - Symplectic integrators and chaos
- [Quantum Mechanics](quantum-mechanics.html) - Numerical solutions to Schrödinger equation
- [Statistical Mechanics](statistical-mechanics.html) - Monte Carlo and molecular dynamics
- [Thermodynamics](thermodynamics.html) - Computational thermodynamics
- [Quantum Field Theory](quantum-field-theory.html) - Lattice field theory simulations

### Related Computational Topics:
- [Condensed Matter Physics](condensed-matter.html) - Band structure calculations
- [String Theory](string-theory.html) - Numerical relativity and AdS/CFT
- [Relativity](relativity.html) - Gravitational wave simulations