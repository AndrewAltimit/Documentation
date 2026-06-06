---
layout: docs
title: "Computational Physics: Visualization, Libraries & Best Practices"
permalink: /docs/physics/computational-physics/tools-and-practices.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Visualization, Libraries &amp; Best Practices</p>

Turning raw arrays into insight, the Python physics ecosystem, and the habits that keep simulations trustworthy.

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

*Previous: [Parallel Computing &amp; Machine Learning](hpc-and-ml.html) · Next: [Computational Physics Hub](./)*

## See Also

- [Computational Physics Hub](./) — back to the overview and numerical-methods foundations.
- [Finite Elements &amp; Fluid Dynamics](fem-and-cfd.html) — FEniCS in action for solving PDEs.
- [Statistical Mechanics](../statistical-mechanics/) — the physics behind autocorrelation and structure factors.
