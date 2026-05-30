---
layout: docs
title: "Computational Physics: Parallel Computing & Machine Learning"
permalink: /docs/physics/computational-physics/hpc-and-ml.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Parallel Computing &amp; Machine Learning</p>

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Parallel Computing &amp; Machine Learning</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Scaling simulations across cores and GPUs, and learning the physics directly from data.</p>
</div>

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

*Previous: [Quantum Computational Methods](quantum-methods.html) · Next: [Visualization, Libraries &amp; Best Practices](tools-and-practices.html)*

## See Also

- [Finite Elements &amp; Fluid Dynamics](fem-and-cfd.html) — the Navier-Stokes solvers that FNOs learn to emulate.
- [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) — embarrassingly parallel sampling and neural-network potentials.
- [Classical Mechanics](../classical-mechanics/) — the $N$-body dynamics behind GPU gravitational simulations.
