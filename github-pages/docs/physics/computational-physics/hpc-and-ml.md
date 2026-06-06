---
layout: docs
title: "Computational Physics: Parallel & High-Performance Computing"
permalink: /docs/physics/computational-physics/hpc-and-ml.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Parallel &amp; High-Performance Computing</p>

Scaling scientific simulations across cores, nodes, and GPUs with MPI, CUDA, and sparse iterative solvers.

Modern physics simulations routinely exceed what a single core can handle: a
three-dimensional fluid solver on a $1024^3$ grid holds billions of unknowns, an
exact-diagonalization study of a quantum spin chain confronts a Hilbert space that
doubles in size with every added spin, and an $N$-body cosmology run tracks
trillions of particles. **High-performance computing (HPC)** is the discipline of
mapping these problems onto parallel hardware — many cores within a node, many
nodes across a cluster interconnect, and the thousands of lightweight threads on a
GPU — while keeping communication, memory traffic, and load imbalance from
destroying the speedup.

> **Looking for machine learning?** The physics-ML material (physics-informed
> neural networks, Fourier neural operators, neural-network potentials,
> equivariant models) now lives on its own page:
> [Machine Learning for Physics](ml-for-physics.html). This page is dedicated to
> classical parallel and high-performance scientific computing.

## Why Parallelism Is Hard: Amdahl and Gustafson

Before writing a line of MPI, it pays to know the ceiling. If a fraction $p$ of a
program's runtime is parallelizable and the rest is inherently serial, then on $N$
processors **Amdahl's law** caps the speedup at

$$
S(N) = \frac{1}{(1-p) + \dfrac{p}{N}} \xrightarrow{N \to \infty} \frac{1}{1-p}.
$$

A code that is 95% parallel can never run more than $20\times$ faster, no matter
how many cores you throw at it — the serial 5% dominates. This is the pessimistic,
*strong-scaling* view: fixed problem size, more processors.

**Gustafson's law** reframes the goal. In practice scientists scale the *problem*
with the machine — a bigger cluster runs a finer grid, not the same grid faster.
For a fixed time budget the scaled speedup is

$$
S(N) = N - (1-p)(N - 1),
$$

which grows almost linearly in $N$. This *weak-scaling* perspective is why
exascale machines remain useful despite Amdahl: the serial fraction stays roughly
constant while the parallel work grows. Real codes are benchmarked against both
curves — strong scaling at fixed size to expose communication overhead, weak
scaling at fixed work-per-rank to expose network and memory bottlenecks.

## MPI for Distributed Computing

The **Message Passing Interface (MPI)** is the lingua franca of cluster computing.
Each process (a *rank*) owns a private address space and exchanges data with peers
only through explicit messages — there is no shared memory, so the model scales
from a laptop to hundreds of thousands of nodes. The core primitives fall into two
families:

- **Point-to-point**: `Send`/`Recv` (and the combined `Sendrecv`) move data
  between two named ranks. These dominate halo exchanges in stencil codes.
- **Collectives**: `Bcast`, `Scatter`, `Gather`, `Reduce`, and `Allreduce`
  involve every rank in a communicator. A reduction such as
  $\sum_r x_r$ over $N$ ranks completes in $O(\log N)$ communication steps via a
  binary tree, not $O(N)$.

A natural first example is **embarrassingly parallel** Monte Carlo: each rank draws
independent samples (with a distinct random seed to avoid correlated streams) and a
single reduction combines the partial counts. The only communication is the final
`reduce`, so the code scales nearly perfectly.

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

### Domain Decomposition and Ghost Cells

For PDE solvers the dominant parallel strategy is **domain decomposition**: the
global grid is partitioned into contiguous subdomains, one per rank, and each rank
updates its own interior. A finite-difference or finite-volume stencil, however,
reads neighboring cells — at a subdomain boundary those neighbors live on another
rank. The fix is a layer of **ghost cells** (also called *halo* cells): each
subdomain is padded by one or more rows that mirror the boundary data of its
neighbors. Every time step begins with a *halo exchange* (the `Sendrecv` calls
above) that refreshes these ghost layers, after which the local update proceeds as
if the data were contiguous.

The performance of this pattern is governed by the **surface-to-volume ratio**. A
cubic subdomain of side $n$ holds $n^3$ interior cells (computation) but exposes
$6n^2$ boundary faces (communication), so the communication-to-computation ratio
scales as

$$
\frac{T_{\text{comm}}}{T_{\text{comp}}} \sim \frac{6 n^2}{n^3} = \frac{6}{n}.
$$

Larger subdomains amortize communication better, which is exactly why **weak
scaling** (fixed $n$ per rank) holds up far better than strong scaling (shrinking
$n$ as ranks grow). The message cost itself follows the classic linear model

$$
T_{\text{msg}}(m) = \alpha + \beta\, m,
$$

where $\alpha$ is the per-message *latency* and $\beta$ the inverse *bandwidth* and
$m$ the message size in bytes. Latency $\alpha$ punishes many small messages, so
high-performance halo exchanges aggregate boundary data into a few large buffers and
overlap communication with computation using non-blocking `Isend`/`Irecv`.

### Choosing a Decomposition

A 1D slab decomposition is trivial to code but its message size grows with the
domain face; a 2D or 3D *blocked* decomposition (the $p_x \times p_y$ processor grid
in the code) minimizes the per-rank surface area and is the standard choice at
scale. The general rule: pick the decomposition that minimizes total surface area
subject to the constraint that subdomains tile the grid evenly. Tools like
`MPI_Cart_create` build a Cartesian communicator that automates neighbor-rank
lookup and can let the MPI runtime map ranks onto the physical network topology for
locality.

## GPU Computing with CUDA/CuPy

A modern GPU is a throughput machine: thousands of arithmetic units organized into
*streaming multiprocessors*, fed by very high memory bandwidth (terabytes per
second) but with comparatively high latency. Code is written as a **kernel** — a
single function executed by a grid of threads, grouped into *blocks* of (typically)
128–512 threads. The CUDA execution model is *Single Instruction, Multiple Threads*
(SIMT): threads within a 32-lane *warp* execute in lockstep, so divergent branches
within a warp serialize and hurt throughput.

GPUs excel precisely on the data-parallel kernels that pervade physics: applying
the same stencil to every grid point, computing pairwise forces between every pair
of particles, or transforming a field by FFT. Two performance facts dominate kernel
design:

- **Coalesced memory access.** Consecutive threads should read consecutive
  addresses so the hardware fuses them into one wide transaction; strided or random
  access wastes bandwidth.
- **Occupancy.** Enough resident warps must be in flight to hide the hundreds-of-cycle
  latency of a global-memory load behind the arithmetic of other warps.

The two kernels below illustrate the two canonical regimes. The direct $N$-body
force loop is **compute-bound** — each thread reads $O(N)$ positions but does
$O(N)$ floating-point operations, so arithmetic dominates. The spectral heat-equation
solver is **bandwidth-bound** — the FFT moves the whole field through memory several
times per step, so it lives or dies on memory throughput, which is exactly what the
GPU's FFT library is tuned for.

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

### The Roofline Model

Whether a kernel is worth porting to the GPU — and whether it is running near peak —
is answered by the **roofline model**. Plot achievable performance (FLOP/s) against
*arithmetic intensity* $I$, the ratio of floating-point operations to bytes moved
from memory:

$$
P = \min\left( P_{\text{peak}},\; B \cdot I \right),
$$

where $P_{\text{peak}}$ is the hardware's peak compute rate and $B$ its peak memory
bandwidth. Kernels with low intensity (a stencil reads several neighbors and does a
handful of flops, $I \lesssim 1$) sit on the *bandwidth-limited* sloped roof — the
direct $N$-body kernel above, with $I = O(N)$, sits on the flat *compute-limited*
roof. The model tells you the optimization that matters: improving cache reuse and
data layout for bandwidth-bound code, or reducing instruction count and improving
occupancy for compute-bound code. The data-transfer cost of moving fields across the
PCIe/NVLink bus to and from the GPU is a separate, often dominant, overhead — the
practical rule is to keep data resident on the device across many time steps, which
is exactly why the loops above never copy back until the simulation finishes.

## Sparse Linear Algebra and Iterative Solvers

Discretizing a PDE or a quantum Hamiltonian produces an enormous matrix $A$ that is
**sparse**: a 7-point Laplacian stencil on a $10^6$-cell grid yields a $10^6 \times
10^6$ matrix with only $\sim 7 \times 10^6$ nonzeros — storing it densely would need
$10^{12}$ entries, which is hopeless. Sparse formats (CSR, CSC, COO) store only the
nonzeros, and the central operation becomes the **sparse matrix–vector product**
(SpMV) $y = Ax$, which costs $O(\text{nnz})$ rather than $O(n^2)$. SpMV is the
workhorse — and the bottleneck — of essentially every large-scale physics solver,
and because it touches each matrix entry exactly once it is firmly bandwidth-bound.

A direct factorization ($LU$, Cholesky) of such a matrix suffers **fill-in**: the
factors are far denser than $A$, exhausting memory. The alternative is **iterative
solvers**, which only ever need to *apply* $A$ to a vector and so preserve sparsity.

### Stationary Iterations: Jacobi and Gauss-Seidel

The simplest iterative scheme splits $A = D + L + U$ into its diagonal, strictly
lower, and strictly upper parts. The **Jacobi iteration**,

$$
x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \ne i} a_{ij}\, x_j^{(k)} \right),
$$

updates every unknown from the *previous* sweep's values, so all updates are
independent — it is naturally parallel, which is why the implementation below shards
rows across a process pool. It converges only when the spectral radius of the
iteration matrix $-D^{-1}(L+U)$ is below one (guaranteed for diagonally dominant
$A$), and even then slowly: the error decays by a factor set by that spectral radius
each sweep. Gauss-Seidel and successive over-relaxation (SOR) converge faster by
reusing updated values within a sweep, at the cost of a sequential dependence that
must be broken with red-black colouring to parallelize.

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

### Krylov Subspace Methods: Conjugate Gradient

Stationary iterations are easy to parallelize but converge slowly. The
state-of-the-art for large symmetric positive-definite systems is the **conjugate
gradient (CG)** method, the canonical *Krylov subspace* solver. CG builds its
approximate solution from the Krylov space

$$
\mathcal{K}_k(A, r_0) = \mathrm{span}\{ r_0,\, A r_0,\, A^2 r_0,\, \dots,\, A^{k-1} r_0 \},
$$

choosing at each step the iterate that minimizes the $A$-norm of the error over that
space. Each iteration costs one SpMV plus a few vector inner products, and the
convergence rate is governed by the condition number $\kappa = \lambda_{\max} /
\lambda_{\min}$:

$$
\frac{\lVert e_k \rVert_A}{\lVert e_0 \rVert_A} \le 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^{k}.
$$

The error therefore shrinks like $(\sqrt{\kappa})^{-1}$ per step — dramatically
faster than Jacobi when $\kappa$ is moderate. For ill-conditioned systems (a fine
grid gives $\kappa \sim h^{-2}$) the cure is a **preconditioner** $M \approx A$ that
is cheap to invert; solving $M^{-1}Ax = M^{-1}b$ instead clusters the spectrum and
collapses the iteration count. Incomplete-Cholesky and multigrid preconditioners are
standard, and for nonsymmetric systems GMRES and BiCGSTAB play the analogous role.
Production libraries (PETSc, Trilinos, hypre) provide these solvers in
MPI-parallel form, where the only communication per iteration is the halo exchange
inside SpMV and the global `Allreduce` inside each inner product.

### The Lanczos Algorithm for Eigenvalues

Many physics problems need the *extremal* eigenpairs of a giant sparse Hermitian
matrix rather than the solution of a linear system: the ground-state energy of a
quantum Hamiltonian, the lowest vibrational modes of a structure, or the dominant
PageRank-like modes of a network. The **Lanczos algorithm** is the symmetric
Krylov method for this task. Starting from a random vector $v_1$, it builds an
orthonormal basis of the Krylov space and, in that basis, *projects* $A$ onto a tiny
tridiagonal matrix

$$
T_k = \begin{pmatrix}
\alpha_1 & \beta_1 & & \\
\beta_1 & \alpha_2 & \beta_2 & \\
& \beta_2 & \ddots & \ddots \\
& & \ddots & \alpha_k
\end{pmatrix},
$$

whose entries come from the three-term recurrence

$$
\beta_{j}\, v_{j+1} = A v_j - \alpha_j v_j - \beta_{j-1} v_{j-1},
\qquad \alpha_j = v_j^{\top} A v_j .
$$

The extreme eigenvalues of $T_k$ (the *Ritz values*, found with a dense solver since
$k \ll n$) converge to the extreme eigenvalues of $A$ after only $k \sim 10$–$100$
iterations, even when $A$ has dimension $10^9$. Crucially, the algorithm never forms
$A$ explicitly — it needs only the action $A v$, supplied as a function, which is why
the `lanczos_eigenvalues` routine above accepts a `H_func` with a `matvec`. In finite
precision the basis loses orthogonality once a Ritz value converges, producing
spurious "ghost" eigenvalues; robust implementations restore orthogonality with
selective or full reorthogonalization (this is the engine inside SciPy's `eigsh` and
ARPACK). The same machinery, run on the action of a spin Hamiltonian, is the basis of
exact-diagonalization studies of quantum lattice models.

## Performance Optimization in Practice

Hardware peak is rarely reached by accident. The optimizations that matter most, in
rough order of impact:

- **Choose the right algorithm first.** A Barnes-Hut tree code turns the $O(N^2)$
  direct $N$-body sum into $O(N \log N)$; a multigrid solver turns an $O(n^2)$
  iterative solve into $O(n)$. No amount of low-level tuning beats a better
  asymptotic complexity.
- **Respect the memory hierarchy.** Registers and L1 cache are orders of magnitude
  faster than DRAM. *Cache blocking* (tiling loops so each tile fits in cache) and
  *structure-of-arrays* layouts (so vector lanes load contiguous data) turn
  bandwidth-bound kernels into compute-bound ones.
- **Vectorize.** Modern CPUs execute SIMD instructions (AVX-512 processes 16
  single-precision lanes at once); aligned, unit-stride, branch-free inner loops let
  the compiler auto-vectorize.
- **Overlap communication and computation.** Non-blocking MPI (`Isend`/`Irecv`)
  lets a rank update its subdomain interior while halo messages are in flight,
  hiding the network behind useful work.
- **Exploit hybrid parallelism.** The standard pattern at scale is **MPI + X**: MPI
  across nodes, OpenMP threads within a node's shared memory, and CUDA on the GPUs —
  matching each level of parallelism to the corresponding level of the hardware
  hierarchy.
- **Measure, don't guess.** Profilers (VTune, Nsight, `perf`, TAU) and the roofline
  model reveal whether a kernel is bound by compute, bandwidth, latency, or load
  imbalance — and therefore which of the above to apply. Premature micro-optimization
  of a kernel that is only 2% of runtime is wasted effort.

A useful mental model is to classify every hot kernel by its limiting resource and
attack that resource specifically: bandwidth-bound code wants better locality,
compute-bound code wants fewer or cheaper instructions and higher occupancy,
latency-bound code wants more concurrency in flight, and communication-bound code
wants larger, fewer, overlapped messages.

---

*Previous: [Quantum Computational Methods](quantum-methods.html) · Next: [Visualization, Libraries &amp; Best Practices](tools-and-practices.html)*

## See Also

- [Machine Learning for Physics](ml-for-physics.html) — physics-informed neural networks, neural operators, and learned potentials (the ML material formerly on this page).
- [Finite Elements &amp; Fluid Dynamics](fem-and-cfd.html) — the sparse linear systems and stencil solvers that these methods accelerate.
- [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) — embarrassingly parallel sampling, the natural first target for MPI.
- [Visualization, Libraries &amp; Best Practices](tools-and-practices.html) — profiling tools and the libraries (PETSc, CuPy, mpi4py) used here.
- [Classical Mechanics](../classical-mechanics/) — the $N$-body dynamics behind GPU gravitational simulations.
