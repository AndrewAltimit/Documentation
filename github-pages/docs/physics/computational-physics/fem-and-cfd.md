---
layout: docs
title: "Computational Physics: Finite Elements & Fluid Dynamics"
permalink: /docs/physics/computational-physics/fem-and-cfd.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Finite Elements &amp; Fluid Dynamics</p>

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Finite Elements &amp; Fluid Dynamics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Discretizing complex geometries with finite elements, and simulating flow with Navier-Stokes and lattice methods.</p>
</div>

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

*Previous: [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) · Next: [Quantum Computational Methods](quantum-methods.html)*

## See Also

- [Classical Mechanics](../classical-mechanics/) — the continuum mechanics underlying fluid dynamics.
- [Quantum Computational Methods](quantum-methods.html) — spectral and finite-difference solvers for the Schrödinger equation.
- [Visualization, Libraries &amp; Best Practices](tools-and-practices.html) — FEniCS and tools for visualizing flow fields.
