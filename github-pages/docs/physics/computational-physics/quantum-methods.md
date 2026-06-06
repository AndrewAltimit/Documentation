---
layout: docs
title: "Computational Physics: Quantum Computational Methods"
permalink: /docs/physics/computational-physics/quantum-methods.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Quantum Computational Methods</p>

Evolving wavefunctions in time and finding electronic ground states — the numerical heart of quantum simulation.

## Time-Dependent Schrödinger Equation

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

## Density Functional Theory (DFT)

<div class="notice--info">
  <p>The snippet below is <strong>illustrative, simplified pseudocode</strong> meant to show the shape of a self-consistent Kohn-Sham loop — building the effective potential, solving for a density, and mixing. It is <em>not</em> a numerically valid DFT solver: a real implementation diagonalizes a Hamiltonian whose kinetic operator is diagonal in momentum space while the potential is diagonal in position space (so the two are never simply added in the same basis), and it constructs the new density from the resulting eigenfunctions.</p>
</div>

```python
class SimpleDFT:
    """Simplified 1D DFT implementation (schematic — see note above)"""
    
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
        
        # SCHEMATIC ONLY: a real Kohn-Sham solver diagonalizes a Hamiltonian
        # whose kinetic term T_k is diagonal in momentum space and whose
        # potential V_eff is diagonal in position space. The two operators
        # live in different bases and must NOT be added directly; the line
        # below is a placeholder standing in for that diagonalization.
        H_diag = self.T_k + np.fft.fft(V_eff)  # not physical — schematic placeholder
        
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

*Previous: [Finite Elements &amp; Fluid Dynamics](fem-and-cfd.html) · Next: [Parallel Computing &amp; Machine Learning](hpc-and-ml.html)*

## See Also

- [Quantum Mechanics](../quantum-mechanics/) — the physics behind wavefunction evolution and the Schrödinger equation.
- [Condensed Matter Physics](../condensed-matter/) — density functional theory and electronic structure in solids.
- [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) — variational and quantum Monte Carlo methods.
