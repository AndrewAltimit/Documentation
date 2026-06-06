---
layout: docs
title: "Computational Physics: Monte Carlo & Molecular Dynamics"
permalink: /docs/physics/computational-physics/monte-carlo-and-md.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Monte Carlo &amp; Molecular Dynamics</p>

Sampling huge state spaces with randomness, and following particles through time with Newton's laws.

## Monte Carlo Methods

Monte Carlo methods trade exactness for scalability: instead of evaluating a sum or integral over every point in a vast space, they *sample* it randomly and average. The estimate's error falls as $1/\sqrt{N}$ no matter how many dimensions the space has — the property that makes Monte Carlo indispensable for statistical mechanics (summing over $2^N$ spin configurations), quantum field theory (path integrals), and Bayesian inference. The trade-off is statistical noise: to halve the error you must quadruple the samples.

**The curse — and blessing — of dimensionality.** A grid with 100 points per axis needs $100^d$ evaluations in $d$ dimensions: hopeless beyond a handful of axes. Monte Carlo sidesteps this entirely — its $1/\sqrt{N}$ error is *independent of dimension*. This is why simulating $10^{23}$-particle systems is even thinkable.

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

Plain Monte Carlo needs you to draw samples directly from the target distribution — easy for a uniform square, impossible for the Boltzmann distribution $e^{-\beta E}$ over $10^{23}$ interacting particles. **MCMC** solves this by constructing a random walk whose *stationary distribution* is the target: take many small, correlated steps, and the chain spends time in each state in proportion to its probability. The **Metropolis-Hastings** rule makes this concrete — propose a move, then accept it with probability $\min(1, p_{\text{new}}/p_{\text{old}})$. Moves to more probable states are always accepted; moves to less probable ones are accepted just often enough to sample the tails correctly, never requiring the (usually intractable) normalization constant.

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
    
    def ewald_summation(self, charges, alpha=None):
        """Ewald summation for long-range electrostatics"""
        # Default screening parameter depends on the box size; compute it here
        # rather than in the signature (a default arg cannot reference `self`).
        if alpha is None:
            alpha = 5.0 / self.box_size
        
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

*Previous: [Computational Physics Hub](./) · Next: [Finite Elements &amp; Fluid Dynamics](fem-and-cfd.html)*

## See Also

- [Statistical Mechanics](../statistical-mechanics/) — the physics behind Monte Carlo sampling and ensembles.
- [Quantum Computational Methods](quantum-methods.html) — split-operator evolution and DFT.
- [Parallel Computing &amp; Machine Learning](hpc-and-ml.html) — scaling MD and Monte Carlo across cores and GPUs.
