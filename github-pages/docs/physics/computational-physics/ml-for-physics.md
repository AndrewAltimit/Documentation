---
layout: docs
title: "Computational Physics: Machine Learning for Physics"
permalink: /docs/physics/computational-physics/ml-for-physics.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">Computational Physics</a> › Machine Learning for Physics</p>

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Machine Learning for Physics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Learning the physics directly from data — solving PDEs, emulating simulators, and respecting symmetry by construction.</p>
</div>

Machine learning entered computational physics not to replace the differential equations but to *complement* them. A traditional solver discretizes a known equation and grinds through it for one set of boundary conditions; a learned model can absorb a known equation as a soft constraint (PINNs), amortize an entire family of solutions into a single forward pass (neural operators), or replace an intractable quantum-chemistry calculation with a cheap surrogate that still obeys the right invariances (neural-network potentials). The unifying theme is **inductive bias**: the more physics you bake into the architecture or loss, the less data you need and the better the model generalizes.

<div class="insight-card">
  <h4>Why physics priors matter</h4>
  <p>A generic neural network is a universal approximator, but in the small-data, high-stakes regime of physics that universality is a liability — it will happily fit a function that violates energy conservation or rotational symmetry. Encoding the symmetry group, the governing PDE, or the conservation law <em>into</em> the model shrinks the hypothesis space to functions that are physically admissible, which is why a symmetry-aware network can match a generic one with orders of magnitude less data.</p>
</div>

## Recent Advances in Physics-ML Integration

The intersection of machine learning and physics has seen explosive growth, with four broad threads driving most of the progress:

**Major themes:**
- **Neural operators** — learning the *solution operator* of an entire family of PDEs, so one trained model solves new initial/boundary conditions in a single forward pass (Fourier Neural Operators, DeepONet, graph neural operators).
- **Equivariant neural networks** — networks whose outputs transform correctly under physical symmetries (rotations, translations, permutations), giving exact conservation and dramatically better data efficiency.
- **Differentiable physics engines** — simulators written in autodiff frameworks so gradients flow end-to-end through the solver, enabling inverse design, control, and learned closures.
- **Foundation models for science** — large models pre-trained on diverse simulation/experimental data (notably for protein structure, materials, and weather) that are then fine-tuned for specific tasks.

A few concrete landmarks worth knowing: **DeepMind's GraphCast** (2023) showed a graph neural network beating the gold-standard numerical weather model on a 10-day forecast while running thousands of times faster; **AlphaFold 2/3** turned protein and biomolecular structure prediction into a largely solved problem using an equivariant attention architecture; and **neural-network interatomic potentials** (MACE, NequIP, Allegro) now reach near-DFT accuracy on molecular dynamics at a fraction of the cost, all built on equivariant message passing.

## Physics-Informed Neural Networks (PINNs)

A PINN represents the unknown field $u(x,t)$ directly as a neural network $u_\theta(x,t)$ and trains it by minimizing the *residual of the governing PDE* evaluated at sampled collocation points, plus penalties for the boundary and initial conditions. Crucially, the spatial and temporal derivatives in the PDE are computed by **automatic differentiation** of the network — there is no mesh and no finite-difference stencil. For a PDE written abstractly as

$$ \mathcal{N}[u](x,t) = 0, \qquad x \in \Omega,\ t \in [0,T], $$

with boundary operator $\mathcal{B}$ and initial condition $u_0$, the composite loss is

$$ L(\theta) = \lambda_r L_r + \lambda_b L_b + \lambda_i L_i, $$

$$ L_r = \frac{1}{N_r}\sum_{k=1}^{N_r} \left| \mathcal{N}[u_\theta](x_k, t_k) \right|^2, $$

$$ L_b = \frac{1}{N_b}\sum_{k=1}^{N_b} \left| \mathcal{B}[u_\theta](x_k, t_k) \right|^2, \qquad L_i = \frac{1}{N_i}\sum_{k=1}^{N_i} \left| u_\theta(x_k, 0) - u_0(x_k) \right|^2. $$

Each term is just a mean-squared residual, and the $\lambda$ weights balance how aggressively the optimizer enforces the equation versus the data. For the 1D heat equation $u_t = \alpha\, u_{xx}$ the residual the code below minimizes is

$$ \mathcal{N}[u] = \frac{\partial u}{\partial t} - \alpha\, \frac{\partial^2 u}{\partial x^2}. $$

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

### When PINNs shine — and when they struggle

PINNs are most attractive when a mesh is awkward: high-dimensional PDEs (where grid points scale as $N^d$), inverse problems where an unknown coefficient is learned jointly with the field, and problems where sparse experimental data must be fused with a known equation. Because the solution is a smooth analytic function of its inputs, it can be queried at any point without interpolation.

The failure modes are equally important to know:

- **Stiff / multiscale dynamics.** Sharp fronts and turbulent spectra are hard for a smooth tanh network to represent; the residual loss develops pathological gradients. Fourier-feature embeddings and curriculum/time-marching schemes help.
- **Loss balancing.** The relative magnitude of $L_r$, $L_b$, $L_i$ controls everything, and a poorly weighted PINN happily satisfies the PDE in the interior while ignoring the boundary. Adaptive weighting (e.g., learning-rate annealing on the gradients of each term, or the neural-tangent-kernel-based schemes) is often essential.
- **Optimization, not approximation, is the bottleneck.** A network *can* represent the solution, but Adam frequently stalls in a bad minimum; a second-order optimizer (L-BFGS) after Adam warm-up is the standard remedy.

<div class="insight-card">
  <h4>Hard vs. soft constraints</h4>
  <p>Penalizing boundary/initial conditions in the loss is a <em>soft</em> constraint — the network only approximately satisfies them. You can instead enforce them <em>exactly</em> by construction: write the output as <code>u(x,t) = u0(x) + t·g(x) + t·(x+1)(x-1)·N(x,t)</code> so the ansatz already obeys the IC and Dirichlet BCs for any network N. Hard constraints remove the corresponding loss terms entirely and usually train far more reliably.</p>
</div>

## Equivariant and Symmetry-Aware Networks

Physics is full of exact symmetries: the energy of a molecule does not change if you rotate, translate, or relabel its atoms; a force vector *rotates with* the molecule. A generic network has to *learn* these invariances from data, wasting capacity and data on facts that are known a priori. **Equivariant networks** build the symmetry into the architecture so it holds exactly, for free.

Formally, a map $f$ is **invariant** under a group $G$ acting via representations $\rho$ if it ignores the transformation, and **equivariant** if it transforms in step with it:

$$ f(\rho_{\text{in}}(g)\, x) = f(x) \quad \text{(invariant)}, \qquad f(\rho_{\text{in}}(g)\, x) = \rho_{\text{out}}(g)\, f(x) \quad \text{(equivariant)}, \qquad \forall g \in G. $$

Energy is a scalar and should be *invariant* under the Euclidean group $E(3)$ (rotations, translations, reflections); forces and dipoles are vectors and should be *equivariant*. The modern recipe for $E(3)$-equivariance represents features as **spherical tensors** labelled by angular-momentum order $\ell$ (scalars $\ell=0$, vectors $\ell=1$, ...) and combines them with the **Clebsch–Gordan tensor product**, exactly as angular momenta couple in quantum mechanics:

$$ (u^{(\ell_1)} \otimes v^{(\ell_2)})^{(\ell)}_m = \sum_{m_1, m_2} C^{\ell\, m}_{\ell_1 m_1\, \ell_2 m_2}\, u^{(\ell_1)}_{m_1}\, v^{(\ell_2)}_{m_2}, \qquad |\ell_1 - \ell_2| \le \ell \le \ell_1 + \ell_2. $$

Because the Clebsch–Gordan coefficients $C$ encode how irreducible representations of $SO(3)$ combine, any network built from these products is rotation-equivariant by construction. This is the engine inside NequIP, MACE, and Allegro.

The code below shows the simpler but widely used special case: a **permutation- and translation-invariant** message-passing network in the style of SchNet, where invariance to rotation is obtained by using only interatomic *distances* (themselves rotation-invariant) as edge features.

```python
import torch
import torch.nn as nn

class InvariantMessagePassing(nn.Module):
    """E(3)-invariant message-passing layer (SchNet-style).

    Atoms carry scalar features; messages depend only on interatomic
    distances, so the output is invariant to rotation/translation and
    equivariant (here: invariant) to permutation of identical atoms.
    """

    def __init__(self, n_features=64, n_rbf=20, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        # Radial basis: expand each distance into smooth Gaussians
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.width = (cutoff / n_rbf)

        # Filter-generating network: distance-expansion -> per-feature weights
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features), nn.SiLU(),
            nn.Linear(n_features, n_features),
        )
        self.update = nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(),
            nn.Linear(n_features, n_features),
        )

    def rbf(self, distances):
        """Smoothly expand distances onto a Gaussian radial basis."""
        diff = distances[..., None] - self.centers
        return torch.exp(-(diff ** 2) / (2 * self.width ** 2))

    def cosine_cutoff(self, distances):
        """Smoothly zero out interactions beyond the cutoff radius."""
        fc = 0.5 * (torch.cos(torch.pi * distances / self.cutoff) + 1.0)
        return fc * (distances < self.cutoff).float()

    def forward(self, x, positions, edge_index):
        """x: (n_atoms, n_features); edge_index: (2, n_edges)."""
        src, dst = edge_index
        r_vec = positions[dst] - positions[src]
        r = torch.norm(r_vec, dim=-1)                 # rotation-invariant
        # Build continuous filters from distances only
        W = self.filter_net(self.rbf(r)) * self.cosine_cutoff(r)[:, None]
        messages = x[src] * W                         # element-wise filter
        # Aggregate messages onto destination atoms (permutation-invariant sum)
        agg = torch.zeros_like(x).index_add_(0, dst, messages)
        return x + self.update(agg)                   # residual update
```

The crucial design choices — using only distances, summing (not concatenating) neighbor messages, and applying a smooth cutoff — are precisely what make the layer respect the symmetries exactly rather than approximately. A full $E(3)$-equivariant model additionally carries $\ell \ge 1$ vector/tensor features and combines them with the tensor product above so that *directional* quantities like forces come out equivariant.

## Neural-Network Interatomic Potentials

Ab-initio molecular dynamics is limited by the cost of evaluating the energy with density-functional theory at every timestep. A **neural-network potential (NNP)** learns a surrogate $E_\theta(\{\mathbf{r}_i\})$ trained to reproduce DFT energies and forces, then drives MD at a tiny fraction of the cost while retaining near-quantum accuracy. The foundational **Behler–Parrinello** architecture decomposes the total energy into a sum of atomic contributions,

$$ E = \sum_{i=1}^{N_{\text{atoms}}} E_i\big(\mathbf{G}_i\big), $$

where each atomic energy $E_i$ is produced by a shared network fed a fixed-length descriptor $\mathbf{G}_i$ of atom $i$'s local environment. This decomposition makes the model **size-extensive** (energy scales with system size) and **transferable** between systems. The descriptors are **symmetry functions**: rotation- and permutation-invariant functions of the neighbor geometry. A radial symmetry function aggregates pairwise distances,

$$ G_i^{\text{rad}} = \sum_{j \neq i} e^{-\eta (r_{ij} - R_s)^2}\, f_c(r_{ij}), $$

and the smooth cutoff $f_c$ ensures the descriptor and its derivatives vanish continuously at the cutoff radius $R_c$:

$$ f_c(r) = \begin{cases} \tfrac{1}{2}\left[\cos\!\left(\dfrac{\pi r}{R_c}\right) + 1\right], & r \le R_c, \\[4pt] 0, & r > R_c. \end{cases} $$

Forces follow exactly from the analytic gradient of the (smooth, differentiable) energy network — automatic differentiation gives them for free,

$$ \mathbf{F}_i = -\nabla_{\mathbf{r}_i} E. $$

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

The state of the art has since moved from hand-crafted symmetry functions to **learned equivariant descriptors**. Message-passing potentials like **NequIP** and **MACE** carry $E(3)$-equivariant features (the spherical-tensor machinery from the previous section), which lets them reach the same accuracy with one to two orders of magnitude less training data and capture many-body angular information that two-body radial descriptors miss. The trade-off is cost per evaluation: descriptor-based potentials are cheaper per step, equivariant ones are far more data-efficient.

## Fourier Neural Operators (FNO)

Where a PINN learns *one* solution, a **neural operator** learns the *solution operator* of a PDE — a map between infinite-dimensional function spaces, e.g. from an initial-condition function $a(x)$ to the solution $u(x)$. Once trained, it solves a new instance in a single forward pass, with **discretization invariance**: train on a coarse grid, evaluate on a fine one. The **Fourier Neural Operator** realizes this by parameterizing the integral kernel operator in the spectral domain. Each layer applies the update

$$ u^{(l+1)}(x) = \sigma\!\Big( W u^{(l)}(x) + \big(\mathcal{K} u^{(l)}\big)(x) \Big), $$

where $W$ is a pointwise (local) linear map and $\mathcal{K}$ is a global convolution implemented as a multiplication in Fourier space. Concretely, with $\mathcal{F}$ the Fourier transform and $R_\phi$ a learned, complex-valued weight applied to the lowest $k_{\max}$ modes,

$$ \big(\mathcal{K} u\big)(x) = \mathcal{F}^{-1}\!\big( R_\phi \cdot \mathcal{F} u \big)(x). $$

Truncating to the lowest modes acts as a learnable low-pass filter: it captures the smooth, long-range structure that dominates most PDE solutions while keeping the parameter count fixed and *independent of resolution*. Because the FFT is global, a single layer mixes information across the whole domain — unlike a convolution, which only sees its local stencil.

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

The two-channel split in `SpectralConv2d` (low-frequency modes from both the positive and negative ends of the spectrum) reflects the conjugate symmetry of the real FFT. The complementary $1\times 1$ convolution `w` lets the operator represent the high-frequency, local residual that the truncated spectral path discards — the same low-pass-plus-local-correction structure that makes FNOs both expressive and resolution-robust. Beyond the FNO, **DeepONet** offers an alternative operator-learning paradigm (a "branch" network encodes the input function and a "trunk" network encodes the query location), and **graph neural operators** generalize the construction to irregular meshes where the FFT no longer applies.

<div class="insight-card">
  <h4>PINN vs. neural operator: which to reach for</h4>
  <p>Use a <strong>PINN</strong> when you need <em>one</em> high-accuracy solution to a known PDE, possibly with an unknown parameter to infer, and have little or no labeled data. Use a <strong>neural operator</strong> when you must solve the <em>same</em> PDE family thousands of times (uncertainty quantification, design optimization, real-time control) and can afford to generate a training set of paired input–output functions once.</p>
</div>

## See Also

- [Parallel Computing](hpc-and-ml.html) — the MPI and GPU infrastructure that trains these models and generates their data.
- [Finite Elements &amp; Fluid Dynamics](fem-and-cfd.html) — the Navier-Stokes solvers that FNOs learn to emulate.
- [Monte Carlo &amp; Molecular Dynamics](monte-carlo-and-md.html) — the MD trajectories that neural-network potentials accelerate.
- [Quantum Computational Methods](quantum-methods.html) — the DFT calculations that supply training labels for NNPs.
- [Visualization, Libraries &amp; Best Practices](tools-and-practices.html) — frameworks and tooling for the surrounding workflow.
