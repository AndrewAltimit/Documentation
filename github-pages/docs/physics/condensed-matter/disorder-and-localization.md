---
layout: docs
title: "Condensed Matter: Disorder & Localization"
permalink: /docs/physics/condensed-matter/disorder-and-localization.html
toc: true
toc_sticky: true
hide_title: true
---

[Condensed Matter Physics](./) &raquo; Disorder &amp; Localization

<!-- Custom styles are now loaded via main.scss -->

## Disorder & Localization

## Why Disorder Changes Everything

Bloch's theorem is the foundation of band theory: in a *perfect* crystal, the eigenstates are extended plane-wave-like states $\psi_{n\mathbf{k}}$ that spread coherently over the entire sample. A perfect crystal with a partially filled band is therefore a *perfect conductor* — at zero temperature an electron in an extended state propagates forever, and the only thing that gives a real metal finite resistance is scattering off lattice vibrations (phonons) and impurities.

The conventional picture treats that impurity scattering perturbatively. An electron travels ballistically for a **mean free path** $\ell = v_F \tau$ (with $\tau$ the elastic scattering time and $v_F$ the Fermi velocity), then scatters into a new direction; many such events randomize its momentum and the motion becomes diffusive with diffusion constant $D = v_F^2 \tau / d$ in $d$ dimensions. The Drude/Boltzmann conductivity follows,

$$\sigma = \frac{n e^2 \tau}{m} = e^2 \, \nu(E_F) \, D,$$

where $n$ is the carrier density, $m$ the effective mass, and $\nu(E_F)$ the density of states at the Fermi level. In this **semiclassical** picture disorder merely *limits* the conductivity to a finite value; it never destroys metallic behavior outright.

In 1958 P. W. Anderson showed this picture is incomplete. If the disorder is strong enough — or, as we will see, in low enough dimension for *any* disorder — quantum interference between the many scattered partial waves can cause the electron wavefunction to become **exponentially localized**:

$$|\psi(\mathbf{r})| \sim \exp\left(-\frac{|\mathbf{r} - \mathbf{r}_0|}{\xi}\right),$$

bound to a region of size $\xi$ (the **localization length**) around some point $\mathbf{r}_0$. A localized electron cannot carry current at $T = 0$: $\sigma \to 0$ even though the density of states at $E_F$ is finite and nonzero. This is **Anderson localization** — a genuine *insulator made entirely by disorder*, with no band gap, no broken symmetry, and no interactions required. It is one of the cleanest examples of a quantum phase transition driven purely by quantum interference.

This page develops the physics in four steps: the Anderson model and the localization transition; the *weak-localization* quantum correction that survives even in good metals; the **mobility edge** that separates extended from localized states and drives the metal-insulator transition; the **scaling theory** that ties it all together; and finally a note on the modern, interacting cousin, **many-body localization**.

## Anderson Localization

### The Anderson Tight-Binding Model

The canonical model is a tight-binding lattice with random on-site energies. On a $d$-dimensional lattice with sites $i$,

$$H = \sum_i \varepsilon_i \, c_i^\dagger c_i - t \sum_{\langle i j \rangle} \left( c_i^\dagger c_j + \text{h.c.} \right),$$

where $t$ is the hopping amplitude between nearest neighbours and the on-site energies $\varepsilon_i$ are *independent random variables*, usually drawn from a uniform box distribution of width $W$:

$$P(\varepsilon_i) = \frac{1}{W} \quad \text{for} \quad -\frac{W}{2} \le \varepsilon_i \le \frac{W}{2}.$$

The single dimensionless control parameter is the **disorder strength** $W/t$. When $W = 0$ the model is a clean tight-binding band of width $2zt$ (with $z$ the coordination number), all states extended. The question is what happens as $W/t$ grows.

### The Localization Transition

The physical content is a competition between two effects:

- **Hopping** $t$ delocalizes: it wants to spread the electron over the lattice into extended Bloch-like states.
- **Disorder** $W$ localizes: large random energy mismatches between neighbouring sites suppress resonant hopping. An electron on a site of energy $\varepsilon_i$ can only hop efficiently to a neighbour whose energy lies within $\sim t$ of $\varepsilon_i$; when the typical mismatch $\sim W$ greatly exceeds $t$, such resonant neighbours are rare and the electron is trapped.

Anderson's insight, sharpened by the locator expansion and later by self-consistent and scaling treatments, is that above a critical disorder $W_c$ *all* eigenstates become localized:

$$\frac{W}{t} > \left(\frac{W}{t}\right)_c \;\;\Longrightarrow\;\; \text{all states localized} \;\;\Longrightarrow\;\; \text{insulator}.$$

For the 3D cubic-lattice Anderson model with a box distribution the critical value is $(W/t)_c \approx 16.5$. Below threshold, states in the band centre remain extended while states in the band tails are localized — the two are separated by a *mobility edge* (next section but one).

### Inverse Participation Ratio: A Diagnostic

A convenient quantitative measure of how spread out an eigenstate $\psi_n(i)$ is across $N$ sites is the **inverse participation ratio** (IPR),

$$P_2^{(n)} = \sum_i |\psi_n(i)|^4 .$$

If the state spreads uniformly over $M$ sites, each amplitude is $|\psi|^2 \sim 1/M$ and $P_2 \sim 1/M$. Hence:

- **Extended state:** $M \sim N$, so $P_2 \sim 1/N \to 0$ as the system grows.
- **Localized state:** $M \sim \xi^d$, independent of $N$, so $P_2 \to \text{const} > 0$ in the thermodynamic limit.

The IPR is the workhorse diagnostic in numerical studies of localization. The following code builds a 1D Anderson chain, diagonalizes it, and shows that *every* eigenstate is localized — the hallmark of one dimension.

```python
import numpy as np

def anderson_1d(N, W, t=1.0, seed=0):
    """Build the 1D Anderson Hamiltonian: random on-site energies + hopping."""
    rng = np.random.default_rng(seed)
    eps = rng.uniform(-W / 2, W / 2, size=N)   # disordered on-site energies
    H = np.diag(eps)
    off = -t * np.ones(N - 1)
    H += np.diag(off, 1) + np.diag(off, -1)    # nearest-neighbour hopping
    return H

def inverse_participation_ratio(psi):
    """IPR of a normalized eigenvector; ~1/N if extended, O(1) if localized."""
    p = np.abs(psi) ** 2
    return np.sum(p ** 2)

# In 1D, ANY nonzero disorder localizes every state.
for N in (500, 1000, 2000):
    H = anderson_1d(N, W=1.0)
    evals, evecs = np.linalg.eigh(H)
    mid = N // 2                       # a band-centre state
    ipr = inverse_participation_ratio(evecs[:, mid])
    # IPR stays ~constant as N grows -> localized (a 1/N trend would mean extended)
    print(f"N={N:5d}  IPR(mid)={ipr:.4e}  ->  effective extent ~ {1/ipr:.1f} sites")
```

Running this shows the IPR (and hence the localization length read off as $1/\text{IPR}$) saturating to an $N$-independent value: the state occupies a fixed number of sites no matter how long the chain. Contrast this with a clean chain ($W = 0$), where the IPR would fall off as $1/N$.

## Weak Localization & Quantum Corrections to Conductivity

Anderson localization in its strong form needs large disorder. But quantum interference leaves a fingerprint *even in good metals* with $k_F \ell \gg 1$, in the form of a small, temperature- and field-dependent correction to the Drude conductivity called **weak localization**. It is the precursor of full localization and one of the most experimentally accessible interference effects in solids.

### The Coherent Backscattering Picture

Consider the quantum amplitude for an electron to return to its starting point after diffusing through the disordered medium. The return probability is a sum over all closed diffusive paths. For any such loop there is a **time-reversed partner** that traverses exactly the same impurities in the opposite order. In the absence of magnetic fields or magnetic impurities, the two amplitudes $A_+$ and $A_- = A_+$ are *identical* (time-reversal symmetry), so when we square the total amplitude the cross term survives:

$$|A_+ + A_-|^2 = |A_+|^2 + |A_-|^2 + 2\,\mathrm{Re}(A_+ A_-^*) = 4|A_+|^2,$$

twice the classical (incoherent) value $2|A_+|^2$. The electron is therefore *more* likely to return to its origin than classical diffusion predicts. Enhanced backscattering means reduced forward transport, so the conductivity is *reduced* below the Drude value. This **constructive interference of time-reversed loops** is the microscopic origin of weak localization, and in diagrammatic language it is the sum of maximally crossed ("Cooperon") diagrams.

### Magnitude and Dimensionality

Summing over diffusive loops of all durations from the elastic time $\tau$ up to the **phase-coherence time** $\tau_\phi$ (beyond which inelastic scattering randomizes the phase and kills the interference) gives the weak-localization correction. In two dimensions it is logarithmic:

$$\Delta\sigma = -\frac{e^2}{\pi h} \ln\!\left(\frac{\tau_\phi}{\tau}\right) = -\frac{e^2}{\pi h} \ln\!\left(\frac{L_\phi}{\ell}\right),$$

where $L_\phi = \sqrt{D \tau_\phi}$ is the phase-coherence length. The general dimensional dependence of the correction (per unit volume in 3D, per square in 2D, per length in 1D) is

$$\Delta\sigma \propto
\begin{cases}
-(L_\phi - \ell) & d = 1, \\
-\ln(L_\phi / \ell) & d = 2, \\
+(1/\ell - 1/L_\phi) & d = 3.
\end{cases}$$

The crucial observation is that in $d \le 2$ the correction *diverges* as $\tau_\phi \to \infty$ (i.e. as $T \to 0$, since $\tau_\phi$ grows on cooling). The metallic state in one and two dimensions is therefore *unstable*: lowering the temperature drives $\sigma$ down without bound, and the system flows toward an insulator. This is the perturbative signature of the scaling result that there is **no true metal in 1D or 2D** for noninteracting electrons with time-reversal symmetry.

### Magnetoresistance: The Experimental Smoking Gun

Weak localization is most cleanly identified through its response to a magnetic field. A perpendicular field $B$ threads the time-reversed loops with opposite Aharonov-Bohm phases, breaking the equality $A_+ = A_-$ once the flux through a typical coherent loop reaches a flux quantum. The field thus *destroys* the interference correction and *restores* the Drude conductivity:

$$\Delta\sigma(B) - \Delta\sigma(0) > 0 \quad\Longrightarrow\quad \text{negative magnetoresistance.}$$

A characteristic **negative magnetoresistance** at low fields and low temperatures, with a magnitude of order $e^2/h$ per square and a width set by $B_\phi \sim \hbar/(e L_\phi^2)$, is the experimental signature. Measuring its field and temperature dependence is the standard way to extract the phase-coherence length $L_\phi(T)$ in mesoscopic metals and 2D electron gases.

### Spin-Orbit Coupling and Antilocalization

The sign of the effect depends on the symmetry class. Strong **spin-orbit scattering** rotates the electron spin around the loop in a way that flips the relative sign of the time-reversed amplitudes, so the interference becomes *destructive* and *enhances* the conductivity. This is **weak antilocalization**, which produces a *positive* low-field magnetoresistance instead — a hallmark of materials with strong spin-orbit coupling (heavy-element metals, topological-insulator surface states). The three canonical behaviours map onto the Wigner-Dyson symmetry classes: orthogonal (weak localization), symplectic (weak antilocalization), and unitary (no correction, time-reversal already broken by $B$).

## The Mobility Edge & Metal-Insulator Transition

### Mott's Mobility Edge

In three dimensions, weak and moderate disorder does not localize everything. Instead, as N. F. Mott emphasized, localized and extended states *coexist in energy but never at the same energy* — they are separated by a sharp boundary called the **mobility edge** $E_c$:

- States with energy $|E| > E_c$ (in the band tails, where the density of states is low) are **localized**.
- States with energy $|E| < E_c$ (near the band centre, high density of states) are **extended**.

The mobility edge cannot lie at an energy where extended and localized states overlap, because an extended and a localized state at the same energy would hybridize and the localized one would delocalize. The transition at $E_c$ is therefore sharp.

### Driving the Metal-Insulator Transition

Whether the material is a metal or an insulator at $T = 0$ depends on where the Fermi level sits relative to the mobility edge:

$$E_F > E_c \;\Rightarrow\; \text{metal} \qquad\qquad E_F < E_c \;\Rightarrow\; \text{insulator}.$$

There are two experimental knobs:

1. **Tune the disorder** $W$. Increasing disorder pushes the mobility edge *inward* toward the band centre. At the critical disorder the two mobility edges (one in each tail) merge at the band centre, $E_c \to 0$, and *all* states localize — the Anderson transition described earlier.
2. **Tune the carrier density / Fermi level.** In doped semiconductors (the textbook example is phosphorus-doped silicon, Si:P) one sweeps $E_F$ through a fixed $E_c$ by changing the dopant concentration. Below a critical density $n_c$ the system is a disorder-driven insulator; above it, a metal.

This **Anderson metal-insulator transition** is a genuine $T = 0$ quantum critical point. It is distinct from the **Mott metal-insulator transition**, which is driven by electron-electron *interaction* ($U/t$ in the Hubbard model) rather than disorder; real materials such as Si:P near the transition involve *both* effects, and disentangling them (the "Anderson-Mott" problem) remains an active research question.

### Critical Behaviour

Near the transition the localization length on the insulating side and the dc conductivity on the metallic side scale as power laws in the distance $|E - E_c|$ from the mobility edge:

$$\xi \sim |E - E_c|^{-\nu} \qquad (\text{insulating side}), \qquad
\sigma \sim |E - E_c|^{s} \qquad (\text{metallic side}).$$

For the 3D orthogonal class numerics give a critical exponent $\nu \approx 1.57$. Wegner's scaling relation ties the conductivity exponent to the localization-length exponent,

$$s = \nu \, (d - 2),$$

so in three dimensions $s = \nu$. The vanishing of $s$ at $d = 2$ is yet another statement that two dimensions is the **lower critical dimension** for the Anderson transition.

## Scaling Theory of Localization

### The Single-Parameter Scaling Hypothesis

The unifying framework is the **scaling theory of localization** of Abrahams, Anderson, Licciardello, and Ramakrishnan (the "Gang of Four", 1979). Its central postulate is that the dimensionless conductance of a block of size $L$,

$$g(L) = \frac{G(L)}{e^2/h} = \frac{\sigma L^{d-2}}{e^2/h},$$

controls everything: when blocks are combined into larger blocks, the *only* quantity needed to predict the new conductance is the old conductance. The way $g$ changes with system size is captured by a single **beta function**,

$$\beta(g) = \frac{d \ln g}{d \ln L},$$

which depends on $g$ alone (not separately on microscopic details). The sign of $\beta$ decides the fate of the system as $L \to \infty$:

- $\beta(g) > 0$: $g$ *grows* with size $\to$ flows to large $g$ $\to$ **metal** (extended states).
- $\beta(g) < 0$: $g$ *shrinks* with size $\to$ flows to $g \to 0$ $\to$ **insulator** (localized states).

### Limits Fix the Shape of $\beta$

Two limits pin down the beta function:

- **Good metal (large $g$):** Ohm's law holds, $g = \sigma L^{d-2}$, so
  $$\beta(g) = d - 2 + O(1/g).$$
  The $O(1/g)$ correction is exactly the weak-localization term computed above, and it is *negative*, pushing $\beta$ below its classical value $d-2$.

- **Strong insulator (small $g$):** conductance is set by exponentially small tunnelling, $g \sim e^{-L/\xi}$, so
  $$\beta(g) = \ln g + \text{const} \;\xrightarrow{g \to 0}\; -\infty.$$

Interpolating smoothly between these limits gives the celebrated scaling diagram:

```
   beta(g) = d ln g / d ln L
        ^
        |                              ___________  d = 3  (beta = +1 at large g)
   +1 --|--------------------_________/
        |               ____/        UNSTABLE FIXED POINT g_c
        |          ____/  *............... d = 2  (beta -> 0 from below)
    0 --|------*--/-------------------------------------------> ln g
        |     /:
        |    / :
        |   /  :                      d = 1  (always below: beta < 0)
        |  /   :
        | /    : metal  --->  if beta>0, g grows  ->  EXTENDED
        |/     : insulator -> if beta<0, g shrinks -> LOCALIZED
   -inf *      :
              g_c (mobility edge, d=3 only)
```

### What the Diagram Tells Us

Reading off the three dimensionalities:

- **$d = 1$:** $\beta < 0$ for all $g$. *Every* state is localized for *any* disorder. A one-dimensional disordered wire is always an insulator at $T = 0$, no matter how weak the disorder — confirming the exact-diagonalization result above.

- **$d = 2$:** $\beta(g) \to 0^-$ from below as $g \to \infty$. The classical term $d - 2 = 0$ vanishes, and the weak-localization correction tips $\beta$ slightly *negative* for all $g$. So $g$ always flows (logarithmically slowly) to zero: **two dimensions is the marginal, lower critical dimension** — noninteracting electrons in 2D with time-reversal symmetry are *always* localized, though the localization length can be astronomically large. (Strong spin-orbit coupling changes the symmetry class and can rescue a 2D metal — the antilocalization story.)

- **$d = 3$:** $\beta$ crosses zero at an *unstable fixed point* $g = g_c$. This sign change *is* the mobility edge. If a sample starts with $g > g_c$ it flows to a metal; if $g < g_c$ it flows to an insulator. The fixed point is a genuine **quantum critical point**, and linearizing $\beta$ about $g_c$ gives the localization-length exponent $\nu = 1/\beta'(g_c)$, recovering the critical scaling $\xi \sim |E - E_c|^{-\nu}$ of the previous section.

Single-parameter scaling thus explains, from one curve, why disorder always wins in low dimensions, why three dimensions has a true metal-insulator transition, and how the weak-localization correction and the strong-localization insulator are two ends of the same flow.

## A Note on Many-Body Localization

Everything above concerns *noninteracting* electrons (or, in the weak-localization story, interactions treated only as a source of dephasing). A profound modern question is what survives when strong interactions are present at $T > 0$. The answer is **many-body localization (MBL)** — a phase in which an *interacting, isolated* quantum system fails to reach thermal equilibrium because of disorder.

### Localization Meets Thermalization

An isolated quantum many-body system is normally expected to act as its own heat bath: under unitary evolution local subsystems reach a thermal (Gibbs) state, a statement formalized by the **eigenstate thermalization hypothesis** (ETH). MBL is the dramatic *failure* of ETH. In a strongly disordered, interacting system the highly excited many-body eigenstates remain *non-thermal*: the system retains memory of its initial conditions forever, transport is absent at *finite* energy density, and the usual machinery of statistical mechanics simply does not apply.

Whereas Anderson localization concerns single-particle wavefunctions at $T = 0$, MBL is a property of *all* (or many) eigenstates across the *entire* many-body spectrum, surviving at nonzero energy density (effectively $T > 0$) even with interactions switched on.

### Signatures of the MBL Phase

The MBL phase is characterized by a cluster of related properties:

- **Emergent integrability / local integrals of motion (l-bits).** Deep in the MBL phase the Hamiltonian can be written in terms of an extensive set of *quasi-local* conserved quantities $\tau_i^z$ ("l-bits"),
  $$H = \sum_i h_i\, \tau_i^z + \sum_{ij} J_{ij}\, \tau_i^z \tau_j^z + \sum_{ijk} K_{ijk}\, \tau_i^z \tau_j^z \tau_k^z + \cdots,$$
  with couplings $J_{ij}$ that decay exponentially with distance. These conserved l-bits are dressed versions of the bare single-particle localized states and are the structural reason equilibration fails.
- **Logarithmic entanglement growth.** Starting from a product state, the entanglement entropy grows only *logarithmically* in time, $S(t) \sim \ln t$, in stark contrast to the *ballistic* (linear-in-$t$) growth of a thermalizing system. Crucially, this slow growth happens even though transport of energy and charge is frozen — interactions dephase the l-bits without moving them.
- **Poisson level statistics.** Energy levels in the MBL phase are *uncorrelated* and obey Poisson statistics (no level repulsion), whereas a thermalizing system shows Wigner-Dyson (random-matrix) level repulsion. The crossover in the level-spacing ratio is a standard numerical diagnostic of the MBL transition.
- **Area-law excited states.** Highly excited MBL eigenstates obey an *area law* for entanglement entropy, like ground states, rather than the *volume law* expected of thermal states.

### Why It Matters and Open Questions

MBL is significant because it is one of very few known mechanisms that protects quantum coherence at finite temperature in a generic interacting system — it can shelter quantum order and even forbidden phases (such as time crystals and certain symmetry-protected topological orders) in highly excited states where thermal fluctuations would normally destroy them.

The field remains unsettled. The existence and stability of a true MBL *phase* in the thermodynamic limit, especially in dimensions above one and in the presence of rare thermal "avalanche" inclusions, is actively debated; much of the firm evidence comes from one-dimensional numerics and from cold-atom and trapped-ion experiments that realize isolated, tunable disordered chains. MBL sits at the intersection of localization physics, quantum thermodynamics, and quantum information, and is treated further among the non-equilibrium frontiers in [Graduate-Level Formalism &amp; Experiment](advanced-formalism.html).

## See Also

- [Condensed Matter Physics (Hub)](./) — crystal structure, band theory, and the Bloch states that disorder localizes.
- [Superconductivity, Quantum Hall & Topological Phases](emergent-phases.html) — quantum Hall plateaus rely on disorder-localized states between extended edge channels.
- [Graduate-Level Formalism & Experiment](advanced-formalism.html) — Green's functions, ARPES/transport probes, and the many-body-localization frontier.
- [Quantum Field Theory](../quantum-field-theory.html) — field-theoretic (nonlinear sigma model) treatments of the localization classes.
- [Computational Physics](../computational-physics/) — exact diagonalization and transfer-matrix methods used to map localization transitions.
