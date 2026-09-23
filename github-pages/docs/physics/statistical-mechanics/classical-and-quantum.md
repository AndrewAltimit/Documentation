---
layout: docs
title: "Statistical Mechanics: Classical & Quantum Statistical Mechanics"
description: "Phase space and Liouville's theorem, equipartition, the density operator, Fermi–Dirac and Bose–Einstein statistics, ideal classical and quantum gases, and the first tools for interacting systems."
permalink: /docs/physics/statistical-mechanics/classical-and-quantum.html
toc: true
toc_sticky: true
---

[Statistical Mechanics](./)

This page develops equilibrium statistical mechanics for classical and quantum systems. It starts from phase space and the classical partition function, generalizes to the density operator and quantum statistics, solves the ideal classical, Fermi and Bose gases, and introduces the two standard approaches to interacting systems: the virial expansion and mean-field theory. The ensembles and the thermodynamic role of the partition function are set out on the [hub page](./). Phase transitions and critical phenomena continue on [Phase Transitions & Graduate Formalism](phase-transitions-and-advanced.html).

## Classical statistical mechanics

### Phase space

The microstate of $N$ classical particles in three dimensions is a point in the $6N$-dimensional **phase space** $\Gamma = (\mathbf{r}_1, \dots, \mathbf{r}_N, \mathbf{p}_1, \dots, \mathbf{p}_N)$. As the system evolves under Hamilton's equations, the point traces a trajectory. Energy conservation confines it to the $(6N-1)$-dimensional surface $H(\Gamma) = E$. The volume element is

$$d\Gamma = \prod_{i=1}^{N} d^3\mathbf{r}_i \, d^3\mathbf{p}_i .$$

To count states rather than measure volumes, phase space is divided into cells of volume $h^{3N}$. The choice of Planck's constant is not arbitrary. It is the cell size for which the classical count agrees with the high-temperature limit of the quantum count, and it makes the entropy absolute rather than defined only up to a constant.

An ensemble is described by a density $\rho(\Gamma, t)$ on phase space, normalized so that $\int \rho \, d\Gamma = 1$. Averages are $\langle A \rangle = \int A(\Gamma)\,\rho(\Gamma)\, d\Gamma$.

### Liouville's theorem

Hamiltonian flow is incompressible: a region of phase space changes shape as it evolves but keeps its volume. Equivalently, the density is constant along trajectories:

$$\frac{d\rho}{dt} = \frac{\partial \rho}{\partial t} + \{\rho, H\} = 0 ,$$

where $\lbrace \cdot,\cdot \rbrace$ is the Poisson bracket.

<figure class="svg-figure">
<svg viewBox="0 0 520 190" role="img" aria-label="A compact blob of phase-space points evolves into a stretched, folded filament of the same area" style="max-width: 520px; width: 100%; color: inherit;">
<g fill="none" stroke="currentColor" stroke-width="1.5">
<line x1="30" y1="160" x2="500" y2="160"/>
<line x1="30" y1="160" x2="30" y2="15"/>
<path d="M95 60 C130 55 150 85 140 115 C130 140 90 140 75 120 C60 100 65 65 95 60 Z" fill="currentColor" fill-opacity="0.15"/>
<path d="M290 120 C320 70 360 40 400 45 C440 50 470 80 455 100 C445 112 430 95 415 85 C390 70 360 80 335 110 C320 128 300 140 290 120 Z" fill="currentColor" fill-opacity="0.15"/>
<path d="M170 95 L260 95" stroke-dasharray="5 4"/>
<path d="M252 89 L262 95 L252 101"/>
</g>
<g fill="currentColor" font-size="13" text-anchor="middle">
<text x="265" y="180">position q</text>
<text x="14" y="90" transform="rotate(-90 14 90)">momentum p</text>
<text x="108" y="40">t = 0</text>
<text x="400" y="30">t &gt; 0: same area</text>
<text x="215" y="85">Hamiltonian flow</text>
</g>
</svg>
<figcaption>Liouville's theorem: an ensemble of initial conditions is sheared and stretched by the dynamics, but its phase-space volume is conserved. In chaotic (mixing) systems the region becomes a fine filament that, at coarse resolution, appears to fill the energy surface. This is the mechanical picture of approach to equilibrium.</figcaption>
</figure>

Two consequences follow:

- **Stationary ensembles.** An equilibrium ensemble has $\partial \rho / \partial t = 0$, so $\lbrace \rho, H \rbrace = 0$. Any $\rho$ that depends on $\Gamma$ only through $H$ (and other conserved quantities) is stationary. The microcanonical $\rho \propto \delta(H - E)$ and canonical $\rho \propto e^{-\beta H}$ distributions are the standard choices.
- **Fine-grained entropy is constant.** Because volume is conserved, the Gibbs entropy $-k_B \int \rho \ln \rho \, d\Gamma$ does not change under Hamiltonian evolution. The entropy increase of the Second Law appears only after coarse-graining, or through the molecular-chaos assumption of the [Boltzmann equation](phase-transitions-and-advanced.html#kinetic-theory-and-the-boltzmann-equation).

### The classical partition function

For $N$ identical particles in the canonical ensemble:

$$Z = \frac{1}{N!\,h^{3N}} \int e^{-\beta H(\mathbf{r}, \mathbf{p})} \, d\Gamma .$$

The factor $1/N!$ is the **Gibbs correction**. Permuting identical particles does not produce a new microstate. Without it the entropy is not extensive, and mixing two samples of the same gas would appear to increase the entropy (the Gibbs paradox).

For a Hamiltonian $H = \sum_i \mathbf{p}_i^2/2m + U(\mathbf{r}_1, \dots, \mathbf{r}_N)$, the momentum integrals are Gaussian and factor out:

$$Z = \frac{Q_N}{N!\,\lambda^{3N}}, \qquad Q_N = \int e^{-\beta U} \, d^3\mathbf{r}_1 \cdots d^3\mathbf{r}_N, \qquad \lambda = \sqrt{\frac{2\pi\hbar^2}{m k_B T}} .$$

The **thermal de Broglie wavelength** $\lambda$ is the typical quantum wavelength of a particle at temperature $T$. All the difficulty of classical statistical mechanics sits in the configurational integral $Q_N$. For an ideal gas it is simply $V^N$.

### Maxwell–Boltzmann velocity distribution

Because the kinetic energy separates from the potential energy, the velocity distribution of a classical particle is independent of interactions. It is the same in a dilute gas and in a liquid:

$$f(v) \, dv = 4\pi v^2 \left(\frac{m}{2\pi k_B T}\right)^{3/2} e^{-m v^2 / 2 k_B T} \, dv .$$

| Characteristic speed | Formula | N$_2$ at 300 K |
|---|---|---|
| Most probable | $v_p = \sqrt{2 k_B T / m}$ | 422 m/s |
| Mean | $\langle v \rangle = \sqrt{8 k_B T / \pi m}$ | 476 m/s |
| Root mean square | $v_{\text{rms}} = \sqrt{3 k_B T / m}$ | 517 m/s |

### Equipartition and its failure

The **generalized equipartition theorem** states that, for any phase-space coordinates $x_i, x_j$ (under mild boundary conditions),

$$\left\langle x_i \frac{\partial H}{\partial x_j} \right\rangle = \delta_{ij}\, k_B T .$$

For a Hamiltonian containing a term $a x_i^2$, this gives $\langle a x_i^2 \rangle = \tfrac12 k_B T$: *every quadratic degree of freedom contributes $\tfrac12 k_B T$ to the mean energy and $\tfrac12 k_B$ to the heat capacity.* Applied to $\mathbf{r}\cdot\nabla U$, the same identity yields the virial theorem.

| System | Quadratic terms per particle | Classical $C_V$ per particle |
|---|---|---|
| Monatomic gas | 3 translational | $\tfrac32 k_B$ |
| Diatomic gas, rotating | 3 translational + 2 rotational | $\tfrac52 k_B$ |
| Diatomic gas, also vibrating | + 2 vibrational (kinetic and potential) | $\tfrac72 k_B$ |
| Crystalline solid | 3 kinetic + 3 potential | $3 k_B$ (Dulong–Petit law) |

Equipartition fails whenever $k_B T$ is small compared with the quantum level spacing of a mode. The mode is then "frozen out" and contributes nothing. Diatomic hydrogen shows $C_V \approx \tfrac32 k_B$ below roughly 50 K, $\tfrac52 k_B$ at room temperature, and approaches $\tfrac72 k_B$ only above a few thousand kelvin. The heat capacity of solids falls to zero as $T \to 0$, and applied to the modes of the electromagnetic field, equipartition gives the divergent "ultraviolet catastrophe". Resolving these failures was a major route to quantum theory. The [photon and phonon gases](#photons-and-phonons) below show how quantum statistics fixes them.

## Quantum statistical mechanics

### The density operator

A quantum system in thermal contact with its surroundings is not in a pure state but in a statistical mixture, described by the **density operator**

$$\hat\rho = \sum_i p_i \, |\psi_i\rangle\langle\psi_i| , \qquad \mathrm{Tr}\,\hat\rho = 1, \qquad \hat\rho \geq 0 .$$

A state is pure if and only if $\mathrm{Tr}\,\hat\rho^2 = 1$. The classical formalism carries over through a direct dictionary:

| Classical | Quantum |
|---|---|
| Phase-space density $\rho(\mathbf{r}, \mathbf{p})$ | Density operator $\hat\rho$ |
| $\dfrac{1}{N!\,h^{3N}} \displaystyle\int d\Gamma$ | $\mathrm{Tr}$ over the (anti)symmetrized Hilbert space |
| $\langle A \rangle = \int A \rho \, d\Gamma$ | $\langle A \rangle = \mathrm{Tr}(\hat\rho \hat A)$ |
| Liouville equation $\partial_t \rho = -\lbrace \rho, H \rbrace$ | von Neumann equation $i\hbar\, \partial_t \hat\rho = [\hat H, \hat\rho]$ |
| Gibbs entropy $-k_B \int \rho \ln \rho$ | von Neumann entropy $-k_B \,\mathrm{Tr}(\hat\rho \ln \hat\rho)$ |

The equilibrium ensembles become

$$\hat\rho_{\text{can}} = \frac{e^{-\beta \hat H}}{Z}, \quad Z = \mathrm{Tr}\, e^{-\beta \hat H} = \sum_n e^{-\beta E_n}; \qquad \hat\rho_{\text{gc}} = \frac{e^{-\beta(\hat H - \mu \hat N)}}{\mathcal{Z}} .$$

The canonical $\hat\rho$ is diagonal in the energy eigenbasis. Its diagonal elements are the populations $e^{-\beta E_n}/Z$, and it has no coherences between energy eigenstates. The Gibbs $1/N!$ no longer has to be inserted by hand: it is replaced by the exact (anti)symmetry of the many-particle Hilbert space. The classical limit recovers the phase-space integral when $\lambda$ is small compared with the interparticle spacing.

### Example: the quantum harmonic oscillator

With levels $E_n = \hbar\omega(n + \tfrac12)$:

$$Z = \sum_{n=0}^{\infty} e^{-\beta\hbar\omega(n + 1/2)} = \frac{1}{2\sinh(\beta\hbar\omega/2)}, \qquad U = \frac{\hbar\omega}{2} + \frac{\hbar\omega}{e^{\beta\hbar\omega} - 1} .$$

For $k_B T \gg \hbar\omega$, $U \to k_B T$, the classical equipartition value. For $k_B T \ll \hbar\omega$ the thermal part is exponentially suppressed. This freeze-out explains the vibrational heat capacity of molecules. Treating each atom of a solid as three such oscillators with one frequency (the Einstein model, 1907) gives a heat capacity that vanishes at low $T$. The Debye model, described below, corrects its low-temperature form.

### Identical particles and quantum statistics

For non-interacting identical particles, a many-body state is specified by the **occupation numbers** $\lbrace n_k \rbrace$ of the single-particle levels $\varepsilon_k$. The spin–statistics theorem restricts them:

- **Fermions** (half-integer spin; electrons, protons, neutrons, $^3$He, $^6$Li): $n_k \in \lbrace 0, 1 \rbrace$, the Pauli exclusion principle.
- **Bosons** (integer spin; photons, phonons, $^4$He, $^{87}$Rb): $n_k \in \lbrace 0, 1, 2, \dots \rbrace$.

Fixing the total $N = \sum_k n_k$ couples the levels, so the grand canonical ensemble is used. There the grand partition function factorizes over levels:

$$\mathcal{Z} = \prod_k \sum_{n_k} e^{-\beta(\varepsilon_k - \mu) n_k} = \prod_k \left(1 \pm e^{-\beta(\varepsilon_k - \mu)}\right)^{\pm 1},$$

with the upper sign for fermions and the lower sign for bosons. Differentiating $\ln \mathcal{Z}$ with respect to $\mu$ gives the mean occupations:

$$\langle n_k \rangle_{\text{FD}} = \frac{1}{e^{\beta(\varepsilon_k - \mu)} + 1}, \qquad \langle n_k \rangle_{\text{BE}} = \frac{1}{e^{\beta(\varepsilon_k - \mu)} - 1}, \qquad \langle n_k \rangle_{\text{MB}} = e^{-\beta(\varepsilon_k - \mu)} .$$

<figure class="svg-figure">
<svg viewBox="0 0 500 270" role="img" aria-label="Mean occupation versus (epsilon minus mu) over kT for Fermi-Dirac, Bose-Einstein and Maxwell-Boltzmann statistics" style="max-width: 520px; width: 100%; color: inherit;">
<g fill="none" stroke="currentColor">
<line x1="60" y1="220" x2="470" y2="220" stroke-width="1.5"/>
<line x1="60" y1="220" x2="60" y2="15" stroke-width="1.5"/>
<line x1="237.8" y1="220" x2="237.8" y2="15" stroke-width="1" stroke-dasharray="3 4" opacity="0.6"/>
<line x1="60" y1="140" x2="470" y2="140" stroke-width="1" stroke-dasharray="3 4" opacity="0.4"/>
<path stroke-width="2.5" d="M60.0 141.4 L70.0 141.8 L80.0 142.2 L90.0 142.8 L100.0 143.4 L110.0 144.3 L120.0 145.3 L130.0 146.5 L140.0 148.0 L150.0 149.7 L160.0 151.8 L170.0 154.3 L180.0 157.1 L190.0 160.4 L200.0 164.0 L210.0 167.9 L220.0 172.1 L230.0 176.5 L240.0 181.0 L250.0 185.5 L260.0 189.8 L270.0 193.9 L280.0 197.7 L290.0 201.1 L300.0 204.2 L310.0 206.8 L320.0 209.1 L330.0 211.1 L340.0 212.7 L350.0 214.1 L360.0 215.2 L380.0 216.9 L400.0 218.0 L430.0 219.0 L460.0 219.5"/>
<path stroke-width="2.5" stroke-dasharray="8 5" d="M252.9 22.4 L254.6 46.3 L256.3 65.7 L258.1 81.7 L259.8 95.2 L261.5 106.7 L263.2 116.6 L265.0 125.2 L266.7 132.7 L268.4 139.4 L271.9 150.7 L275.3 159.7 L278.8 167.2 L282.2 173.5 L285.7 178.7 L289.1 183.2 L294.3 188.8 L299.5 193.4 L304.7 197.2 L311.6 201.2 L318.5 204.5 L327.1 207.6 L337.5 210.5 L349.5 213.0 L363.3 215.0 L380.6 216.6 L401.3 217.9 L427.2 218.9 L460.0 219.5"/>
<path stroke-width="2" stroke-dasharray="2 4" d="M197.3 21.3 L201.7 39.9 L206.1 56.8 L210.5 72.1 L214.8 86.0 L219.2 98.5 L223.6 109.9 L228.0 120.3 L232.4 129.6 L236.7 138.1 L241.1 145.8 L245.5 152.7 L249.9 159.1 L254.2 164.8 L258.6 169.9 L263.0 174.6 L269.6 180.9 L276.1 186.2 L282.7 190.9 L289.3 194.9 L298.0 199.4 L306.8 203.1 L317.7 206.8 L330.9 210.1 L346.2 213.0 L363.7 215.3 L385.6 217.1 L409.7 218.3 L440.3 219.2 L460.0 219.5"/>
</g>
<g fill="currentColor" font-size="13">
<text x="52" y="224" text-anchor="end">0</text>
<text x="52" y="184" text-anchor="end">0.5</text>
<text x="52" y="144" text-anchor="end">1</text>
<text x="52" y="64" text-anchor="end">2</text>
<text x="104" y="238" text-anchor="middle">−3</text>
<text x="237.8" y="238" text-anchor="middle">0</text>
<text x="371" y="238" text-anchor="middle">3</text>
<text x="265" y="262" text-anchor="middle">(ε − μ) / k_B T</text>
<text x="18" y="120" text-anchor="middle" transform="rotate(-90 18 120)">mean occupation ⟨n⟩</text>
<text x="90" y="132">Fermi–Dirac (solid)</text>
<text x="272" y="60">Bose–Einstein (dashed)</text>
<text x="120" y="40">Maxwell–Boltzmann (dotted)</text>
</g>
</svg>
<figcaption>Mean occupation of a level as a function of $(\varepsilon - \mu)/k_B T$. Fermi–Dirac occupation never exceeds 1 and equals ½ at $\varepsilon = \mu$. Bose–Einstein occupation diverges as $\varepsilon \to \mu^+$, so $\mu$ must lie below the lowest level. All three curves coincide when $\varepsilon - \mu \gg k_B T$, the classical limit.</figcaption>
</figure>

| | Maxwell–Boltzmann | Fermi–Dirac | Bose–Einstein |
|---|---|---|---|
| Particles | Distinguishable or dilute | Half-integer spin | Integer spin |
| Allowed $n_k$ | any (with $1/N!$) | 0 or 1 | 0, 1, 2, … |
| $\langle n_k \rangle$ | $e^{-\beta(\varepsilon - \mu)}$ | $\left(e^{\beta(\varepsilon - \mu)} + 1\right)^{-1}$ | $\left(e^{\beta(\varepsilon - \mu)} - 1\right)^{-1}$ |
| Constraint on $\mu$ | none | none | $\mu < \varepsilon_0$ (lowest level) |
| Low-$T$ behaviour | freezes into ground state | filled Fermi sea up to $E_F$ | Bose–Einstein condensation |
| Occupation-number variance | $\langle n \rangle$ | $\langle n \rangle(1 - \langle n \rangle)$ | $\langle n \rangle(1 + \langle n \rangle)$ |

The last row shows how quantum statistics changes fluctuations. Fermions are anti-bunched, with variance suppressed below the Poisson value. Bosons are bunched, with variance enhanced, which is the origin of the Hanbury Brown–Twiss effect. In two dimensions, exchange statistics can interpolate between these cases (**anyons**). Anyons are realized as quasiparticles in fractional quantum Hall states.

### The classical limit

Quantum statistics matters when particles' wavepackets overlap. This is measured by the **degeneracy parameter** $n\lambda^3$, where $n = N/V$:

- $n\lambda^3 \ll 1$ (hot or dilute): $e^{\beta\mu} \approx n\lambda^3 \ll 1$, so $\varepsilon - \mu \gg k_B T$ for every level and both quantum distributions reduce to Maxwell–Boltzmann.
- $n\lambda^3 \gtrsim 1$ (cold or dense): the gas is **degenerate**, and fermions and bosons behave completely differently.

Air at room temperature has $n\lambda^3 \sim 10^{-7}$. Conduction electrons in a metal, being light and dense, have $n\lambda^3 \sim 10^{3}$–$10^{4}$ even at room temperature. Ultracold atomic gases reach $n\lambda^3 \sim 1$ at densities near $10^{13}$–$10^{14}\,\text{cm}^{-3}$ and temperatures from tens of nanokelvin to about a microkelvin.

## Ideal gases

### Classical ideal gas

With $U = 0$, $Q_N = V^N$ and

$$Z = \frac{1}{N!}\left(\frac{V}{\lambda^3}\right)^{N} .$$

Using Stirling's approximation, the thermodynamics follows: $PV = N k_B T$, $U = \tfrac32 N k_B T$, and the **Sackur–Tetrode entropy**

$$S = N k_B \left[\ln\!\left(\frac{V}{N \lambda^3}\right) + \frac52\right] .$$

This entropy is extensive only because of the $1/N!$ factor. It becomes negative, which is unphysical, exactly when $n\lambda^3 \gtrsim 1$. That is a sign that the classical treatment has broken down.

### The degenerate Fermi gas

At $T = 0$ fermions fill every single-particle state up to the **Fermi energy** $E_F$, forming a Fermi sphere of radius $k_F$ in momentum space. For spin-½ particles (two spin states per momentum state):

$$k_F = (3\pi^2 n)^{1/3}, \qquad E_F = \frac{\hbar^2 k_F^2}{2m} = \frac{\hbar^2}{2m}\left(3\pi^2 n\right)^{2/3} .$$

The ground state has nonzero energy and pressure, even at absolute zero:

$$U_0 = \frac35 N E_F, \qquad P_0 = \frac25 n E_F .$$

This **degeneracy pressure** has a purely quantum origin. At $T > 0$ only particles within about $k_B T$ of $E_F$ can be excited. The Sommerfeld expansion then gives a heat capacity linear in $T$:

$$C_V = \frac{\pi^2}{2} N k_B \frac{T}{T_F}, \qquad T_F = E_F / k_B .$$

This explains why conduction electrons contribute almost nothing to the room-temperature heat capacity of metals, a long-standing puzzle of classical theory. At low temperature the linear electronic term dominates over the phonon $T^3$ term.

| System | Typical $T_F$ | Consequence |
|---|---|---|
| Conduction electrons in Cu ($E_F \approx 7.0$ eV) | $\approx 8 \times 10^4$ K | Metals are deeply degenerate at room temperature |
| White dwarf electrons | $\sim 10^9$ K | Degeneracy pressure supports the star, up to the Chandrasekhar limit of about $1.4\,M_\odot$ |
| Neutron star neutrons | $\sim 10^{11}$–$10^{12}$ K | Neutron degeneracy plus nuclear forces support the star |
| Trapped ultracold $^{40}$K or $^6$Li | $\sim 0.1$–$1\ \mu$K | Tunable degenerate Fermi gases, first reached in 1999 (DeMarco and Jin) |

### The Bose gas and Bose–Einstein condensation

For bosons the chemical potential must stay below the lowest level ($\mu < 0$ for free particles). Sum the occupations of the excited states with the three-dimensional density of states $g(\varepsilon) \propto \sqrt{\varepsilon}$. At $\mu = 0$ this sum reaches a maximum:

$$n_{\text{exc}}^{\max} = \frac{\zeta(3/2)}{\lambda^3}, \qquad \zeta(3/2) \approx 2.612 .$$

When $n\lambda^3$ exceeds $\zeta(3/2)$, the excited states cannot hold all the particles. The excess occupies the single ground state with macroscopic occupation. This is **Bose–Einstein condensation**, which sets in below

$$T_c = \frac{2\pi\hbar^2}{m k_B}\left(\frac{n}{\zeta(3/2)}\right)^{2/3}, \qquad \frac{N_0}{N} = 1 - \left(\frac{T}{T_c}\right)^{3/2} \quad (T < T_c) .$$

Condensation happens in momentum space, not position space, and it occurs without interactions: it is a pure statistics effect. In a harmonic trap of mean frequency $\bar\omega$ the density of states changes, giving $k_B T_c \approx 0.94\,\hbar\bar\omega N^{1/3}$ and a condensate fraction $1 - (T/T_c)^3$.

Experimental landmarks:

- **1938** — Superfluidity of liquid $^4$He below the λ point at 2.17 K, interpreted by London as a (strongly interacting) Bose condensate. Only about 10% of the atoms occupy the zero-momentum state.
- **1995** — BEC in dilute alkali gases: $^{87}$Rb at JILA (Cornell and Wieman) and $^{23}$Na at MIT (Ketterle), recognized by the 2001 Nobel Prize.
- **2006–2018** — Condensates of exciton-polaritons in semiconductor microcavities (2006), of photons in dye-filled microcavities (2010), and in microgravity aboard NASA's Cold Atom Laboratory on the ISS (2018).
- **2024** — First BEC of dipolar *molecules* (NaCs), reaching about 60% condensate fraction near 6 nK, using microwave shielding to suppress collisional losses (Bigagli *et al.*, *Nature* 2024).

### Photons and phonons

Photons and phonons are bosons whose number is not conserved, so $\mu = 0$. For blackbody radiation, counting two polarizations per mode gives the **Planck spectrum** and the Stefan–Boltzmann law:

$$u(\omega)\, d\omega = \frac{\hbar \omega^3}{\pi^2 c^3} \frac{d\omega}{e^{\beta\hbar\omega} - 1}, \qquad u = \frac{U}{V} = \frac{\pi^2 k_B^4}{15 \hbar^3 c^3}\, T^4 .$$

At low frequency this reduces to the classical Rayleigh–Jeans form $u \propto \omega^2 k_B T$. At high frequency it is cut off exponentially, which removes the ultraviolet catastrophe.

Lattice vibrations of a solid are treated the same way. The **Debye model** uses acoustic phonons with a linear dispersion up to a cutoff frequency $\omega_D$, defining the Debye temperature $\Theta_D = \hbar\omega_D/k_B$. It gives the Dulong–Petit value $3Nk_B$ at high $T$ and the Debye $T^3$ law at low $T$:

$$C_V \approx \frac{12\pi^4}{5} N k_B \left(\frac{T}{\Theta_D}\right)^3 \qquad (T \ll \Theta_D) .$$

### Summary of ideal gases

| | Classical gas | Fermi gas ($T \ll T_F$) | Bose gas ($T < T_c$) | Photon gas |
|---|---|---|---|---|
| Equation of state | $P = n k_B T$ | $P \approx \tfrac25 n E_F$, nearly $T$-independent | $P \propto T^{5/2}$, independent of $n$ | $P = u/3 \propto T^4$ |
| Heat capacity | $\tfrac32 N k_B$ | $\propto T$ | $\propto T^{3/2}$, cusp at $T_c$ | $\propto V T^3$ |
| Chemical potential | $k_B T \ln(n\lambda^3)$ | $\approx E_F$ | $= 0$ | $= 0$ |

## Interacting systems

Interactions couple the particles, so the partition function no longer factorizes, and exact solutions are rare. There are two standard starting points: expand in the density (the virial expansion, for dilute gases), or replace each particle's environment with an average field (mean-field theory, for ordered phases).

### Virial and cluster expansions

The equation of state of a real gas is expanded in powers of the density:

$$\frac{P}{n k_B T} = 1 + B_2(T)\, n + B_3(T)\, n^2 + \cdots$$

For a pair potential $u(r)$, write the Boltzmann factor as $e^{-\beta u} = 1 + f(r)$. The **Mayer function** $f(r) = e^{-\beta u(r)} - 1$ is small except where particles interact. The second virial coefficient is then

$$B_2(T) = -\frac12 \int f(r)\, d^3 r = -2\pi \int_0^\infty \left(e^{-\beta u(r)} - 1\right) r^2 \, dr .$$

Higher coefficients are sums of connected Mayer cluster diagrams. Two examples:

- **Hard spheres** of diameter $\sigma$: $B_2 = \tfrac{2\pi}{3}\sigma^3$, which is positive and independent of $T$.
- **Hard core plus weak attraction:** to leading order in $\beta$, $B_2 \approx b - a/(k_B T)$. This reproduces the second virial coefficient of the van der Waals equation $\left(P + a n^2\right)(1 - n b) = n k_B T$. $B_2$ changes sign at the Boyle temperature $T_B = a/(b k_B)$.

The virial series converges only at low density. It cannot describe condensation, which requires a non-perturbative treatment.

### Mean-field theory

The **Ising model** is the standard lattice model of an interacting system:

$$H = -J \sum_{\langle ij \rangle} s_i s_j - h \sum_i s_i, \qquad s_i = \pm 1 ,$$

with a sum over nearest-neighbour pairs on a lattice of coordination number $z$. Mean-field (Weiss) theory replaces each neighbour $s_j$ with its average $m = \langle s \rangle$. Each spin then sees an effective field $h + zJm$, and self-consistency requires

$$m = \tanh\!\big(\beta (zJm + h)\big) .$$

<figure class="svg-figure">
<svg viewBox="0 0 330 260" role="img" aria-label="Graphical solution of the mean-field equation: the line y = m intersects tanh only at the origin above Tc, and at three points below Tc" style="max-width: 360px; width: 100%; color: inherit;">
<g fill="none" stroke="currentColor">
<line x1="30" y1="130" x2="295" y2="130" stroke-width="1.2"/>
<line x1="160" y1="245" x2="160" y2="15" stroke-width="1.2"/>
<path stroke-width="1.5" stroke-dasharray="3 3" d="M56 234 L264 26"/>
<path stroke-width="2.2" stroke-dasharray="8 5" d="M40.0 190.9 L60.0 184.6 L80.0 176.6 L100.0 167.0 L120.0 155.7 L140.0 143.2 L160.0 130.0 L180.0 116.8 L200.0 104.3 L220.0 93.0 L240.0 83.4 L260.0 75.4 L280.0 69.1"/>
<path stroke-width="2.5" d="M40.0 208.9 L60.0 207.6 L80.0 204.5 L90.0 201.8 L100.0 197.9 L110.0 192.3 L120.0 184.6 L126.0 178.8 L132.0 172.0 L138.0 164.3 L144.0 155.7 L150.0 146.4 L156.0 136.7 L160.0 130.0 L164.0 123.3 L170.0 113.6 L176.0 104.3 L182.0 95.7 L188.0 88.0 L194.0 81.2 L200.0 75.4 L210.0 67.7 L220.0 62.1 L230.0 58.2 L240.0 55.5 L260.0 52.4 L280.0 51.1"/>
</g>
<g fill="currentColor">
<circle cx="232.6" cy="57.4" r="4"/>
<circle cx="87.4" cy="202.6" r="4"/>
<circle cx="160" cy="130" r="4" fill-opacity="0.35"/>
</g>
<g fill="currentColor" font-size="12">
<text x="298" y="134">m</text>
<text x="252" y="20">y = m</text>
<text x="226" y="44">T &lt; T_c (solid)</text>
<text x="222" y="98">T &gt; T_c (dashed)</text>
<text x="168" y="148">m = 0</text>
<text x="240" y="72">+m₀</text>
<text x="58" y="222">−m₀</text>
</g>
</svg>
<figcaption>Graphical solution of $m = \tanh(zJm/k_BT)$ at $h = 0$. Above $T_c = zJ/k_B$ the curve's slope at the origin is below 1 and $m = 0$ is the only solution. Below $T_c$, two stable solutions $\pm m_0$ appear (filled dots). The $m = 0$ solution becomes unstable: this is spontaneous symmetry breaking.</figcaption>
</figure>

A nonzero solution exists when the slope of the right-hand side at the origin, $\beta z J$, exceeds 1. This gives $k_B T_c = zJ$. Expanding near $T_c$ yields the mean-field critical behaviour:

$$m \simeq \sqrt{3}\left(1 - \frac{T}{T_c}\right)^{1/2}, \qquad \chi = \left.\frac{\partial m}{\partial h}\right|_{h=0} = \frac{1}{k_B(T - T_c)} \quad (T > T_c, \text{ Curie–Weiss law}) .$$

Mean-field theory predicts that a transition exists but is quantitatively unreliable, especially in low dimensions, because it neglects fluctuations:

| Lattice | $z$ | Mean-field $k_B T_c / J$ | Exact or numerical $k_B T_c / J$ |
|---|---|---|---|
| 1D chain | 2 | 2 | 0 (no transition) |
| 2D square | 4 | 4 | $2/\ln(1+\sqrt2) \approx 2.269$ (Onsager) |
| 3D simple cubic | 6 | 6 | $\approx 4.512$ (Monte Carlo) |

It also gets the critical exponents wrong below four dimensions: for example, $\beta = 1/2$ instead of $1/8$ in 2D or about $0.326$ in 3D. Why, and how the renormalization group fixes this, is the subject of [Landau theory and critical phenomena](phase-transitions-and-advanced.html#landau-theory).

### Correlation functions and the fluctuation–response relation

The **connected two-point correlation function** measures how much a fluctuation at one site influences another:

$$G(\mathbf{r}_i - \mathbf{r}_j) = \langle s_i s_j \rangle - \langle s_i \rangle\langle s_j \rangle .$$

It is linked exactly to a thermodynamic response. Differentiating $\ln Z$ twice with respect to $h$ gives the susceptibility per spin

$$\chi = \beta \sum_{\mathbf{r}} G(\mathbf{r}) ,$$

so a divergent susceptibility requires correlations that extend over a divergent range. Away from criticality, correlations decay exponentially with a **correlation length** $\xi$. In the Ornstein–Zernike approximation the structure factor is Lorentzian, $S(\mathbf{k}) \propto 1/(k^2 + \xi^{-2})$. At the critical point $\xi \to \infty$ and $G(r) \sim r^{-(d-2+\eta)}$. The strong long-wavelength density fluctuations near a liquid–gas critical point scatter light strongly: this is **critical opalescence**.

## See also

- [Statistical Mechanics Hub](./) — microstates, entropy, the ensembles and the partition function.
- [Phase Transitions & Graduate Formalism](phase-transitions-and-advanced.html) — Landau theory, critical phenomena, the renormalization group, non-equilibrium and quantum dynamics.
- [Thermodynamics](../thermodynamics.html) — the macroscopic laws these results reproduce.
- [Quantum Mechanics](../quantum-mechanics/) — identical particles and the foundation of quantum statistics.
- [Condensed Matter Physics](../condensed-matter/) — Fermi liquids, phonons, magnetism and superconductivity.
- [Classical Mechanics](../classical-mechanics/) — Hamiltonian dynamics and phase space.

**Next:** [Phase Transitions & Graduate Formalism](phase-transitions-and-advanced.html) →
