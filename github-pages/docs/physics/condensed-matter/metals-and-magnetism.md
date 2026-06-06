---
layout: docs
title: "Condensed Matter: Metals & Magnetism"
permalink: /docs/physics/condensed-matter/metals-and-magnetism.html
toc: true
toc_sticky: true
hide_title: true
---

[Condensed Matter Physics](./) &raquo; Metals &amp; Magnetism

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Metals &amp; Magnetism</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">From the electron sea to spontaneous magnetic order</p>
</div>

<div class="intro-card">
  <p class="lead-text">Two of the oldest puzzles in solid-state physics — why metals conduct and why iron is magnetic — turn out to share a common moral: a metal is not a box of independent electrons, and a magnet is not a box of independent spins. Both are <strong>collective</strong> quantum states. This page follows the historical refinement of the metallic state (Drude → Sommerfeld → Landau's Fermi liquid), introduces the Fermi surface and electronic screening, and then builds magnetism from the exchange interaction up through the Heisenberg and Ising models, spin waves, and the mean-field theory of magnetic phase transitions.</p>
</div>

<div class="principle-card">
  <h4>The big idea: the Pauli principle does the heavy lifting</h4>
  <p>Almost everything surprising on this page traces back to fermion statistics. Electrons cannot share a quantum state, so at low temperature they stack up to the Fermi energy, leaving only a thin shell near the Fermi surface thermally active — that single fact rescues the metallic heat capacity, fixes the spin susceptibility, and (through the exchange energy that the antisymmetry of the wavefunction demands) makes magnetism possible in the first place. Magnetism is the Pauli principle made macroscopic: the spins align not because they are tiny bar magnets feeling each other's fields — that effect is a thousand times too weak — but because aligning them lets the electrons avoid each other and lower their electrostatic energy.</p>
</div>

## Metals and Fermi Liquids

Our picture of a metal evolved in three stages, each fixing a failure of the last: a classical gas of electrons (Drude), then a quantum gas obeying the Pauli principle (Sommerfeld), then an interacting fluid of "dressed" electrons (Landau). Each step kept the successes of the previous one while resolving a glaring discrepancy with experiment.

### Drude Model

The earliest picture treats conduction electrons as a classical gas of point particles bouncing off the ions, with a mean time $\tau$ between collisions. The equation of motion for the average drift momentum $\mathbf{p}$ under fields $\mathbf{E}$ and $\mathbf{B}$, with friction represented by the relaxation time, is

$$\frac{d\mathbf{p}}{dt} = -e\left(\mathbf{E} + \frac{\mathbf{p}}{m}\times\mathbf{B}\right) - \frac{\mathbf{p}}{\tau}.$$

In steady state with $\mathbf{B}=0$ this gives Ohm's law and the DC conductivity, and including a magnetic field gives the Hall coefficient:

$$\sigma = \frac{ne^2\tau}{m}, \qquad R_H = -\frac{1}{ne}.$$

The Hall coefficient $R_H$ even gives the carrier density and sign correctly for simple metals. Two more successes follow almost for free:

- **AC response.** Solving for an oscillating field $\mathbf{E}(t) = \mathbf{E}_0 e^{-i\omega t}$ gives the frequency-dependent conductivity $\sigma(\omega) = \sigma_0 / (1 - i\omega\tau)$, which feeds directly into the dielectric function and predicts the **plasma frequency** $\omega_p = \sqrt{ne^2/\epsilon_0 m}$ below which a metal reflects light — the reason metals are shiny.
- **Wiedemann–Franz law.** The ratio of thermal to electrical conductivity is predicted to be proportional to temperature, $\kappa/\sigma T = L$, a constant. The numerical value comes out roughly right.

But Drude fails badly on two counts. It predicts a huge electronic contribution to the heat capacity — each electron should carry $\tfrac{3}{2}k_B$ of thermal energy, giving a constant $C_V$ that experiments flatly contradict. And it gets the **Lorenz number** $L$ in Wiedemann–Franz off by a factor of order two, for compensating wrong reasons (a too-large heat capacity multiplied by a too-small mean velocity). Both failures have the same root: electrons are not a classical gas.

### Sommerfeld Model

The fix is quantum statistics. Electrons are fermions obeying the **Fermi–Dirac distribution**

$$f(E) = \frac{1}{e^{(E-\mu)/k_BT} + 1},$$

so at $T=0$ they fill every state up to the **Fermi energy** $E_F$ and none above it. For a free-electron gas of density $n$ the filled states form a sphere in $\mathbf{k}$-space — the **Fermi sphere** — of radius $k_F = (3\pi^2 n)^{1/3}$, and

$$E_F = \frac{\hbar^2 k_F^2}{2m} = \frac{\hbar^2}{2m}(3\pi^2 n)^{2/3}.$$

For a typical metal $E_F \sim 5$ eV, corresponding to a **Fermi temperature** $T_F = E_F/k_B \sim 50{,}000$ K — far above room temperature, which is why the electron gas is always strongly *degenerate* and the classical picture never applies.

Only the thin shell of electrons within $\sim k_B T$ of $E_F$ can be thermally excited — the rest are Pauli-blocked. This single observation fixes the heat-capacity disaster. The fraction of thermally active electrons is $\sim k_B T / E_F$, each carrying $\sim k_B T$ of energy, so the thermal energy scales as $N (k_B T)^2/E_F$ and its derivative — the heat capacity — is **linear** in temperature, exactly as measured:

$$C_V = \gamma T, \qquad \gamma = \frac{\pi^2 k_B^2 g(E_F)}{3}.$$

The coefficient $\gamma$ (the **Sommerfeld coefficient**) is directly proportional to the density of states at the Fermi level, $g(E_F)$, making low-temperature heat-capacity measurements a clean experimental probe of $g(E_F)$. The same factor $g(E_F)$ governs the Pauli spin susceptibility below, so the two quantities can be cross-checked.

<div class="insight-card">
  <h4>Why the Sommerfeld correction is small</h4>
  <p>At room temperature $k_BT/E_F \approx 300/50{,}000 \approx 0.6\%$. Only that tiny fraction of the conduction electrons is "awake"; the rest are frozen deep inside the Fermi sea, unable to change state because every nearby level is already occupied. Drude's mistake was to let all $N$ electrons absorb heat classically — overcounting the thermally active population by a factor of $E_F/k_BT \sim 100$.</p>
</div>

### The Fermi Surface

The boundary in $\mathbf{k}$-space between occupied and empty states at $T=0$ is the **Fermi surface**. For free electrons it is a sphere, but in a real crystal the periodic potential distorts it: bands bend, the surface can neck, pinch through Brillouin-zone boundaries, and split into multiple sheets (electron pockets and hole pockets). Only electrons *at* the Fermi surface participate in low-energy phenomena — conduction, heat capacity, magnetism, screening — so the geometry of this surface, not the bulk of the filled states beneath it, controls a metal's behavior.

Two facts make the Fermi surface central:

- **It carries the current.** An applied field $\mathbf{E}$ shifts the whole Fermi sphere by $\delta\mathbf{k} = -e\mathbf{E}\tau/\hbar$. The interior states cancel pairwise; only the *displaced* surface produces a net current. This is why conductivity depends on the Fermi velocity $v_F = \hbar k_F/m$ and the surface area, not on the total electron count in any simple way.
- **It is directly measurable.** In the **de Haas–van Alphen effect**, the magnetization of a metal oscillates periodically in $1/B$ as Landau levels sweep through the Fermi energy. The oscillation period gives the extremal cross-sectional area $A$ of the Fermi surface perpendicular to $\mathbf{B}$ via the Onsager relation,

$$\Delta\!\left(\frac{1}{B}\right) = \frac{2\pi e}{\hbar A}.$$

By rotating the crystal and tracking how $A$ changes, experimentalists reconstruct the full three-dimensional Fermi surface — one of the great triumphs of mid-20th-century solid-state physics.

### Screening

Drop a test charge into a metal and the mobile electron sea rushes to surround it, neutralizing its field beyond a short distance. This **screening** is why a metal's interior is field-free and why electron–electron and electron–impurity interactions are short-ranged despite the bare Coulomb force being long-ranged.

In the **Thomas–Fermi** approximation, a slowly varying potential $\phi(\mathbf{r})$ locally shifts the bands, the electrons redistribute to flatten the electrochemical potential, and the induced charge density is $\delta n = -e^2 g(E_F)\phi$. Inserting this into Poisson's equation turns the long-range Coulomb potential of a point charge into a **screened (Yukawa) potential**:

$$\phi(r) = \frac{Q}{4\pi\epsilon_0 r}\, e^{-r/\lambda_{TF}}, \qquad \frac{1}{\lambda_{TF}^2} = \frac{e^2 g(E_F)}{\epsilon_0}.$$

The **Thomas–Fermi screening length** $\lambda_{TF}$ is typically less than an ångström in a good metal — interactions are screened on the scale of the interatomic spacing. The fuller quantum treatment (the **Lindhard dielectric function** $\epsilon(\mathbf{q},\omega)$) reproduces Thomas–Fermi at long wavelength but adds a subtlety: the sharp edge of the Fermi sphere produces a weak non-analyticity at $q = 2k_F$, which in real space gives slowly decaying **Friedel oscillations** $\propto \cos(2k_F r)/r^3$ in the screened charge density. These oscillations are the fingerprint of the Fermi surface in the response function, and they mediate the RKKY interaction between magnetic impurities discussed below.

### Fermi Liquid Theory

Real electrons repel each other strongly, so why does the free-electron picture work at all? Landau's answer: interactions "dress" each electron into a **quasiparticle** — an electron carrying a cloud of disturbance in the surrounding sea — that behaves like a free particle with a renormalized effective mass $m^*$. The crucial claim, **adiabatic continuity**, is that if you imagine slowly turning on the interactions, each free-electron state evolves smoothly into a quasiparticle state without level crossings, so the low-energy excitations remain in one-to-one correspondence with those of the free gas. That is why Sommerfeld's results survive — they just acquire renormalized coefficients ($m \to m^*$).

Residual interactions are encoded in the Landau parameters $f_{\mathbf{k}\mathbf{k}'}^{\sigma\sigma'}$, which give the change in energy when one quasiparticle is added in the presence of others:

$$\delta E = \sum_{\mathbf{k}\sigma} \epsilon_{\mathbf{k}} n_{\mathbf{k}\sigma} + \frac{1}{2V}\sum_{\mathbf{k}\mathbf{k}'\sigma\sigma'} f_{\mathbf{k}\mathbf{k}'}^{\sigma\sigma'} n_{\mathbf{k}\sigma} n_{\mathbf{k}'\sigma'}.$$

Expanding the angular dependence of $f$ in Legendre polynomials yields the dimensionless **Landau parameters** $F_\ell^s$ (symmetric, charge channel) and $F_\ell^a$ (antisymmetric, spin channel). These compress all the residual interactions into a handful of numbers that renormalize measurable quantities:

| Quantity | Free gas | Fermi liquid |
|----------|----------|--------------|
| Effective mass | $m$ | $m^* = m(1 + F_1^s/3)$ |
| Heat capacity $\gamma$ | $\propto g(E_F)$ | $\propto m^*$ (enhanced) |
| Compressibility | $\propto g(E_F)$ | $\div (1 + F_0^s)$ |
| Spin susceptibility | $\chi_{Pauli}$ | $\div (1 + F_0^a)$ |

The theory also predicts a distinctive **collective mode** — zero sound, an oscillation of the Fermi surface shape that propagates even in the collisionless regime where ordinary (first) sound cannot. A divergence in one of these renormalizations signals an instability: $1 + F_0^a \to 0$ heralds the **Stoner ferromagnetic instability** (the spin susceptibility diverges and the metal magnetizes), tying Fermi-liquid theory directly to the magnetism discussed next.

<div class="insight-card">
  <h4>When the Fermi liquid breaks down</h4>
  <p>The whole edifice rests on quasiparticles being long-lived: phase space restricts the electron–electron scattering rate to $1/\tau \sim (E - E_F)^2$, so excitations near $E_F$ are arbitrarily sharp. This fails in one dimension (replaced by the <em>Luttinger liquid</em>, where excitations fractionalize into separate spin and charge waves) and apparently fails in the "strange metal" phase of high-$T_c$ cuprates, where resistivity rises <em>linearly</em> in $T$ rather than quadratically — one of the central open problems in condensed matter.</p>
</div>

## Magnetism

Magnetism is a purely quantum, purely collective effect — a classical system in thermal equilibrium cannot be magnetic at all (the **Bohr–van Leeuwen theorem**, which follows because the magnetic field enters the classical partition function only through a shift of integration variables that the momentum integral undoes). The phenomena below differ in *how the atomic moments respond*: independently and weakly (paramagnetism), or by locking into collective order through exchange interactions (ferro-, antiferro-, and ferrimagnetism).

| Order | Moment arrangement | Net moment | Hallmark temperature |
|-------|--------------------|------------|----------------------|
| Paramagnet | random, align weakly with field | zero at $H=0$ | none |
| Ferromagnet | parallel | large, spontaneous | Curie $T_C$ |
| Antiferromagnet | alternating up/down | zero (cancels) | Néel $T_N$ |
| Ferrimagnet | alternating but unequal | nonzero (partial cancel) | Curie $T_C$ |

### The Exchange Interaction

The force that aligns spins is *not* the magnetic dipole interaction between moments — that is some $10^{-3}$ eV and would order magnets only below $\sim 1$ K, whereas iron stays magnetic past 1000 K. The real driver is **exchange**: a purely electrostatic effect dressed in quantum clothing. Because the total electronic wavefunction must be antisymmetric, the spin configuration controls the spatial configuration. Two electrons in a spin *triplet* (parallel spins) have an antisymmetric spatial wavefunction that keeps them apart, lowering their Coulomb repulsion; a spin *singlet* (antiparallel) does the opposite. The energy difference between the two arrangements is the **exchange integral** $J$.

For two localized spins this is captured exactly by the **Heisenberg form**,

$$\hat{H}_{ij} = -2J\,\hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j,$$

with the sign convention that $J > 0$ favors parallel spins (ferromagnetic) and $J < 0$ favors antiparallel (antiferromagnetic). Several physical mechanisms generate an effective $J$:

- **Direct exchange** between overlapping orbitals on neighboring atoms.
- **Superexchange** through an intervening non-magnetic ion (e.g. the O$^{2-}$ bridging two Mn$^{2+}$ ions in MnO), usually antiferromagnetic — described by the Goodenough–Kanamori rules.
- **Itinerant/Stoner exchange** in metals, where it acts on the conduction electrons themselves and can split the spin-up and spin-down bands.
- **RKKY interaction**, where localized moments couple indirectly through the conduction sea; the coupling oscillates in sign with distance as $\cos(2k_F r)/r^3$ — the same Friedel oscillation seen in screening.

### The Heisenberg and Ising Models

Summing the pairwise exchange over a lattice gives the **Heisenberg model**, the workhorse Hamiltonian of localized magnetism:

$$\hat{H} = -\sum_{\langle ij\rangle} J_{ij}\,\hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j - g\mu_B \mu_0 H \sum_i \hat{S}_i^z,$$

where $\langle ij\rangle$ runs over bonds and the second term is the Zeeman coupling to an applied field. When strong spin–orbit or crystal-field anisotropy pins the spins to a single axis, the transverse components freeze out and the model reduces to the **Ising model**, in which each spin is a classical $\pm 1$ variable:

$$\hat{H} = -\sum_{\langle ij\rangle} J_{ij}\, s_i s_j - h\sum_i s_i, \qquad s_i = \pm 1.$$

The Ising model is the most-studied model in statistical physics. Its lessons are foundational:

- **1D:** Onsager-soluble; no order at any finite temperature, because a single domain wall costs only finite energy and entropy always wins (a defining instance of the Mermin–Wagner intuition).
- **2D:** Onsager's exact solution (1944) gives a genuine phase transition at $k_B T_c = 2J/\ln(1+\sqrt{2})$, with the magnetization, susceptibility, and heat capacity showing power-law critical behavior.
- **3D:** No closed-form solution; critical exponents come from renormalization-group and Monte Carlo methods. The Ising universality class describes not only magnets but the liquid–gas critical point and binary-alloy ordering.

### Paramagnetism

Independent atomic moments align only weakly with an applied field, and thermal agitation fights that alignment — so the susceptibility falls off as $1/T$ (the **Curie law**):

$$\chi = \frac{C}{T}, \qquad C = \frac{N\mu_0\mu_B^2 g^2 J(J+1)}{3k_B}.$$

This comes from expanding the **Brillouin function** $B_J(x)$ (the exact magnetization of a free spin-$J$ moment in a field) for small argument. In a metal the story is different: the Pauli principle blocks most spins from flipping, so only electrons near the Fermi surface contribute, giving a temperature-*independent* **Pauli paramagnetism**,

$$\chi = \mu_0\mu_B^2 g(E_F).$$

Once interactions are switched on, this bare susceptibility is enhanced by the **Stoner factor** $1/(1 - U g(E_F))$, where $U$ is the on-site repulsion. When $U g(E_F) \to 1$ the susceptibility diverges and the metal spontaneously magnetizes — the **Stoner criterion** for itinerant ferromagnetism, the band-electron counterpart of the Fermi-liquid instability $F_0^a \to -1$ noted above.

### Ferromagnetism and Mean-Field Theory

When the exchange interaction is strong enough, moments align *spontaneously* even with no applied field. The Heisenberg Hamiltonian is intractable in general, so **Weiss mean-field theory** replaces the fluctuating neighbor spins felt by a given spin with their average, packaged as an effective internal field $\lambda M$ proportional to the magnetization itself. Each spin then behaves like a paramagnet in the total field $H + \lambda M$, giving a *self-consistent* equation:

$$M = Ng\mu_B J\, B_J\!\left(\frac{g\mu_B J(H + \lambda M)}{k_B T}\right).$$

Below the **Curie temperature**

$$T_C = \frac{g^2\mu_B^2 J(J+1)\,\lambda}{3k_B}$$

this equation has a nonzero solution at $H = 0$ — spontaneous magnetization, the order parameter switching on continuously from zero. Above $T_C$ thermal disorder wins and the material reverts to a paramagnet, now obeying the **Curie–Weiss law** $\chi = C/(T - T_C)$, whose shifted denominator (compared with the bare Curie law) is the experimental signature of ferromagnetic correlations.

Near $T_C$ the mean-field theory predicts universal **critical exponents** — $M \sim (T_C - T)^{1/2}$, $\chi \sim |T - T_C|^{-1}$, $C$ a finite jump — which is the magnetic instance of Landau theory. These mean-field exponents are correct above the upper critical dimension ($d \ge 4$) but quantitatively wrong in 2D and 3D, where fluctuations the mean field ignores become decisive; the correct exponents come from the renormalization group and define the universality classes shared with the Ising and Heisenberg fixed points.

### Antiferromagnetism

Here the exchange favors *anti*-alignment ($J < 0$): neighboring moments point opposite ways, so the net magnetization cancels even though the system is fully ordered. The structure is best described by splitting the lattice into two interpenetrating **sublattices** A and B with opposite **staggered magnetization** $M_A = -M_B$, the true order parameter. Order sets in below the **Néel temperature** $T_N$. Applying the same mean-field machinery to the two-sublattice model gives, above $T_N$,

$$\chi = \frac{2C}{T + T_N},$$

with the telltale $T + T_N$ in the denominator — the opposite sign to a ferromagnet's $T - T_C$. Below $T_N$ the susceptibility becomes anisotropic: it stays finite (and roughly constant) for fields perpendicular to the sublattice axis but drops toward zero parallel to it, since aligned spins resist a field that would have to flip half of them. Antiferromagnetic order is invisible to a bulk magnetometer but shows up directly in **neutron diffraction**, which sees magnetic Bragg peaks at the staggered wavevector — the technique with which Shull first confirmed Néel order in MnO.

### Ferrimagnetism

Ferrimagnets are antiferromagnets whose two sublattices carry *unequal* moments (different ions or different site multiplicities), so the cancellation is incomplete and a net spontaneous magnetization survives. Magnetite (Fe$_3$O$_4$) — the original lodestone — is the canonical example: Fe$^{3+}$ and Fe$^{2+}$ ions on inequivalent spinel sites couple antiparallel through superexchange but do not cancel. Because the two sublattice magnetizations fall off differently with temperature, a ferrimagnet can even pass through a **compensation point** where they momentarily cancel and the net moment vanishes before reappearing — a feature exploited in some magneto-optical recording media.

### Spin Waves and Magnons

Just as a crystal's lowest-energy excitations are quantized lattice vibrations (phonons), an ordered magnet's lowest excitations are quantized waves of precessing, slightly tilted spins — **magnons**. Rather than flipping a single spin (which costs the full exchange energy), the system lowers the cost by spreading the deviation coherently over a long-wavelength wave. The **Holstein–Primakoff transformation** maps spin operators onto bosonic creation/annihilation operators and, to leading order in $1/S$, diagonalizes the Heisenberg ferromagnet into independent magnon modes with dispersion

$$\hbar\omega_{\mathbf{k}} = 2JS\,(1 - \cos(ka)) \approx 2JSa^2 k^2 \quad (ka \ll 1).$$

The long-wavelength dispersion is **quadratic**, $\omega \sim k^2$ — in sharp contrast to the *linear* dispersion of acoustic phonons (and of antiferromagnetic magnons, which are linear, $\omega \sim k$). This difference is observable: a quadratic magnon spectrum produces the **Bloch $T^{3/2}$ law** for the temperature decay of the spontaneous magnetization,

$$M(T) = M(0)\left[1 - \left(\frac{T}{T^*}\right)^{3/2}\right],$$

and a corresponding $T^{3/2}$ magnon heat capacity, both confirmed by experiment. Magnon dispersions are mapped directly by **inelastic neutron scattering**, making spin waves one of the most precisely tested predictions in magnetism.

<div class="insight-card">
  <h4>Why dimension and symmetry decide whether order survives</h4>
  <p>The <strong>Mermin–Wagner theorem</strong> states that a continuous symmetry cannot be spontaneously broken at finite temperature in one or two dimensions: the soft, long-wavelength magnons (Goldstone modes) are so easy to excite that they destroy the order. This is exactly why the 2D Heisenberg model has no finite-temperature order, while the 2D <em>Ising</em> model — whose discrete symmetry has no Goldstone mode — does. The gapless quadratic magnon is the loophole and the executioner at once.</p>
</div>

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Three models, one metal</h4>
    <p>Drude gives Ohm's law and the Hall effect; Sommerfeld fixes the heat capacity with Fermi statistics; Landau rescues the picture for interacting electrons via quasiparticles.</p>
  </div>
  <div class="takeaway-card">
    <h4>Only the Fermi surface matters</h4>
    <p>Conduction, heat capacity, magnetism, and screening are all controlled by the thin shell of states near $E_F$, not by the filled sea beneath.</p>
  </div>
  <div class="takeaway-card">
    <h4>Screening tames the Coulomb force</h4>
    <p>The mobile electron sea converts long-range $1/r$ interactions into short-range Yukawa potentials, with Friedel oscillations betraying the Fermi surface.</p>
  </div>
  <div class="takeaway-card">
    <h4>Exchange, not dipoles, makes magnets</h4>
    <p>Magnetic order is an electrostatic effect enforced by the Pauli principle — orders of magnitude stronger than dipole–dipole coupling.</p>
  </div>
  <div class="takeaway-card">
    <h4>Mean-field theory predicts transitions</h4>
    <p>The Weiss field gives $T_C$, $T_N$, the Curie–Weiss law, and mean-field critical exponents — corrected by the renormalization group near the transition.</p>
  </div>
  <div class="takeaway-card">
    <h4>Magnons are the magnetic phonons</h4>
    <p>Quantized spin waves carry the low-energy excitations of an ordered magnet, giving the Bloch $T^{3/2}$ law and direct neutron-scattering signatures.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="./">Condensed Matter Physics (Hub)</a> — crystal structure, band theory, and the electronic properties that underpin metals and magnetism.</li>
    <li><a href="emergent-phases.html">Superconductivity, Quantum Hall &amp; Topological Phases</a> — what happens when the electron sea or the spin lattice condenses into a new ordered state.</li>
    <li><a href="advanced-formalism.html">Graduate-Level Formalism &amp; Experiment</a> — second quantization, Green's functions, and the many-body machinery behind Fermi-liquid theory and spin waves.</li>
    <li><a href="../statistical-mechanics/">Statistical Mechanics</a> — the Ising model, mean-field theory, and the physics of phase transitions and critical exponents.</li>
    <li><a href="../quantum-mechanics/">Quantum Mechanics</a> — the Pauli principle and exchange that make both metals and magnetism quantum from the start.</li>
    <li><a href="../">Physics Hub</a> — browse all physics topics.</li>
  </ul>
</div>
