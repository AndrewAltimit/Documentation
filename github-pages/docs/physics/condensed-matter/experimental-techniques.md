---
layout: docs
title: "Condensed Matter: Experimental Techniques"
permalink: /docs/physics/condensed-matter/experimental-techniques.html
toc: true
toc_sticky: true
hide_title: true
---

[Condensed Matter Physics](./) &raquo; Experimental Techniques

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Experimental Techniques</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">How we actually see band structure, order, and excitations in real materials</p>
</div>

## Why probes, and what they each "see"

A theory of a solid is only as good as our ability to test it. Every experimental
technique in condensed matter is, at bottom, a way of coupling an external field —
photons, electrons, neutrons, a magnetic field, a temperature gradient — to a
material and reading off the response. What makes a probe powerful is **which
correlation function it measures** and **in what variables it is resolved** (energy,
momentum, position, or temperature).

The single most useful organizing idea is the **linear-response / fluctuation–dissipation**
picture. A weak external perturbation that couples to some operator $\hat{B}$ produces
a measurable change in the conjugate observable $\hat{A}$, governed by the retarded
response function

$$\chi_{AB}(\mathbf{q},\omega) = -\frac{i}{\hbar}\int_0^\infty dt\, e^{i\omega t}\,\langle [\hat{A}(\mathbf{q},t),\hat{B}(-\mathbf{q},0)]\rangle .$$

Scattering and spectroscopic experiments measure the **dynamical structure factor**,
the imaginary part of $\chi$ weighted by a Bose factor,

$$S(\mathbf{q},\omega) = \frac{1}{\pi}\frac{1}{1 - e^{-\hbar\omega/k_B T}}\,\mathrm{Im}\,\chi(\mathbf{q},\omega),$$

so that a scattering experiment with momentum transfer $\mathbf{q}$ and energy transfer
$\hbar\omega$ directly maps out the spectrum of the material's excitations. The table
below summarizes the division of labor; the rest of the page treats each probe in detail.

<div class="comparison-table" markdown="1">

| Probe | Couples to | Resolved in | Reveals |
|---|---|---|---|
| **ARPES** | Single-electron removal | $(\mathbf{k},\omega)$ | Band structure, Fermi surface, self-energy, gaps |
| **STM / STS** | Single-electron tunneling | $(\mathbf{r},\omega)$ | Local DOS, atomic-scale order, QPI, gap maps |
| **Neutron scattering** | Nuclear positions, spins | $(\mathbf{q},\omega)$ | Crystal & magnetic structure, phonons, magnons |
| **Inelastic light (Raman/IR)** | Charge density, bonds | $\omega$ (near $\mathbf{q}=0$) | Phonons, magnons, electronic continua, symmetry |
| **Quantum oscillations** | Landau quantization | $1/B$ | Fermi-surface areas, effective mass, scattering |
| **Transport** | Charge / heat currents | $T, B, \omega$ | Carrier type & density, mobility, gaps, scattering |
| **Thermodynamics** | Entropy, magnetization | $T, B$ | DOS at $E_F$, phase transitions, degrees of freedom |

</div>

## ARPES — mapping band structure and self-energy

**Angle-resolved photoemission spectroscopy** is the most direct experimental window
onto the electronic band structure $E(\mathbf{k})$. A monochromatic photon of energy
$h\nu$ ejects an electron from the solid; by measuring the photoelectron's kinetic
energy and emission angle, one reconstructs the energy and crystal momentum the
electron had *inside* the material.

### Kinematics

Energy conservation fixes the binding energy relative to the Fermi level,

$$E_B = h\nu - W - E_{kin},$$

where $W$ is the work function. Crucially, the **component of momentum parallel to the
surface is conserved** across the sample boundary (the surface breaks translational
symmetry only along the normal):

$$k_\parallel = \frac{1}{\hbar}\sqrt{2 m E_{kin}}\,\sin\theta .$$

Sweeping the emission angle $\theta$ thus sweeps $k_\parallel$, and a modern hemispherical
analyzer records intensity as a 2D image over $(E_{kin},\theta)$ — i.e. a slice of the
band structure $E(k_\parallel)$ in a single shot. The perpendicular component $k_\perp$ is
not conserved and must be inferred (e.g. by varying $h\nu$, often using a nearly-free-electron
final state), which is why ARPES is at its sharpest for quasi-2D materials (cuprates,
graphene, transition-metal dichalcogenides) where $k_\perp$ dispersion is weak.

### What is actually measured: the spectral function

In the sudden approximation the photocurrent is proportional to the **single-particle
spectral function** times the Fermi function and a matrix element:

$$I(\mathbf{k},\omega) \propto |M_{fi}(\mathbf{k})|^2\, f(\omega)\, A(\mathbf{k},\omega).$$

The spectral function is the imaginary part of the retarded Green's function,

$$A(\mathbf{k},\omega) = -\frac{1}{\pi}\,\mathrm{Im}\,G^R(\mathbf{k},\omega)
= \frac{1}{\pi}\,\frac{|\Sigma''(\mathbf{k},\omega)|}{[\omega - \epsilon^0_\mathbf{k} - \Sigma'(\mathbf{k},\omega)]^2 + [\Sigma''(\mathbf{k},\omega)]^2}.$$

For a non-interacting band $A$ is a delta function pinned to the bare dispersion
$\epsilon^0_\mathbf{k}$. Interactions enter entirely through the **self-energy**
$\Sigma = \Sigma' + i\Sigma''$, and ARPES is essentially a machine for measuring it:

- The **real part** $\Sigma'$ shifts and renormalizes the dispersion. The renormalized
  band crosses $E_F$ where $\omega - \epsilon^0_\mathbf{k} - \Sigma'=0$; the slope ratio
  gives the mass enhancement $m^*/m = 1 + \lambda$, with $\lambda = -\partial\Sigma'/\partial\omega$.
  A "kink" in the dispersion at a phonon or magnon energy is a textbook fingerprint of
  electron–boson coupling.
- The **imaginary part** $\Sigma''$ sets the linewidth: the peak width in a **momentum
  distribution curve** (MDC, fixed $\omega$) is $\Delta k = 2\Sigma''/(\hbar v_F)$, giving
  the quasiparticle scattering rate directly. A Fermi liquid shows $\Sigma'' \propto \omega^2 + (\pi k_B T)^2$;
  cuprate "strange metals" famously show $\Sigma'' \propto |\omega|$ (marginal Fermi liquid).

### Practical analysis: MDCs vs EDCs

Two orthogonal cuts through the $I(\mathbf{k},\omega)$ image are used:

- **EDC** (energy distribution curve, fixed $\mathbf{k}$): natural for reading off
  **gaps** — superconducting gaps, the cuprate pseudogap, charge-density-wave gaps —
  as a suppression of weight at $E_F$ and a coherence peak at $\pm\Delta$.
- **MDC** (momentum distribution curve, fixed $\omega$): a Lorentzian whose center
  traces the dispersion and whose width gives $\Sigma''$. Because the bare dispersion is
  locally linear, MDCs decouple $\Sigma'$ (peak position) from $\Sigma''$ (peak width)
  cleanly, which is why self-energy extraction is usually done from MDCs.

### Fermi surfaces and modern variants

Integrating $I(\mathbf{k},\omega)$ over a narrow window at $E_F$ and plotting versus
$(k_x,k_y)$ produces a direct **image of the Fermi surface** — bright contours wherever a
band crosses $E_F$. Spin-resolved ARPES adds a Mott or VLEED spin detector to map the
**spin texture**, the decisive evidence for spin–momentum locking on topological-insulator
surfaces. Time-resolved ARPES (pump–probe) populates and watches *unoccupied* states relax,
accessing the band structure above $E_F$ and ultrafast dynamics.

<div class="info-box" markdown="1">
**What ARPES uniquely delivers:** the only probe that resolves the electronic spectral
function in both energy and momentum, hence the gold standard for band dispersions, Fermi
surfaces, anisotropic gaps, and the self-energy of correlated electrons.
**Limitations:** surface sensitive (probing depth of a few atomic layers, demands atomically
clean cleaved surfaces and UHV), needs $\mathbf{k}$-conserving geometry (best for 2D
systems), and reads only *occupied* states unless pumped.
</div>

## STM / STS — real-space local density of states

Where ARPES resolves momentum, **scanning tunneling microscopy** resolves *position* —
down to single atoms. A sharp metal tip is brought within a nanometer of a conducting
surface; the exponentially small overlap of tip and sample wavefunctions lets electrons
**quantum-tunnel** across the vacuum gap, producing a current that is exponentially
sensitive to the tip–sample separation $d$:

$$I \propto e^{-2\kappa d}, \qquad \kappa = \frac{\sqrt{2m\phi}}{\hbar}.$$

A typical $\kappa \approx 1\ \text{\AA}^{-1}$ means the current changes by an order of
magnitude per Angstrom — the origin of STM's sub-atomic vertical resolution. Holding $I$
constant with a feedback loop while raster-scanning gives a **topographic** image of the
surface (more precisely a contour of constant integrated LDOS).

### Tunneling spectroscopy and the LDOS

Within the Tersoff–Hamann picture and a flat tip DOS, the tunneling current integrates the
**local density of states** of the sample over the bias window:

$$I(\mathbf{r},V) \propto \int_0^{eV} \rho_s(\mathbf{r},\omega)\, T(\omega,eV)\, d\omega .$$

Differentiating with respect to bias gives the centerpiece of **scanning tunneling
spectroscopy (STS)** — the differential conductance is, to good approximation, the LDOS at
energy $eV$ measured at the atomic position $\mathbf{r}$:

$$\left.\frac{dI}{dV}\right|_{\mathbf{r},V} \propto \rho_s(\mathbf{r},\, eV).$$

By taking a full $dI/dV$ spectrum at every pixel one builds a **spectroscopic map**:
the spatial variation of the density of states at a chosen energy. This is how
superconducting gap inhomogeneity in cuprates, vortex cores, and impurity-bound states are
imaged atom by atom.

### Quasiparticle interference (QPI)

Defects scatter Bloch electrons, and the interference of incoming and outgoing waves prints
standing-wave ripples in the LDOS at wavevector $\mathbf{q} = \mathbf{k}_f - \mathbf{k}_i$.
Fourier-transforming a $dI/dV$ map turns these ripples into bright spots whose positions
encode the **joint density of states** of the band structure:

$$\rho(\mathbf{q},\omega) \;\leftrightarrow\; \text{scattering between } \mathbf{k}_i,\mathbf{k}_f \text{ on the contour } E(\mathbf{k})=\omega .$$

QPI thereby recovers momentum-space information (constant-energy contours, gap anisotropy,
even the sign structure of an order parameter) from a real-space measurement — a beautiful
complement to ARPES, and applicable to buried or non-cleavable Fermi surfaces ARPES cannot
reach.

<div class="info-box" markdown="1">
**What STM/STS uniquely delivers:** atomic-scale real-space imaging of the LDOS, with
spectroscopic energy resolution set by temperature ($\sim 3.5\,k_B T$) and bias modulation.
Indispensable for inhomogeneous states, single-impurity physics, and (via QPI) momentum-resolved
gap structure. **Limitations:** surface-only, requires an atomically clean conducting surface,
and measures a convolution of sample and tip DOS.
</div>

## Neutron scattering — structure and magnetic order

Neutrons are the workhorse for **structure** because they carry no charge (so they
penetrate deep into bulk samples and scatter from nuclei, not the electron cloud) and
because they carry a **magnetic moment** (so they scatter directly from electronic spins).
Their de Broglie wavelength at thermal energies is $\sim 1\text{–}2\ \text{\AA}$ — comparable to
interatomic spacings — and their energy at those wavelengths is $\sim$ meV — comparable to
phonon and magnon energies. A single instrument therefore resolves both *where* atoms and
spins sit and *how* they move.

### Elastic scattering: crystal and magnetic structure

In a diffraction (elastic) experiment the scattered intensity is concentrated at reciprocal-lattice
vectors $\mathbf{G}$, weighted by the structure factor:

$$I(\mathbf{Q}) \propto |F(\mathbf{Q})|^2, \qquad F(\mathbf{Q}) = \sum_j b_j\, e^{i\mathbf{Q}\cdot\mathbf{r}_j}\, e^{-W_j},$$

where $b_j$ is the **nuclear scattering length** of atom $j$ and $e^{-W_j}$ is the Debye–Waller
factor. Because $b_j$ varies erratically (not monotonically) with atomic number, neutrons
locate **light atoms next to heavy ones** (hydrogen, oxygen, lithium) where X-rays struggle,
and distinguish neighboring elements and isotopes.

The magnetic moment of the neutron adds **magnetic Bragg peaks**. When spins order with a
periodicity different from the lattice, new peaks appear at the magnetic propagation vector
$\mathbf{k}$, often at half-integer positions for an antiferromagnet. Their intensity is
governed by the magnetic structure factor with the all-important polarization factor:

$$I_{mag}(\mathbf{Q}) \propto |f(\mathbf{Q})|^2\, \sum_{\alpha\beta}\big(\delta_{\alpha\beta} - \hat{Q}_\alpha\hat{Q}_\beta\big)\, S^\alpha(\mathbf{Q})\,S^\beta(-\mathbf{Q}).$$

The transverse projector $\delta_{\alpha\beta}-\hat{Q}_\alpha\hat{Q}_\beta$ means **neutrons
only see the spin component perpendicular to $\mathbf{Q}$** — by measuring at several $\mathbf{Q}$
one reconstructs the full spin *direction*, making neutron diffraction the definitive probe
of magnetic order (the very tool with which antiferromagnetism was first confirmed). Polarized
neutrons further separate nuclear from magnetic and longitudinal from transverse channels.

### Inelastic scattering: phonons and magnons

When the neutron exchanges energy with the sample it measures the **dynamical structure
factor** directly:

$$\frac{d^2\sigma}{d\Omega\, dE} \propto \frac{k_f}{k_i}\, S(\mathbf{Q},\omega),$$

with $S(\mathbf{Q},\omega)$ as defined in the overview. Scanning $(\mathbf{Q},\omega)$ on a
triple-axis or time-of-flight spectrometer maps **dispersion relations**: acoustic and optical
**phonon** branches from nuclear (coherent) scattering, and **magnon / spin-wave** branches
from the magnetic cross-section. Spin liquids and quantum-critical systems instead show a broad
*continuum* of $S(\mathbf{Q},\omega)$ — the smoking gun of fractionalized (e.g. spinon)
excitations — which is one of the most distinctive results only neutrons can deliver.

<div class="info-box" markdown="1">
**What neutron scattering uniquely delivers:** bulk-sensitive, quantitative crystal *and*
magnetic structure, plus the full $(\mathbf{Q},\omega)$ map of phonons and magnons. The only
routine probe of magnetic structure and spin dynamics in absolute units.
**Limitations:** weak cross-section demands large single crystals and reactor/spallation
sources; energy/momentum resolution and flux are perennial trade-offs.
</div>

## Inelastic & Raman scattering — excitations near zero momentum

Light-scattering probes complement neutrons by accessing excitations with extreme **energy
resolution** but at essentially **zero momentum transfer** (the photon wavevector is tiny on
the scale of the Brillouin zone). They are table-top, fast, and exquisitely sensitive to
**symmetry**.

### Raman scattering

In Raman scattering a visible photon is inelastically scattered, shifting in frequency by the
energy of an excitation it creates (Stokes) or absorbs (anti-Stokes):

$$\hbar\omega_{scattered} = \hbar\omega_{incident} \mp \hbar\Omega_{excitation}.$$

The Stokes/anti-Stokes intensity ratio is set by thermal occupation,
$I_{aS}/I_{S} = e^{-\hbar\Omega/k_B T}$, providing an internal thermometer. The measured
cross-section is governed by the Raman response $\chi''_{\gamma}(\omega)$, where the symmetry
of the light-polarization geometry selects a particular irreducible representation:

$$\frac{d^2\sigma}{d\Omega\,d\omega} \propto \big[1 + n(\omega)\big]\, \chi''_\gamma(\omega).$$

Because each excitation transforms as a definite **irreducible representation of the crystal
point group**, choosing incident/scattered polarizations ($A_{1g}$, $B_{1g}$, $B_{2g}$, …)
filters phonons, magnons, and electronic continua by symmetry. This is how Raman fingerprints
**phonon modes** (and through them lattice symmetry, strain, layer number in 2D materials),
**two-magnon** scattering in antiferromagnets, the **electronic continuum** and pair-breaking
$2\Delta$ peak in superconductors, and amplitude (Higgs) modes of order parameters.

### Infrared and optical conductivity

Infrared spectroscopy measures absorption/reflection, from which a Kramers–Kronig analysis
yields the **complex optical conductivity** $\sigma(\omega)=\sigma_1+i\sigma_2$. The
low-frequency Drude peak,

$$\sigma_1(\omega) = \frac{\sigma_0}{1 + (\omega\tau)^2},$$

gives the scattering rate $1/\tau$ and plasma frequency (hence carrier density), while gaps
appear as a clean suppression of $\sigma_1$ below a threshold. The optical sum rule
$\int_0^\infty \sigma_1(\omega)\,d\omega = \pi n e^2/2m$ ties spectral-weight transfer to
correlation physics. IR thus reads off charge gaps, the Drude weight, phonon and interband
features, and the redistribution of spectral weight at phase transitions.

<div class="info-box" markdown="1">
**What light scattering uniquely delivers:** meV-to-sub-meV energy resolution, symmetry
selectivity via polarization, and direct access to $\mathbf{q}\approx 0$ excitations
(zone-center phonons, magnons, Higgs/amplitude modes, electronic continua) — fast and on small
samples. **Limitations:** restricted to $\mathbf{q}\approx 0$; shallow optical penetration; and
the response is convolved with light–matter matrix elements.
</div>

## Quantum oscillations — the Fermi surface

In a strong magnetic field the electronic states condense into **Landau levels**, and as the
field is swept these levels pass through the Fermi energy one by one. Every time a level
crosses $E_F$ the density of states at $E_F$ spikes, and essentially every physical property —
magnetization (**de Haas–van Alphen**), resistivity (**Shubnikov–de Haas**), magnetostriction,
sound velocity — oscillates. These quantum oscillations are the cleanest, most quantitative
map of the **Fermi surface** available.

### Onsager relation: oscillation frequency = Fermi-surface area

The oscillations are **periodic in $1/B$**, and the Onsager relation ties their frequency $F$
directly to an extremal cross-sectional area $A_{ext}$ of the Fermi surface perpendicular to
the field:

$$F = \frac{\hbar}{2\pi e}\, A_{ext}(E_F).$$

Rotating the sample maps how $F(\theta)$ varies, tracing out the **full 3D shape of the Fermi
surface**. Each distinct extremal orbit contributes its own frequency, so a Fourier transform
of the signal in $1/B$ reveals the Fermi-surface "fingerprint" of the metal.

### Lifshitz–Kosevich: mass and scattering

The amplitude of each oscillation is described by the **Lifshitz–Kosevich** formula,

$$M \propto \left(\frac{B}{T}\right)^{1/2} R_T\, R_D\, R_S \,\sin\!\left(\frac{2\pi F}{B} + \phi\right),$$

with three damping factors that turn the amplitude into a measurement of microscopic
quantities:

- **Thermal factor** $R_T = X/\sinh X$ with $X = 2\pi^2 k_B T\, m^*/\hbar e B$. Fitting the
  temperature dependence of the amplitude yields the **cyclotron effective mass** $m^*$ — a
  direct measure of mass renormalization, hence correlation strength.
- **Dingle factor** $R_D = e^{-2\pi^2 k_B T_D\, m^*/\hbar e B}$. The field dependence gives the
  **Dingle temperature** $T_D$ and thus the quantum scattering rate / mean free path.
- **Spin factor** $R_S = \cos(\pi g m^*/2 m_e)$, encoding Zeeman splitting and the $g$-factor.

The **phase** $\phi$ carries a Berry-phase offset: a $\pi$ Berry phase (shifting $\phi$ by
$1/2$) is a hallmark of **Dirac/Weyl** fermions, so quantum oscillations also test topological
band structure.

<div class="info-box" markdown="1">
**What quantum oscillations uniquely deliver:** bulk, quantitative Fermi-surface geometry plus
effective masses, scattering rates, and Berry phase — the benchmark against which band-structure
calculations and ARPES Fermi surfaces are checked.
**Limitations:** demand high-purity samples ($\omega_c\tau \gg 1$), high fields, and low
temperatures; large or open Fermi-surface orbits can be hard to observe.
</div>

## Transport measurements

Transport is the most accessible and historically the first window onto a solid's electronic
state — it asks how charge and heat flow in response to electric fields, magnetic fields, and
temperature gradients.

### Resistivity and the four-probe method

DC resistivity is measured with a **four-probe** geometry: current is driven through the outer
two contacts and voltage read across the inner two, so that contact and lead resistances drop
out of the measurement. The temperature dependence is diagnostic of the ground state and its
excitations:

- **Metal:** $\rho(T) = \rho_0 + AT^2$ (Fermi-liquid electron–electron scattering) or
  $\rho \propto T^5$ (phonon-limited, Bloch–Grüneisen) at low $T$; the residual $\rho_0$ measures
  disorder.
- **Semiconductor / insulator:** activated, $\rho \propto e^{E_g/2k_B T}$, giving the **gap**;
  variable-range hopping $\rho \propto e^{(T_0/T)^{1/4}}$ signals localization.
- **Superconductor:** $\rho \to 0$ below $T_c$.
- **Strange metal:** linear $\rho \propto T$ over a wide range — a defining anomaly of cuprates
  and other quantum-critical systems.

### Hall effect: carrier type and density

A magnetic field perpendicular to the current deflects carriers, building a transverse Hall
voltage. In the simplest single-band picture the Hall coefficient gives the **sign and density**
of carriers directly:

$$R_H = \frac{E_y}{j_x B_z} = \frac{1}{n q}.$$

A positive $R_H$ signals hole-like, negative electron-like conduction. Combined with the
conductivity it yields the **mobility** $\mu = |R_H|\,\sigma$. The Hall response is also the
gateway to topological transport — the **quantized** Hall plateaus $\sigma_{xy} = \nu e^2/h$,
and the **anomalous** Hall effect proportional to Berry curvature in magnetic conductors.

### Magnetotransport

Beyond the Hall effect, the field dependence of the longitudinal resistance (magnetoresistance)
reveals multiband effects, Fermi-surface topology (via Shubnikov–de Haas oscillations, above),
weak localization/antilocalization (a quantum-coherence and spin–orbit diagnostic), and chiral
anomalies in Weyl semimetals (negative longitudinal magnetoresistance).

<div class="info-box" markdown="1">
**What transport uniquely delivers:** the macroscopic ground-state response — metal vs insulator
vs superconductor, carrier sign/density/mobility, gaps, and (via Hall) topological quantization.
Fast, on tiny samples, over huge ranges of $T$ and $B$.
**Limitations:** integrates over the whole Fermi surface (no momentum resolution); multiband and
inhomogeneity complicate interpretation.
</div>

## Thermodynamic measurements

Thermodynamic probes count **degrees of freedom and entropy**. They do not resolve momentum or
position, but they are unmatched at identifying phase transitions and at counting the states
available at the Fermi level.

### Specific heat

At low temperature the specific heat of a metal separates cleanly into electronic and lattice
parts:

$$C(T) = \gamma T + \beta T^3.$$

Plotting $C/T$ versus $T^2$ gives a straight line whose **intercept $\gamma$** is the Sommerfeld
coefficient — proportional to the **density of states at the Fermi level**,
$\gamma = \tfrac{\pi^2}{3} k_B^2\, g(E_F)$. A giant $\gamma$ is the defining signature of
**heavy-fermion** materials (effective masses of hundreds of $m_e$). The **slope $\beta$** gives
the Debye temperature and thus the phonon spectrum. A phase transition shows up as a sharp
**anomaly**: a mean-field jump $\Delta C$ at a superconducting $T_c$ (whose size,
$\Delta C/\gamma T_c \approx 1.43$ in BCS, tests the pairing), a $\lambda$-shaped peak at a
continuous magnetic transition, or a latent-heat spike at a first-order one. The entropy
$S(T)=\int_0^T (C/T')\,dT'$ released across a transition counts the participating degrees of
freedom (e.g. $R\ln 2$ per spin-$\tfrac12$ moment).

### Magnetization and susceptibility

The magnetization $M(H,T)$ and susceptibility $\chi = \partial M/\partial H$ classify magnetic
ground states. A Curie–Weiss law,

$$\chi(T) = \frac{C}{T - \theta_{CW}},$$

extracts the local-moment size from the Curie constant $C$ and the dominant exchange (sign and
scale) from the Weiss temperature $\theta_{CW}$. A temperature-independent **Pauli** susceptibility
signals itinerant moments; sharp features locate ferromagnetic ($\theta_{CW}>0$) and
antiferromagnetic ($\theta_{CW}<0$, Néel kink) transitions; hysteresis loops quantify coercivity
and ordered moment. The frustration ratio $|\theta_{CW}|/T_N \gg 1$ flags candidate spin liquids.

### Thermal expansion and magnetocalorics

Thermal expansion $\alpha = \tfrac1L\,\partial L/\partial T$ couples the entropy to volume
(Grüneisen analysis) and is acutely sensitive to **pressure-tuned quantum critical points**, while
the magnetocaloric effect ($\partial T/\partial H$ at fixed entropy) sharpens the detection of
field-induced transitions where specific heat alone is ambiguous.

<div class="info-box" markdown="1">
**What thermodynamics uniquely delivers:** bulk, model-independent counting of entropy and states
— $g(E_F)$ via $\gamma$, moment size and exchange via Curie–Weiss, and an unambiguous, quantitative
locator of phase transitions and their order.
**Limitations:** no momentum or spatial resolution; signals average over the whole sample, so they
constrain rather than uniquely determine microscopic mechanisms.
</div>

## Choosing and combining probes

No single technique tells the whole story; the art of condensed-matter experiment is
**triangulation**. A Fermi surface inferred from ARPES is confirmed in absolute terms by quantum
oscillations and tested for momentum-averaged consistency against the Hall coefficient and
$\gamma$ from specific heat. A superconducting gap seen as a coherence peak in ARPES EDCs is
mapped in real space by STS, its symmetry pinned down by Raman and its $\Delta C/T_c$ jump by
calorimetry. Magnetic order proposed from a Curie–Weiss susceptibility is **proven** — direction
and all — only by neutron diffraction, with its dynamics filled in by inelastic neutron and
two-magnon Raman scattering. Reading the same physics through complementary correlation functions
is what turns a measurement into an understanding.

## See Also

<div class="see-also-card">
  <h4>Where to go next</h4>
  <ul>
    <li><a href="advanced-formalism.html">Graduate-Level Formalism &amp; Experiment</a> — the Green's-function and self-energy machinery these probes measure.</li>
    <li><a href="emergent-phases.html">Superconductivity, Quantum Hall &amp; Topological Phases</a> — the phases these techniques were built to characterize.</li>
    <li><a href="./">Condensed Matter Physics (Hub)</a> — crystal structure, band theory, and magnetism.</li>
    <li><a href="../quantum-field-theory.html">Quantum Field Theory</a> — linear response and correlation functions in field-theoretic language.</li>
  </ul>
</div>
