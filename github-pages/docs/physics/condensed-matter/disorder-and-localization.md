---
layout: docs
title: "Condensed Matter: Disorder & Localization"
description: "How disorder and quantum interference turn metals into insulators: Anderson localization, weak localization and antilocalization, the mobility edge, the scaling theory of localization, hopping transport, and many-body localization."
permalink: /docs/physics/condensed-matter/disorder-and-localization.html
toc: true
toc_sticky: true
hide_title: true
---

[Condensed Matter Physics](./) &raquo; Disorder &amp; Localization

## Disorder & Localization

Disorder in a solid (impurities, vacancies, alloy randomness, rough interfaces)
does more than add resistance. Quantum interference between waves scattered by
random impurities can trap electrons completely, producing an insulator with no
band gap and no interactions. This page covers that phenomenon, **Anderson
localization**, from its weak precursor in good metals (weak localization and
antilocalization) through the metal-insulator transition at the mobility edge,
the single-parameter scaling theory that organizes the whole subject, transport
in the localized phase, and **many-body localization**, the interacting,
finite-temperature extension whose status is still debated. Band theory and
Bloch states are covered on the [hub](./); the field-theoretic and Green's-function
methods on [Graduate-Level Formalism](advanced-formalism.html).

## Why disorder matters

In a perfect crystal Bloch's theorem guarantees extended eigenstates, so a
partially filled band is a perfect conductor at $T = 0$. Real metals have finite
resistance because electrons scatter from phonons and defects. In the
semiclassical (Drude-Boltzmann) picture an electron travels a **mean free path**
$\ell = v_F\tau$ between elastic collisions, and over many collisions its motion
becomes diffusive with diffusion constant $D = v_F^2\tau/d$ in $d$ dimensions. The
conductivity is

$$\sigma = \frac{ne^2\tau}{m} = e^2\,\nu(E_F)\,D,$$

the second form (the Einstein relation) using the density of states per unit
volume $\nu(E_F)$. In this picture disorder only *limits* the conductivity.

In 1958 P. W. Anderson showed that the picture is incomplete. When disorder is
strong enough, or in low enough dimension for *any* disorder, interference between
scattered partial waves makes eigenstates **exponentially localized**:

$$|\psi(\mathbf{r})| \sim \exp\left(-\frac{|\mathbf{r} - \mathbf{r}_0|}{\xi}\right),$$

with **localization length** $\xi$. A localized electron carries no dc current at
$T = 0$, so $\sigma \to 0$ even though the density of states at $E_F$ is finite.
The result is an insulator produced by disorder alone. Anderson shared the 1977
Nobel Prize partly for this work, and the effect has since been observed for
electrons, light, sound, and ultracold atoms.

### Length scales and regimes

Localization physics is organized by comparing a few lengths with each other and
with the sample size $L$:

| Length | Definition | Physical meaning |
|---|---|---|
| Fermi wavelength $\lambda_F$ | $2\pi/k_F$ | Scale of quantum interference |
| Mean free path $\ell$ | $v_F\tau$ | Distance between elastic collisions |
| Phase-coherence length $L_\phi$ | $\sqrt{D\tau_\phi}$ | Distance over which phase memory survives inelastic scattering |
| Localization length $\xi$ | decay length of eigenstates | Size of a localized state |
| Thermal length $L_T$ | $\sqrt{\hbar D/k_BT}$ | Scale set by thermal energy smearing |

| Regime | Condition | Behaviour |
|---|---|---|
| Ballistic | $L < \ell$ | Conductance set by the number of channels (Landauer) |
| Diffusive, weakly disordered | $\ell < L$, $k_F\ell \gg 1$, $L < \xi$ | Ohmic, with small interference corrections (weak localization) |
| Strongly localized | $L > \xi$ | Conductance $\sim e^{-2L/\xi}$; insulating at $T = 0$ |
| Mesoscopic | $L \lesssim L_\phi$ | Sample-specific fluctuations of order $e^2/h$ (universal conductance fluctuations) |

The **Ioffe-Regel criterion** $k_F\ell \sim 1$ marks where the semiclassical
picture must fail: the mean free path cannot be shorter than a wavelength.

A useful dimensionless measure is the **Thouless conductance**
$g = E_{\text{Th}}/\delta$, the ratio of the Thouless energy $E_{\text{Th}} = \hbar D/L^2$
(the inverse time to diffuse across the sample) to the mean level spacing
$\delta = 1/(\nu L^d)$. It equals the conductance in units of $e^2/h$ up to a
numerical factor. Eigenstates are extended when $g \gg 1$ (levels hybridize across
the sample) and localized when $g \ll 1$. How $g$ changes with $L$ is the content
of the [scaling theory](#scaling-theory-of-localization).

## Anderson localization

### The Anderson model

The canonical model is a tight-binding lattice with random on-site energies,

$$H = \sum_i \varepsilon_i\, c_i^\dagger c_i - t\sum_{\langle ij\rangle}\left(c_i^\dagger c_j + \text{h.c.}\right),$$

where $t$ is the nearest-neighbour hopping and the $\varepsilon_i$ are independent
random variables, conventionally drawn from a box distribution of width $W$:

$$P(\varepsilon_i) = \frac{1}{W} \quad \text{for} \quad -\frac{W}{2} \le \varepsilon_i \le \frac{W}{2}.$$

The control parameter is the dimensionless disorder strength $W/t$. At $W = 0$ the
model is a clean band of width $2zt$ ($z$ the coordination number) with all states
extended.

### The localization transition

Hopping delocalizes; disorder localizes. An electron on a site of energy
$\varepsilon_i$ tunnels efficiently only to a neighbour whose energy lies within
about $t$ of its own. When the typical mismatch $W$ greatly exceeds $t$, such
resonances are rare and the electron is trapped. Anderson's analysis of the
convergence of the locator (strong-disorder) expansion showed that above a
critical disorder every eigenstate is localized:

$$\frac{W}{t} > \left(\frac{W}{t}\right)_c \;\Longrightarrow\; \text{all states localized} \;\Longrightarrow\; \text{insulator}.$$

The outcome depends on dimension:

| Dimension | Critical disorder | Localization length at weak disorder |
|---|---|---|
| $d = 1$ | $W_c = 0$: any disorder localizes all states | $\xi \approx 24(4t^2 - E^2)/W^2$ lattice spacings (Thouless); anomalous near $E = 0$, where $\xi \approx 105\,t^2/W^2$ |
| $d = 2$ (orthogonal class) | $W_c = 0$: marginal | $\xi \sim \ell\,\exp(\pi k_F\ell/2)$, exponentially large |
| $d = 3$ (cubic, box distribution) | $W_c \approx 16.5\,t$ | States in the band centre stay extended for $W < W_c$ |

For $W < W_c$ in three dimensions, band-centre states are extended while the
band tails are localized, and the two are separated by a mobility edge
([below](#the-mobility-edge-and-the-metal-insulator-transition)). Precisely at the
transition, eigenstates are neither extended nor localized but **multifractal**:
their moments scale as $\sum_i |\psi(i)|^{2q} \sim L^{-\tau_q}$ with a nonlinear
spectrum $\tau_q$.

### Diagnostics: participation ratio and level statistics

The **inverse participation ratio** (IPR) of a normalized eigenstate on $N$ sites,

$$P_2 = \sum_i |\psi(i)|^4,$$

measures the number of sites it occupies, $1/P_2$. An extended state has
$P_2 \sim 1/N \to 0$; a localized state has $P_2 \to \text{const}$ independent of
$N$. The code below diagonalizes the 1D Anderson chain and compares the
disorder-averaged IPR of band-centre states with the clean chain.

```python
import numpy as np

def anderson_1d(N, W, t=1.0, rng=None):
    """1D Anderson Hamiltonian: box-distributed on-site energies + NN hopping."""
    eps = rng.uniform(-W / 2, W / 2, size=N)
    return np.diag(eps) - t * (np.eye(N, k=1) + np.eye(N, k=-1))

def mean_ipr(N, W, window=0.5, samples=20, seed=0):
    """Average IPR of eigenstates with |E| < window over disorder realizations."""
    rng = np.random.default_rng(seed)
    iprs = []
    for _ in range(samples):
        evals, evecs = np.linalg.eigh(anderson_1d(N, W, rng=rng))
        sel = np.abs(evals) < window
        iprs.extend(np.sum(np.abs(evecs[:, sel]) ** 4, axis=0))
    return np.mean(iprs)

for W in (0.0, 2.0):
    for N in (250, 500, 1000, 2000):
        p2 = mean_ipr(N, W, samples=1 if W == 0 else 20)
        print(f"W={W:3.1f}  N={N:5d}  <IPR>={p2:.3e}  N*<IPR>={N * p2:7.1f}")
```

Typical output:

```text
W=0.0  N=  250  <IPR>=5.976e-03  N*<IPR>=    1.5
W=0.0  N= 2000  <IPR>=7.496e-04  N*<IPR>=    1.5
W=2.0  N=  250  <IPR>=4.511e-02  N*<IPR>=   11.3
W=2.0  N= 2000  <IPR>=3.798e-02  N*<IPR>=   76.0
```

In the clean chain $N \cdot P_2$ is constant (it equals $3/2$ for standing
waves), so the IPR falls as $1/N$. With $W = 2t$ the IPR saturates at about
$0.04$: each state occupies a few tens of sites however long the chain. (For
localization lengths comparable to $N$ the IPR still drifts, which is why
numerical work uses finite-size scaling or the transfer-matrix method, where $\xi$
is the inverse of the smallest Lyapunov exponent of a product of random
$2\times 2$ transfer matrices.)

A second diagnostic uses the spectrum alone. Localized states at nearby energies
sit in different places and do not repel, so their level spacings follow Poisson
statistics; extended states mix and show random-matrix (Wigner-Dyson) level
repulsion. The mean ratio of consecutive spacings,
$\langle r\rangle = \langle\min(s_n, s_{n+1})/\max(s_n, s_{n+1})\rangle$, needs no
unfolding of the density of states and takes the values $2\ln 2 - 1 \approx 0.386$
(Poisson), $\approx 0.531$ (GOE), $\approx 0.600$ (GUE), and $\approx 0.674$ (GSE).

## Weak localization and antilocalization

Strong localization needs strong disorder, but interference leaves a measurable
fingerprint even in good metals with $k_F\ell \gg 1$: a small, temperature- and
field-dependent correction to the Drude conductivity called **weak
localization**. It is the perturbative precursor of Anderson localization.

### Coherent backscattering

The probability for a diffusing electron to return to its starting point is a
sum over closed paths. Every closed path has a **time-reversed partner** that
visits the same impurities in the opposite order. With time-reversal symmetry the
two amplitudes are equal, $A_+ = A_-$, so

$$|A_+ + A_-|^2 = |A_+|^2 + |A_-|^2 + 2\,\text{Re}(A_+A_-^*) = 4|A_+|^2,$$

twice the classical value $2|A_+|^2$. Enhanced return probability means reduced
forward transport, so the conductivity falls below the Drude value. In
diagrammatic language this is the sum of maximally crossed diagrams (the
**Cooperon**). The same effect appears in optics as the coherent-backscattering
cone of light reflected from a random medium.

### Size of the correction

Loops contribute from the elastic time $\tau$ up to the **phase-coherence time**
$\tau_\phi$, beyond which inelastic scattering (electron-electron,
electron-phonon) randomizes the phase. In two dimensions the correction per
square is logarithmic:

$$\Delta\sigma_{2D} = -\frac{e^2}{2\pi^2\hbar}\ln\frac{\tau_\phi}{\tau} = -\frac{e^2}{\pi h}\ln\frac{\tau_\phi}{\tau} = -\frac{2e^2}{\pi h}\ln\frac{L_\phi}{L_e},$$

with $L_e = \sqrt{D\tau}$ (spin degeneracy included). In general dimension, up to
positive numerical prefactors,

$$\Delta\sigma \propto -\frac{e^2}{h}\times
\begin{cases}
L_\phi - L_e & d = 1 \;(\text{conductance times length}), \\
\ln(L_\phi/L_e) & d = 2, \\
1/L_e - 1/L_\phi & d = 3 .
\end{cases}$$

The correction is negative in every dimension. In $d = 3$ it stays finite as
$L_\phi \to \infty$, but in $d \le 2$ it diverges. Because $\tau_\phi$ grows on
cooling (typically $\tau_\phi \propto T^{-p}$ with $p$ between 1 and 3), the
conductivity of a 1D or 2D metal decreases without bound as $T \to 0$. This is the
perturbative signature of the scaling-theory result that non-interacting,
time-reversal-symmetric electrons have **no true metal in $d \le 2$**.

### Magnetoresistance as the fingerprint

A perpendicular magnetic field gives time-reversed loops opposite Aharonov-Bohm
phases, destroying the interference once the flux through a typical coherent loop
reaches about one flux quantum. The field therefore *restores* conductance, giving
a **negative magnetoresistance** of order $e^2/h$ per square at low field. For a
2D film, Hikami, Larkin, and Nagaoka (HLN) derived

$$\sigma(B) - \sigma(0) = \alpha\,\frac{e^2}{2\pi^2\hbar}\left[\psi\!\left(\frac{1}{2} + \frac{B_\phi}{B}\right) - \ln\frac{B_\phi}{B}\right], \qquad B_\phi = \frac{\hbar}{4eL_\phi^2},$$

where $\psi$ is the digamma function. The bracket is positive and grows with $B$.
In the simplest limits $\alpha = 1$ for weak localization (conductance rises with
field) and $\alpha = -1/2$ for weak antilocalization (conductance falls). Fitting
this form to low-field data is the standard way to extract $L_\phi(T)$ in thin
metal films, semiconductor heterostructures, graphene, and topological-insulator
surfaces.

### Symmetry classes and antilocalization

Strong spin-orbit scattering rotates the spin along a loop so that the
time-reversed amplitudes acquire a relative phase of $\pi$ (a Berry phase from
the $2\pi$ spin rotation), turning constructive interference into destructive.
This **weak antilocalization** *raises* the conductivity and produces a positive
low-field magnetoresistance. The three behaviours correspond to the Wigner-Dyson
symmetry classes:

| Class | Time reversal | Spin rotation | $\beta$ | Interference correction | 2D non-interacting ground state | Examples |
|---|---|---|---|---|---|---|
| Orthogonal | yes | yes | 1 | weak localization | insulator | ordinary metal films, Si MOSFETs |
| Unitary | broken | either | 2 | suppressed at leading order | insulator (except quantum Hall transitions) | samples in a field, magnetic impurities |
| Symplectic | yes ($\Theta^2 = -1$) | broken | 4 | weak antilocalization | metal-insulator transition possible | heavy-metal films, InAs/InSb, topological-insulator surfaces |

The index $\beta$ is the Dyson index of the corresponding random-matrix ensemble.
The Altland-Zirnbauer extension to ten classes (adding particle-hole and chiral
symmetries) underlies the classification of topological insulators and
superconductors: topological surface states are exactly those that cannot be
localized by disorder respecting the protecting symmetry.

### Interaction corrections

Electron-electron interactions in a disordered metal add a separate correction,
the **Altshuler-Aronov** effect: diffusive motion makes the Coulomb interaction
more effective, producing a $\ln T$ correction to $\sigma$ in 2D and a zero-bias
anomaly (a dip in the tunneling density of states at $E_F$). It is distinguished
experimentally from weak localization because it is insensitive to weak
perpendicular fields. Interactions also supply the dephasing that sets
$\tau_\phi$.

## The mobility edge and the metal-insulator transition

### Mott's mobility edge

In three dimensions moderate disorder localizes only part of the spectrum. As N.
F. Mott emphasized, localized and extended states are separated in energy by a
sharp **mobility edge** $E_c$: states in the band tails, where the density of
states is low, are localized; states in the band centre are extended. The two
cannot coexist at the same energy, because a localized state degenerate with an
extended one would hybridize with it and delocalize.

<figure style="margin: 1.5rem auto; max-width: 480px;">
<svg viewBox="0 0 480 250" role="img" aria-labelledby="me-title me-desc" style="width: 100%; height: auto; color: currentColor;">
  <title id="me-title">Mobility edges in a disordered band</title>
  <desc id="me-desc">Density of states versus energy for a disordered 3D band. The central region between two mobility edges, minus E_c and plus E_c, contains extended states; the two tails beyond them contain localized states. A Fermi level inside a tail gives an insulator; inside the central region a metal.</desc>
  <defs>
    <marker id="me-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor"/></marker>
    <pattern id="me-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="currentColor" stroke-width="1" stroke-opacity="0.45"/></pattern>
  </defs>
  <line x1="30" y1="200" x2="460" y2="200" stroke="currentColor" stroke-width="1.5" marker-end="url(#me-arrow)"/>
  <text x="455" y="222" font-size="14" fill="currentColor" text-anchor="end">E</text>
  <text x="40" y="40" font-size="13" fill="currentColor">DOS</text>
  <path d="M40,200 C90,199 120,190 150,165 C185,130 205,60 245,60 C285,60 305,130 340,165 C370,190 400,199 450,200 Z" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="1.8"/>
  <path d="M40,200 C90,199 120,190 150,165 L150,200 Z" fill="url(#me-hatch)"/>
  <path d="M340,165 C370,190 400,199 450,200 L340,200 Z" fill="url(#me-hatch)"/>
  <line x1="150" y1="200" x2="150" y2="120" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5,4"/>
  <line x1="340" y1="200" x2="340" y2="120" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5,4"/>
  <text x="150" y="112" font-size="13" fill="currentColor" text-anchor="middle">−E_c</text>
  <text x="340" y="112" font-size="13" fill="currentColor" text-anchor="middle">+E_c</text>
  <text x="245" y="150" font-size="14" fill="currentColor" text-anchor="middle">extended</text>
  <text x="85" y="160" font-size="13" fill="currentColor" text-anchor="middle">localized</text>
  <text x="405" y="160" font-size="13" fill="currentColor" text-anchor="middle">localized</text>
  <text x="245" y="238" font-size="12" fill="currentColor" text-anchor="middle">E_F in the extended region: metal. E_F in a hatched tail: insulator.</text>
</svg>
<figcaption style="text-align: center; font-size: 0.9em;">Increasing disorder moves the mobility edges toward the band centre; at W = W_c they meet and every state is localized.</figcaption>
</figure>

### Driving the transition

Whether a sample is a metal at $T = 0$ depends on where the Fermi level sits
relative to $E_c$. There are two ways to cross the transition:

1. **Tune the disorder.** Increasing $W$ moves both mobility edges inward. At $W_c$ they meet at the band centre and all states localize.
2. **Tune the carrier density.** In doped semiconductors, the textbook case being phosphorus-doped silicon (Si:P), changing the dopant concentration sweeps $E_F$ through $E_c$. Below a critical density $n_c$ the material is insulating; above it, metallic. Mott's criterion $n_c^{1/3}a_B^\ast \approx 0.26$, with $a_B^\ast$ the effective Bohr radius, holds across many semiconductors.

This **Anderson transition** is a $T = 0$ quantum phase transition. It is distinct
from the **Mott transition**, driven by electron-electron repulsion ($U/t$ in the
Hubbard model) rather than disorder. Real doped semiconductors involve both, and
the combined Anderson-Mott problem remains incompletely understood.

### Critical behaviour

Near the transition the localization length on the insulating side and the dc
conductivity on the metallic side follow power laws:

$$\xi \sim |E - E_c|^{-\nu} \quad (\text{insulating side}), \qquad
\sigma \sim |E - E_c|^{s} \quad (\text{metallic side}).$$

Wegner's scaling relation ties the exponents together,

$$s = \nu\,(d - 2),$$

so $s = \nu$ in three dimensions and $s \to 0$ as $d \to 2$, another sign that
two is the lower critical dimension. Transfer-matrix and finite-size-scaling
numerics for the 3D orthogonal class give $\nu \approx 1.57$ (Slevin and
Ohtsuki), and the $2 + \epsilon$ expansion gives $\nu \approx 1/\epsilon$ near two
dimensions. Experiments on doped semiconductors have long reported smaller
conductivity exponents in some systems (early Si:P data gave $s \approx 1/2$), a
discrepancy usually attributed to Coulomb interactions placing real materials in
a different universality class.

### Transport in the localized phase

An Anderson insulator still conducts at $T > 0$, by phonon-assisted tunneling
("hopping") between localized states. Mott argued that at low temperature the
optimal hop balances distance against energy mismatch, giving **variable-range
hopping**:

$$\sigma(T) \propto \exp\left[-\left(\frac{T_0}{T}\right)^{1/(d+1)}\right].$$

Long-range Coulomb interactions open a soft **Coulomb gap** in the single-particle
density of states at $E_F$, $\nu(E) \propto |E - E_F|^{d-1}$, which changes the
law to the Efros-Shklovskii form

$$\sigma(T) \propto \exp\left[-\left(\frac{T_{\text{ES}}}{T}\right)^{1/2}\right]$$

in every dimension. A crossover from Mott to Efros-Shklovskii behaviour on cooling
is commonly observed.

## Scaling theory of localization

### Single-parameter scaling

The unifying framework is the scaling theory of Abrahams, Anderson, Licciardello,
and Ramakrishnan (the "Gang of Four", 1979). Its central hypothesis is that the
dimensionless conductance of a block of size $L$,

$$g(L) = \frac{G(L)}{e^2/h}, \qquad G(L) = \sigma L^{d-2} \text{ in the ohmic regime},$$

is the only parameter that matters: when blocks are combined into larger blocks,
the new conductance depends only on the old one. Its flow is captured by a single
**beta function**

$$\beta(g) = \frac{d\ln g}{d\ln L}.$$

If $\beta > 0$, $g$ grows with size and the system is a metal; if $\beta < 0$, $g$
shrinks toward zero and the system is an insulator.

### Limits fix the shape of the beta function

- **Good metal ($g \gg 1$).** Ohm's law gives $g \propto L^{d-2}$, so
  $\beta(g) = d - 2 - a/g + \ldots$ The $-a/g$ term ($a > 0$ in the orthogonal
  class) is exactly the weak-localization correction computed above.
- **Strong insulator ($g \ll 1$).** Conductance is set by tunneling through the
  sample, $g \sim e^{-L/\xi}$, so $\beta(g) = \ln g + \text{const} \to -\infty$.

Assuming $\beta$ is continuous and monotonic between these limits gives the
standard scaling diagram:

<figure style="margin: 1.5rem auto; max-width: 520px;">
<svg viewBox="0 0 500 320" role="img" aria-labelledby="beta-title beta-desc" style="width: 100%; height: auto; color: currentColor;">
  <title id="beta-title">Scaling function beta(g) for d = 1, 2, 3</title>
  <desc id="beta-desc">beta equals d ln g over d ln L plotted against ln g. All three curves go to minus infinity at small g. For d = 3 the curve crosses zero at a critical conductance g_c and approaches plus one at large g; for d = 2 it approaches zero from below; for d = 1 it approaches minus one. Arrows on the ln g axis show flow away from g_c in d = 3.</desc>
  <defs>
    <marker id="beta-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor"/></marker>
  </defs>
  <line x1="40" y1="140" x2="480" y2="140" stroke="currentColor" stroke-width="1.5" marker-end="url(#beta-arrow)"/>
  <line x1="60" y1="305" x2="60" y2="20" stroke="currentColor" stroke-width="1.5" marker-end="url(#beta-arrow)"/>
  <text x="475" y="160" font-size="14" fill="currentColor" text-anchor="end">ln g</text>
  <text x="68" y="28" font-size="14" fill="currentColor">β(g) = d ln g / d ln L</text>
  <line x1="60" y1="60" x2="470" y2="60" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,4" stroke-opacity="0.6"/>
  <line x1="60" y1="220" x2="470" y2="220" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,4" stroke-opacity="0.6"/>
  <text x="52" y="64" font-size="12" fill="currentColor" text-anchor="end">+1</text>
  <text x="52" y="144" font-size="12" fill="currentColor" text-anchor="end">0</text>
  <text x="52" y="224" font-size="12" fill="currentColor" text-anchor="end">−1</text>
  <path d="M70,285 C170,110 250,62 460,61" fill="none" stroke="currentColor" stroke-width="2.4"/>
  <path d="M70,292 C170,165 260,146 460,143" fill="none" stroke="currentColor" stroke-width="2.4" stroke-dasharray="9,5"/>
  <path d="M70,299 C170,240 260,223 460,222" fill="none" stroke="currentColor" stroke-width="2.4" stroke-dasharray="2,4"/>
  <text x="462" y="54" font-size="13" fill="currentColor" text-anchor="end">d = 3</text>
  <text x="462" y="134" font-size="13" fill="currentColor" text-anchor="end">d = 2</text>
  <text x="462" y="214" font-size="13" fill="currentColor" text-anchor="end">d = 1</text>
  <circle cx="179" cy="140" r="5" fill="currentColor"/>
  <text x="186" y="130" font-size="13" fill="currentColor">g_c</text>
  <line x1="170" y1="125" x2="120" y2="125" stroke="currentColor" stroke-width="1.5" marker-end="url(#beta-arrow)"/>
  <line x1="215" y1="118" x2="265" y2="118" stroke="currentColor" stroke-width="1.5" marker-end="url(#beta-arrow)"/>
  <text x="100" y="112" font-size="11" fill="currentColor">insulator</text>
  <text x="235" y="108" font-size="11" fill="currentColor">metal</text>
</svg>
<figcaption style="text-align: center; font-size: 0.9em;">Single-parameter scaling for the orthogonal class. Arrows show the direction of flow as the system size grows in d = 3; g_c is an unstable fixed point.</figcaption>
</figure>

### What the diagram says

- **$d = 1$.** $\beta < 0$ for all $g$: every state is localized for any disorder, confirming the numerics above.
- **$d = 2$.** The classical term $d - 2$ vanishes and the weak-localization correction makes $\beta$ slightly negative for all $g$. Conductance flows logarithmically slowly to zero, so two dimensions is the lower critical dimension and non-interacting 2D electrons with time-reversal and spin-rotation symmetry are always localized, though $\xi$ can be astronomically large. In the symplectic class $a < 0$, $\beta$ is positive at large $g$, and a genuine 2D metal-insulator transition exists.
- **$d = 3$.** $\beta$ crosses zero at an unstable fixed point $g_c$. Samples starting above $g_c$ flow to a metal, those below to an insulator; the fixed point is the mobility edge. Linearizing, $\beta(g) \approx (g - g_c)/(\nu g_c)$, so the slope at the crossing gives the localization-length exponent $\nu = 1/[g_c\,\beta'(g_c)]$.

One curve therefore explains why disorder always wins in low dimensions, why three
dimensions has a true metal-insulator transition, and how weak localization and
strong localization are two ends of the same flow. The field-theoretic
formulation is the nonlinear sigma model of Wegner and Efetov, in which $\beta(g)$
is the RG beta function of the coupling $1/g$.

### Where single-parameter scaling is modified

- **Quantum Hall systems.** In a strong field (unitary class) almost all states in a disordered Landau level are localized, which produces the plateaus of the [quantum Hall effect](emergent-phases.html#quantum-hall-effects), but one critical energy per level carries extended states. The plateau transition is a two-parameter flow in $(\sigma_{xx}, \sigma_{xy})$; numerical studies of the Chalker-Coddington network model give $\nu \approx 2.6$, and the critical theory is still not established.
- **Interacting 2D systems.** Clean, low-density silicon MOSFETs and other 2D carrier systems show an apparent metal-insulator transition (Kravchenko and co-workers, 1994 onward), contrary to the non-interacting prediction. Strong interactions ($r_s \gg 1$) are believed responsible; the nature of the metallic phase remains debated.
- **Topological protection.** Surface states of 3D topological insulators and edge states of quantum spin Hall insulators are not localized by time-reversal-symmetric disorder. The quantum spin Hall edge, for example, cannot backscatter without breaking time-reversal symmetry.

## Experimental realizations

Anderson localization is a wave phenomenon and has been observed well beyond
electrons in solids:

| System | Observation | Notes |
|---|---|---|
| Doped semiconductors (Si:P, Si:B, Ge:Sb) | Density-tuned metal-insulator transition; variable-range hopping | Coulomb interactions always present |
| Thin metal films, 2D electron gases, graphene | Weak localization and antilocalization magnetoresistance | Standard measurement of $L_\phi$ |
| Light in disordered photonic lattices | Transverse localization of a beam (2007) | Clean control of disorder; earlier 3D claims were contested because absorption mimics localization |
| Ultrasound in elastic networks | 3D localization of classical waves (2008) | No interactions, no dephasing |
| Ultracold atoms | 1D localization of Bose-Einstein condensates in speckle and quasi-periodic potentials (2008); 3D localization and mobility-edge measurements (2011-2015) | Direct imaging of exponential profiles; tunable interactions |

## Many-body localization

Everything above concerns non-interacting particles, with interactions entering
only as a source of dephasing. **Many-body localization (MBL)** asks whether
localization survives in an isolated, interacting system at nonzero energy
density. If it does, the system never reaches thermal equilibrium on its own.

### Thermalization and its failure

An isolated quantum many-body system is normally its own heat bath: under unitary
evolution, small subsystems approach a thermal state. The **eigenstate
thermalization hypothesis** (ETH) explains this by positing that individual highly
excited eigenstates already look thermal to local observables. MBL is the proposed
failure of ETH. Basko, Aleiner, and Altshuler (2006) argued perturbatively that
weakly interacting electrons with localized single-particle states remain
insulating up to a finite temperature, and numerical studies of spin chains such
as the random-field Heisenberg model

$$H = J\sum_i \mathbf{S}_i\cdot\mathbf{S}_{i+1} + \sum_i h_i S_i^z, \qquad h_i \in [-W, W],$$

found an apparent transition from ergodic to localized behaviour near
$W/J \approx 3.5$-$4$.

| Property | Thermal (ETH) | Anderson (non-interacting) | Many-body localized |
|---|---|---|---|
| Local memory of initial state | lost | retained | retained |
| dc transport at $T > 0$ | yes | none without a bath | none |
| Eigenstate entanglement | volume law | area law | area law |
| Entanglement growth after a quench | linear in $t$ | saturates quickly | logarithmic in $t$ |
| Level statistics | Wigner-Dyson | Poisson | Poisson |

### Phenomenology of the MBL regime

Deep in the localized regime the Hamiltonian can be written in terms of an
extensive set of quasi-local conserved quantities $\tau_i^z$, the **l-bits**,
which are dressed versions of the local degrees of freedom:

$$H = \sum_i h_i\,\tau_i^z + \sum_{ij}J_{ij}\,\tau_i^z\tau_j^z + \sum_{ijk}K_{ijk}\,\tau_i^z\tau_j^z\tau_k^z + \cdots,$$

with couplings decaying exponentially with distance. Because each $\tau_i^z$ is
conserved, nothing is transported, yet the exponentially small couplings
$J_{ij}$ dephase distant l-bits over times $t \sim e^{|i-j|/\xi}$. That is why
entanglement grows as $S(t) \sim \ln t$ even though energy and particles do not
move, the sharpest difference between MBL and Anderson localization.

### Experiments

Signatures of MBL have been reported in several quantum simulators: persistent
density imbalance of fermions in a 1D quasi-periodic optical lattice (Munich,
2015) and of bosons in 2D (2016), and localization with slow entanglement growth
in trapped-ion chains and superconducting-qubit processors. Related
disorder-free forms exist: **Stark MBL** in a linear potential, and
**quasi-periodic** localization (Aubry-André), which has a sharp transition even
in 1D. Many-body localization was also the first proposed mechanism for
stabilizing **discrete time crystals**, observed in trapped ions and NV-centre
ensembles in 2017 and on a superconducting processor in 2021-22.

### The stability debate

Whether MBL is a true phase of matter in the thermodynamic limit, rather than a
very long-lived transient, has been the central question since about 2020.

- **Finite-size drifts.** Šuntajs, Bonča, Prosen, and Vidmar (2020) pointed out that the apparent critical disorder in exact-diagonalization studies drifts upward with system size, so the thermodynamic-limit transition could lie at much stronger disorder, or not exist.
- **Avalanches.** De Roeck and Huveneers argued that rare, locally ergodic regions can act as baths that thermalize their surroundings, with the thermal region growing like an avalanche. This destabilizes MBL in $d > 1$ and with sufficiently slowly decaying interactions, and in 1D pushes the critical disorder well above the early estimates.
- **Prethermal MBL.** The emerging picture distinguishes a **prethermal MBL regime**, in which dynamics are frozen on all experimentally and numerically accessible time scales, from asymptotic MBL at infinite times. The existence of the latter in 1D at very strong disorder is still unresolved, and in two or more dimensions it is widely thought not to exist.

For experiments and quantum devices, which always operate at finite times, the
prethermal regime already delivers the practical payoff: slow relaxation that
protects local quantum information and non-equilibrium order. Sierant, Lewenstein,
Scardicchio, Vidmar, and Zakrzewski, "Many-body localization in the age of
classical computing" (*Rep. Prog. Phys.*, 2025), review the current state of the
problem.

## References

1. P. W. Anderson, "Absence of diffusion in certain random lattices," *Phys. Rev.* 109, 1492 (1958).
2. E. Abrahams, P. W. Anderson, D. C. Licciardello, T. V. Ramakrishnan, "Scaling theory of localization," *Phys. Rev. Lett.* 42, 673 (1979).
3. P. A. Lee and T. V. Ramakrishnan, "Disordered electronic systems," *Rev. Mod. Phys.* 57, 287 (1985).
4. F. Evers and A. D. Mirlin, "Anderson transitions," *Rev. Mod. Phys.* 80, 1355 (2008).
5. E. Akkermans and G. Montambaux, *Mesoscopic Physics of Electrons and Photons* (2007).
6. B. I. Shklovskii and A. L. Efros, *Electronic Properties of Doped Semiconductors* (1984).
7. D. A. Abanin, E. Altman, I. Bloch, M. Serbyn, "Colloquium: Many-body localization, thermalization, and entanglement," *Rev. Mod. Phys.* 91, 021001 (2019).

## See Also

- [Condensed Matter Physics (Hub)](./) — crystal structure, band theory, and the Bloch states that disorder localizes.
- [Metals & Magnetism](metals-and-magnetism.html) — Drude and Fermi-liquid theory of the clean metal.
- [Superconductivity, Quantum Hall & Topological Phases](emergent-phases.html) — quantum Hall plateaus rely on localized states between extended ones.
- [Graduate-Level Formalism](advanced-formalism.html) — Green's functions, the tenfold way, and scaling theory of quantum phase transitions.
- [Experimental Techniques](experimental-techniques.html) — magnetotransport and the measurement of $L_\phi$.
- [Quantum Field Theory](../quantum-field-theory.html) — field-theoretic methods, including the nonlinear sigma model.
- [Computational Physics](../computational-physics/) — exact diagonalization and transfer-matrix methods used to map localization transitions.
