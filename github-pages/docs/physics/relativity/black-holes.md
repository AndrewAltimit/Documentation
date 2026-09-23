---
layout: docs
title: "Relativity: Black Holes"
description: "Schwarzschild, Reissner–Nordström and Kerr black holes: horizons, orbits, singularities, Penrose diagrams, black-hole thermodynamics, the information paradox, and the observational record through 2025."
permalink: /docs/physics/relativity/black-holes.html
toc: true
toc_sticky: true
---

[Relativity](./) &raquo; [Graduate Topics](advanced.html) &raquo; Black Holes

## Black Holes

A **black hole** is a region of spacetime from which no causal signal can reach distant observers; its boundary is the **event horizon**. This page covers the stationary black-hole solutions of general relativity — **Schwarzschild** (mass only), **Reissner–Nordström** (mass and charge) and **Kerr** (mass and spin) — together with their horizons, orbits and singularities, the Penrose diagrams that summarize their causal structure, the laws of **black-hole thermodynamics**, the **information paradox**, and the present observational evidence. It assumes [General Relativity](general-relativity.html) and uses the language of [Tensor Formalism](tensor-formalism.html).

**Conventions.** Geometric units $G = c = 1$ throughout unless SI factors are shown explicitly, so mass, length and time share a dimension ($M_\odot \approx 1.48\ \text{km} \approx 4.93\ \mu\text{s}$). Signature $(-,+,+,+)$; $d\Omega^2 = d\theta^2 + \sin^2\theta\, d\phi^2$ is the unit 2-sphere metric. Charge $Q$ is in geometrized Gaussian units and spin enters through $a = J/M$. To restore SI units substitute $M \to GM/c^2$, $Q^2 \to GQ^2/(4\pi\varepsilon_0 c^4)$ and $a \to J/(Mc)$.

### The stationary family at a glance

| Solution | Parameters | Horizons | Singularity | Found |
|----------|-----------|----------|-------------|-------|
| Schwarzschild | $M$ | $r = 2M$ | Spacelike point-like, $r = 0$ | 1916 |
| Reissner–Nordström | $M, Q$ | $r_\pm = M \pm \sqrt{M^2 - Q^2}$ | Timelike, $r = 0$ | 1916–1918 |
| Kerr | $M, J$ | $r_\pm = M \pm \sqrt{M^2 - a^2}$ | Timelike ring, $r = 0,\ \theta = \pi/2$ | 1963 |
| Kerr–Newman | $M, J, Q$ | $r_\pm = M \pm \sqrt{M^2 - a^2 - Q^2}$ | Timelike ring | 1965 |

Astrophysical black holes are described, to excellent approximation, by the Kerr metric: any net charge is rapidly neutralized by surrounding plasma, while angular momentum is inherited from the collapsing matter and is hard to shed.

## The Schwarzschild Solution

Karl Schwarzschild found the static, spherically symmetric vacuum solution in 1916, weeks after Einstein published the field equations:

$$ds^2 = -\left(1-\frac{2M}{r}\right)dt^2 + \left(1-\frac{2M}{r}\right)^{-1}dr^2 + r^2\,d\Omega^2 .$$

By **Birkhoff's theorem** this is the *unique* spherically symmetric vacuum solution, and it is automatically static. It therefore describes the exterior of any non-rotating spherical body — a star, a planet, or a black hole — even one that is pulsating or collapsing radially, since spherically symmetric motion cannot radiate gravitational waves. For $r \gg 2M$ the metric approaches Minkowski space and $g_{tt} \approx -(1 + 2\Phi)$ with Newtonian potential $\Phi = -M/r$.

### Schwarzschild radius and event horizon

The metric factor $1 - 2M/r$ vanishes at the **Schwarzschild radius**

$$r_s = 2M = \frac{2GM}{c^2} \approx 2.95\ \text{km}\times\frac{M}{M_\odot}.$$

For the Earth $r_s \approx 9\ \text{mm}$. A body compressed inside its Schwarzschild radius must collapse, and the sphere $r = r_s$ becomes its **event horizon**: the boundary of the set of events that can send signals to future null infinity.

The horizon is a **coordinate singularity**, not a physical one. The curvature invariant (Kretschmann scalar)

$$K = R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} = \frac{48M^2}{r^6}$$

is finite at $r = 2M$, and an infalling observer crosses the horizon in finite proper time with nothing locally remarkable happening. The divergence of $g_{rr}$ reflects only the failure of the static coordinates, much as longitude fails at the poles.

**Tidal forces at the horizon.** The radial tidal acceleration across a body of length $L$ is $\Delta a \approx 2GML/r^3$. Evaluated at $r = r_s$ it scales as $1/M^2$:

| Black hole | Mass | Tidal stretch across 2 m at the horizon |
|------------|------|-----------------------------------------|
| Stellar | $10\,M_\odot$ | $\sim 2\times10^{8}\ \text{m/s}^2$ (lethal long before the horizon) |
| Sagittarius A* | $4.3\times10^{6}\,M_\odot$ | $\sim 10^{-3}\ \text{m/s}^2$ (imperceptible) |
| M87* | $6.5\times10^{9}\,M_\odot$ | $\sim 5\times10^{-10}\ \text{m/s}^2$ |

Large black holes can therefore be entered intact; small ones "spaghettify" infalling matter well outside the horizon.

### Orbits, the photon sphere and the ISCO

Geodesics in Schwarzschild conserve energy $E = (1 - 2M/r)\,dt/d\tau$ and angular momentum $L = r^2\,d\phi/d\tau$ per unit mass. Radial motion then reduces to a one-dimensional problem,

$$\left(\frac{dr}{d\tau}\right)^2 = E^2 - V_{\text{eff}}(r), \qquad V_{\text{eff}}(r) = \left(1 - \frac{2M}{r}\right)\left(\epsilon + \frac{L^2}{r^2}\right),$$

with $\epsilon = 1$ for massive particles and $\epsilon = 0$ for light. The extra $-2ML^2/r^3$ term relative to Newtonian gravity is responsible for perihelion precession and, near the hole, for two characteristic radii:

- **Photon sphere**, $r = 3M$: the unstable circular orbit of light. Photons with impact parameter $b < b_c = 3\sqrt{3}\,M$ are captured, so a black hole backlit by a luminous source casts a **shadow** of angular radius $\approx 3\sqrt{3}\,M/D$ — about $2.6$ times the horizon's coordinate size.
- **Innermost stable circular orbit (ISCO)**, $r = 6M$: circular orbits exist for massive particles down to $r = 3M$, but they are unstable inside $6M$. The ISCO sets the inner edge of a thin accretion disk and hence the **radiative efficiency** $\eta = 1 - E_{\text{ISCO}} = 1 - \sqrt{8/9} \approx 5.7\%$ — the fraction of rest-mass energy radiated by gas spiralling in to the ISCO. Nuclear fusion, by comparison, releases $0.7\%$.

Rotation shifts these radii substantially (see [Kerr orbits](#kerr-orbits-and-radiative-efficiency) below).

### Eddington–Finkelstein and Kruskal–Szekeres coordinates

Define the **tortoise coordinate** and the ingoing null coordinate

$$r_* = r + 2M \ln\left|\frac{r}{2M} - 1\right|, \qquad v = t + r_* .$$

In **ingoing Eddington–Finkelstein coordinates** $(v, r, \theta, \phi)$,

$$ds^2 = -\left(1-\frac{2M}{r}\right)dv^2 + 2\,dv\,dr + r^2\,d\Omega^2 ,$$

which is regular at $r = 2M$ ($\det g = -r^4\sin^2\theta$). Ingoing radial light rays are the lines $v = \text{const}$; the outgoing family tilts over and, for $r < 2M$, also moves to smaller $r$ — the horizon is where outgoing light stands still.

**Kruskal–Szekeres coordinates** $(T, X)$ cover the entire **maximal analytic extension**. In the exterior region ($r > 2M$):

$$T = \sqrt{\frac{r}{2M}-1}\; e^{r/4M}\sinh\frac{t}{4M}, \qquad X = \sqrt{\frac{r}{2M}-1}\; e^{r/4M}\cosh\frac{t}{4M},$$

$$ds^2 = \frac{32M^3}{r}\,e^{-r/2M}\left(-dT^2 + dX^2\right) + r^2\,d\Omega^2, \qquad X^2 - T^2 = \left(\frac{r}{2M} - 1\right)e^{r/2M}.$$

Radial light rays travel at $45^\circ$, the horizon is the pair of null lines $X = \pm T$, and the singularity $r = 0$ is the hyperbola $T^2 - X^2 = 1$. The extension has four regions: our exterior (I), the black-hole interior (II), a time-reversed **white hole** (IV), and a second exterior (III) joined to ours through the **Einstein–Rosen bridge**, a non-traversable wormhole. Regions III and IV are artefacts of an *eternal* black hole; a hole formed by collapse has only regions I and II (see [Penrose diagrams](#penrose-diagrams)).

### Formation and the singularity theorems

Oppenheimer and Snyder (1939) showed that a homogeneous pressureless ball collapses through its horizon to $r = 0$ in finite proper time. Whether this depended on the idealized symmetry was settled by **Penrose's singularity theorem** (1965): if spacetime contains a **trapped surface** (a closed 2-surface from which both ingoing and outgoing light converge), the null energy condition holds, and space has a non-compact Cauchy surface, then spacetime is **null-geodesically incomplete** — some light ray ends after finite affine parameter. Hawking and Penrose extended the argument in 1970. The theorems predict *that* singularities form, not what they are like; this work earned Penrose half of the 2020 Nobel Prize in Physics.

### The central singularity

At $r = 0$ the Kretschmann scalar diverges: this is a genuine curvature singularity where tidal forces become infinite and classical general relativity stops making predictions. Inside the horizon $g_{tt}$ and $g_{rr}$ change sign, so $r$ is a *time* coordinate: decreasing $r$ is as unavoidable as the passage of time, and the singularity is a moment in every interior observer's future, not a place that can be avoided.

**Worked example: time to the singularity.** For radial free fall from rest at infinity ($E = 1$), $dr/d\tau = -\sqrt{2M/r}$, so the proper time from horizon to singularity is

$$\Delta\tau = \int_0^{2M}\sqrt{\frac{r}{2M}}\,dr = \frac{4M}{3} = \frac{4GM}{3c^3}.$$

The longest possible proper time inside is $\pi M$, achieved by falling freely from rest *at* the horizon; firing rockets in any direction only shortens it. In SI units $\pi GM/c^3 \approx 1.5\times10^{-4}\ \text{s}$ for $M = 10\,M_\odot$ and about $66\ \text{s}$ for Sagittarius A* ($4.3\times10^6\,M_\odot$).

## The Reissner–Nordström Solution

Adding electric charge gives the **Reissner–Nordström** (RN) solution, the unique static, spherically symmetric solution of the Einstein–Maxwell equations:

$$ds^2 = -f(r)\,dt^2 + f(r)^{-1}dr^2 + r^2\,d\Omega^2, \qquad f(r) = 1-\frac{2M}{r}+\frac{Q^2}{r^2},$$

with potential $A_\mu\,dx^\mu = -(Q/r)\,dt$. The roots of $f(r) = 0$ are

$$r_\pm = M \pm \sqrt{M^2 - Q^2}.$$

- **Outer (event) horizon** $r_+$: the surface of no return.
- **Inner (Cauchy) horizon** $r_-$: beyond it the future is no longer determined by initial data, because the singularity is **timelike** and can inject arbitrary information. Radiation falling in behind an observer is infinitely blueshifted at $r_-$, driving **mass inflation** (Poisson–Israel, 1990); the smooth RN interior is not expected to survive in realistic collapse. Whether the resulting singularity is strong enough to enforce *strong cosmic censorship* is still debated, and the answer is known to depend on the matter content and on a positive cosmological constant.

| Regime | Condition | Structure |
|--------|-----------|-----------|
| Sub-extremal | $\lvert Q\rvert < M$ | Two horizons $r_- < r_+$; timelike singularity at $r=0$ |
| Extremal | $\lvert Q\rvert = M$ | One degenerate horizon at $r = M$; zero surface gravity and temperature |
| Super-extremal | $\lvert Q\rvert > M$ | No horizon: a **naked singularity** |

**Cosmic censorship.** Penrose's *weak* cosmic censorship conjecture asserts that generic gravitational collapse of reasonable matter never produces a singularity visible from infinity. It is unproven; fine-tuned counterexamples exist (e.g. critical collapse at the threshold of black-hole formation), but attempts to overcharge or overspin a black hole past extremality by throwing in test particles fail once self-force and back-reaction are included. Extremality behaves as an unreachable limit, in parallel with the third law below.

## The Kerr Solution

Roy Kerr found the rotating vacuum solution in 1963. In **Boyer–Lindquist coordinates**,

$$ds^2 = -\left(1-\frac{2Mr}{\rho^2}\right)dt^2 - \frac{4Mar\sin^2\theta}{\rho^2}\,dt\,d\phi + \frac{\rho^2}{\Delta}\,dr^2 + \rho^2\,d\theta^2 + \left(r^2+a^2+\frac{2Ma^2r\sin^2\theta}{\rho^2}\right)\sin^2\theta\,d\phi^2,$$

$$\rho^2 = r^2 + a^2\cos^2\theta, \qquad \Delta = r^2 - 2Mr + a^2, \qquad a = \frac{J}{M}.$$

The off-diagonal $dt\,d\phi$ term encodes **frame dragging**: zero-angular-momentum observers rotate with angular velocity $\omega = -g_{t\phi}/g_{\phi\phi}$ relative to infinity. Setting $a = 0$ recovers Schwarzschild; setting $M = 0$ gives flat space in oblate spheroidal coordinates. Unlike Schwarzschild, Kerr is **not** the unique exterior of a rotating star — real stars carry higher multipole moments — but it is the unique end state of collapse (see [No-hair theorem](#the-no-hair-theorem)).

### Horizons and the ergosphere

The horizons are the roots of $\Delta = 0$,

$$r_\pm = M \pm \sqrt{M^2 - a^2},$$

which exist only for $a \le M$ (the **Kerr bound**, $J \le M^2$ in geometric units or $J \le GM^2/c$ in SI). The **static limit** or ergosurface is where $g_{tt} = 0$:

$$r_{\text{E}}(\theta) = M + \sqrt{M^2 - a^2\cos^2\theta}.$$

It touches the horizon at the poles and bulges out to $r = 2M$ at the equator. The region between it and $r_+$ is the **ergosphere**. There $\partial_t$ is spacelike, so *no observer can remain at rest* relative to distant stars — every worldline must co-rotate — yet escape to infinity is still possible, because the event horizon has not been crossed.

<figure style="margin: 1.5em auto; max-width: 640px;">
<svg viewBox="0 0 640 320" role="img" aria-label="Meridional cross-section of a Kerr black hole with a = 0.9M showing the ergosphere, outer and inner horizons, and ring singularity" style="max-width:640px;width:100%;height:auto;color:inherit;font-family:inherit">
  <line x1="210" y1="18" x2="210" y2="302" stroke="currentColor" stroke-width="1" stroke-dasharray="3 4" opacity="0.6"/>
  <text x="216" y="26" font-size="12" fill="currentColor">spin axis</text>
  <path d="M 210.0 45.1 L 221.9 45.0 L 233.9 44.7 L 246.1 44.5 L 258.7 44.7 L 271.4 45.6 L 284.4 47.3 L 297.3 50.2 L 310.0 54.3 L 322.4 59.8 L 334.2 66.6 L 345.2 74.8 L 355.2 84.3 L 364.1 94.9 L 371.6 106.6 L 377.5 119.2 L 381.9 132.4 L 384.6 146.1 L 385.5 160.0 L 384.6 173.9 L 381.9 187.6 L 377.5 200.8 L 371.6 213.4 L 364.1 225.1 L 355.2 235.7 L 345.2 245.2 L 334.2 253.4 L 322.4 260.2 L 310.0 265.7 L 297.3 269.8 L 284.4 272.7 L 271.4 274.4 L 258.7 275.3 L 246.1 275.5 L 233.9 275.3 L 221.9 275.0 L 210.0 274.9 L 198.1 275.0 L 186.1 275.3 L 173.9 275.5 L 161.3 275.3 L 148.6 274.4 L 135.6 272.7 L 122.7 269.8 L 110.0 265.7 L 97.6 260.2 L 85.8 253.4 L 74.8 245.2 L 64.8 235.7 L 55.9 225.1 L 48.4 213.4 L 42.5 200.8 L 38.1 187.6 L 35.4 173.9 L 34.5 160.0 L 35.4 146.1 L 38.1 132.4 L 42.5 119.2 L 48.4 106.6 L 55.9 94.9 L 64.8 84.3 L 74.8 74.8 L 85.8 66.6 L 97.6 59.8 L 110.0 54.3 L 122.7 50.2 L 135.6 47.3 L 148.6 45.6 L 161.3 44.7 L 173.9 44.5 L 186.1 44.7 L 198.1 45.0 L 210.0 45.1 Z" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1.5" stroke-dasharray="7 4"/>
  <ellipse cx="210" cy="160" rx="135.6" ry="114.9" fill="currentColor" fill-opacity="0.16" stroke="currentColor" stroke-width="2.2"/>
  <ellipse cx="210" cy="160" rx="85.0" ry="45.1" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <circle cx="138.0" cy="160" r="4.5" fill="currentColor"/>
  <circle cx="282.0" cy="160" r="4.5" fill="currentColor"/>
  <line x1="138.0" y1="160" x2="282.0" y2="160" stroke="currentColor" stroke-width="1" stroke-dasharray="2 3" opacity="0.6"/>
  <g font-size="12.5" fill="currentColor" stroke="currentColor" stroke-width="0.8">
    <line x1="374.9" y1="120.0" x2="420" y2="60" fill="none"/>
    <text x="424" y="58" stroke="none">static limit (ergosurface)</text>
    <text x="424" y="73" stroke="none" opacity="0.8">g_tt = 0</text>
    <line x1="318.5" y1="91.1" x2="420" y2="112" fill="none"/>
    <text x="424" y="116" stroke="none">outer (event) horizon r+</text>
    <line x1="269.5" y1="127.5" x2="420" y2="170" fill="none"/>
    <text x="424" y="174" stroke="none">inner (Cauchy) horizon r−</text>
    <line x1="286.0" y1="163" x2="420" y2="258" fill="none"/>
    <text x="424" y="262" stroke="none">ring singularity</text>
    <text x="424" y="277" stroke="none" opacity="0.8">(radius a, seen edge-on)</text>
  </g>
  <text x="210" y="308.9" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.85">ergosphere between dashed and bold surfaces</text>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">Meridional cross-section of a Kerr black hole with $a = 0.9M$, drawn in Kerr–Schild-like coordinates where surfaces of constant $r$ are oblate spheroids. The shaded band between the dashed ergosurface and the outer horizon is the ergosphere; the ring singularity appears as two points at distance $a$ from the axis.</figcaption>
</figure>

### The Penrose process and superradiance

Because the Killing vector $\xi_{(t)} = \partial_t$ is spacelike in the ergosphere, a particle there can have **negative conserved energy** $E = -\xi_{(t)}\cdot p < 0$ as measured from infinity. In the **Penrose process** (1969) a particle entering the ergosphere splits in two; one fragment falls through the horizon on a negative-energy orbit and the other escapes with *more* energy than the original. The difference comes from the hole's rotational energy, and the hole spins down.

The wave analogue is **superradiance**: a mode $e^{-i\omega t + im\phi}$ scattering off the hole is amplified when

$$0 < \omega < m\,\Omega_H, \qquad \Omega_H = \frac{a}{r_+^2 + a^2},$$

where $\Omega_H$ is the angular velocity of the horizon. If a massive bosonic field has a Compton wavelength comparable to the hole's size, superradiant modes can be trapped and grow exponentially into a **boson cloud**, spinning the hole down. The observed spins of stellar and supermassive black holes are therefore used to exclude ranges of ultralight axion-like particle masses.

### Irreducible mass

Energy extraction cannot reduce the horizon area (see the [area theorem](#the-four-laws)). Christodoulou (1970) showed that this leaves an **irreducible mass**

$$M_{\text{irr}}^2 = \frac{A}{16\pi} = \frac{1}{2}\left(M^2 + \sqrt{M^4 - J^2}\right), \qquad M^2 = M_{\text{irr}}^2 + \frac{J^2}{4M_{\text{irr}}^2}.$$

For a maximally rotating hole ($J = M^2$), $M_{\text{irr}} = M/\sqrt{2}$, so up to $1 - 1/\sqrt{2} \approx 29\%$ of the mass-energy is extractable rotational energy. Magnetic analogues of the Penrose process — above all the **Blandford–Znajek mechanism** (1977), in which magnetic field lines threading the horizon are wound up by frame dragging — are the leading explanation for the relativistic jets of active galactic nuclei and are the standard engine in general-relativistic MHD simulations of M87*.

### Kerr orbits and radiative efficiency

Rotation pulls prograde orbits inward and pushes retrograde orbits outward:

| Spin | Horizon $r_+$ | Prograde photon orbit | Prograde ISCO | Efficiency $\eta$ | Retrograde ISCO |
|------|---------------|-----------------------|---------------|-------------------|-----------------|
| $a = 0$ | $2M$ | $3M$ | $6M$ | $5.7\%$ | $6M$ |
| $a = M$ | $M$ | $M$ | $M$ | $1 - 1/\sqrt{3} \approx 42\%$ | $9M$ ($\eta \approx 3.8\%$) |

(For $a \to M$ the Boyer–Lindquist radii coincide, but the proper distances between horizon, photon orbit and ISCO remain finite and distinct.) Because the ISCO depends so strongly on spin, X-ray **continuum fitting** and relativistically broadened **iron-line** spectroscopy of the inner disk are the main electromagnetic spin measurements. Thorne (1974) showed that photon capture limits disk-fed spin-up to about $a \approx 0.998\,M$.

### The ring singularity

The curvature singularity sits where $\rho^2 = 0$: $r = 0$ **and** $\theta = \pi/2$. In Kerr–Schild coordinates this is a **ring** of radius $a$ in the equatorial plane. The analytic extension continues through the disk bounded by the ring into a region of negative $r$ that contains closed timelike curves. As with the RN inner horizon, this interior structure is believed to be destroyed by mass inflation in any realistic collapse.

### The no-hair theorem

The uniqueness theorems of Israel (1967), Carter (1971), Hawking (1972) and Robinson (1975) establish that a stationary, asymptotically flat, non-singular-outside-the-horizon black hole in Einstein–Maxwell theory is a member of the **Kerr–Newman** family, fixed by three numbers:

$$M, \qquad J, \qquad Q .$$

All other information about the collapsed matter — composition, shape, baryon number — is radiated away as gravitational and electromagnetic waves (the **ringdown**) or hidden behind the horizon. A concrete, testable consequence: every multipole moment of a Kerr hole is fixed by $M$ and $a$, e.g. the mass quadrupole is $Q_2 = -Ma^2$, and every ringdown **quasinormal-mode** frequency is a function of $(M, a)$ alone. Measuring two or more modes and checking that they imply the same $(M, a)$ — **black-hole spectroscopy** — is a direct test of the Kerr hypothesis. The theorem assumes four-dimensional Einstein–Maxwell theory; with other matter fields (Yang–Mills, some scalar couplings) "hairy" black holes exist.

## Penrose Diagrams

A **Penrose (conformal) diagram** compactifies spacetime by a conformal rescaling $\tilde g_{\mu\nu} = \Omega^2 g_{\mu\nu}$ that brings infinity to a finite distance while preserving null geodesics. With spherical symmetry suppressed (each point is a 2-sphere), radial light rays run at $45^\circ$ and the whole causal structure can be read off by eye. The boundary of an asymptotically flat spacetime consists of:

| Symbol | Name | What reaches it |
|--------|------|-----------------|
| $i^+$ / $i^-$ | future / past timelike infinity | Endpoints / origins of massive worldlines |
| $i^0$ | spatial infinity | Ends of spacelike slices |
| $\mathscr{I}^+$ / $\mathscr{I}^-$ | future / past null infinity ("scri") | Outgoing / incoming light rays |

<figure style="margin: 1.5em auto; max-width: 720px;">
<svg viewBox="0 0 720 360" role="img" aria-label="Penrose diagrams: left, radial Minkowski spacetime as a triangle; right, the maximally extended Schwarzschild spacetime with exterior regions I and III, black-hole interior II, white hole IV, and spacelike singularities at top and bottom" style="max-width:720px;width:100%;height:auto;color:inherit;font-family:inherit">
  <g stroke="currentColor" fill="none" stroke-linecap="round">
    <!-- Minkowski (radial) -->
    <path d="M 110 50 L 250 190 L 110 330 Z" stroke-width="2"/>
    <path d="M 110 330 C 150 270, 160 200, 110 50" stroke-width="2.2"/>
    <path d="M 110 260 L 215 155" stroke-width="1.5" stroke-dasharray="5 4"/>
    <path d="M 215 225 L 110 120 L 145 85" stroke-width="1.5" stroke-dasharray="5 4"/>
    <!-- Schwarzschild -->
    <path d="M 450 100 L 610 260 M 450 260 L 610 100" stroke-width="1.8"/>
    <path d="M 610 100 L 690 180 L 610 260 M 450 100 L 370 180 L 450 260" stroke-width="2"/>
    <path d="M 450 100 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7" stroke-width="2"/>
    <path d="M 450 260 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7 l 10 7 l 10 -7" stroke-width="2"/>
    <path d="M 610 260 Q 650 190 575 101" stroke-width="2.4"/>
      </g>
  <path d="M 450 100 L 530 180 L 610 100 Z" fill="currentColor" fill-opacity="0.12"/>
  <path d="M 450 260 L 530 180 L 610 260 Z" fill="currentColor" fill-opacity="0.05"/>
  <g font-size="13" fill="currentColor" text-anchor="middle">
    <text x="175" y="22" font-weight="bold">Minkowski (radial)</text>
    <text x="110" y="42">i⁺</text>
    <text x="110" y="348">i⁻</text>
    <text x="264" y="194" text-anchor="start">i⁰</text>
    <text x="195" y="108" font-size="16" transform="rotate(45 195 108)">ℐ⁺</text>
    <text x="195" y="282" font-size="16" transform="rotate(-45 195 282)">ℐ⁻</text>
    <text x="96" y="190" transform="rotate(-90 96 190)">r = 0</text>
    <text x="158" y="252" font-size="11.5" text-anchor="start" opacity="0.85">worldline</text>
    <text x="530" y="22" font-weight="bold">Schwarzschild (maximal extension)</text>
    <text x="530" y="78">singularity r = 0</text>
    <text x="530" y="298">singularity r = 0</text>
    <text x="530" y="138">II black hole</text>
    <text x="530" y="232">IV white hole</text>
    <text x="668" y="185">I</text>
    <text x="420" y="185">III</text>
    <text x="700" y="184" text-anchor="start">i⁰</text>
    <text x="350" y="184" text-anchor="end">i⁰</text>
    <text x="655" y="130" font-size="16" transform="rotate(45 655 130)">ℐ⁺</text>
    <text x="655" y="238" font-size="16" transform="rotate(-45 655 238)">ℐ⁻</text>
    <text x="578" y="164" font-size="11.5" transform="rotate(-45 578 164)">r = 2M</text>
    <text x="572" y="214" font-size="11.5" text-anchor="start" opacity="0.85">infaller</text>
  </g>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">Left: radial Minkowski spacetime; dashed lines are light rays (one reflecting through the origin), the solid curve a timelike worldline running from $i^-$ to $i^+$. Right: the maximally extended Schwarzschild spacetime. The diagonals are the horizons $r = 2M$; the zig-zag lines are the spacelike singularities. An observer who crosses from region I into region II must reach the singularity.</figcaption>
</figure>

The Schwarzschild diagram makes three facts visible:

- The **event horizon** is the boundary of the causal past of $\mathscr{I}^+$. Once inside region II, every future-directed ($\le 45^\circ$ from vertical) path ends on the singularity.
- The **singularity is spacelike** (horizontal): it is a moment of time, not a location.
- The eternal solution contains a white hole (IV) and a second exterior (III). A black hole formed by stellar collapse has neither: its diagram is region I and II with the collapsing star's surface replacing the left half.

For **Reissner–Nordström** and **Kerr** the singularity is timelike (vertical) and the maximally extended diagram is an infinite tower of alternating exterior, between-horizon and inner regions. An observer could in principle avoid the singularity and pass through the Cauchy horizon into another universe — precisely the structure that mass inflation is expected to destroy.

**Reading rules:** light moves at $45^\circ$; your causal future is the upward-opening $45^\circ$ wedge; a horizon is a null line crossable in one direction only; a horizontal singularity is unavoidable, a vertical one is avoidable in principle.

## Black-Hole Thermodynamics

In the early 1970s Bardeen, Carter and Hawking proved four laws of black-hole mechanics with the same structure as thermodynamics. Bekenstein (1972) argued that the analogy must be literal — otherwise dropping entropy into a black hole would violate the second law — and Hawking's discovery of black-hole radiation (1974) fixed the constants.

### The four laws

| Law | Thermodynamics | Black holes |
|-----|----------------|-------------|
| Zeroth | $T$ uniform in equilibrium | Surface gravity $\kappa$ constant over a stationary horizon |
| First | $dE = T\,dS + \text{work terms}$ | $dM = \dfrac{\kappa}{8\pi}\,dA + \Omega_H\,dJ + \Phi_H\,dQ$ |
| Second | $dS \ge 0$ | $dA \ge 0$ (Hawking area theorem, classical) |
| Third | $T = 0$ unreachable in finitely many steps | $\kappa = 0$ (extremality) unreachable by a finite process |

Here $\kappa$ is the **surface gravity** — the force per unit mass, measured at infinity, needed to hold a particle static just outside the horizon — $A$ the horizon area, $\Omega_H$ the horizon angular velocity and $\Phi_H$ the horizon electric potential. For Kerr–Newman

$$\kappa = \frac{r_+ - r_-}{2\left(r_+^2 + a^2\right)}, \qquad A = 4\pi\left(r_+^2 + a^2\right),$$

reducing to $\kappa = 1/4M$ and $A = 16\pi M^2$ for Schwarzschild, where one checks directly that $dM = (\kappa/8\pi)\,dA$. Integrating the first law with Euler's theorem for homogeneous functions gives the **Smarr formula**

$$M = \frac{\kappa A}{4\pi} + 2\,\Omega_H J + \Phi_H Q .$$

The area theorem was tested directly in 2025: in the gravitational-wave event **GW250114** the final horizon area inferred from the ringdown exceeded the summed areas of the two initial black holes, as required, at a significance above $4\sigma$.

### Hawking temperature

Quantizing matter fields on a collapsing black-hole background, Hawking (1974) found that the horizon emits a thermal flux at the **Hawking temperature**

$$T_H = \frac{\hbar\,\kappa}{2\pi\,c\,k_B}, \qquad T_H^{\text{Schw}} = \frac{\hbar c^3}{8\pi G M k_B} \approx 6.2 \times 10^{-8}\ \text{K} \times \frac{M_\odot}{M}.$$

The temperature is inversely proportional to mass. Every astrophysical black hole is far colder than the $2.7\ \text{K}$ cosmic microwave background and today absorbs more than it emits. The spectrum at infinity is a black body modified by **greybody factors**, the frequency-dependent transmission through the curvature potential around the hole.

**Origin of the radiation.** The picture of virtual pairs split at the horizon is a heuristic. The precise statement is that the notion of particle is observer-dependent: the quantum state that is regular (vacuum-like) for freely falling observers at the horizon is thermally populated for static observers far away. The same effect in flat space is the **Unruh effect** — a uniformly accelerated observer in the Minkowski vacuum sees a thermal bath at $T_U = \hbar a/(2\pi c k_B)$ — and $\kappa$ plays the role of the acceleration. Hawking radiation has not been observed from any astrophysical black hole, but its kinematics have been reproduced in analogue systems; in Bose–Einstein-condensate "sonic black holes" Steinhauer's group reported spontaneous Hawking phonons with a thermal spectrum at the predicted temperature (2016, 2019).

### Bekenstein–Hawking entropy

With $T_H$ fixed, the first law determines the **Bekenstein–Hawking entropy**

$$S_{BH} = \frac{k_B\,c^3\,A}{4\,G\,\hbar} = \frac{k_B\,A}{4\,\ell_P^2}, \qquad \ell_P = \sqrt{\frac{G\hbar}{c^3}} \approx 1.6\times10^{-35}\ \text{m}.$$

The entropy is one quarter of the horizon area in Planck units. Ordinary thermodynamic entropy is extensive, scaling with volume; black-hole entropy scales with area. This is the basis of the **holographic principle** — the maximum information content of a region is bounded by its boundary area — made precise in AdS/CFT (see [Toward Quantum Gravity](quantum-gravity.html#the-holographic-principle)). For certain supersymmetric extremal black holes in string theory, Strominger and Vafa (1996) counted the microstates and reproduced $A/4$ exactly.

**Worked example: a stellar black hole.** For $M = 10\,M_\odot$, $r_s \approx 2.95\times10^4\ \text{m}$, $A = 4\pi r_s^2 \approx 1.1\times10^{10}\ \text{m}^2$, and $\ell_P^2 \approx 2.6\times10^{-70}\ \text{m}^2$:

$$S_{BH} \approx \frac{1.1\times10^{10}}{4 \times 2.6\times10^{-70}}\,k_B \approx 10^{79}\,k_B .$$

The progenitor star had an entropy of order $10^{58}\,k_B$; collapse increased it by some twenty orders of magnitude. The supermassive black holes dominate the entropy budget of the observable universe.

### Evaporation and lifetime

Losing energy raises $T_H$, so black holes have **negative heat capacity** and evaporation runs away. Treating the hole as a black body of area $\propto M^2$ at $T_H \propto 1/M$ gives $dM/dt \propto -M^{-2}$ and a finite lifetime. For emission of photons only,

$$t_{\text{evap}} = \frac{5120\,\pi\,G^2 M^3}{\hbar\,c^4} \approx 2.1\times10^{67}\ \text{yr}\times\left(\frac{M}{M_\odot}\right)^3 ,$$

with order-unity corrections once greybody factors and all particle species light enough to be emitted are included. Primordial black holes with initial mass near $5\times10^{11}\ \text{kg}$ would be completing their evaporation today, ending in a burst of gamma rays; searches by Fermi-LAT and ground-based Cherenkov and water-Cherenkov observatories have not found such bursts and bound their local rate. The final Planck-scale stage lies outside known physics.

## The Information Paradox

If a black hole formed from matter in a pure quantum state evaporates completely into **exactly thermal** radiation, the final state is mixed. Unitary quantum evolution never maps a pure state to a mixed one, so either information is lost — a modification of quantum mechanics — or Hawking's semiclassical calculation fails somewhere it was expected to be reliable. This is the **black-hole information paradox** (Hawking, 1976).

The sharpest form is an entanglement argument:

1. **No drama.** By the equivalence principle, an infalling observer sees vacuum at a large horizon, which requires each outgoing Hawking mode to be entangled with a partner mode just inside.
2. **Unitarity.** For the final radiation to be pure, late Hawking quanta must be entangled with the *earlier* radiation.
3. **Monogamy of entanglement.** A system cannot be maximally entangled with two others at once.

Page (1993) showed that unitarity requires the entanglement entropy of the radiation to follow the **Page curve**: rising while the hole is young, peaking at the **Page time** (roughly when the hole has lost half its initial Bekenstein–Hawking entropy), then falling to zero. Hawking's calculation instead gives a monotonically increasing entropy.

<figure style="margin: 1.5em auto; max-width: 540px;">
<svg viewBox="0 0 540 340" role="img" aria-label="Schematic Page curve: Hawking's semiclassical radiation entropy rises monotonically, the black hole's Bekenstein-Hawking entropy falls, and the unitary Page curve follows the minimum of the two, peaking at the Page time" style="max-width:540px;width:100%;height:auto;color:inherit;font-family:inherit">
  <g stroke="currentColor" fill="none">
    <line x1="60" y1="250" x2="490" y2="250" stroke-width="1.5"/>
    <line x1="60" y1="250" x2="60" y2="30" stroke-width="1.5"/>
    <path d="M 60.0 250.0 L 68.8 247.2 L 77.5 244.4 L 86.2 241.6 L 95.0 238.7 L 103.8 235.9 L 112.5 233.0 L 121.2 230.0 L 130.0 227.1 L 138.8 224.1 L 147.5 221.2 L 156.2 218.1 L 165.0 215.1 L 173.8 212.0 L 182.5 208.9 L 191.2 205.8 L 200.0 202.6 L 208.7 199.4 L 217.5 196.2 L 226.2 192.9 L 235.0 189.6 L 243.8 186.3 L 252.5 182.9 L 261.2 179.5 L 270.0 176.0 L 278.8 172.5 L 287.5 168.9 L 296.2 165.3 L 305.0 161.6 L 313.8 157.8 L 322.5 154.0 L 331.2 150.1 L 340.0 146.1 L 348.8 142.1 L 357.5 138.0 L 366.2 133.7 L 375.0 129.4 L 383.7 124.9 L 392.5 120.3 L 401.2 115.5 L 410.0 110.6 L 418.8 105.4 L 427.5 100.0 L 436.2 94.3 L 445.0 88.2 L 453.8 81.5 L 462.5 74.0 L 471.2 65.1 L 480.0 50.0" stroke-width="2" stroke-dasharray="7 5"/>
    <path d="M 60.0 50.0 L 68.8 52.8 L 77.5 55.6 L 86.2 58.4 L 95.0 61.3 L 103.8 64.1 L 112.5 67.0 L 121.2 70.0 L 130.0 72.9 L 138.8 75.9 L 147.5 78.8 L 156.2 81.9 L 165.0 84.9 L 173.8 88.0 L 182.5 91.1 L 191.2 94.2 L 200.0 97.4 L 208.7 100.6 L 217.5 103.8 L 226.2 107.1 L 235.0 110.4 L 243.8 113.7 L 252.5 117.1 L 261.2 120.5 L 270.0 124.0 L 278.8 127.5 L 287.5 131.1 L 296.2 134.7 L 305.0 138.4 L 313.8 142.2 L 322.5 146.0 L 331.2 149.9 L 340.0 153.9 L 348.8 157.9 L 357.5 162.0 L 366.2 166.3 L 375.0 170.6 L 383.7 175.1 L 392.5 179.7 L 401.2 184.5 L 410.0 189.4 L 418.8 194.6 L 427.5 200.0 L 436.2 205.7 L 445.0 211.8 L 453.8 218.5 L 462.5 226.0 L 471.2 234.9 L 480.0 250.0" stroke-width="1.6" stroke-dasharray="2 4"/>
    <path d="M 60.0 250.0 L 68.8 247.2 L 77.5 244.4 L 86.2 241.6 L 95.0 238.7 L 103.8 235.9 L 112.5 233.0 L 121.2 230.0 L 130.0 227.1 L 138.8 224.1 L 147.5 221.2 L 156.2 218.1 L 165.0 215.1 L 173.8 212.0 L 182.5 208.9 L 191.2 205.8 L 200.0 202.6 L 208.7 199.4 L 217.5 196.2 L 226.2 192.9 L 235.0 189.6 L 243.8 186.3 L 252.5 182.9 L 261.2 179.5 L 270.0 176.0 L 278.8 172.5 L 287.5 168.9 L 296.2 165.3 L 305.0 161.6 L 313.8 157.8 L 322.5 154.0 L 331.2 150.1 L 340.0 153.9 L 348.8 157.9 L 357.5 162.0 L 366.2 166.3 L 375.0 170.6 L 383.7 175.1 L 392.5 179.7 L 401.2 184.5 L 410.0 189.4 L 418.8 194.6 L 427.5 200.0 L 436.2 205.7 L 445.0 211.8 L 453.8 218.5 L 462.5 226.0 L 471.2 234.9 L 480.0 250.0" stroke-width="3.2"/>
    <line x1="331.5" y1="250" x2="331.5" y2="150.0" stroke-width="1" opacity="0.5"/>
  </g>
  <g font-size="12.5" fill="currentColor">
    <text x="270.0" y="284" text-anchor="middle">time since formation → evaporation</text>
    <text x="46" y="150.0" text-anchor="middle" transform="rotate(-90 46 150.0)">entropy</text>
    <text x="331.5" y="266" text-anchor="middle">Page time</text>
    <text x="476" y="46.0" text-anchor="end">Hawking (semiclassical): keeps rising</text>
    <text x="68" y="46.0">S_BH of the hole</text>
    <text x="257.4" y="230.0" text-anchor="middle" font-weight="bold">Page curve (unitary)</text>
    <text x="480" y="266" text-anchor="end">end</text>
  </g>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">Schematic entanglement entropy of the Hawking radiation. The semiclassical calculation (dashed) keeps rising; unitarity caps the radiation entropy by the black hole's own coarse-grained entropy (dotted), so the true fine-grained entropy follows the Page curve (solid).</figcaption>
</figure>

| Proposal | Year | Idea | Status |
|----------|------|------|--------|
| Black-hole complementarity (Susskind, Thorlacius, Uglum; 't Hooft) | 1993 | Information is both reflected at the horizon and carried inside, but no single observer can compare the two copies | Challenged by AMPS |
| AdS/CFT (Maldacena) | 1997 | Black-hole evaporation in anti-de Sitter space is dual to unitary evolution of a boundary quantum field theory | Strong evidence information is preserved, but no bulk mechanism |
| Firewalls (Almheiri, Marolf, Polchinski, Sully) | 2012 | Preserving unitarity and monogamy after the Page time requires a high-energy barrier at the horizon | Abandons no-drama; widely considered a sign of a missing ingredient |
| ER = EPR (Maldacena, Susskind) | 2013 | Entangled systems are connected by non-traversable wormholes; interior partners are built from the radiation | Conjectural framework |
| Soft hair (Hawking, Perry, Strominger) | 2016 | Horizons carry infinitely many soft charges from asymptotic symmetries | Probably cannot store all the information |
| Islands and replica wormholes (Penington; Almheiri, Engelhardt, Marolf, Maxfield; and others) | 2019 | The fine-grained radiation entropy must include "island" regions inside the horizon, found as quantum extremal surfaces; replica-wormhole saddles in the gravitational path integral justify the rule | Reproduces the Page curve in controlled models |

The island computations show that the semiclassical gravitational path integral, evaluated correctly, already knows about unitarity. What they do not provide is a detailed account of *how* information is encoded in individual Hawking quanta, nor a derivation for realistic four-dimensional, asymptotically flat black holes. The problem remains a principal testing ground for quantum gravity; see [Toward Quantum Gravity](quantum-gravity.html) and [String Theory](../string-theory/).

## Observational Status

Black holes have gone from mathematical curiosities to precisely measured astrophysical objects.

| Evidence | Key results |
|----------|-------------|
| X-ray binaries | Dynamical masses above the neutron-star limit since Cygnus X-1 (early 1970s); modern mass $\approx 21\,M_\odot$. Spins from continuum fitting and iron lines, several near the Kerr bound. |
| Stellar orbits at the Galactic Centre | Two decades of orbit tracking (Keck, VLT/GRAVITY) give Sagittarius A* a mass of $4.3\times10^6\,M_\odot$ at 8.2 kpc; the star S2 shows gravitational redshift and Schwarzschild precession. Nobel Prize 2020 (Genzel, Ghez). |
| Gravitational waves | First detection GW150914 (2015). The LIGO–Virgo–KAGRA fourth observing run (O4) ended on 18 November 2025; its first-part catalog GWTC-4.0 (2025) added about 130 candidates. Notable events: **GW231123** (remnant $\approx 225\,M_\odot$, component spins $\approx 0.9$ and $0.8$, component masses in or above the expected pair-instability gap) and **GW250114** (network SNR $\approx 80$; area theorem confirmed; two ringdown modes consistent with a single Kerr $(M, a)$). |
| Horizon-scale imaging | Event Horizon Telescope images of M87* (2019) and Sagittarius A* (2022), with ring diameters matching the Kerr shadow prediction $\approx 2\sqrt{27}\,M$ to within about 10%. Polarimetric images reveal ordered magnetic fields near both horizons, and multi-year M87* campaigns show a persistent ring whose brightness pattern and polarization change between epochs. |
| Early universe | JWST and Chandra found an X-ray-luminous black hole in the galaxy UHZ1 at redshift $z \approx 10$, less than 500 Myr after the Big Bang, adding pressure to models of how supermassive black holes grew so quickly. |

**Next steps.** Ground-based detectors plan a further observing period in 2026 and the O5 run from around late 2027. ESA's **LISA** mission, adopted in 2024 and targeted for launch in the mid-2030s, will observe mergers of $10^5$–$10^7\,M_\odot$ black holes and extreme-mass-ratio inspirals that map the Kerr geometry orbit by orbit. Next-generation ground detectors (Einstein Telescope, Cosmic Explorer) and space-extended very-long-baseline interferometry are aimed at precision black-hole spectroscopy and movies of horizon-scale plasma.

---

## Continue Reading

- **Previous:** [Tensor Formalism & the Field Equations](tensor-formalism.html) — the curvature machinery these solutions satisfy.
- **Next:** [Gravitational Waves](gravitational-waves.html) — linearized gravity, binary inspiral, and the ringdown of newly formed black holes.
- **Up:** [Graduate Topics](advanced.html) — the relativity deep-dive hub.

## See Also

- [General Relativity](general-relativity.html) — the equivalence principle, the field equations, and the classic tests.
- [Relativistic Cosmology](cosmology.html) — horizons in expanding spacetimes and de Sitter temperature.
- [Toward Quantum Gravity](quantum-gravity.html) — holography, AdS/CFT, and the programmes that aim to resolve the singularity.
- [String Theory](../string-theory/) — microscopic state counting that reproduces the Bekenstein–Hawking entropy.
- [Quantum Field Theory](../quantum-field-theory.html) — the field theory behind Hawking and Unruh radiation.
- [Thermodynamics](../thermodynamics.html) — the four laws that black holes mirror.
- [Physics Hub](../) — browse all physics topics.
