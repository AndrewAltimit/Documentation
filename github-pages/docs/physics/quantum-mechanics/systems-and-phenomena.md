---
layout: docs
title: "Quantum Mechanics: Systems & Phenomena"
description: "The exactly solvable quantum systems (infinite well, harmonic oscillator, hydrogen, spin-1/2), the characteristic quantum phenomena (tunneling, interference, entanglement, identical particles), and the experiments that established them."
permalink: /docs/physics/quantum-mechanics/systems-and-phenomena.html
toc: true
toc_sticky: true
---

## Systems & Phenomena

[Quantum Mechanics](./) &raquo; Systems &amp; Phenomena

This page collects the exactly solvable systems that anchor quantum mechanics — the infinite square well, the harmonic oscillator, the hydrogen atom and the spin-1/2 particle — and the phenomena with no classical counterpart: tunneling, single-particle interference, entanglement and the statistics of identical particles. Each is paired with the experiments that established it. The operators, the Schrödinger equation and the measurement postulates are developed on [States, Operators & Dynamics](formalism.html); here they are applied.

## Bound States: The Solvable Systems

A particle confined by a potential has a **discrete** energy spectrum: only standing waves that satisfy the boundary conditions are allowed. How the levels are spaced depends on the shape of the potential, and the three classic systems show the three characteristic patterns.

<figure style="margin: 1.5em auto; max-width: 720px;">
<svg viewBox="0 0 720 330" role="img" aria-label="Energy levels of the infinite square well (spreading as n squared), harmonic oscillator (evenly spaced), and hydrogen atom (crowding toward the ionization limit)" style="max-width:720px;width:100%;height:auto;color:inherit;font-family:inherit">
  <g stroke="currentColor" fill="none"><path d="M 50 40 L 50 260 L 190 260 L 190 40" stroke-width="2.2"/><line x1="50" y1="248" x2="190" y2="248" stroke-width="1.6"/><line x1="50" y1="212" x2="190" y2="212" stroke-width="1.6"/><line x1="50" y1="152" x2="190" y2="152" stroke-width="1.6"/><line x1="50" y1="68" x2="190" y2="68" stroke-width="1.6"/><path d="M 250.0 50.0 L 255.5 70.5 L 261.0 89.9 L 266.5 108.3 L 272.0 125.6 L 277.5 141.9 L 283.0 157.1 L 288.5 171.3 L 294.0 184.4 L 299.5 196.5 L 305.0 207.5 L 310.5 217.5 L 316.0 226.4 L 321.5 234.3 L 327.0 241.1 L 332.5 246.9 L 338.0 251.6 L 343.5 255.3 L 349.0 257.9 L 354.5 259.5 L 360.0 260.0 L 365.5 259.5 L 371.0 257.9 L 376.5 255.3 L 382.0 251.6 L 387.5 246.9 L 393.0 241.1 L 398.5 234.3 L 404.0 226.4 L 409.5 217.5 L 415.0 207.5 L 420.5 196.5 L 426.0 184.4 L 431.5 171.3 L 437.0 157.1 L 442.5 141.9 L 448.0 125.6 L 453.5 108.3 L 459.0 89.9 L 464.5 70.5 L 470.0 50.0" stroke-width="2.2"/><line x1="326.9" y1="241.0" x2="393.1" y2="241.0" stroke-width="1.6"/><line x1="302.7" y1="203.0" x2="417.3" y2="203.0" stroke-width="1.6"/><line x1="286.0" y1="165.0" x2="434.0" y2="165.0" stroke-width="1.6"/><line x1="272.5" y1="127.0" x2="447.5" y2="127.0" stroke-width="1.6"/><line x1="260.7" y1="89.0" x2="459.3" y2="89.0" stroke-width="1.6"/><path d="M 608.8 286.0 L 610.5 258.3 L 612.2 231.9 L 613.9 212.0 L 615.7 196.4 L 617.4 183.9 L 619.1 173.7 L 620.8 165.2 L 622.5 157.9 L 624.2 151.7 L 626.0 146.3 L 627.7 141.6 L 629.4 137.4 L 631.1 133.7 L 632.8 130.3 L 634.5 127.3 L 636.2 124.6 L 638.0 122.2 L 639.7 119.9 L 641.4 117.8 L 643.1 115.9 L 644.8 114.2 L 646.5 112.5 L 648.3 111.0 L 650.0 109.6 L 651.7 108.3 L 653.4 107.1 L 655.1 105.9 L 656.8 104.8 L 658.5 103.8 L 660.3 102.9 L 662.0 101.9 L 663.7 101.1 L 665.4 100.3 L 667.1 99.5 L 668.8 98.8 L 670.5 98.1 L 672.3 97.4 L 674.0 96.8 L 675.7 96.2 L 677.4 95.6 L 679.1 95.0 L 680.8 94.5 L 682.6 94.0 L 684.3 93.5 L 686.0 93.0 L 687.7 92.6 L 689.4 92.1 L 691.1 91.7 L 692.8 91.3 L 694.6 90.9 L 696.3 90.6 L 698.0 90.2 L 699.7 89.9 L 701.4 89.5 L 703.1 89.2 L 704.9 88.9 L 706.6 88.6 L 708.3 88.3 L 710.0 88.0" stroke-width="2.2"/><path d="M 591.2 286.0 L 589.5 258.3 L 587.8 231.9 L 586.1 212.0 L 584.3 196.4 L 582.6 183.9 L 580.9 173.7 L 579.2 165.2 L 577.5 157.9 L 575.8 151.7 L 574.0 146.3 L 572.3 141.6 L 570.6 137.4 L 568.9 133.7 L 567.2 130.3 L 565.5 127.3 L 563.8 124.6 L 562.0 122.2 L 560.3 119.9 L 558.6 117.8 L 556.9 115.9 L 555.2 114.2 L 553.5 112.5 L 551.7 111.0 L 550.0 109.6 L 548.3 108.3 L 546.6 107.1 L 544.9 105.9 L 543.2 104.8 L 541.5 103.8 L 539.7 102.9 L 538.0 101.9 L 536.3 101.1 L 534.6 100.3 L 532.9 99.5 L 531.2 98.8 L 529.5 98.1 L 527.7 97.4 L 526.0 96.8 L 524.3 96.2 L 522.6 95.6 L 520.9 95.0 L 519.2 94.5 L 517.4 94.0 L 515.7 93.5 L 514.0 93.0 L 512.3 92.6 L 510.6 92.1 L 508.9 91.7 L 507.2 91.3 L 505.4 90.9 L 503.7 90.6 L 502.0 90.2 L 500.3 89.9 L 498.6 89.5 L 496.9 89.2 L 495.1 88.9 L 493.4 88.6 L 491.7 88.3 L 490.0 88.0" stroke-width="2.2"/><line x1="480" y1="70" x2="720" y2="70" stroke-width="1" stroke-dasharray="4 4"/><line x1="589.0" y1="250.0" x2="611.0" y2="250.0" stroke-width="1.6"/><line x1="556.0" y1="115.0" x2="644.0" y2="115.0" stroke-width="1.6"/><line x1="501.0" y1="90.0" x2="699.0" y2="90.0" stroke-width="1.6"/><line x1="490.0" y1="81.2" x2="710.0" y2="81.2" stroke-width="1.6"/></g>
  <g fill="currentColor"><text x="196.0" y="252.0" font-size="11.5">n = 1</text><text x="196.0" y="216.0" font-size="11.5">n = 2</text><text x="196.0" y="156.0" font-size="11.5">n = 3</text><text x="196.0" y="72.0" font-size="11.5">n = 4</text><text x="403.1" y="245.0" font-size="11.5">n = 0</text><text x="427.3" y="207.0" font-size="11.5">n = 1</text><text x="444.0" y="169.0" font-size="11.5">n = 2</text><text x="457.5" y="131.0" font-size="11.5">n = 3</text><text x="469.3" y="93.0" font-size="11.5">n = 4</text><text x="617.0" y="254.0" font-size="11.5">n = 1</text><text x="650.0" y="119.0" font-size="11.5">n = 2</text><text x="480.0" y="62.0" font-size="11.5">E = 0: ionization limit (n → ∞)</text>
    <text x="120" y="318" font-size="13" text-anchor="middle" font-weight="bold">Infinite well: E ∝ n²</text>
    <text x="360" y="318" font-size="13" text-anchor="middle" font-weight="bold">Oscillator: E ∝ n + ½</text>
    <text x="600" y="318" font-size="13" text-anchor="middle" font-weight="bold">Hydrogen: E ∝ −1/n²</text>
  </g>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">The lowest energy levels of the three canonical potentials, drawn to scale within each panel. Level lines end where they meet the potential (the classical turning points).</figcaption>
</figure>

| System | Potential | Energy levels | Spacing | Ground-state energy | Realized in |
|--------|-----------|---------------|---------|---------------------|-------------|
| Infinite square well | $0$ inside, $\infty$ outside | $\dfrac{n^2\pi^2\hbar^2}{2mL^2}$, $n = 1, 2, \ldots$ | Grows as $2n+1$ | Nonzero | Quantum dots, conjugated molecules, nanowires |
| Harmonic oscillator | $\tfrac{1}{2}m\omega^2 x^2$ | $\hbar\omega\left(n+\tfrac{1}{2}\right)$, $n = 0, 1, \ldots$ | Constant $\hbar\omega$ | $\tfrac{1}{2}\hbar\omega$ | Molecular vibrations, phonons, photons in a cavity mode, trapped ions |
| Hydrogen atom | $-\dfrac{e^2}{4\pi\varepsilon_0 r}$ | $-\dfrac{13.6\ \text{eV}}{n^2}$, $n = 1, 2, \ldots$ | Shrinks as $1/n^3$ | $-13.6\ \text{eV}$ | Atomic spectra, chemistry, astrophysics |

Hard walls push higher states apart; a quadratic well gives evenly spaced rungs; the Coulomb potential weakens with distance, so levels crowd together below the ionization limit. In every case the ground-state energy lies *above* the potential minimum — the **zero-point energy** required by the uncertainty principle.

### Infinite square well

For a particle confined to $0 \le x \le L$ by infinitely high walls, the wave function must vanish at both walls, so only whole numbers of half-wavelengths fit:

$$\psi_n(x) = \sqrt{\frac{2}{L}}\,\sin\left(\frac{n\pi x}{L}\right), \qquad E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}, \qquad n = 1, 2, 3, \ldots$$

The state $\psi_n$ has $n - 1$ interior nodes, a pattern (more nodes, more energy) that holds for any one-dimensional bound-state problem.

**Worked example.** An electron in a $1\ \text{nm}$ well has

$$E_1 = \frac{\pi^2\left(1.055\times10^{-34}\ \text{J s}\right)^2}{2\left(9.11\times10^{-31}\ \text{kg}\right)\left(10^{-9}\ \text{m}\right)^2} \approx 6.0\times10^{-20}\ \text{J} \approx 0.38\ \text{eV},$$

comparable to chemical and semiconductor energy scales. Because $E \propto 1/L^2$, shrinking a semiconductor nanocrystal shifts its emission to shorter wavelengths — the size-tuned colours of **quantum dots**, whose synthesis was recognized by the 2023 Nobel Prize in Chemistry.

A **finite** well of depth $V_0$ always has at least one bound state in one dimension, and its wave functions leak into the walls as $e^{-\kappa |x|}$ with $\kappa = \sqrt{2m(V_0 - E)}/\hbar$. That evanescent tail is the origin of tunneling (below).

### Harmonic oscillator

The oscillator $V(x) = \tfrac{1}{2}m\omega^2x^2$ is the model for any system near a stable equilibrium, and for every normal mode of a free field. It is solved most cleanly with the **ladder operators**

$$\hat a = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat x + \frac{i\hat p}{m\omega}\right), \qquad \left[\hat a, \hat a^\dagger\right] = 1, \qquad \hat H = \hbar\omega\left(\hat a^\dagger\hat a + \tfrac{1}{2}\right).$$

$\hat a^\dagger$ raises the energy by one quantum $\hbar\omega$ and $\hat a$ lowers it, with $\hat a|n\rangle = \sqrt{n}\,|n-1\rangle$ and $\hat a^\dagger|n\rangle = \sqrt{n+1}\,|n+1\rangle$. The ground state is fixed by $\hat a|0\rangle = 0$, which gives

$$\psi_n(x) = \frac{1}{\sqrt{2^n\,n!}}\left(\frac{m\omega}{\pi\hbar}\right)^{1/4} H_n(\xi)\,e^{-\xi^2/2}, \qquad \xi = \sqrt{\frac{m\omega}{\hbar}}\,x,$$

where $H_n$ are the Hermite polynomials ($H_0 = 1$, $H_1 = 2\xi$, $H_2 = 4\xi^2 - 2$). The ground state is a Gaussian that saturates the uncertainty bound, $\Delta x\,\Delta p = \hbar/2$.

Reinterpreting $n$ as a *number of quanta* rather than an excitation level is the step from single-particle quantum mechanics to quantum field theory: a photon is one quantum of an electromagnetic-field oscillator mode, a phonon one quantum of a lattice vibration. The zero-point energy $\tfrac{1}{2}\hbar\omega$ per mode is physical; it contributes to the Casimir force and keeps helium liquid down to absolute zero at ordinary pressure.

### Hydrogen atom

The Coulomb problem separates in spherical coordinates, $\psi_{n\ell m}(r,\theta,\phi) = R_{n\ell}(r)\,Y_\ell^m(\theta,\phi)$, with energies

$$E_n = -\frac{m_e e^4}{2\left(4\pi\varepsilon_0\right)^2\hbar^2}\,\frac{1}{n^2} = -\frac{1}{2}\,\frac{m_e c^2\alpha^2}{n^2} \approx -\frac{13.6\ \text{eV}}{n^2},$$

where $\alpha \approx 1/137$ is the fine-structure constant. The states are labelled by

| Quantum number | Range | Physical meaning |
|----------------|-------|------------------|
| $n$ | $1, 2, 3, \ldots$ | Principal: sets the energy and the size ($\langle r\rangle \sim n^2 a_0$) |
| $\ell$ | $0, 1, \ldots, n-1$ (s, p, d, f, ...) | Orbital angular momentum, $L^2 = \hbar^2\ell(\ell+1)$ |
| $m_\ell$ | $-\ell, \ldots, +\ell$ | Projection $L_z = m_\ell\hbar$ |
| $m_s$ | $\pm\tfrac{1}{2}$ | Electron spin projection |

The ground state is

$$\psi_{100}(r) = \frac{1}{\sqrt{\pi a_0^3}}\,e^{-r/a_0}, \qquad a_0 = \frac{4\pi\varepsilon_0\hbar^2}{m_e e^2} \approx 0.529\ \mathring{\mathrm{A}} = 5.29\times10^{-11}\ \text{m},$$

with most probable radius $a_0$ and mean radius $\tfrac{3}{2}a_0$.

**Degeneracy and its lifting.** The energy depends only on $n$, so each level holds $n^2$ orbital states ($2n^2$ with spin). The extra degeneracy between different $\ell$ is special to the pure $1/r$ potential — it reflects a conserved Laplace–Runge–Lenz vector, the same symmetry that closes Kepler orbits. Real atoms lift it in stages:

| Effect | Origin | Size (hydrogen) |
|--------|--------|-----------------|
| Fine structure | Relativistic kinetic energy, spin–orbit coupling, Darwin term | $\sim\alpha^2 E_n$; $E_{nj} = E_n\left[1 + \dfrac{\alpha^2}{n^2}\left(\dfrac{n}{j + 1/2} - \dfrac{3}{4}\right)\right]$ |
| Lamb shift | Vacuum fluctuations of the quantized field (QED) | $2S_{1/2}$–$2P_{1/2}$ splitting $\approx 1.06\ \text{GHz}$ |
| Hyperfine structure | Electron spin coupled to proton spin | Ground-state splitting $1420.4\ \text{MHz}$: the 21 cm line of radio astronomy |

In multi-electron atoms, electron–electron repulsion splits levels with different $\ell$ even without relativity, which is why $4s$ fills before $3d$.

**Spectra.** Photons emitted in transitions $n_2 \to n_1$ obey the **Rydberg formula** $1/\lambda = R_\infty\left(1/n_1^2 - 1/n_2^2\right)$ with $R_\infty \approx 1.097\times10^7\ \text{m}^{-1}$ (corrected slightly by the reduced mass). The Lyman series ($n_1 = 1$) is ultraviolet; the Balmer series ($n_1 = 2$) includes the red H-alpha line at $656\ \text{nm}$. Precision spectroscopy of the $1S$–$2S$ transition, now known to about 15 significant figures, is among the most stringent tests of quantum electrodynamics and a key input to the proton-radius measurements.

### Spin-1/2 and the Stern–Gerlach experiment

Spin is angular momentum with no classical orbital counterpart. For spin-1/2 the state space is two-dimensional and the spin operators are $\hat S_i = \tfrac{\hbar}{2}\sigma_i$ with the Pauli matrices

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \qquad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \qquad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.$$

In 1922 Stern and Gerlach sent silver atoms through an inhomogeneous magnetic field. A classical magnetic moment with random orientation would smear into a continuous band; the beam split into **two** discrete spots. The result, later understood as the spin of the single unpaired electron (Uhlenbeck and Goudsmit, 1925), is the textbook demonstration of quantized angular momentum. Sequential Stern–Gerlach devices also illustrate non-commuting observables: selecting $S_z = +\hbar/2$, then measuring $S_x$, then $S_z$ again yields both $S_z$ outcomes with equal probability — the $S_x$ measurement erased the earlier $S_z$ information.

Any two-level system — a spin, a photon's polarization, two atomic levels, a superconducting circuit — is mathematically the same object, a **qubit**, developed on [Quantum Computing](qm-computing.html).

## Quantum Phenomena

### Tunneling

A classical particle with energy $E$ below a barrier height $V_0$ is reflected. A quantum wave function instead decays exponentially inside the barrier, and if the barrier is thin some amplitude emerges on the far side.

<figure style="margin: 1.5em auto; max-width: 680px;">
<svg viewBox="0 0 680 260" role="img" aria-label="A wave incident on a rectangular barrier: it oscillates on the left, decays exponentially inside the barrier, and continues with reduced amplitude on the right" style="max-width:680px;width:100%;height:auto;color:inherit;font-family:inherit">
  <rect x="300" y="60" width="80" height="170" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="1.8"/>
  <line x1="20" y1="230" x2="660" y2="230" stroke="currentColor" stroke-width="1.8"/>
  <line x1="20" y1="150" x2="660" y2="150" stroke="currentColor" stroke-width="1" stroke-dasharray="5 5" opacity="0.7"/>
  <path d="M 30.0 167.8 L 33.0 180.9 L 36.1 190.5 L 39.1 195.7 L 42.1 195.9 L 45.2 190.9 L 48.2 181.5 L 51.2 168.6 L 54.3 153.6 L 57.3 138.2 L 60.3 124.2 L 63.4 113.0 L 66.4 105.8 L 69.4 103.6 L 72.5 106.4 L 75.5 114.1 L 78.5 125.7 L 81.6 140.1 L 84.6 155.5 L 87.6 170.3 L 90.7 182.8 L 93.7 191.8 L 96.7 196.1 L 99.8 195.4 L 102.8 189.6 L 105.8 179.5 L 108.9 166.1 L 111.9 150.9 L 114.9 135.7 L 118.0 122.0 L 121.0 111.4 L 124.0 105.1 L 127.1 103.7 L 130.1 107.4 L 133.1 115.9 L 136.2 128.1 L 139.2 142.7 L 142.2 158.1 L 145.3 172.6 L 148.3 184.7 L 151.3 192.9 L 154.4 196.3 L 157.4 194.7 L 160.4 188.1 L 163.5 177.4 L 166.5 163.6 L 169.6 148.3 L 172.6 133.2 L 175.6 119.9 L 178.7 110.0 L 181.7 104.5 L 184.7 104.0 L 187.8 108.6 L 190.8 117.7 L 193.8 130.4 L 196.9 145.3 L 199.9 160.7 L 202.9 174.9 L 206.0 186.4 L 209.0 193.8 L 212.0 196.4 L 215.1 193.9 L 218.1 186.6 L 221.1 175.2 L 224.2 161.0 L 227.2 145.6 L 230.2 130.7 L 233.3 118.0 L 236.3 108.7 L 239.3 104.0 L 242.4 104.4 L 245.4 109.8 L 248.4 119.7 L 251.5 132.9 L 254.5 148.0 L 257.5 163.3 L 260.6 177.1 L 263.6 188.0 L 266.6 194.6 L 269.7 196.4 L 272.7 193.0 L 275.7 184.9 L 278.8 172.9 L 281.8 158.4 L 284.8 143.0 L 287.9 128.3 L 290.9 116.1 L 293.9 107.6 L 297.0 103.7 L 300.0 105.0 L 300.0 105.0 L 304.2 110.0 L 308.4 114.5 L 312.6 118.4 L 316.8 121.9 L 321.1 125.0 L 325.3 127.8 L 329.5 130.3 L 333.7 132.5 L 337.9 134.4 L 342.1 136.2 L 346.3 137.7 L 350.5 139.1 L 354.7 140.3 L 358.9 141.4 L 363.2 142.3 L 367.4 143.2 L 371.6 143.9 L 375.8 144.6 L 380.0 145.2 L 380.0 145.2 L 383.0 145.9 L 386.1 147.0 L 389.1 148.4 L 392.1 150.1 L 395.2 151.7 L 398.2 153.1 L 401.2 154.2 L 404.3 154.8 L 407.3 154.9 L 410.3 154.5 L 413.4 153.5 L 416.4 152.2 L 419.4 150.6 L 422.5 149.0 L 425.5 147.4 L 428.5 146.2 L 431.6 145.4 L 434.6 145.1 L 437.6 145.3 L 440.7 146.0 L 443.7 147.2 L 446.7 148.7 L 449.8 150.3 L 452.8 151.9 L 455.8 153.3 L 458.9 154.3 L 461.9 154.9 L 464.9 154.9 L 468.0 154.3 L 471.0 153.3 L 474.0 151.9 L 477.1 150.3 L 480.1 148.7 L 483.1 147.2 L 486.2 146.0 L 489.2 145.3 L 492.2 145.1 L 495.3 145.4 L 498.3 146.2 L 501.3 147.5 L 504.4 149.0 L 507.4 150.6 L 510.4 152.2 L 513.5 153.5 L 516.5 154.5 L 519.6 154.9 L 522.6 154.8 L 525.6 154.2 L 528.7 153.1 L 531.7 151.7 L 534.7 150.1 L 537.8 148.4 L 540.8 147.0 L 543.8 145.9 L 546.9 145.2 L 549.9 145.1 L 552.9 145.5 L 556.0 146.4 L 559.0 147.7 L 562.0 149.3 L 565.1 150.9 L 568.1 152.5 L 571.1 153.7 L 574.2 154.6 L 577.2 154.9 L 580.2 154.7 L 583.3 154.0 L 586.3 152.9 L 589.3 151.4 L 592.4 149.8 L 595.4 148.2 L 598.4 146.8 L 601.5 145.7 L 604.5 145.1 L 607.5 145.1 L 610.6 145.6 L 613.6 146.6 L 616.6 148.0 L 619.7 149.5 L 622.7 151.2 L 625.7 152.7 L 628.8 153.9 L 631.8 154.7 L 634.8 154.9 L 637.9 154.7 L 640.9 153.9 L 643.9 152.6 L 647.0 151.1 L 650.0 149.5" fill="none" stroke="currentColor" stroke-width="2.4"/>
  <g font-size="12.5" fill="currentColor">
    <text x="340.0" y="52" text-anchor="middle">barrier V₀</text>
    <text x="560" y="136" text-anchor="middle">energy E &lt; V₀</text>
    <text x="340.0" y="248" text-anchor="middle">width a</text>
    <text x="150" y="222" text-anchor="middle">incident + reflected</text>
    <text x="340.0" y="205" text-anchor="middle">decays ~ e^(−κx)</text>
    <text x="520" y="222" text-anchor="middle">transmitted, amplitude ~ e^(−κa)</text>
  </g>
</svg>
<figcaption style="font-size: 0.9em; text-align: center;">Real part of the wave function for a particle with $E &lt; V_0$ incident from the left on a rectangular barrier. Inside the barrier it decays as $e^{-\kappa x}$; the transmitted wave has the same wavelength but an amplitude reduced by roughly $e^{-\kappa a}$.</figcaption>
</figure>

For a rectangular barrier of height $V_0$ and width $a$, the exact transmission probability is

$$T = \left[1 + \frac{V_0^2\,\sinh^2(\kappa a)}{4E\left(V_0 - E\right)}\right]^{-1} \;\approx\; \frac{16E\left(V_0-E\right)}{V_0^2}\,e^{-2\kappa a} \quad (\kappa a \gg 1), \qquad \kappa = \frac{\sqrt{2m\left(V_0-E\right)}}{\hbar}.$$

For a smooth barrier the WKB approximation gives $T \approx \exp\left(-2\int \kappa(x)\,dx\right)$ over the classically forbidden region. The exponential dependence on $a\sqrt{m(V_0 - E)}$ is why tunneling is routine for electrons across angstrom gaps and negligible for macroscopic objects.

| Phenomenon | What tunnels | Consequence |
|------------|--------------|-------------|
| Alpha decay (Gamow; Gurney and Condon, 1928) | Alpha particle through the Coulomb barrier | Half-lives from under a microsecond to $10^{10}$ years; the Geiger–Nuttall law |
| Stellar fusion | Protons through their mutual repulsion | The Sun burns at $1.5\times10^7\ \text{K}$, far below the classical threshold |
| Scanning tunnelling microscope (Binnig and Rohrer, 1981) | Electrons across a tip–surface vacuum gap | Current changes about tenfold per $0.1\ \text{nm}$, giving atomic resolution |
| Flash memory | Electrons through an oxide (Fowler–Nordheim tunnelling) | Charge written to and erased from a floating gate |
| Transistor scaling | Leakage through thin gate oxides | Forced the move to high-k gate dielectrics in the 2000s |
| Josephson junction | Cooper pairs through a thin insulator | SQUID magnetometers, voltage standards, superconducting qubits |

**Macroscopic quantum tunnelling.** In the mid-1980s Clarke, Devoret and Martinis showed that the phase difference across a current-biased Josephson junction — a collective coordinate describing billions of Cooper pairs — escapes its metastable state by tunnelling, and that it occupies discrete energy levels that can be driven with microwaves. This established that a macroscopic electrical circuit can behave as a single quantum degree of freedom, the basis of today's superconducting qubits, and was recognized by the 2025 Nobel Prize in Physics.

### Superposition and interference

If a system can be in states $|0\rangle$ and $|1\rangle$, it can be in any normalized combination

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1,$$

with $|\alpha|^2$ and $|\beta|^2$ the probabilities of the two measurement outcomes. The complex *phases* of the amplitudes are physical: they produce interference whenever two paths lead to the same outcome.

The **double-slit experiment** is the standard demonstration. Particles of de Broglie wavelength $\lambda = h/p$ passing slits a distance $d$ apart produce fringes of spacing $\Delta y = \lambda L/d$ on a screen at distance $L$, even when sent one at a time — each particle lands at a single point, and the pattern builds up statistically. Tonomura's 1989 single-electron experiment showed this build-up directly. The same interference has been observed for neutrons, atoms, $\text{C}_{60}$ fullerenes (1999), and molecules of more than $25\,000$ atomic mass units (2019).

If *which-path* information is available anywhere — in a detector, or imprinted in the environment — the fringes vanish, because the two paths lead to orthogonal states of the larger system and no longer interfere. This is the mechanism of **decoherence**, developed on [States, Operators & Dynamics](formalism.html#decoherence).

### Entanglement

A two-particle state is **entangled** if it cannot be written as a product $|\phi\rangle_A \otimes |\chi\rangle_B$. The four maximally entangled **Bell states** are

$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}\left(|00\rangle \pm |11\rangle\right), \qquad |\Psi^\pm\rangle = \frac{1}{\sqrt{2}}\left(|01\rangle \pm |10\rangle\right).$$

In the spin singlet $|\Psi^-\rangle$, measurements of the two spins along any common axis always give opposite results, however far apart the particles are. Each particle on its own, however, is in a maximally mixed state: local measurement statistics are completely random, and the **no-communication theorem** guarantees that nothing done to one particle changes the statistics of the other. Entanglement produces correlations, not signals.

What makes these correlations non-classical is **Bell's theorem** (1964): no local hidden-variable model can reproduce them. In the CHSH form, local models obey $|S| \le 2$ while quantum mechanics reaches $2\sqrt{2}$. Tests began with Freedman and Clauser (1972) and Aspect (1982); in 2015 three groups (Delft, Vienna, NIST) closed the locality and detection loopholes simultaneously. The 2022 Nobel Prize in Physics went to Aspect, Clauser and Zeilinger for these experiments. The full derivation and experimental history are on [Bell's Theorem & Experimental Tests](bell-inequalities-and-tests.html).

### Identical particles

Quantum particles of the same species are strictly indistinguishable, so exchanging two of them can change the wave function by at most a sign:

$$\psi(x_2, x_1) = \pm\,\psi(x_1, x_2).$$

The **spin–statistics theorem** of relativistic quantum field theory fixes the sign: integer-spin **bosons** take $+$, half-integer-spin **fermions** take $-$. For fermions the antisymmetry forces $\psi = 0$ when two particles share a state — the **Pauli exclusion principle** — which builds the periodic table, makes matter incompressible and supports white dwarfs and neutron stars against collapse. Bosons tend to bunch into the same state, giving laser light, superfluidity and Bose–Einstein condensation (first achieved in dilute atomic gases in 1995).

## Where Quantum Mechanics Shows Up

| Technology or phenomenon | Quantum principle |
|--------------------------|-------------------|
| Lasers | Stimulated emission into a single mode; bosonic enhancement |
| Transistors, LEDs, solar cells | Electronic band structure and band gaps of crystals |
| MRI and NMR | Precession and coherent control of nuclear spins |
| Atomic clocks, GPS | Hyperfine transitions; the SI second is defined by the caesium-133 hyperfine frequency, $9\,192\,631\,770\ \text{Hz}$ |
| Scanning tunnelling microscopy, flash memory | Tunnelling |
| Superconducting magnets, SQUIDs, superconducting qubits | Macroscopic quantum coherence of Cooper pairs; the Josephson effect |
| Chemistry and the periodic table | Hydrogen-like orbitals plus the Pauli exclusion principle |
| Quantum dots in displays | Size quantization, $E \propto 1/L^2$ |

---

## Continue Reading

- **Previous:** [States, Operators & Dynamics](formalism.html) — the equations and operators behind these systems.
- **Next:** [Computing, Information & Advanced Formalism](computing-and-advanced.html) — qubits, algorithms, and the graduate machinery.
- **Up:** [Quantum Mechanics Hub](./)

## See Also

- [Bell's Theorem & Experimental Tests](bell-inequalities-and-tests.html) — the CHSH inequality and the loophole-free experiments.
- [Research Frontiers](qm-research-frontiers.html) — many-body physics, topological matter, and open foundational questions.
- [Condensed Matter Physics](../condensed-matter/) — tunnelling, superconductivity, and many-body phases in solids.
- [Statistical Mechanics](../statistical-mechanics/) — quantum statistics behind lasers and condensates.
