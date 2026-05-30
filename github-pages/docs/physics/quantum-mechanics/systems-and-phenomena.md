---
layout: docs
title: "Quantum Mechanics: Systems & Phenomena"
permalink: /docs/physics/quantum-mechanics/systems-and-phenomena.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Systems &amp; Phenomena</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The canonical solvable systems and the distinctively quantum effects they reveal: confinement and energy levels, tunneling, entanglement, and superposition.</p>
</div>

[Quantum Mechanics](./) &raquo; Systems &amp; Phenomena

## Quantum Systems

### Particle in a Box

For an infinite potential well of width L:

**Wave functions:**
$$
\psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right) \text{ for } 0 \leq x \leq L, \quad \psi_n(x) = 0 \text{ elsewhere}
$$

**Energy levels:**
$$E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$

Where $n = 1, 2, 3, \ldots$

As a concrete example, an electron confined to a $1\,\text{nm}$ box has ground-state energy
$$E_1 = \frac{\pi^2(1.05\times10^{-34})^2}{2 \times 9.1\times10^{-31} \times (10^{-9})^2} \approx 6\times10^{-20}\ \text{J} \approx 0.38\ \text{eV}.$$
The energy is quantized: only the discrete values $E_n$ are allowed.

### Harmonic Oscillator

**Potential:** $V(x) = \tfrac{1}{2}m\omega^2 x^2$

**Energy levels:**
$$E_n = \hbar\omega\left(n + \tfrac{1}{2}\right)$$

Where $n = 0, 1, 2, \ldots$

**Ground state wave function:**
$$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \exp\left(-\frac{m\omega x^2}{2\hbar}\right)$$
Note: the prefactor $\left(m\omega/\pi\hbar\right)^{1/4}$ ensures normalization $\int|\psi_0|^2\,dx = 1$.

### Hydrogen Atom

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://en.wikipedia.org/wiki/Hydrogen_atom"> Article: <b><i>Hydrogen Atom Electron Orbitals - Wikipedia</i></b></a></p>

**Energy levels:**
$$E_n = -\frac{13.6\ \text{eV}}{n^2}$$

**Wave functions characterized by quantum numbers:**
- $n$: principal quantum number ($1, 2, 3, \ldots$)
- $\ell$: orbital angular momentum ($0, 1, \ldots, n-1$)
- $m$: magnetic quantum number ($-\ell, \ldots, +\ell$)
- $s$: spin quantum number ($\pm\tfrac{1}{2}$)

**Ground state (1s):**
$$\psi_{100}(r,\theta,\phi) = \frac{1}{\sqrt{\pi}}\left(\frac{1}{a_0}\right)^{3/2} e^{-r/a_0}$$

Where $a_0$ = Bohr radius = 0.529 Å = $5.29 \times 10^{-11}$ m.

Note: this is properly normalized: $\iiint |\psi_{100}|^2\, r^2 \sin\theta\,dr\,d\theta\,d\phi = 1$.

<div class="tip-card">
  <h4>Why hydrogen is the keystone</h4>
  <p>The hydrogen atom is the only atom solved exactly, and almost everything in chemistry borrows its vocabulary. Two features carry the physics. First, the energy depends <em>only</em> on $n$ (a special "accidental" degeneracy of the pure $1/r$ Coulomb potential), so the $2s$ and $2p$ states share an energy in this idealization — degeneracies that real atoms lift through electron–electron repulsion and relativistic corrections. Second, the levels converge to $E \to 0$ as $n \to \infty$: that ceiling is the <strong>ionization energy</strong> (13.6 eV from the ground state), the work needed to free the electron entirely. The transitions between these levels produce the Balmer and Lyman spectral lines that first revealed quantization, and the Bohr radius $a_0$ sets the natural size of the atom.</p>
</div>

### Comparing the Three Canonical Systems

These three solvable systems anchor most of quantum mechanics. Notice how the *spacing* of energy levels — not just their values — reflects the shape of the confining potential.

| System | Potential | Energy levels | Level spacing | Where it shows up |
|--------|-----------|---------------|---------------|-------------------|
| Particle in a box | Infinite walls | $E_n = \dfrac{n^2\pi^2\hbar^2}{2mL^2}$ | Grows like $n^2$ (spreads out) | Quantum dots, conjugated molecules |
| Harmonic oscillator | $\tfrac{1}{2}m\omega^2 x^2$ | $E_n = \hbar\omega\left(n+\tfrac{1}{2}\right)$ | Constant $\hbar\omega$ (evenly spaced) | Vibrations, phonons, quantum fields |
| Hydrogen atom | $-\dfrac{e^2}{4\pi\varepsilon_0 r}$ | $E_n = -\dfrac{13.6\ \text{eV}}{n^2}$ | Shrinks like $1/n^2$ (converges) | Atomic spectra, chemistry |

<div class="tip-card">
  <h4>Why the patterns differ</h4>
  <p>A steep, hard-walled box pushes higher states up rapidly, so levels spread apart. A spring-like (quadratic) well gives perfectly even rungs — the hallmark of the harmonic oscillator and the reason it underlies field quantization. The hydrogen atom's $-1/r$ attraction weakens with distance, so levels bunch up and converge to the ionization threshold ($E \to 0$) as $n \to \infty$.</p>
</div>

## Quantum Phenomena

### Tunneling

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://en.wikipedia.org/wiki/Quantum_tunnelling"> Article: <b><i>Quantum Tunneling - Wikipedia</i></b></a></p>

Particles can penetrate classically forbidden regions. For a rectangular barrier:

**Transmission coefficient:**
$$T \approx \frac{16E(V_0-E)}{V_0^2}\, e^{-2\kappa a}$$

Where $\kappa = \sqrt{2m(V_0-E)}/\hbar$ and $a$ is the barrier width.

The physics lives in the exponential. A classical particle with energy $E < V_0$ simply bounces off the wall; quantum mechanically the wave function does not stop dead at the barrier but *decays* inside it as $e^{-\kappa x}$, and if the barrier is thin enough a small amplitude survives on the far side. The transmission probability therefore plummets exponentially with the barrier width $a$ and with $\kappa \propto \sqrt{V_0 - E}$ — the square root of how far the barrier height rises *above* the particle's energy. That is why tunneling is dramatic for electrons across an atomic-scale gap yet utterly negligible for a tennis ball against a wall.

<div class="tip-card">
  <h4>Tunneling in the real world</h4>
  <p>This single exponential explains a remarkable range of phenomena. <strong>Alpha decay</strong>: an alpha particle escapes a heavy nucleus by tunneling through the Coulomb barrier, and the steep dependence on barrier height is why nuclear half-lives span from microseconds to billions of years. <strong>The scanning tunneling microscope</strong> measures the tunneling current between a sharp tip and a surface; because $T$ changes by a factor of ~10 for every 0.1 nm of gap, it resolves individual atoms. <strong>Fusion in the Sun</strong> proceeds only because protons tunnel through their mutual repulsion — without quantum tunneling, stars could not shine. And <strong>flash memory</strong> stores data by tunneling electrons onto an isolated gate.</p>
</div>

### Quantum Entanglement
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-pdf-fill.svg" class="icon"><a href="https://cds.cern.ch/record/111654/files/vol1p195-200_001.pdf"> Paper: <b><i>On the Einstein Podolsky Rosen Paradox</i></b> - John Bell</a></p>
<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/play-btn-fill.svg" class="icon"><a href="https://www.youtube.com/watch?v=ZuvK-od647c"> Video: <b><i>Quantum Entanglement Explained</i></b></a></p>

Non-local correlations between particles. Example — the singlet Bell state:

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}\left(|\!\uparrow\downarrow\rangle - |\!\downarrow\uparrow\rangle\right)$$

This is one of the four maximally entangled Bell states. It is properly normalized:
$$\langle\Psi^-|\Psi^-\rangle = \tfrac{1}{2}\left(\langle\uparrow\downarrow| - \langle\downarrow\uparrow|\right)\left(|\!\uparrow\downarrow\rangle - |\!\downarrow\uparrow\rangle\right) = \tfrac{1}{2}(1 + 1) = 1.$$

Measurement of one particle instantly determines the state of the other, regardless of distance. The correlations are real, but the no-communication theorem prevents using them to transmit information faster than light.

### Quantum Superposition

A system can exist in multiple states simultaneously:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

**Normalization requirement:** $|\alpha|^2 + |\beta|^2 = 1$
- $|\alpha|^2$ = probability of measuring state $|0\rangle$
- $|\beta|^2$ = probability of measuring state $|1\rangle$
- $\alpha$ and $\beta$ are complex numbers (amplitudes)

## Where Quantum Mechanics Shows Up

Quantum mechanics is not confined to the laboratory — it underlies a great deal of working technology and natural phenomena.

- **Laser light:** a coherent quantum state of photons, all in the same mode, exhibiting bosonic statistics.
- **Computer chips:** transistor operation and band structure are quantum; tunneling sets fundamental scaling limits.
- **MRI:** nuclear spins are manipulated through quantum coherence, with RF pulses creating superpositions.
- **Superconductivity and superfluidity:** macroscopic quantum phenomena where Cooper pairs or condensates behave as a single quantum state.
- **Quantum chemistry:** molecular orbitals, reaction tunneling, and spectroscopic transitions all follow from the systems above.

## Experimental Techniques

### Double-Slit Experiment

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/file-text-fill.svg" class="icon"><a href="https://www.feynmanlectures.caltech.edu/III_01.html"> Lecture: <b><i>The Feynman Lectures - Quantum Behavior</i></b></a></p>

Demonstrates wave-particle duality:
- Single particles create interference patterns
- Observation destroys interference

### Stern-Gerlach Experiment
Demonstrates quantization of angular momentum:
- Atoms split into discrete beams
- Proves space quantization

### Bell's Inequality Tests
Confirms quantum entanglement:
- Violates local hidden variable theories
- Supports quantum non-locality

---

## Continue Reading

- **Previous:** [States, Operators & Dynamics](formalism.html) — the equations and operators behind these systems.
- **Next:** [Computing, Information & Advanced Formalism](computing-and-advanced.html) — qubits, algorithms, and the graduate machinery.
- **Up:** [Quantum Mechanics Hub](./)

## See Also

- [Condensed Matter Physics](../condensed-matter/) — tunneling, superconductivity, and many-body phases in solids.
- [Statistical Mechanics](../statistical-mechanics/) — quantum statistics behind lasers and condensates.
