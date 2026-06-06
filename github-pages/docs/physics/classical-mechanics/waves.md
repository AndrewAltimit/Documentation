---
layout: docs
title: "Classical Mechanics: Oscillations & Waves"
permalink: /docs/physics/classical-mechanics/waves.html
toc: true
toc_sticky: true
hide_title: true
---

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Oscillations &amp; Waves</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">From a single mass on a spring to coupled oscillators, normal modes, the wave equation, dispersion, and the first hints of nonlinearity.</p>
</div>

[Classical Mechanics](./) &raquo; Oscillations &amp; Waves

## Why Oscillations Are Everywhere

Look around: pendulum clocks tick, guitar strings vibrate, atoms in solids jiggle, electrons slosh back and forth in antennas, and the ground itself rings after an earthquake. Oscillatory motion is ubiquitous because it emerges whenever a system sits near a stable equilibrium and feels a **restoring force** that pushes it back when displaced. Expand *any* smooth potential $V(x)$ about a stable minimum $x_0$:

$$V(x) \approx V(x_0) + \tfrac{1}{2}V''(x_0)\,(x - x_0)^2 + \cdots$$

The linear term vanishes (it is a minimum), so to leading order every stable system is a Hooke's-law spring with effective stiffness $k = V''(x_0)$. This is the deep reason the **simple harmonic oscillator** is the single most important model in physics: it is the universal lowest-order description of small motions about equilibrium, and it reappears in mechanics, optics, circuits, and — quantized — in every quantum field.

When many such oscillators are coupled and arranged in space, their collective small motions organize into **waves**. This page builds that ladder rung by rung: one oscillator, then two, then a chain, then a continuum (the wave equation), and finally the phenomena — standing versus traveling waves, dispersion, group versus phase velocity, energy transport, Fourier decomposition — that the continuum supports, closing with a glimpse of what happens when the small-amplitude approximation fails and nonlinearity takes over.

## Simple Harmonic Motion: A Recap

<p class="referenceBoxes type3"><img src="https://andrewaltimit.github.io/Documentation/images/git.svg" class="icon"><a href="https://github.com/matplotlib/matplotlib/blob/main/galleries/examples/animation/simple_anim.py"> Code: <b><i>SHM Animation with Matplotlib</i></b></a></p>

A mass $m$ subject to a restoring force proportional to displacement obeys **Hooke's law**:

$$F = -kx$$

Newton's second law then gives the defining equation of simple harmonic motion (SHM):

$$m\ddot{x} + kx = 0 \qquad\Longleftrightarrow\qquad \ddot{x} + \omega^2 x = 0, \quad \omega = \sqrt{\frac{k}{m}}$$

The general solution is sinusoidal:

$$x(t) = A\cos(\omega t + \varphi)$$

Where:
- $A$ = amplitude (set by initial conditions)
- $\omega$ = angular frequency $= \sqrt{k/m}$ (set by the system)
- $\varphi$ = phase constant (set by initial conditions)

The hallmark of SHM is that the frequency is **independent of amplitude** (isochronism), a direct consequence of the force being linear in $x$. The period and frequency are

$$T = \frac{2\pi}{\omega} = 2\pi\sqrt{\frac{m}{k}}, \qquad f = \frac{1}{T} = \frac{\omega}{2\pi}.$$

### Energy in SHM

Energy sloshes back and forth between kinetic and potential while the total stays fixed:

$$E = \tfrac{1}{2}m\dot{x}^2 + \tfrac{1}{2}kx^2 = \tfrac{1}{2}kA^2 = \text{constant}.$$

At the turning points all the energy is potential; at the equilibrium point all of it is kinetic. The time-averaged kinetic and potential energies are equal, $\langle T\rangle = \langle V\rangle = \tfrac{1}{4}kA^2$ — a fact (the virial theorem for a quadratic potential) that recurs throughout physics.

### Damped Oscillations

Real oscillations do not last forever. Adding a velocity-proportional drag $-b\dot{x}$ gives the **damped harmonic oscillator**:

$$m\ddot{x} + b\dot{x} + kx = 0.$$

Trying $x = e^{\lambda t}$ yields the characteristic roots

$$\lambda = -\gamma \pm \sqrt{\gamma^2 - \omega_0^2}, \qquad \gamma = \frac{b}{2m}, \quad \omega_0 = \sqrt{\frac{k}{m}}.$$

The behavior splits into three regimes set by the discriminant:

| Regime | Condition | Behavior |
|--------|-----------|----------|
| Underdamped | $\gamma < \omega_0$ | Oscillates at $\omega_d = \sqrt{\omega_0^2 - \gamma^2}$ with amplitude decaying as $e^{-\gamma t}$ |
| Critically damped | $\gamma = \omega_0$ | Returns to equilibrium in the shortest time without overshoot |
| Overdamped | $\gamma > \omega_0$ | Returns to equilibrium slowly, without oscillating |

The quality factor $Q = \omega_0/(2\gamma)$ measures how many radians of oscillation occur before the energy falls by $e^{-1}$; high-$Q$ systems (a tuning fork, a microwave cavity) ring for a long time.

### Driven Resonance

Push a damped oscillator with a periodic force $F_0\cos(\omega t)$ and the steady-state amplitude is

$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + 4\gamma^2\omega^2}}.$$

The amplitude peaks sharply near $\omega \approx \omega_0$ — **resonance**, the phenomenon behind everything from pushing a swing to the operation of an LC radio tuner, and the reason engineers fear matching a structure's natural frequency.

## Coupled Oscillators and Normal Modes

A single oscillator is the seed; the moment we couple two of them, the new physics of **normal modes** appears, and these modes are the conceptual bridge to waves.

### Two Coupled Masses

Consider two equal masses $m$ connected to fixed walls by springs of stiffness $k$ and to each other by a coupling spring $k_c$. Let $x_1, x_2$ be their displacements from equilibrium. Newton's laws give two *coupled* equations:

$$\begin{aligned}
m\ddot{x}_1 &= -kx_1 - k_c(x_1 - x_2),\\
m\ddot{x}_2 &= -kx_2 - k_c(x_2 - x_1).
\end{aligned}$$

The coupling makes neither mass a simple oscillator on its own. The trick is to find combinations of coordinates that *do* decouple. Add and subtract the equations:

$$\begin{aligned}
\eta_+ &= x_1 + x_2: \qquad m\ddot{\eta}_+ = -k\,\eta_+,\\
\eta_- &= x_1 - x_2: \qquad m\ddot{\eta}_- = -(k + 2k_c)\,\eta_-.
\end{aligned}$$

Each combination is now an independent SHM. These special coordinates are the **normal modes**, and their frequencies are the **normal frequencies**:

$$\omega_+ = \sqrt{\frac{k}{m}} \quad (\text{symmetric: masses move together}), \qquad \omega_- = \sqrt{\frac{k + 2k_c}{m}} \quad (\text{antisymmetric: masses move opposite}).$$

In the symmetric mode the coupling spring is never stretched, so it does not contribute — hence the lower frequency. In the antisymmetric mode the coupling spring is maximally stressed, raising the frequency. Any motion whatsoever is a superposition of these two modes. The slow exchange of energy between two weakly coupled identical pendulums (**beats**) is exactly the interference of two nearly equal normal frequencies.

### The General Eigenvalue Picture

For $N$ coupled oscillators the equations of motion in matrix form are

$$\mathbf{M}\ddot{\mathbf{x}} = -\mathbf{K}\mathbf{x},$$

with mass matrix $\mathbf{M}$ and stiffness matrix $\mathbf{K}$. Substituting $\mathbf{x} = \mathbf{a}\,e^{i\omega t}$ turns this into a generalized eigenvalue problem:

$$\left(\mathbf{K} - \omega^2 \mathbf{M}\right)\mathbf{a} = 0.$$

Nontrivial solutions require $\det(\mathbf{K} - \omega^2\mathbf{M}) = 0$, an order-$N$ polynomial in $\omega^2$ whose roots are the **normal-mode frequencies** $\omega_n$ and whose eigenvectors $\mathbf{a}_n$ are the **mode shapes**. The crucial structural result: a system of $N$ coupled linear oscillators has exactly $N$ normal modes, each behaving as an independent simple oscillator, and **every** motion is a linear superposition of them. Solving the coupled dynamics reduces to diagonalizing a matrix.

### From a Chain to a Continuum

Now line up $N$ identical masses connected by springs, like beads on a stretched string. Label the displacement of the $n$-th mass $u_n$. The force on it comes from its two neighbors:

$$m\ddot{u}_n = k(u_{n+1} - u_n) - k(u_n - u_{n-1}) = k(u_{n+1} - 2u_n + u_{n-1}).$$

Look for traveling-wave solutions $u_n = A\,e^{i(qna - \omega t)}$, where $a$ is the spacing between masses and $q$ is a wavenumber. Substituting gives the **dispersion relation of the discrete chain**:

$$\omega(q) = 2\sqrt{\frac{k}{m}}\,\left|\sin\!\left(\frac{qa}{2}\right)\right|.$$

This single formula is rich. For long wavelengths ($qa \ll 1$) it linearizes to $\omega \approx \sqrt{k/m}\,a\,|q| = c\,|q|$ with a constant speed $c = a\sqrt{k/m}$ — **non-dispersive sound-like waves**. For short wavelengths the relation bends over and saturates: the discreteness of the lattice matters, and waves of different wavelength travel at different speeds. Letting the spacing $a \to 0$ while holding the long-wavelength speed fixed turns the chain into a continuous medium, and the discrete difference $u_{n+1} - 2u_n + u_{n-1}$ becomes a second spatial derivative — giving the wave equation. The normal modes of the chain become the harmonics of a string; the $N$ discrete modes become a continuum of allowed wavelengths. This chain is also the launching point for phonons in [solid-state physics](../../technology/) and lattice field theory.

## The Wave Equation

The continuum limit of the coupled chain, and equally the small-amplitude motion of a stretched string, an air column, or an electromagnetic field, is governed by the **classical wave equation**:

$$\frac{\partial^2 y}{\partial t^2} = v^2\,\frac{\partial^2 y}{\partial x^2}.$$

Here $y(x,t)$ is the local displacement and $v$ is the wave speed, fixed by the medium's restoring stiffness and inertia. For a string of tension $\mathcal{T}$ and linear mass density $\mu$,

$$v = \sqrt{\frac{\mathcal{T}}{\mu}}.$$

### Derivation from a String

Consider a small element of a taut string between $x$ and $x + dx$. The vertical component of tension at each end is $\mathcal{T}\,\partial y/\partial x$; the net upward force is the difference, $\mathcal{T}\,(\partial^2 y/\partial x^2)\,dx$. Newton's second law for the element, mass $\mu\,dx$, gives

$$\mu\,dx\,\frac{\partial^2 y}{\partial t^2} = \mathcal{T}\,\frac{\partial^2 y}{\partial x^2}\,dx,$$

which is the wave equation with $v^2 = \mathcal{T}/\mu$. The same structure — inertia on the left, a restoring spatial curvature on the right — recurs for sound (pressure and bulk modulus), for light (Maxwell's equations give $v = c$), and for the small transverse waves of the coupled chain above.

### d'Alembert's General Solution

The one-dimensional wave equation has a remarkably simple general solution, due to d'Alembert:

$$y(x,t) = f(x - vt) + g(x + vt).$$

*Any* twice-differentiable shape $f$ rigidly translates to the right at speed $v$, and any $g$ translates to the left, without changing form. The wave equation does not pick out sinusoids — it propagates arbitrary profiles. Initial displacement $y(x,0)$ and velocity $\partial_t y(x,0)$ determine $f$ and $g$ uniquely. The fact that shapes propagate undistorted is special to the *non-dispersive* wave equation; we will see below that dispersion breaks it.

### Sinusoidal Waves and the Basic Vocabulary

Among all solutions, the sinusoidal traveling wave

$$y(x,t) = A\cos(kx - \omega t + \phi)$$

is the natural building block, because Fourier analysis lets us assemble any wave from these. The standard quantities:

| Symbol | Name | Relation |
|--------|------|----------|
| $\lambda$ | wavelength | $\lambda = 2\pi/k$ |
| $k$ | wavenumber | $k = 2\pi/\lambda$ |
| $f$ | frequency | $f = \omega/2\pi$ |
| $\omega$ | angular frequency | $\omega = 2\pi f$ |
| $T$ | period | $T = 1/f$ |
| $v$ | wave (phase) speed | $v = f\lambda = \omega/k$ |

Substituting the sinusoid into the wave equation forces $\omega = vk$ — a **linear** dispersion relation, the signature of a non-dispersive medium.

## Standing Waves versus Traveling Waves

The same wave equation supports two qualitatively different behaviors depending on boundary conditions.

A **traveling wave** carries energy and a recognizable shape across space: $y = A\cos(kx - \omega t)$ moves rightward forever in an unbounded medium. A **standing wave** carries no net energy across the medium; it oscillates in place, the result of two equal counter-propagating traveling waves interfering. Add a right- and left-moving wave of equal amplitude:

$$A\cos(kx - \omega t) + A\cos(kx + \omega t) = 2A\cos(kx)\cos(\omega t).$$

Space and time have **separated**: the shape $\cos(kx)$ is fixed and the whole pattern breathes in time with $\cos(\omega t)$. Points where $\cos(kx) = 0$ never move (**nodes**); points of maximal swing are **antinodes**.

### Boundary Conditions Quantize the Modes

Confine a string of length $L$ between two fixed ends, $y(0,t) = y(L,t) = 0$. Only standing waves with a node at each end survive, which requires

$$k_n = \frac{n\pi}{L}, \qquad \lambda_n = \frac{2L}{n}, \qquad f_n = \frac{n}{2L}\sqrt{\frac{\mathcal{T}}{\mu}}, \qquad n = 1, 2, 3, \ldots$$

These are the **normal modes** of the string — the continuum descendants of the coupled-oscillator modes above. The lowest, $n=1$, is the **fundamental**; the rest are **harmonics**, integer multiples of the fundamental. Their integer ratios are precisely why plucked strings and blown pipes sound musical. A pipe open at one end and closed at the other instead admits only odd harmonics, giving it a distinctive timbre. The discreteness of allowed wavelengths from confinement is also the direct classical ancestor of energy quantization in the quantum particle-in-a-box.

## Dispersion, Phase Velocity, and Group Velocity

In the ideal string the relation $\omega = vk$ is linear and *all* wavelengths travel at the same speed $v$. Most real media are not so kind: stiffness, geometry, or the underlying lattice make $\omega$ a nonlinear function of $k$, the **dispersion relation** $\omega = \omega(k)$. When that happens we must distinguish two velocities.

### Phase Velocity

The phase velocity is the speed of a single sinusoidal crest:

$$v_p = \frac{\omega}{k}.$$

It tells you how fast a point of constant phase (a crest) moves. For the non-dispersive string it equals $v$ for every wave; in a dispersive medium it depends on $k$.

### Group Velocity

A real signal is never a single infinite sinusoid — it is a **wave packet**, a superposition of nearby wavenumbers. The packet's envelope, which carries the energy and the information, moves at the **group velocity**:

$$v_g = \frac{d\omega}{dk}.$$

To see why, superpose two waves of nearly equal wavenumbers $k \pm \Delta k$ and frequencies $\omega \pm \Delta\omega$:

$$\cos\big((k+\Delta k)x - (\omega+\Delta\omega)t\big) + \cos\big((k-\Delta k)x - (\omega-\Delta\omega)t\big) = 2\cos(\Delta k\,x - \Delta\omega\,t)\,\cos(kx - \omega t).$$

The fast carrier $\cos(kx - \omega t)$ moves at $v_p = \omega/k$, while the slow envelope $\cos(\Delta k\,x - \Delta\omega\,t)$ moves at $\Delta\omega/\Delta k \to d\omega/dk = v_g$. In a **non-dispersive** medium $\omega = vk$ so $v_p = v_g = v$ and packets keep their shape (this is d'Alembert's undistorted propagation). In a **dispersive** medium the two differ:

$$v_g = \frac{d\omega}{dk} = v_p + k\frac{dv_p}{dk}.$$

The component sinusoids slide through the envelope, and the packet spreads and distorts as it travels — the reason a sharp pulse on a stiff rod, or a light pulse in an optical fiber, broadens. When $dv_p/dk < 0$ ("normal dispersion") $v_g < v_p$; when $dv_p/dk > 0$ ("anomalous dispersion") $v_g > v_p$. Information and energy always travel at $v_g$, which is why a phase velocity exceeding $c$ (possible in some media) violates no relativity.

### Worked Example: Deep-Water Waves

Surface gravity waves on deep water obey $\omega = \sqrt{gk}$. Then

$$v_p = \frac{\omega}{k} = \sqrt{\frac{g}{k}}, \qquad v_g = \frac{d\omega}{dk} = \frac{1}{2}\sqrt{\frac{g}{k}} = \tfrac{1}{2}v_p.$$

The envelope of a wave group moves at *half* the speed of the individual crests: watch a patch of swell and you see crests appear at the back of the group, race forward through it, and vanish at the front. This concrete, observable consequence of dispersion is a textbook demonstration that $v_g \ne v_p$.

## Energy Transport

Waves carry energy without carrying matter. For a wave $y(x,t)$ on a string, the energy is stored partly as kinetic energy of the moving elements and partly as potential energy of the stretched string:

$$\frac{dE}{dx} = \underbrace{\tfrac{1}{2}\mu\left(\frac{\partial y}{\partial t}\right)^2}_{\text{kinetic}} + \underbrace{\tfrac{1}{2}\mathcal{T}\left(\frac{\partial y}{\partial x}\right)^2}_{\text{potential}}.$$

For a sinusoidal traveling wave $y = A\cos(kx - \omega t)$, both densities average to the same value, and the time-averaged **power** transmitted past a point — the rate at which energy flows down the string — is

$$\langle P\rangle = \tfrac{1}{2}\mu\,v\,\omega^2 A^2.$$

The two universal scalings here are central: transmitted power is proportional to the **square of the amplitude** and to the **square of the frequency**. The amplitude-squared law is why intensity (energy flux) is the physically measured quantity in optics and acoustics, and why a wave's "loudness" or "brightness" tracks $A^2$ rather than $A$. For three-dimensional waves spreading from a point source, energy conservation over expanding spherical shells of area $4\pi r^2$ forces the intensity to fall as $1/r^2$ (the inverse-square law) and the amplitude as $1/r$.

The continuity of energy flow is captured locally by a conservation law: the energy density $u$ and the energy flux (Poynting-like) $S$ satisfy

$$\frac{\partial u}{\partial t} + \frac{\partial S}{\partial x} = 0,$$

stating that energy is neither created nor destroyed, only transported — the wave analog of mass conservation in fluid flow.

## Fourier Decomposition

The reason the sinusoidal wave deserves its privileged status is **Fourier's theorem**: because the wave equation is *linear*, any solution is a superposition of sinusoids, and any reasonable initial shape can be built from them. A periodic profile of period $L$ expands as a **Fourier series**:

$$y(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}\left[a_n\cos\!\left(\frac{2\pi n x}{L}\right) + b_n\sin\!\left(\frac{2\pi n x}{L}\right)\right],$$

with coefficients extracted by projecting onto each harmonic:

$$a_n = \frac{2}{L}\int_0^L y(x)\cos\!\left(\frac{2\pi n x}{L}\right)dx, \qquad b_n = \frac{2}{L}\int_0^L y(x)\sin\!\left(\frac{2\pi n x}{L}\right)dx.$$

A non-periodic profile uses the continuous **Fourier transform**:

$$\tilde{y}(k) = \int_{-\infty}^{\infty} y(x)\,e^{-ikx}\,dx, \qquad y(x) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \tilde{y}(k)\,e^{ikx}\,dk.$$

This is the master tool of wave physics, and its consequences are large:

- **Mode independence.** Each Fourier component is a normal mode that evolves on its own, $e^{ikx} \to e^{i(kx - \omega(k)t)}$. Solving a wave problem reduces to (i) Fourier-decomposing the initial data, (ii) advancing each mode by its own phase, (iii) summing back. This is precisely how dispersion spreads a packet: different $k$ accumulate different phases $\omega(k)t$.
- **Timbre.** The Fourier amplitudes of a vibrating string are exactly the strengths of its harmonics — the spectrum that distinguishes a violin from a flute playing the same pitch.
- **Bandwidth–duration reciprocity.** A pulse narrow in space (small $\Delta x$) requires a broad band of wavenumbers (large $\Delta k$), with $\Delta x\,\Delta k \gtrsim 1$. This classical Fourier inequality is the mathematical skeleton on which the quantum Heisenberg uncertainty principle is built.

Fourier decomposition is the workhorse behind signal processing, optics, spectroscopy, and the numerical solution of wave equations (spectral methods), and it is the natural language connecting classical waves to the [wavefunctions of quantum mechanics](../quantum-mechanics/).

### Code: Building a Wave from Its Fourier Modes

A square wave assembled from its odd harmonics makes the decomposition concrete and shows the characteristic Gibbs overshoot near the jumps:

```python
import numpy as np

def square_wave_fourier(x, n_terms):
    """Fourier-series approximation of a square wave (period 2*pi).

    Only odd harmonics appear; each evolves independently under the
    wave equation, which is what makes superposition so powerful.
    """
    y = np.zeros_like(x)
    for n in range(1, 2 * n_terms, 2):   # n = 1, 3, 5, ...
        y += (4 / (np.pi * n)) * np.sin(n * x)
    return y

x = np.linspace(0, 2 * np.pi, 1000)
y1  = square_wave_fourier(x, 1)    # fundamental only
y5  = square_wave_fourier(x, 5)    # 5 odd harmonics
y50 = square_wave_fourier(x, 50)   # converges toward the square wave
```

## Introduction to Nonlinear Waves

Everything above rests on one assumption: small amplitude, so the restoring force is linear and the wave equation holds. Push harder and the linear approximation fails. Two new and characteristically nonlinear effects appear.

**Amplitude-dependent speed and steepening.** In many media the wave speed depends on the local amplitude. Crests then travel faster than troughs, the leading face of a wave steepens, and a smooth profile can sharpen into a **shock** — the mechanism behind sonic booms and breaking ocean waves. A minimal model is the inviscid Burgers equation,

$$\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = 0,$$

whose $u\,\partial_x u$ term is the prototype nonlinearity: the wave advects itself, so taller parts overtake shorter ones and the profile self-steepens until it breaks.

**Solitons: when nonlinearity and dispersion balance.** Dispersion spreads a packet; nonlinear steepening sharpens it. When the two effects exactly cancel, a wave can propagate without changing shape at all — a **soliton**. The classic example is the Korteweg–de Vries (KdV) equation for shallow-water waves,

$$\frac{\partial u}{\partial t} + 6u\,\frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0,$$

where the nonlinear $6u\,\partial_x u$ steepening is balanced against the dispersive $\partial_x^3 u$ spreading. KdV famously admits an exact solitary-wave solution,

$$u(x,t) = \tfrac{1}{2}c\,\operatorname{sech}^2\!\left[\tfrac{1}{2}\sqrt{c}\,(x - ct)\right],$$

a localized hump that travels at constant speed $c$, keeps its shape indefinitely, and even survives collisions with other solitons. John Scott Russell first observed such a wave chasing a canal boat in 1834. Solitons now appear in optical fibers, plasmas, Bose–Einstein condensates, and integrable field theories, and they mark the frontier where the tidy linear world of this page gives way to the [nonlinear dynamics and chaos](chaos-and-computational.html) of the next.

The linear theory remains the indispensable foundation: every nonlinear analysis begins by understanding the linear normal modes, then asks how weak nonlinearity couples them. Oscillations and waves are thus not a closed topic but the gateway from classical mechanics into optics, acoustics, condensed matter, field theory, and quantum mechanics.

---

## Continue

| Previous | Next |
|----------|------|
| [&larr; Newtonian Mechanics &amp; Conservation Laws](newtonian.html) | [Lagrangian &amp; Hamiltonian Mechanics &rarr;](lagrangian-hamiltonian.html) |

## See Also

- [Newtonian Mechanics &amp; Conservation Laws](newtonian.html) — the force-based foundation where the simple harmonic oscillator and the wave equation first appear.
- [Lagrangian &amp; Hamiltonian Mechanics](lagrangian-hamiltonian.html) — normal modes re-derived as the small-oscillation limit of the Lagrangian, and action-angle variables.
- [Chaos, Modern Topics &amp; Computation](chaos-and-computational.html) — where nonlinear waves and driven oscillators become chaotic.
- [Quantum Mechanics](../quantum-mechanics/) — where standing-wave quantization, Fourier analysis, and the harmonic oscillator carry directly into wavefunctions.
- [Computational Physics](../computational-physics/) — numerical methods for solving wave equations and oscillator dynamics.
