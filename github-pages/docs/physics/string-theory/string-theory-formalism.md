---
layout: docs
title: "String Theory: Graduate Formalism"
permalink: /docs/physics/string-theory/string-theory-formalism.html
toc: true
toc_sticky: true
---

<!-- Custom styles for string theory visualizations -->
<link rel="stylesheet" href="{{ '/assets/css/physics-string-theory.css' | relative_url }}">

[String Theory](./) &raquo; Graduate Formalism

## Graduate Formalism

This page is a self-contained, graduate-level treatment of the machinery of string theory. It assumes familiarity with quantum field theory, group theory, and differential geometry. We build the **worldsheet conformal field theory**, quantize the bosonic and super-strings in both the **RNS** and **Green-Schwarz** formalisms, develop **BRST cohomology**, treat **D-branes** and their effective actions, compactify on **Calabi-Yau** manifolds, organize the **duality web** and **M-theory**, derive the **AdS/CFT** dictionary, count **black-hole microstates**, and survey **topological strings** and **modern amplitude methods**. Every formula is written so it renders correctly under MathJax; conceptual narratives of the same material live on the sibling pages linked at the foot.

## Worldsheet Conformal Field Theory

The relativistic string sweeps out a two-dimensional **worldsheet** $\Sigma$ embedded in $D$-dimensional spacetime by the maps $X^{\mu}(\sigma^0,\sigma^1)$. Quantizing the string is, after gauge fixing, exactly the problem of a two-dimensional **conformal field theory** living on $\Sigma$. The entire spectrum, interactions, and consistency conditions of the string are dictated by the structure of this CFT.

### Polyakov Path Integral

The dynamics follow from the **Polyakov action**, which introduces an independent worldsheet metric $h_{ab}$:

$$S_P = -\frac{1}{4\pi\alpha'} \int d^2\sigma \, \sqrt{-h} \, h^{ab} \partial_a X^{\mu} \partial_b X^{\nu} G_{\mu\nu}(X)$$

Here $\alpha' = \ell_s^2$ is the **Regge slope** (the inverse string tension up to $2\pi$), and $G_{\mu\nu}$ is the spacetime metric. The action enjoys three local symmetries: $D$-dimensional Poincaré invariance (for flat $G_{\mu\nu}=\eta_{\mu\nu}$), worldsheet **diffeomorphism** invariance, and **Weyl** invariance $h_{ab}\to e^{2\omega(\sigma)}h_{ab}$. The three gauge parameters of diffeomorphisms-plus-Weyl exactly match the three independent components of the symmetric $h_{ab}$, so the metric is pure gauge classically.

**Conformal gauge** fixes $h_{ab} = e^{\phi}\eta_{ab}$. In flat spacetime the Weyl factor $\phi$ decouples (classically), leaving the free action

$$S = \frac{1}{4\pi\alpha'} \int d^2\sigma \, \partial X^{\mu}\bar{\partial}X_{\mu},$$

where we have passed to **complex coordinates** $z = e^{\sigma^0 + i\sigma^1}$ on the Euclidean cylinder mapped to the plane, with $\partial \equiv \partial_z$ and $\bar{\partial} \equiv \partial_{\bar z}$. The equation of motion $\partial\bar{\partial}X^{\mu}=0$ splits each field into independent **holomorphic** (left-moving) and **antiholomorphic** (right-moving) parts.

### Mode Expansion and the OPE

For the closed string the field admits the **mode expansion**

$$X^{\mu}(z,\bar{z}) = x^{\mu} - \frac{i\alpha'}{2} p^{\mu} \ln|z|^2 + i\sqrt{\frac{\alpha'}{2}} \sum_{n\neq 0} \frac{1}{n}\left[\alpha^{\mu}_n z^{-n} + \tilde{\alpha}^{\mu}_n \bar{z}^{-n}\right].$$

Canonical quantization promotes the modes to operators with

$$[\alpha^{\mu}_m, \alpha^{\nu}_n] = m\,\delta_{m+n,0}\,\eta^{\mu\nu}, \qquad [\tilde{\alpha}^{\mu}_m, \tilde{\alpha}^{\nu}_n] = m\,\delta_{m+n,0}\,\eta^{\mu\nu}, \qquad [x^{\mu},p^{\nu}]=i\eta^{\mu\nu}.$$

The negative-norm states from the timelike $\eta^{00}=-1$ are the reason a careful treatment of gauge constraints is essential. The fundamental **operator product expansion** of the free boson is

$$X^{\mu}(z,\bar{z})\,X^{\nu}(0,0) \sim -\frac{\alpha'}{2}\eta^{\mu\nu}\ln|z|^2,$$

from which every correlation function follows by Wick contraction.

### Stress Tensor and the Virasoro Algebra

The holomorphic **stress tensor** is

$$T(z) = -\frac{1}{\alpha'}:\partial X^{\mu}\partial X_{\mu}:, \qquad T(z) = \sum_n \frac{L_n}{z^{n+2}}, \qquad L_n = \frac{1}{2}\sum_m :\alpha_{n-m}\cdot\alpha_m:.$$

The $L_n$ are the **Virasoro generators**. Their OPE encodes the **central charge** $c$ through the most singular term:

$$T(z)\,T(w) \sim \frac{c/2}{(z-w)^4} + \frac{2T(w)}{(z-w)^2} + \frac{\partial T(w)}{z-w},$$

equivalent to the **Virasoro algebra**

$$[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12} m(m^2-1)\delta_{m+n,0}.$$

For $D$ free bosons $c = D$. A **primary field** of conformal weight $(h,\tilde h)$ transforms as

$$T(z)\,\mathcal{O}(w,\bar w) \sim \frac{h\,\mathcal{O}(w,\bar w)}{(z-w)^2} + \frac{\partial\mathcal{O}(w,\bar w)}{z-w},$$

and $L_0 + \tilde L_0$ generates dilations (the worldsheet Hamiltonian), while $L_0 - \tilde L_0$ generates rotations. **Level matching** $L_0 = \tilde L_0$ on physical states follows from invariance under $\sigma^1\to\sigma^1+2\pi$.

### The Critical Dimension and the Weyl Anomaly

Weyl invariance is a **gauge** symmetry; it must survive quantization or the longitudinal mode $\phi$ fails to decouple and the theory is inconsistent. Quantum mechanically the trace of the stress tensor is

$$T^a{}_a = -\frac{c_{\text{tot}}}{12}R^{(2)},$$

where $R^{(2)}$ is the worldsheet Ricci scalar and $c_{\text{tot}}$ sums the matter and ghost central charges. The reparametrization ghosts (next section) contribute $c_{\text{gh}}=-26$, so the **Weyl anomaly** cancels only if

$$c_{\text{matter}} = 26 \quad\Longrightarrow\quad D = 26 \quad(\text{bosonic string}).$$

This is the celebrated **critical dimension**. The same condition appears in light-cone gauge as the requirement that the Lorentz algebra close, fixing simultaneously $D=26$ and the **normal-ordering constant** $a=1$ in the mass formula

$$\alpha' M^2 = 4(N - 1), \qquad N = \sum_{n>0}\alpha_{-n}\cdot\alpha_n.$$

The $N=0$ ground state has $M^2<0$: the bosonic-string **tachyon**, removed by supersymmetry below.

### Vertex Operators

Asymptotic string states map to local operators on $\Sigma$ via the **state-operator correspondence** of radial quantization. Emission/absorption of a string is represented by integrating a **vertex operator** of total weight $(1,1)$ (so it is Weyl-invariant under integration).

The **tachyon** vertex operator is

$$V_T = g_s \int d^2z \, :e^{ik\cdot X}:, \qquad \alpha' k^2 = 4,$$

with weight $(\alpha'k^2/4, \alpha'k^2/4)$. The massless **graviton, B-field, and dilaton** are packaged in

$$V = \zeta_{\mu\nu} \int d^2z \, :\partial X^{\mu}\,\bar{\partial}X^{\nu}\,e^{ik\cdot X}:, \qquad k^2 = 0, \quad k^{\mu}\zeta_{\mu\nu}=0,$$

where the symmetric traceless part of $\zeta$ is the graviton $g_{\mu\nu}$, the antisymmetric part the Kalb-Ramond $B_{\mu\nu}$, and the trace the dilaton $\Phi$. In the superstring the supersymmetric completion adds the fermion bilinear,

$$V^{(0)} = \zeta_{\mu\nu} \int d^2z \, :(\partial X^{\mu} + i\,k\!\cdot\!\psi\,\psi^{\mu})(\bar\partial X^{\nu} + i\,k\!\cdot\!\tilde\psi\,\tilde\psi^{\nu})\,e^{ik\cdot X}:.$$

Vertex operators come in **fixed** and **integrated** forms; the three fixed insertions on the sphere soak up the $SL(2,\mathbb{C})$ conformal Killing volume, and the remaining $n-3$ punctures are integrated:

$$A_n \sim g_s^{n-2}\int \prod_{i=4}^{n} d^2z_i \, \big\langle\, V_1(z_1)V_2(z_2)V_3(z_3)\prod_{i\ge 4} V_i(z_i)\,\big\rangle.$$

### BRST Quantization

Covariant quantization that keeps Lorentz invariance manifest replaces gauge fixing by the **Faddeev-Popov** procedure, introducing anticommuting **ghosts** $(b,c)$ of weights $(2,-1)$ with

$$c(z)\,b(w) \sim \frac{1}{z-w}, \qquad T_{gh} = -2b\,\partial c - (\partial b)\,c, \qquad c_{gh} = -26.$$

The nilpotent **BRST charge** is

$$Q_B = \oint \frac{dz}{2\pi i}\left(c\,T_{\text{matter}} + \tfrac{1}{2}c\,T_{gh}\right) = \oint \frac{dz}{2\pi i}\left(cT + b\,c\,\partial c\right),$$

with the closed-string version adding the antiholomorphic copy $\tilde Q_B$. Nilpotency $Q_B^2=0$ holds **if and only if** $c_{\text{matter}}=26$ — the same critical-dimension condition, now phrased cohomologically. Physical states are the **BRST cohomology**:

$$Q_B\lvert\phi\rangle = 0, \qquad \lvert\phi\rangle \sim \lvert\phi\rangle + Q_B\lvert\chi\rangle,$$

i.e. closed modulo exact. The physical Hilbert space is $H^*(Q_B)$ at ghost number 1, and one proves the **no-ghost theorem**: cohomology classes have positive norm, the timelike and longitudinal oscillator excitations being removed exactly as in light-cone gauge. Amplitudes are BRST-invariant correlators, and the **$b_0 - \tilde b_0 = 0$** condition implements level matching while $b$-ghost insertions provide the measure on moduli space.

## Superstring Theory: RNS Formalism

The bosonic string is sick (a tachyon, no spacetime fermions). The cure is **worldsheet supersymmetry**: adjoin to each $X^{\mu}$ a Majorana fermion $\psi^{\mu}$. The **Ramond-Neveu-Schwarz (RNS)** formalism makes worldsheet SUSY manifest at the cost of needing the **GSO projection** to recover spacetime SUSY.

### Worldsheet Supersymmetry

The gauge-fixed **RNS action** in superconformal gauge is

$$S = \frac{1}{4\pi\alpha'} \int d^2\sigma \left[\partial_{\alpha}X^{\mu}\partial^{\alpha}X_{\mu} - i\,\bar\psi^{\mu}\rho^{\alpha}\partial_{\alpha}\psi_{\mu}\right],$$

with $\rho^{\alpha}$ the two-dimensional Dirac matrices satisfying $\{\rho^{\alpha},\rho^{\beta}\}=2\eta^{\alpha\beta}$. It is invariant under the **worldsheet supersymmetry**

$$\delta X^{\mu} = \bar\epsilon\,\psi^{\mu}, \qquad \delta\psi^{\mu} = -i\,\rho^{\alpha}\partial_{\alpha}X^{\mu}\,\epsilon.$$

In addition to $T(z)$, the theory has a fermionic **supercurrent**

$$G(z) = i\sqrt{\frac{2}{\alpha'}}\,\psi^{\mu}\partial X_{\mu} = \sum_r \frac{G_r}{z^{r+3/2}},$$

whose modes generate the $N=1$ **superconformal algebra**

$$[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12}m(m^2-1)\delta_{m+n,0},$$

$$[L_m, G_r] = \left(\frac{m}{2} - r\right)G_{m+r}, \qquad \{G_r, G_s\} = 2L_{r+s} + \frac{c}{12}\left(4r^2 - 1\right)\delta_{r+s,0}.$$

Each free $(X,\psi)$ pair contributes $c = 1 + \tfrac{1}{2} = \tfrac{3}{2}$, so $c_{\text{matter}} = \tfrac{3D}{2}$. The superconformal ghosts $(b,c)$ and $(\beta,\gamma)$ contribute $c_{gh} = -26 + 11 = -15$, and anomaly cancellation $\tfrac{3D}{2}=15$ fixes the **superstring critical dimension**

$$D = 10.$$

### NS and R Sectors

The worldsheet fermions admit two boundary conditions around the cylinder, labeling two **sectors**:

- **Neveu-Schwarz (NS):** $\psi^{\mu}(\sigma+2\pi)=-\psi^{\mu}(\sigma)$, half-integer modes $\psi_r$, $r\in\mathbb{Z}+\tfrac{1}{2}$. The ground state is a spacetime **boson**; the lowest excitation $\psi^{\mu}_{-1/2}\lvert 0\rangle_{\text{NS}}$ is the would-be vector.
- **Ramond (R):** $\psi^{\mu}(\sigma+2\pi)=+\psi^{\mu}(\sigma)$, integer modes $\psi_n$. The zero modes $\psi^{\mu}_0$ satisfy $\{\psi^{\mu}_0,\psi^{\nu}_0\}=\eta^{\mu\nu}$ — a $D$-dimensional **Clifford algebra** — so the ground state is a spacetime **spinor**. This is the origin of fermions in the spectrum.

The mass formulas are

$$\alpha' M^2 = N - a, \qquad a_{\text{NS}} = \tfrac{1}{2}, \quad a_{\text{R}} = 0,$$

so the NS ground state is again tachyonic until projected out.

### GSO Projection

Consistency (modular invariance, spacetime SUSY, tachyon removal) requires the **Gliozzi-Scherk-Olive (GSO) projection** onto definite worldsheet **fermion parity** $(-1)^F$. Define the NS fermion number

$$F = \sum_{r>0} \psi_{-r}\cdot\psi_r,$$

and the GSO operator $(-1)^{F}$ with the convention that it equals $-1$ on the NS ground state. Keeping the **even** NS states discards the tachyon and retains $\psi^{\mu}_{-1/2}\lvert 0\rangle$ as the massless vector. In the R sector $(-1)^F$ involves the chirality operator $\Gamma_{11}=\Gamma^0\cdots\Gamma^9$, and the GSO choice selects a definite spacetime chirality. Schematically the projection keeps

$$(-1)^{F} = \pm(-1)^{\tilde F},$$

with the relative sign distinguishing the two type-II theories. After projection the massless content assembles into the $D=10$ supergravity multiplets:

- **NS-NS:** $g_{\mu\nu}$ (graviton), $B_{\mu\nu}$, $\Phi$ (dilaton).
- **R-R:** $p$-form potentials $C_p$ sourcing the D-branes.
- **NS-R, R-NS:** the gravitini and dilatini (spacetime fermions).

The choice of relative R-sector chirality yields the **type IIA** (non-chiral, R-R forms $C_1, C_3$) or **type IIB** (chiral, $C_0, C_2, C_4$) theory; orbifolding/orientifolding gives the type I and the two heterotic strings.

### Superstring Spectrum and Spacetime SUSY

After GSO, the NS and R sectors contain **equal numbers** of bosonic and fermionic states at every mass level — the **Gliozzi-Scherk-Olive** observation that the projected spectrum is spacetime supersymmetric. The famous **abstruse identity** of Jacobi,

$$\vartheta_3^4 - \vartheta_4^4 - \vartheta_2^4 = 0,$$

is the partition-function statement that bosons and fermions cancel, giving a vanishing one-loop cosmological constant for the type-II strings.

## Green-Schwarz Formalism

The RNS formalism hides spacetime SUSY (it is only manifest after GSO). The **Green-Schwarz (GS)** formalism instead makes spacetime supersymmetry manifest from the start, treating the Grassmann coordinates $\theta^A$ ($A=1,2$) of superspace as worldsheet fields.

### Spacetime Supersymmetry and Kappa Symmetry

The GS action is

$$S = -\frac{T}{2} \int d^2\sigma \left[\sqrt{-h}\, h^{ab}\,\Pi_a^{\mu}\Pi_{b\mu} - 2i\,\varepsilon^{ab}\,\partial_a X^{\mu}\big(\bar\theta^1\Gamma_{\mu}\partial_b\theta^1 - \bar\theta^2\Gamma_{\mu}\partial_b\theta^2\big) + \dots\right],$$

where the supersymmetric line element is built from

$$\Pi_a^{\mu} = \partial_a X^{\mu} - i\,\bar\theta^A\Gamma^{\mu}\partial_a\theta^A.$$

The second (Wess-Zumino) term exists only when a Fierz identity holds, which requires spacetime dimension $D\in\{3,4,6,10\}$ — and only $D=10$ gives a critical superstring. The action has a fermionic local **kappa symmetry** that gauges away half the components of $\theta^A$, matching the $8$ physical fermionic and $8$ bosonic transverse degrees of freedom. In **light-cone gauge** the GS string becomes a free theory of $8$ transverse bosons $X^i$ and $8$ transverse spacetime spinors $S^a$ (the $\mathbf{8_v}$ and $\mathbf{8_s}$ of $SO(8)$), making spacetime SUSY and the absence of a tachyon manifest. The price is the loss of manifest Lorentz covariance off light-cone gauge; the **pure-spinor formalism** of Berkovits restores covariant quantization by replacing kappa symmetry with a BRST operator built from a constrained ghost $\lambda^{\alpha}$ obeying $\lambda\Gamma^{\mu}\lambda=0$.

## D-Brane Physics

**Dp-branes** are $(p+1)$-dimensional hypersurfaces on which open strings end. They are simultaneously solitonic solutions of supergravity carrying **Ramond-Ramond charge** and dynamical objects with a worldvolume gauge theory.

### Boundary Conditions and T-Duality

Open-string endpoints obey **Neumann** conditions along the brane and **Dirichlet** conditions transverse to it:

$$\text{Neumann:}\quad \partial_n X^{\mu}\big|_{\partial\Sigma} = 0 \ (\mu \parallel \text{brane}), \qquad \text{Dirichlet:}\quad \partial_t X^{\mu}\big|_{\partial\Sigma} = 0 \ (\mu \perp \text{brane}).$$

**T-duality** along a circle exchanges the two: $X = X_L + X_R \leftrightarrow X' = X_L - X_R$ swaps Neumann $\leftrightarrow$ Dirichlet, so a Dp-brane becomes a D$(p\pm1)$-brane. This is why D-branes are unavoidable once T-duality is taken seriously, and why type IIA (even $p$) and IIB (odd $p$) map into each other under T-duality.

### Effective Actions

The low-energy dynamics of a single Dp-brane is governed by the **Dirac-Born-Infeld** action,

$$S_{\text{DBI}} = -T_p \int d^{p+1}\xi \, e^{-\Phi}\sqrt{-\det\left(g_{ab} + B_{ab} + 2\pi\alpha' F_{ab}\right)},$$

with $g_{ab}$ the induced metric, $B_{ab}$ the pullback of the NS-NS two-form, and $F_{ab}$ the worldvolume $U(1)$ field strength. Expanding in $\alpha'$ gives Maxwell theory plus an infinite tower of $\alpha'$ corrections:

$$S_{\text{DBI}} = -T_p\int d^{p+1}\xi \, e^{-\Phi}\left[1 + \frac{(2\pi\alpha')^2}{4} F_{\mu\nu}F^{\mu\nu} + O(F^4)\right] + \dots.$$

The brane also couples to the R-R potentials through the topological **Chern-Simons (Wess-Zumino)** term

$$S_{\text{CS}} = \mu_p \int_{\text{worldvolume}} \sum_q C_q \wedge e^{\,2\pi\alpha' F + B} \wedge \sqrt{\hat{A}(R)},$$

with $\hat A$ the A-roof (Dirac) genus encoding curvature couplings; the leading term $\mu_p\int C_{p+1}$ shows the Dp-brane is an electric source for $C_{p+1}$. BPS saturation relates the tension and charge:

$$T_p = \frac{\mu_p}{1} = \frac{1}{g_s(2\pi)^p \alpha'^{(p+1)/2}}.$$

### Gauge Theory on Branes and Tachyon Condensation

A stack of $N$ coincident Dp-branes carries **Chan-Paton** labels at the open-string endpoints, so the massless vectors form a $U(N)$ adjoint: the worldvolume theory is $(p+1)$-dimensional **$U(N)$ super Yang-Mills**, the dimensional reduction of $\mathcal{N}=1$ SYM in $D=10$. The transverse positions become adjoint scalars $\Phi^i$, and their **noncommutativity** $[\Phi^i,\Phi^j]\neq 0$ encodes bound states and the Myers (dielectric) effect. A coincident **brane-antibrane** pair has an open-string **tachyon** $T$; its condensation $\langle T\rangle\neq 0$ annihilates the pair, and **Sen's conjectures** identify the surviving lower-branes as topological defects. The conserved charges are classified not by ordinary cohomology but by **K-theory** $K(X)$ of spacetime.

## Compactification

To connect the $D=10$ superstring to four-dimensional physics, six dimensions are curled up on a small internal manifold $M_6$. The geometry of $M_6$ fixes the 4D gauge group, matter content, and couplings — string **phenomenology** is the study of which $M_6$ reproduces the Standard Model.

### Calabi-Yau Manifolds

Preserving exactly $\mathcal{N}=1$ supersymmetry in four dimensions requires a covariantly constant spinor on $M_6$, which forces $SU(3)$ **holonomy** — a **Calabi-Yau** threefold. Equivalently, a Calabi-Yau is a compact Kähler manifold with vanishing first Chern class,

$$c_1(M) = 0 \quad\Longleftrightarrow\quad R_{i\bar j} = 0 \ \text{(Ricci-flat, by Yau's theorem)},$$

admitting a nowhere-vanishing holomorphic $(3,0)$-form $\Omega$. Its topology is captured by the **Hodge numbers** $h^{p,q} = \dim H^{p,q}_{\bar\partial}(M)$, of which the independent ones for a CY threefold are

$$h^{1,1} = \#\,\text{Kähler moduli}, \qquad h^{2,1} = \#\,\text{complex-structure moduli}.$$

The Euler character is $\chi = 2(h^{1,1}-h^{2,1})$, and the net number of chiral generations in the simplest heterotic compactification is $\tfrac{1}{2}|\chi|$. **Mirror symmetry** exchanges $h^{1,1}\leftrightarrow h^{2,1}$ between a CY $X$ and its mirror $Y$, trading hard symplectic (Kähler) computations on $X$ for easy complex-geometry computations on $Y$.

### Moduli and Their Stabilization

The massless scalar **moduli** — Kähler $T^i$, complex-structure $U^a$, and the axio-dilaton $S$ — are flat directions of the potential, an embarrassment: they would mediate unobserved fifth forces and leave couplings undetermined. **Flux compactifications** lift them. Turning on quantized three-form fluxes $F_3, H_3$ generates a **Gukov-Vafa-Witten superpotential**

$$W = \int_M \Omega \wedge \big(F_3 - \tau H_3\big), \qquad \tau = C_0 + i e^{-\Phi},$$

whose $F$-term conditions $D_a W = \partial_a W + (\partial_a K)W = 0$ fix the complex-structure moduli and the dilaton. The **KKLT** construction further stabilizes the Kähler moduli using non-perturbative effects (gaugino condensation, Euclidean D3 instantons), $W = W_0 + A e^{-aT}$, yielding (after an uplift) a metastable de Sitter vacuum. The **Large Volume Scenario** instead balances $\alpha'$ corrections against a single non-perturbative term to stabilize an exponentially large internal volume. The astronomical number of flux choices is the origin of the string **landscape**, $\sim 10^{500}$ vacua.

## M-Theory and the Duality Web

The five consistent $D=10$ superstring theories — **type I, type IIA, type IIB, heterotic $SO(32)$, heterotic $E_8\times E_8$** — are not distinct theories but limits of a single underlying structure, related by **dualities** and unified by eleven-dimensional **M-theory**.

### M-Theory and 11D Supergravity

The strong-coupling limit of type IIA grows an **eleventh dimension** of radius $R_{11} = g_s\ell_s$; at low energy the dynamics is the unique **eleven-dimensional supergravity**,

$$S_{11} = \frac{1}{2\kappa_{11}^2}\int d^{11}x\,\sqrt{-g}\left(R - \frac{1}{2}|F_4|^2\right) - \frac{1}{12\kappa_{11}^2}\int C_3\wedge F_4\wedge F_4,$$

with $F_4 = dC_3$. Its charged objects are the **M2-brane** (electric source of $C_3$) and the **M5-brane** (magnetic source, carrying a self-dual three-form on its worldvolume). The type IIA D2 and NS5 descend from the M2 and M5 by reduction on $S^1$; the D0-branes are Kaluza-Klein momentum modes along the M-theory circle.

### S-, T-, and U-Duality

- **T-duality** (perturbative): relates theories on circles of radius $R$ and $\alpha'/R$, exchanging momentum $n/R \leftrightarrow$ winding $wR/\alpha'$. It maps IIA $\leftrightarrow$ IIB and the two heterotic strings into one another.
- **S-duality** (non-perturbative): inverts the coupling $g_s \leftrightarrow 1/g_s$. **Type IIB is self-dual**, with the axio-dilaton transforming under $SL(2,\mathbb{Z})$ as $\tau\to(a\tau+b)/(c\tau+d)$; type I is S-dual to heterotic $SO(32)$.
- **U-duality**: the discrete group generated by S and T together, e.g. $E_{7(7)}(\mathbb{Z})$ for type II on $T^6$, unifying all perturbative and non-perturbative equivalences.

The web is summarized by the reductions of M-theory:

$$\text{M-theory on } S^1 \;\longrightarrow\; \text{Type IIA},$$

$$\text{M-theory on } T^2 \;\longrightarrow\; \text{Type IIB} \ (\text{via shrinking the }T^2;\ SL(2,\mathbb{Z})_\tau = \text{geometric}),$$

$$\text{M-theory on } S^1/\mathbb{Z}_2 \;\longrightarrow\; E_8\times E_8 \ \text{heterotic} \ (\text{Horava-Witten}).$$

## AdS/CFT Correspondence

The **anti-de Sitter / conformal field theory** correspondence is a precise, non-perturbative equivalence between a theory of quantum gravity in AdS and an ordinary gauge theory on its conformal boundary — the sharpest realization of **holography**.

### The Canonical Duality and Dictionary

Maldacena's example is

$$\text{Type IIB on } AdS_5\times S^5 \;\;\longleftrightarrow\;\; \mathcal{N}=4 \text{ SU}(N) \text{ super Yang-Mills in 4D}.$$

The parameters match as

$$g_{\text{YM}}^2 = 4\pi g_s, \qquad \lambda \equiv g_{\text{YM}}^2 N = \frac{L^4}{\alpha'^2}, \qquad \frac{L^4}{\ell_p^4} \sim N,$$

with $L$ the common AdS and sphere radius. Because supergravity is reliable when $L\gg\ell_s$ (i.e. $\lambda\gg1$) while perturbative gauge theory needs $\lambda\ll1$, the duality is **strong-weak**: it computes one side precisely where the other is intractable. The **GKP-Witten dictionary** equates the gravity partition function (with boundary condition $\phi\to\phi_0$ for the bulk field dual to a CFT operator $\mathcal{O}$) to the CFT generating functional,

$$Z_{\text{grav}}\big[\phi\to\phi_0\big] = \Big\langle\, \exp\!\int_{\partial} \phi_0\,\mathcal{O}\,\Big\rangle_{\text{CFT}},$$

so that connected correlators come from functional derivatives of the on-shell gravity action,

$$\langle \mathcal{O}(x_1)\cdots\mathcal{O}(x_n)\rangle = \frac{\delta^n S_{\text{grav}}^{\text{on-shell}}}{\delta\phi_0(x_1)\cdots\delta\phi_0(x_n)}\bigg|_{\phi_0=0}.$$

The conformal dimension $\Delta$ of $\mathcal{O}$ is fixed by the bulk mass via $\Delta(\Delta-d) = m^2 L^2$. Bulk infrared divergences map to boundary ultraviolet divergences, regulated by **holographic renormalization** (covariant counterterms on a cutoff boundary).

### Generalizations

- **$AdS_4/CFT_3$:** M-theory on $AdS_4\times S^7$ is dual to the $\mathcal{N}=6$ Chern-Simons-matter **ABJM** theory.
- **$AdS_3/CFT_2$:** type IIB on $AdS_3\times S^3\times M_4$ is dual to a 2D CFT (the D1-D5 system), where the Cardy formula reproduces black-hole entropy.
- **$AdS_2/CFT_1$:** governs the near-horizon throats of extremal black holes and connects to the **SYK** model.
- **Non-conformal / AdS-CMT:** Dp-branes with $p\neq 3$, and holographic models of superconductors, strange metals, and the quark-gluon plasma.

## Black Holes and Microstate Counting

A central success of the formalism is the **statistical** derivation of black-hole entropy: counting the quantum microstates of a D-brane bound state reproduces the Bekenstein-Hawking area law exactly.

### The Strominger-Vafa Calculation

Consider the **D1-D5-P** system in type IIB compactified on $T^5$: $N_1$ D1-branes and $N_5$ D5-branes wrapping cycles, carrying $n$ units of momentum $P$ along a shared circle. This is a five-dimensional **BPS** (extremal, supersymmetric) black hole. The bound-state degeneracy is the number of ways to partition the momentum $n$ among the $4N_1N_5$ left-moving open-string oscillators; by the Cardy formula the count is $\exp\!\big(2\pi\sqrt{N_1 N_5 n}\big)$, giving

$$S_{\text{micro}} = \ln d(N_1,N_5,n) = 2\pi\sqrt{N_1 N_5 n}.$$

The macroscopic black hole has horizon area $A$ with

$$S_{\text{BH}} = \frac{A}{4G_5} = 2\pi\sqrt{N_1 N_5 n},$$

matching the microscopic count **exactly**, including the factor of $1/4$. Subleading corrections also agree, providing strong evidence that string theory captures the quantum microstructure of horizons.

### Attractor Mechanism

For extremal black holes the moduli $z^i$ flow, under the radial evolution toward the horizon, to fixed points determined purely by the charges, independent of their asymptotic values. The near-horizon geometry is $AdS_2\times S^2$ (or $AdS_2\times S^3$ in 5D), and the moduli sit at the critical points of the central charge / effective potential,

$$\partial_i V_{\text{BH}}(z,\bar z)\big|_{\text{horizon}} = 0, \qquad V_{\text{BH}} = |Z|^2 + g^{i\bar j}D_i Z\,\overline{D_j Z},$$

with $Z(q,p;z)$ the $\mathcal{N}=2$ central charge. This **attractor mechanism** explains why the entropy depends only on quantized charges — exactly as a microstate count must.

## Topological String Theory

**Topological strings** are obtained by **twisting** the worldsheet $\mathcal{N}=(2,2)$ superconformal algebra so that the path integral localizes onto holomorphic/constant maps, computing protected (BPS) quantities exactly.

### A-Model and B-Model

The **A-twist** produces a theory depending only on the **Kähler** structure of the target CY $X$; its free energies are generating functions of **Gromov-Witten** invariants $N_{g,\beta}$ counting holomorphic curves,

$$F_A(t) = \sum_{g\ge0}\sum_{\beta} N_{g,\beta}\,g_s^{2g-2}\,e^{-\beta\cdot t}.$$

The **B-twist** depends only on the **complex structure** of the mirror $Y$ and is computed by period integrals of $\Omega$. **Mirror symmetry** is the statement

$$F_A(X) = F_B(Y),$$

turning intractable curve counts on $X$ into classical-geometry integrals on $Y$. The higher-genus B-model free energies $F^{(g)}$ are governed by the **BCOV holomorphic anomaly equation**

$$\bar\partial_{\bar i} F^{(g)} = \frac{1}{2}\,\bar{C}_{\bar i}^{\,jk}\left(D_j D_k F^{(g-1)} + \sum_{h=1}^{g-1} D_j F^{(h)}\,D_k F^{(g-h)}\right),$$

which recursively determines the $F^{(g)}$ up to a holomorphic ambiguity fixed by boundary conditions. Topological strings compute F-terms in the physical Type II effective action and connect to **Chern-Simons** theory (Gopakumar-Vafa large-$N$ duality) and to BPS state counts.

## Amplitudes and Modern Methods

Beyond the genus-expansion of worldsheet correlators, modern reformulations reorganize string and field-theory amplitudes into strikingly compact forms.

### Scattering Equations and CHY

The **Cachazo-He-Yuan (CHY)** formalism writes tree-level $n$-point massless amplitudes as a contour integral over the moduli space of $n$ punctured spheres, localized on the **scattering equations**

$$\sum_{j\neq i} \frac{k_i\cdot k_j}{\sigma_i - \sigma_j} = 0, \qquad i = 1,\dots,n.$$

The amplitude factorizes into a measure times two half-integrands,

$$A_n = \int d\mu_n \; I_L(\sigma)\,I_R(\sigma), \qquad d\mu_n = \frac{\prod_i d\sigma_i}{\text{vol}\,SL(2,\mathbb{C})}\prod_{i}{}'\,\delta\!\Big(\sum_{j\neq i}\frac{k_i\cdot k_j}{\sigma_i-\sigma_j}\Big),$$

with the choice of $I_L,I_R$ selecting the theory (Yang-Mills, gravity, the bi-adjoint scalar, etc.). The **double copy** — gravity $=$ (gauge theory)$^2$ via Bern-Carrasco-Johansson color-kinematics duality — is manifest here as $I_L=I_R=$ the gauge half-integrand giving gravity.

### Ambitwistor Strings

The CHY formulas are reproduced by a genuine worldsheet model, the **ambitwistor string**, with action

$$S = \int_{\Sigma} P_{\mu}\,\bar\partial X^{\mu} - \frac{e}{2}P^2,$$

a chiral, infinite-tension theory whose path integral localizes precisely onto the scattering equations. Unlike the ordinary string it imposes **no** critical dimension on the bosonic model (the gauge constraint $P^2=0$ does the work), and its loop generalizations compute field-theory loop integrands. Related to these are **twistor strings** (Witten's $\mathcal{N}=4$ SYM model) and the **amplituhedron**, a positive-geometry whose canonical form yields planar $\mathcal{N}=4$ SYM integrands without reference to a Lagrangian.

## Holographic Entanglement and Complexity

AdS/CFT geometrizes quantum-information quantities of the boundary theory, a circle of ideas central to current quantum-gravity research.

### Ryu-Takayanagi

The entanglement entropy of a boundary region $A$ equals the area of the minimal bulk surface $\gamma_A$ homologous to $A$:

$$S_A = \frac{\text{Area}(\gamma_A)}{4 G_N} + S_{\text{bulk}}(\gamma_A) + \dots,$$

the quantum-corrected (FLM/QES) form including bulk entanglement across $\gamma_A$. This identifies the emergent radial direction of AdS with the **entanglement structure** of the dual state — "spacetime built from entanglement." The same surfaces underlie the **subregion duality** and the resolution of the black-hole information problem via the **island** formula and the Page curve.

### Holographic Complexity

Two conjectures relate the **computational complexity** of the boundary state to bulk volumes/actions:

$$\text{Complexity}=\text{Volume:}\quad \mathcal{C}_V = \frac{V(\Sigma_{\max})}{G_N\,\ell}, \qquad \text{Complexity}=\text{Action:}\quad \mathcal{C}_A = \frac{S_{\text{WdW}}}{\pi\hbar},$$

with $\Sigma_{\max}$ a maximal-volume bulk slice and $S_{\text{WdW}}$ the action of the Wheeler-DeWitt patch. These diagnose the late-time growth of the black-hole interior and constrain firewall scenarios.

## The Swampland Program

The complement of the landscape is the **swampland**: low-energy effective theories that look consistent but admit no UV completion in quantum gravity. The program states sharp **conjectures** sorting the consistent from the inconsistent.

- **Distance conjecture:** infinite-distance limits in moduli space spawn an exponentially light tower of states, $m \sim M_P\, e^{-\alpha\, d(\phi)}$, capping the validity of any single EFT.
- **Weak gravity conjecture:** any $U(1)$ must contain a state with $m \le q\, M_P$ (gravity is the weakest force), forbidding stable extremal black-hole remnants.
- **de Sitter conjecture:** scalar potentials of consistent EFTs obey $|\nabla V| \ge c\, V / M_P$ (or a refinement on $\min\nabla^2 V$), disfavoring long-lived metastable de Sitter and constraining inflation.

These tie the otherwise-vast landscape back to falsifiable statements about cosmology and particle physics.

## Computational Tools

The formal structures above feed into concrete computations — Calabi-Yau metrics from Kähler potentials, Yukawa couplings from the holomorphic three-form, Gromov-Witten invariants, and holographic correlators. The following sketches the symbolic scaffolding such calculations rest on.

```python
import numpy as np
from sympy import symbols, Matrix, simplify

def calabi_yau_metric(z, z_bar, kahler_potential):
    """Kähler metric g_{i jbar} = d_i d_jbar K from a Kähler potential."""
    n = len(z)
    g = Matrix.zeros(n, n)
    for i in range(n):
        for j in range(n):
            g[i, j] = kahler_potential.diff(z[i]).diff(z_bar[j])
    return g

def yukawa_coupling(omega, A, B, C):
    """Yukawa Y_ABC = int_X Omega ^ d_A d_B d_C (schematic)."""
    return omega.diff(A).diff(B).diff(C)

def gromov_witten_invariant(degree, genus, marked_points):
    """Placeholder: in practice use localization or mirror symmetry."""
    raise NotImplementedError("Use localization / mirror map for N_{g,beta}.")

def ads_cft_correlator(operators, positions):
    """Placeholder: solve bulk EOM in AdS, extract boundary fall-off."""
    raise NotImplementedError("Solve classical bulk EOM and read the source/vev.")
```

## References and Further Reading

### Classic Textbooks
1. **Polchinski** — *String Theory* (2 volumes) — the standard graduate reference.
2. **Green, Schwarz & Witten** — *Superstring Theory* (2 volumes).
3. **Becker, Becker & Schwarz** — *String Theory and M-Theory*.
4. **Kiritsis** — *String Theory in a Nutshell*.
5. **Blumenhagen, Lüst & Theisen** — *Basic Concepts of String Theory*.

### Advanced Monographs
1. **D'Hoker & Phong** — *Two-loop superstrings* (series).
2. **Hori et al.** — *Mirror Symmetry* (Clay monograph).
3. **Ammon & Erdmenger** — *Gauge/Gravity Duality*.
4. **Nakahara** — *Geometry, Topology and Physics* (for the differential geometry).

### Reviews
1. **Aharony et al.** — *Large N field theories, string theory and gravity* (2000).
2. **Brennan, Carta & Vafa** — *The string landscape, the swampland, and the missing corner* (2017).
3. **Harlow** — *TASI lectures on the emergence of the bulk in AdS/CFT* (2018).
4. **Van Raamsdonk** — *Building up spacetime with quantum entanglement* (2010).
5. **Berkovits** — *Pure spinor formalism* reviews; **Gopakumar & Vafa** — *Topological strings and large N duality*.

---

**Previous:** [D-Branes, Dualities & M-Theory](dualities-and-branes.html) — the narrative treatment of branes, dualities, M-theory, holography, and black holes. **Up:** [String Theory (Overview)](./) — strings, quantization, and the five superstring theories.

## See Also

- [String Theory (Overview)](./) — strings, quantization, and the five superstring theories.
- [D-Branes, Dualities &amp; M-Theory](dualities-and-branes.html) — the narrative treatment of branes, dualities, and holography.
- [Criticisms, Research &amp; Graduate Formalism](frontiers-and-formalism.html) — open problems, current research, and experimental prospects.
- [Quantum Field Theory](../quantum-field-theory.html) — BRST quantization and the field-theory side of AdS/CFT.
- [Computational Physics](../computational-physics/) — numerical and symbolic tools like those above.
