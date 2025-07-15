from manim import *
import numpy as np

class QuantumGraduateLevel(ThreeDScene):
    def construct(self):
        # Opening - start accessible then ramp up
        self.intro_with_formalism()              # 2s
        self.hilbert_space_fundamentals()        # 6s
        self.density_matrices_decoherence()      # 7s
        self.unitary_evolution_hamiltonians()    # 6s
        self.quantum_algorithms_depth()          # 8s
        self.error_correction_topology()         # 6s
        self.quantum_complexity_theory()         # 5s
        
    def intro_with_formalism(self):
        # Start with familiar then quickly advance
        title = Text("QUANTUM COMPUTING", font_size=48, color=BLUE)
        subtitle = Text("A Mathematical Framework", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), run_time=0.5)
        self.play(FadeIn(subtitle), run_time=0.3)
        
        # Transition to formalism
        postulate = Text("Postulate 1: State Space", font_size=20, color=YELLOW)
        postulate.to_edge(UP)
        
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            Write(postulate),
            run_time=0.5
        )
        
        # Hilbert space definition - fixed LaTeX
        hilbert_def = MathTex(
            r"|\psi\rangle \in \mathcal{H} = \mathbb{C}^{2^n}",
            font_size=36
        )
        hilbert_def.shift(UP * 2)
        
        self.play(Write(hilbert_def), run_time=0.7)
        self.wait(0.5)
        
    def hilbert_space_fundamentals(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        
        title = Text("HILBERT SPACE STRUCTURE", font_size=32, color=GREEN)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)
        
        # Single qubit basis
        basis_title = Text("Computational Basis:", font_size=20)
        basis_title.shift(LEFT * 4 + UP * 2)
        
        basis_0 = MathTex(r"|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}", font_size=28)
        basis_1 = MathTex(r"|1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}", font_size=28)
        basis_0.shift(LEFT * 4)
        basis_1.shift(LEFT * 4 + DOWN)
        
        self.play(
            Write(basis_title),
            Write(basis_0),
            Write(basis_1),
            run_time=0.8
        )
        
        # General state
        general_title = Text("General State:", font_size=20)
        general_title.shift(RIGHT * 2 + UP * 2)
        
        general_state = MathTex(
            r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}",
            font_size=24
        )
        general_state.shift(RIGHT * 2)
        
        normalization = MathTex(r"|\alpha|^2 + |\beta|^2 = 1", font_size=20, color=YELLOW)
        normalization.shift(RIGHT * 2 + DOWN)
        
        self.play(
            Write(general_title),
            Write(general_state),
            run_time=0.6
        )
        self.play(Write(normalization), run_time=0.4)
        
        # Bloch sphere parameterization
        bloch_param = MathTex(
            r"|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle",
            font_size=22
        )
        bloch_param.shift(DOWN * 2)
        
        self.play(Write(bloch_param), run_time=0.6)
        
        # Inner product structure
        inner_title = Text("Inner Product:", font_size=20, color=BLUE)
        inner_title.shift(DOWN * 3)
        
        inner_prod = MathTex(
            r"\langle\psi|\phi\rangle = \alpha^*\gamma + \beta^*\delta",
            font_size=20
        )
        inner_prod.shift(DOWN * 3.5)
        
        self.play(Write(inner_title), Write(inner_prod), run_time=0.5)
        
        # Multi-qubit tensor product
        self.wait(0.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title], run_time=0.3)
        
        tensor_title = Text("Tensor Product Structure:", font_size=24)
        tensor_title.shift(UP * 2)
        
        tensor_ex = MathTex(
            r"|00\rangle = |0\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}",
            font_size=24
        )
        tensor_ex.shift(UP * 0.5)
        
        general_2q = MathTex(
            r"|\psi\rangle_{AB} = \sum_{i,j \in \{0,1\}} c_{ij} |i\rangle_A \otimes |j\rangle_B",
            font_size=22
        )
        general_2q.shift(DOWN * 0.5)
        
        dim_scaling = MathTex(
            r"\dim(\mathcal{H}_n) = 2^n",
            font_size=26,
            color=YELLOW
        )
        dim_scaling.shift(DOWN * 2)
        
        self.play(
            Write(tensor_title),
            Write(tensor_ex),
            run_time=0.8
        )
        self.play(Write(general_2q), run_time=0.6)
        self.play(Write(dim_scaling), Flash(dim_scaling), run_time=0.5)
        
        self.wait(0.5)
        
    def density_matrices_decoherence(self):
        # Clear and set up 3D
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        
        title = Text("DENSITY MATRICES & DECOHERENCE", font_size=32, color=RED)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.3)
        
        # Pure state density matrix
        pure_title = Text("Pure State:", font_size=20)
        pure_title.shift(LEFT * 4 + UP * 2)
        self.add_fixed_in_frame_mobjects(pure_title)
        
        pure_def = MathTex(
            r"\rho = |\psi\rangle\langle\psi|", 
            font_size=24
        )
        pure_def.shift(LEFT * 4 + UP)
        self.add_fixed_in_frame_mobjects(pure_def)
        
        pure_matrix = MathTex(
            r"\rho = \begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}",
            font_size=22
        )
        pure_matrix.shift(LEFT * 4)
        self.add_fixed_in_frame_mobjects(pure_matrix)
        
        pure_prop = MathTex(r"\text{Tr}(\rho^2) = 1", font_size=20, color=GREEN)
        pure_prop.shift(LEFT * 4 + DOWN)
        self.add_fixed_in_frame_mobjects(pure_prop)
        
        self.play(
            Write(pure_title),
            Write(pure_def),
            run_time=0.5
        )
        self.play(Write(pure_matrix), run_time=0.5)
        self.play(Write(pure_prop), run_time=0.3)
        
        # Mixed state
        mixed_title = Text("Mixed State:", font_size=20)
        mixed_title.shift(RIGHT * 2 + UP * 2)
        self.add_fixed_in_frame_mobjects(mixed_title)
        
        mixed_def = MathTex(
            r"\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|",
            font_size=22
        )
        mixed_def.shift(RIGHT * 2 + UP)
        self.add_fixed_in_frame_mobjects(mixed_def)
        
        mixed_example = MathTex(
            r"\rho_{\text{mixed}} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}",
            font_size=22
        )
        mixed_example.shift(RIGHT * 2)
        self.add_fixed_in_frame_mobjects(mixed_example)
        
        mixed_prop = MathTex(r"\text{Tr}(\rho^2) < 1", font_size=20, color=RED)
        mixed_prop.shift(RIGHT * 2 + DOWN)
        self.add_fixed_in_frame_mobjects(mixed_prop)
        
        self.play(
            Write(mixed_title),
            Write(mixed_def),
            run_time=0.5
        )
        self.play(Write(mixed_example), run_time=0.5)
        self.play(Write(mixed_prop), run_time=0.3)
        
        # Bloch ball visualization
        self.wait(0.5)
        
        # Create Bloch sphere for pure states
        sphere = Sphere(radius=1, resolution=(20, 20))
        sphere.set_color(BLUE_E)
        sphere.set_opacity(0.2)
        sphere.shift(DOWN * 0.5)
        
        # Inner ball for mixed states
        inner_ball = Sphere(radius=0.6, resolution=(15, 15))
        inner_ball.set_color(RED_E)
        inner_ball.set_opacity(0.3)
        inner_ball.shift(DOWN * 0.5)
        
        self.play(Create(sphere), run_time=0.5)
        self.play(Create(inner_ball), run_time=0.5)
        
        # Decoherence visualization
        decoherence_title = Text("Decoherence Process:", font_size=20, color=PURPLE)
        decoherence_title.shift(DOWN * 3)
        self.add_fixed_in_frame_mobjects(decoherence_title)
        
        # Show state vector shrinking
        state_vec = Arrow3D(
            start=[0, -0.5, 0],
            end=[0, -0.5, 1],
            color=YELLOW,
            thickness=0.05
        )
        
        self.play(Write(decoherence_title), Create(state_vec), run_time=0.5)
        
        # Animate decoherence
        for i in range(3):
            scale = 1 - 0.3 * (i + 1)
            new_vec = Arrow3D(
                start=[0, -0.5, 0],
                end=[0, -0.5, scale],
                color=YELLOW,
                thickness=0.05
            )
            self.play(Transform(state_vec, new_vec), run_time=0.4)
        
        # Decoherence equation
        deco_eq = MathTex(
            r"\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \mathcal{L}_k[\rho]",
            font_size=20
        )
        deco_eq.shift(DOWN * 3.5)
        self.add_fixed_in_frame_mobjects(deco_eq)
        self.play(Write(deco_eq), run_time=0.6)
        
        self.wait(0.5)
        self.set_camera_orientation(phi=0, theta=-90*DEGREES)
        
    def unitary_evolution_hamiltonians(self):
        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        
        title = Text("UNITARY EVOLUTION", font_size=32, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)
        
        # Schrödinger equation
        schrodinger = MathTex(
            r"i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = H|\psi(t)\rangle",
            font_size=28
        )
        schrodinger.shift(UP * 2)
        
        self.play(Write(schrodinger), run_time=0.6)
        
        # Time evolution operator
        evolution_title = Text("Time Evolution Operator:", font_size=20)
        evolution_title.shift(LEFT * 3 + UP * 0.5)
        
        evolution_op = MathTex(
            r"U(t) = e^{-iHt/\hbar}",
            font_size=26
        )
        evolution_op.shift(LEFT * 3 + DOWN * 0.5)
        
        unitary_prop = MathTex(
            r"U^\dagger U = UU^\dagger = I",
            font_size=22,
            color=GREEN
        )
        unitary_prop.shift(LEFT * 3 + DOWN * 1.5)
        
        self.play(
            Write(evolution_title),
            Write(evolution_op),
            run_time=0.6
        )
        self.play(Write(unitary_prop), run_time=0.4)
        
        # Common Hamiltonians
        ham_title = Text("Quantum Gate Hamiltonians:", font_size=20)
        ham_title.shift(RIGHT * 2 + UP * 0.5)
        
        # Pauli matrices
        pauli_x = MathTex(r"H_X = \frac{\pi}{2}\sigma_x", font_size=20)
        pauli_y = MathTex(r"H_Y = \frac{\pi}{2}\sigma_y", font_size=20)
        pauli_z = MathTex(r"H_Z = \frac{\pi}{2}\sigma_z", font_size=20)
        
        pauli_group = VGroup(pauli_x, pauli_y, pauli_z).arrange(DOWN, buff=0.3)
        pauli_group.shift(RIGHT * 2 + DOWN * 0.5)
        
        self.play(Write(ham_title), run_time=0.4)
        self.play(Write(pauli_group), run_time=0.6)
        
        # Gate decomposition
        self.wait(0.5)
        decomp_title = Text("Universal Gate Set:", font_size=24, color=BLUE)
        decomp_title.shift(DOWN * 2.5)
        
        decomp_eq = MathTex(
            r"U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)",
            font_size=22
        )
        decomp_eq.shift(DOWN * 3.2)
        
        self.play(Write(decomp_title), run_time=0.3)
        self.play(Write(decomp_eq), run_time=0.5)
        
        # Show gate matrices explicitly
        self.wait(0.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title], run_time=0.3)
        
        # Gate matrices
        gate_matrices_title = Text("Fundamental Gate Matrices:", font_size=24)
        gate_matrices_title.shift(UP * 2)
        
        x_matrix = MathTex(
            r"X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}",
            font_size=22
        )
        x_matrix.shift(LEFT * 4)
        
        y_matrix = MathTex(
            r"Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}",
            font_size=22
        )
        y_matrix.shift(LEFT * 1.3)
        
        z_matrix = MathTex(
            r"Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}",
            font_size=22
        )
        z_matrix.shift(RIGHT * 1.3)
        
        h_matrix = MathTex(
            r"H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}",
            font_size=22
        )
        h_matrix.shift(RIGHT * 4)
        
        cnot_matrix = MathTex(
            r"\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}",
            font_size=20
        )
        cnot_matrix.shift(DOWN * 1.5)
        
        self.play(Write(gate_matrices_title), run_time=0.4)
        self.play(
            Write(x_matrix),
            Write(y_matrix),
            Write(z_matrix),
            Write(h_matrix),
            run_time=0.8
        )
        self.play(Write(cnot_matrix), run_time=0.5)
        
        # Solovay-Kitaev theorem mention
        sk_theorem = Text("Solovay-Kitaev: Any U approximated to ε with O(log^c(1/ε)) gates", 
                         font_size=18, color=YELLOW)
        sk_theorem.shift(DOWN * 3.5)
        self.play(Write(sk_theorem), run_time=0.6)
        
        self.wait(0.5)
        
    def quantum_algorithms_depth(self):
        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        
        title = Text("QUANTUM ALGORITHMS", font_size=32, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)
        
        # Grover's algorithm
        grover_title = Text("Grover's Search Algorithm", font_size=24, color=GREEN)
        grover_title.shift(UP * 2)
        self.play(Write(grover_title), run_time=0.4)
        
        # Oracle and diffusion operator
        oracle_def = MathTex(
            r"O_f|x\rangle = (-1)^{f(x)}|x\rangle",
            font_size=22
        )
        oracle_def.shift(LEFT * 3 + UP * 0.5)
        
        diffusion_def = MathTex(
            r"D = 2|s\rangle\langle s| - I",
            font_size=22
        )
        diffusion_def.shift(RIGHT * 3 + UP * 0.5)
        
        uniform_super = MathTex(
            r"|s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle",
            font_size=20
        )
        uniform_super.shift(DOWN * 0.3)
        
        self.play(
            Write(oracle_def),
            Write(diffusion_def),
            run_time=0.6
        )
        self.play(Write(uniform_super), run_time=0.5)
        
        # Grover operator
        grover_op = MathTex(
            r"G = D \cdot O_f",
            font_size=24,
            color=YELLOW
        )
        grover_op.shift(DOWN * 1.2)
        
        iterations = MathTex(
            r"k \approx \frac{\pi}{4}\sqrt{N}",
            font_size=22,
            color=GREEN
        )
        iterations.shift(DOWN * 2)
        
        self.play(Write(grover_op), run_time=0.4)
        self.play(Write(iterations), run_time=0.4)
        
        # Amplitude amplification visualization
        amp_viz = VGroup()
        for i in range(5):
            bar = Rectangle(
                width=0.3,
                height=0.2 + 0.4 * i if i < 3 else 0.2,
                color=BLUE if i != 2 else YELLOW,
                fill_opacity=0.7
            )
            bar.shift(LEFT * 2 + RIGHT * 0.5 * i + DOWN * 3)
            amp_viz.add(bar)
        
        self.play(Create(amp_viz), run_time=0.5)
        
        # Quantum Fourier Transform
        self.wait(0.5)
        self.play(
            FadeOut(grover_title),
            FadeOut(oracle_def),
            FadeOut(diffusion_def),
            FadeOut(uniform_super),
            FadeOut(grover_op),
            FadeOut(iterations),
            FadeOut(amp_viz),
            run_time=0.3
        )
        
        qft_title = Text("Quantum Fourier Transform", font_size=24, color=BLUE)
        qft_title.shift(UP * 2)
        self.play(Write(qft_title), run_time=0.4)
        
        qft_def = MathTex(
            r"QFT|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle",
            font_size=22
        )
        qft_def.shift(UP * 0.5)
        
        qft_matrix = MathTex(
            r"F_N = \frac{1}{\sqrt{N}}\begin{pmatrix} 1 & 1 & 1 & \cdots \\ 1 & \omega & \omega^2 & \cdots \\ 1 & \omega^2 & \omega^4 & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}",
            font_size=20
        )
        qft_matrix.shift(DOWN * 0.5)
        
        self.play(Write(qft_def), run_time=0.5)
        self.play(Write(qft_matrix), run_time=0.6)
        
        # Circuit depth
        circuit_depth = MathTex(
            r"\text{Circuit Depth: } O(n^2) \text{ gates}",
            font_size=22,
            color=GREEN
        )
        circuit_depth.shift(DOWN * 2)
        
        self.play(Write(circuit_depth), run_time=0.4)
        
        # Phase estimation mention
        phase_est = Text("Key component: Phase Estimation → Shor's Algorithm", 
                        font_size=20, color=YELLOW)
        phase_est.shift(DOWN * 3)
        self.play(Write(phase_est), run_time=0.5)
        
        # Quantum advantage
        advantage = MathTex(
            r"\text{Classical: } O(N) \rightarrow \text{Quantum: } O(\sqrt{N})",
            font_size=24,
            color=GOLD
        )
        advantage.shift(DOWN * 3.7)
        self.play(Write(advantage), Flash(advantage), run_time=0.6)
        
        self.wait(0.5)
        
    def error_correction_topology(self):
        # Clear and 3D setup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        self.set_camera_orientation(phi=70*DEGREES, theta=-45*DEGREES)
        
        title = Text("QUANTUM ERROR CORRECTION", font_size=32, color=RED)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.3)
        
        # Error types
        error_title = Text("Quantum Errors:", font_size=20)
        error_title.shift(LEFT * 4 + UP * 2)
        self.add_fixed_in_frame_mobjects(error_title)
        
        bit_flip = MathTex(r"X: |0\rangle \leftrightarrow |1\rangle", font_size=20)
        phase_flip = MathTex(r"Z: |+\rangle \leftrightarrow |-\rangle", font_size=20)
        both_flip = MathTex(r"Y = iXZ", font_size=20)
        
        error_group = VGroup(bit_flip, phase_flip, both_flip).arrange(DOWN, buff=0.3)
        error_group.shift(LEFT * 4)
        self.add_fixed_in_frame_mobjects(error_group)
        
        self.play(Write(error_title), Write(error_group), run_time=0.6)
        
        # 3-qubit code
        code_title = Text("3-Qubit Bit Flip Code:", font_size=20)
        code_title.shift(RIGHT * 2 + UP * 2)
        self.add_fixed_in_frame_mobjects(code_title)
        
        encoding = MathTex(
            r"|0\rangle_L = |000\rangle, \quad |1\rangle_L = |111\rangle",
            font_size=22
        )
        encoding.shift(RIGHT * 2 + UP)
        self.add_fixed_in_frame_mobjects(encoding)
        
        self.play(Write(code_title), Write(encoding), run_time=0.5)
        
        # Stabilizer formalism
        stabilizer_title = Text("Stabilizer Formalism:", font_size=20, color=BLUE)
        stabilizer_title.shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(stabilizer_title)
        
        stabilizers = MathTex(
            r"S_1 = Z_1Z_2I_3, \quad S_2 = I_1Z_2Z_3",
            font_size=20
        )
        stabilizers.shift(DOWN * 1.2)
        self.add_fixed_in_frame_mobjects(stabilizers)
        
        self.play(Write(stabilizer_title), Write(stabilizers), run_time=0.5)
        
        # Surface code visualization
        surface_title = Text("Surface Code:", font_size=22, color=PURPLE)
        surface_title.shift(DOWN * 2.5)
        self.add_fixed_in_frame_mobjects(surface_title)
        
        # Create lattice
        lattice = VGroup()
        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:
                    # Data qubit
                    qubit = Sphere(radius=0.15, resolution=(10, 10))
                    qubit.set_color(BLUE)
                else:
                    # Measurement qubit
                    qubit = Sphere(radius=0.15, resolution=(10, 10))
                    qubit.set_color(RED)
                
                qubit.shift(RIGHT * (i - 1) + UP * (j - 1) + DOWN * 3)
                lattice.add(qubit)
        
        # Add connections
        connections = VGroup()
        for i in range(2):
            for j in range(3):
                # Horizontal
                line = Line(
                    RIGHT * (i - 0.5) + UP * (j - 1) + DOWN * 3,
                    RIGHT * (i + 0.5) + UP * (j - 1) + DOWN * 3,
                    color=GRAY,
                    stroke_width=2
                )
                connections.add(line)
                
                # Vertical
                if j < 2:
                    line = Line(
                        RIGHT * (i - 1) + UP * (j - 0.5) + DOWN * 3,
                        RIGHT * (i - 1) + UP * (j + 0.5) + DOWN * 3,
                        color=GRAY,
                        stroke_width=2
                    )
                    connections.add(line)
        
        self.play(
            Write(surface_title),
            Create(connections),
            Create(lattice),
            run_time=0.8
        )
        
        # Threshold theorem
        threshold = MathTex(
            r"p < p_{th} \approx 10^{-2} \Rightarrow \text{Arbitrary accuracy}",
            font_size=20,
            color=GREEN
        )
        threshold.shift(DOWN * 4)
        self.add_fixed_in_frame_mobjects(threshold)
        self.play(Write(threshold), run_time=0.5)
        
        self.wait(0.5)
        self.set_camera_orientation(phi=0, theta=-90*DEGREES)
        
    def quantum_complexity_theory(self):
        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        
        title = Text("QUANTUM COMPLEXITY", font_size=32, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)
        
        # Complexity classes
        classes_title = Text("Complexity Classes:", font_size=24)
        classes_title.shift(UP * 2)
        self.play(Write(classes_title), run_time=0.3)
        
        # Create nested circles for complexity classes
        p_circle = Circle(radius=0.8, color=GREEN)
        p_circle.shift(LEFT * 3)
        p_label = Text("P", font_size=20).move_to(p_circle)
        
        bpp_circle = Circle(radius=1.2, color=BLUE)
        bpp_circle.shift(LEFT * 3)
        bpp_label = Text("BPP", font_size=18).shift(LEFT * 3 + UP * 1)
        
        bqp_circle = Circle(radius=1.6, color=PURPLE)
        bqp_circle.shift(LEFT * 3)
        bqp_label = Text("BQP", font_size=18).shift(LEFT * 3 + UP * 1.4)
        
        pspace_circle = Circle(radius=2, color=RED)
        pspace_circle.shift(LEFT * 3)
        pspace_label = Text("PSPACE", font_size=18).shift(LEFT * 3 + UP * 1.8)
        
        self.play(
            Create(pspace_circle), Write(pspace_label),
            Create(bqp_circle), Write(bqp_label),
            Create(bpp_circle), Write(bpp_label),
            Create(p_circle), Write(p_label),
            run_time=0.8
        )
        
        # BQP definition
        bqp_def = MathTex(
            r"\text{BQP: } \Pr[M(x) = f(x)] \geq \frac{2}{3}",
            font_size=22
        )
        bqp_def.shift(RIGHT * 2)
        
        poly_time = Text("Polynomial time quantum computation", font_size=18, color=PURPLE)
        poly_time.shift(RIGHT * 2 + DOWN * 0.5)
        
        self.play(Write(bqp_def), Write(poly_time), run_time=0.6)
        
        # Quantum supremacy
        supremacy_title = Text("Quantum Supremacy:", font_size=22, color=YELLOW)
        supremacy_title.shift(DOWN * 1.5)
        
        supremacy_def = Text("Task solvable by quantum but not classical polynomial time", 
                           font_size=18)
        supremacy_def.shift(DOWN * 2)
        
        self.play(Write(supremacy_title), Write(supremacy_def), run_time=0.5)
        
        # NISQ era
        nisq = Text("NISQ: Noisy Intermediate-Scale Quantum", font_size=20, color=ORANGE)
        nisq.shift(DOWN * 3)
        
        nisq_props = Text("50-100 qubits, limited coherence, no error correction", 
                         font_size=16, color=GRAY)
        nisq_props.shift(DOWN * 3.5)
        
        self.play(Write(nisq), Write(nisq_props), run_time=0.6)
        
        # Final thought
        future = Text("Path to Fault-Tolerant Quantum Computing", 
                     font_size=24, color=GREEN)
        future.shift(DOWN * 4.5)
        
        self.play(Write(future), Flash(future), run_time=0.6)
        
        self.wait(1)
        
        # End card
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        
        end_title = Text("QUANTUM COMPUTING", font_size=44, color=BLUE, weight=BOLD)
        end_subtitle = Text("From Theory to Implementation", font_size=22, color=GRAY)
        end_group = VGroup(end_title, end_subtitle).arrange(DOWN, buff=0.3)
        
        self.play(Write(end_group), run_time=0.8)
        self.wait(1)