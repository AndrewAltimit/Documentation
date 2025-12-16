import numpy as np
from manim import *


class QuantumGraduateRefined(ThreeDScene):
    def construct(self):
        # Total runtime: ~60 seconds for better comprehension
        self.intro_with_formalism()  # 3s
        self.hilbert_space_fundamentals()  # 10s
        self.density_matrices_decoherence()  # 10s
        self.unitary_evolution_hamiltonians()  # 10s
        self.quantum_algorithms_depth()  # 12s
        self.error_correction_topology()  # 8s
        self.quantum_complexity_theory()  # 7s

    def intro_with_formalism(self):
        # Clear intro with proper pacing
        title = Text("QUANTUM COMPUTING", font_size=56, color=BLUE, weight=BOLD)
        subtitle = Text(
            "Graduate-Level Mathematical Framework", font_size=28, color=GRAY
        )
        subtitle.next_to(title, DOWN)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle), run_time=0.5)
        self.wait(1.2)  # Let viewers read

        # Smooth transition
        self.play(title.animate.scale(0.5).to_edge(UP), FadeOut(subtitle), run_time=0.5)

        # First postulate - clear and centered
        postulate = Text(
            "Postulate 1: Quantum States Live in Hilbert Space",
            font_size=32,
            color=YELLOW,
        )
        postulate.shift(UP * 2)

        self.play(Write(postulate), run_time=0.8)
        self.wait(1)  # Pause to read

        # Hilbert space definition - larger and clearer
        hilbert_def = MathTex(
            r"|\psi\rangle \in \mathcal{H} = \mathbb{C}^{2^n}", font_size=48  # Larger
        )
        hilbert_def.shift(DOWN * 0.5)

        # Explanation
        explanation = Text(
            "n qubits → 2ⁿ dimensional complex vector space", font_size=24, color=GRAY
        )
        explanation.shift(DOWN * 2)

        self.play(Write(hilbert_def), run_time=1)
        self.play(FadeIn(explanation), run_time=0.5)
        self.wait(2)  # Important pause

    def hilbert_space_fundamentals(self):
        # Clear transition
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)

        # Section header with underline
        title = Text("HILBERT SPACE STRUCTURE", font_size=40, color=GREEN)
        title.to_edge(UP)
        underline = Line(
            title.get_left() + DOWN * 0.3, title.get_right() + DOWN * 0.3, color=GREEN
        )

        self.play(Write(title), Create(underline), run_time=0.6)

        # Part 1: Computational Basis - Progressive reveal
        basis_title = Text("1. Computational Basis", font_size=32, color=BLUE)
        basis_title.shift(UP * 2)
        self.play(Write(basis_title), run_time=0.5)
        self.wait(0.5)

        # Show basis vectors one at a time
        basis_0 = MathTex(
            r"|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}", font_size=36  # Larger
        )
        basis_0.shift(LEFT * 3)

        self.play(Write(basis_0), run_time=0.8)
        self.wait(1)  # Pause

        basis_1 = MathTex(
            r"|1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}", font_size=36
        )
        basis_1.shift(RIGHT * 3)

        self.play(Write(basis_1), run_time=0.8)
        self.wait(1.5)  # Let viewers compare

        # Part 2: General State - Clear presentation
        self.play(
            FadeOut(basis_title),
            basis_0.animate.scale(0.8).shift(UP * 2),
            basis_1.animate.scale(0.8).shift(UP * 2),
            run_time=0.5,
        )

        general_title = Text("2. General Qubit State", font_size=32, color=PURPLE)
        general_title.shift(UP * 0.5)
        self.play(Write(general_title), run_time=0.5)

        # Show general state with clear spacing
        general_state = MathTex(
            r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", font_size=40
        )
        general_state.shift(DOWN * 0.5)

        self.play(Write(general_state), run_time=1)
        self.wait(1)

        # Matrix form aligned below
        matrix_form = MathTex(
            r"= \alpha\begin{pmatrix} 1 \\ 0 \end{pmatrix} + \beta\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}",
            font_size=32,
        )
        matrix_form.shift(DOWN * 1.5)

        self.play(Write(matrix_form), run_time=1.2)
        self.wait(1)

        # Normalization constraint - highlight importance
        normalization = MathTex(
            r"|\alpha|^2 + |\beta|^2 = 1", font_size=36, color=YELLOW
        )
        normalization.shift(DOWN * 2.8)

        norm_box = SurroundingRectangle(normalization, color=YELLOW, buff=0.2)

        self.play(Write(normalization), run_time=0.8)
        self.play(Create(norm_box), run_time=0.4)
        self.wait(2)  # Important constraint

        # Part 3: Bloch Sphere - Clean transition
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in [title, underline]],
            run_time=0.5
        )

        bloch_title = Text("3. Bloch Sphere Representation", font_size=32, color=ORANGE)
        bloch_title.shift(UP * 2)
        self.play(Write(bloch_title), run_time=0.5)

        # Bloch parameterization
        bloch_param = MathTex(
            r"|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle",
            font_size=32,
        )
        bloch_param.shift(DOWN * 0.5)

        self.play(Write(bloch_param), run_time=1.5)
        self.wait(2)  # Complex formula needs time

        # Visual Bloch sphere
        sphere = Circle(radius=1.5, color=BLUE_E)
        sphere.shift(DOWN * 2.5)

        # Axes
        z_axis = Line(DOWN * 4, DOWN * 1, color=BLUE)
        z_label_0 = MathTex("|0\\rangle", font_size=24).next_to(
            z_axis.get_start(), DOWN
        )
        z_label_1 = MathTex("|1\\rangle", font_size=24).next_to(z_axis.get_end(), UP)

        self.play(
            Create(sphere),
            Create(z_axis),
            Write(z_label_0),
            Write(z_label_1),
            run_time=1,
        )
        self.wait(1.5)

        # Part 4: Tensor Products - New screen
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in [title, underline]],
            run_time=0.5
        )

        tensor_title = Text(
            "4. Multi-Qubit Systems: Tensor Products", font_size=32, color=RED
        )
        tensor_title.shift(UP * 2)
        self.play(Write(tensor_title), run_time=0.6)
        self.wait(0.5)

        # Example with clear layout
        tensor_ex = MathTex(r"|00\rangle = |0\rangle \otimes |0\rangle", font_size=36)
        tensor_ex.shift(UP * 0.5)

        self.play(Write(tensor_ex), run_time=0.8)
        self.wait(0.8)

        # Show matrix
        tensor_matrix = MathTex(
            r"= \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}",
            font_size=32,
        )
        tensor_matrix.shift(DOWN * 0.5)

        self.play(Write(tensor_matrix), run_time=1.2)
        self.wait(1.5)

        # General formula
        general_2q = MathTex(
            r"|\psi\rangle_{AB} = \sum_{i,j \in \{0,1\}} c_{ij} |i\rangle_A \otimes |j\rangle_B",
            font_size=30,
        )
        general_2q.shift(DOWN * 2)

        self.play(Write(general_2q), run_time=1.2)
        self.wait(1)

        # Dimension scaling - emphasize exponential growth
        dim_scaling = MathTex(
            r"\text{dim}(\mathcal{H}_n) = 2^n", font_size=44, color=YELLOW
        )
        dim_scaling.shift(DOWN * 3.2)

        # Add emphasis animation
        self.play(Write(dim_scaling), run_time=0.8)
        self.play(
            dim_scaling.animate.scale(1.2),
            Flash(dim_scaling, color=YELLOW),
            run_time=0.5,
        )
        self.wait(2)  # Key insight

    def density_matrices_decoherence(self):
        # Clear and setup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        # Title with emphasis
        title = Text("DENSITY MATRICES & DECOHERENCE", font_size=40, color=RED)
        title.to_edge(UP)
        underline = Line(
            title.get_left() + DOWN * 0.3, title.get_right() + DOWN * 0.3, color=RED
        )
        self.add_fixed_in_frame_mobjects(title, underline)
        self.play(Write(title), Create(underline), run_time=0.6)

        # Part 1: Pure States - Clear layout
        pure_section = VGroup()

        pure_title = Text("Pure States", font_size=32, color=GREEN)
        pure_title.shift(LEFT * 4 + UP * 2)
        self.add_fixed_in_frame_mobjects(pure_title)

        # Definition with spacing
        pure_def = MathTex(r"\rho = |\psi\rangle\langle\psi|", font_size=32)
        pure_def.shift(LEFT * 4 + UP * 0.8)
        self.add_fixed_in_frame_mobjects(pure_def)

        self.play(Write(pure_title), run_time=0.5)
        self.play(Write(pure_def), run_time=0.8)
        self.wait(1)

        # Matrix representation
        pure_matrix = MathTex(
            r"\rho = \begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}",
            font_size=28,
        )
        pure_matrix.shift(LEFT * 4 + DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(pure_matrix)

        self.play(Write(pure_matrix), run_time=1)
        self.wait(1.5)

        # Key property
        pure_prop = MathTex(r"\text{Tr}(\rho^2) = 1", font_size=28, color=GREEN)
        pure_prop_box = SurroundingRectangle(pure_prop, color=GREEN, buff=0.15)
        pure_group = VGroup(pure_prop, pure_prop_box)
        pure_group.shift(LEFT * 4 + DOWN * 1.8)
        self.add_fixed_in_frame_mobjects(pure_group)

        self.play(Write(pure_prop), Create(pure_prop_box), run_time=0.8)
        self.wait(1.5)

        # Part 2: Mixed States - Parallel layout
        mixed_title = Text("Mixed States", font_size=32, color=RED)
        mixed_title.shift(RIGHT * 2 + UP * 2)
        self.add_fixed_in_frame_mobjects(mixed_title)

        mixed_def = MathTex(
            r"\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|", font_size=30
        )
        mixed_def.shift(RIGHT * 2 + UP * 0.8)
        self.add_fixed_in_frame_mobjects(mixed_def)

        self.play(Write(mixed_title), run_time=0.5)
        self.play(Write(mixed_def), run_time=1)
        self.wait(1)

        # Example
        mixed_example = MathTex(
            r"\rho_{\text{mixed}} = \frac{I}{2} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}",
            font_size=26,
        )
        mixed_example.shift(RIGHT * 2 + DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(mixed_example)

        self.play(Write(mixed_example), run_time=1)
        self.wait(1)

        # Key property
        mixed_prop = MathTex(r"\text{Tr}(\rho^2) < 1", font_size=28, color=RED)
        mixed_prop_box = SurroundingRectangle(mixed_prop, color=RED, buff=0.15)
        mixed_group = VGroup(mixed_prop, mixed_prop_box)
        mixed_group.shift(RIGHT * 2 + DOWN * 1.8)
        self.add_fixed_in_frame_mobjects(mixed_group)

        self.play(Write(mixed_prop), Create(mixed_prop_box), run_time=0.8)
        self.wait(2)  # Compare pure vs mixed

        # Part 3: Bloch Ball Visualization
        self.play(
            FadeOut(pure_title),
            FadeOut(pure_def),
            FadeOut(pure_matrix),
            FadeOut(pure_group),
            FadeOut(mixed_title),
            FadeOut(mixed_def),
            FadeOut(mixed_example),
            FadeOut(mixed_group),
            run_time=0.5,
        )

        bloch_title = Text(
            "Bloch Ball: Pure vs Mixed States", font_size=32, color=PURPLE
        )
        bloch_title.shift(UP * 2.5)
        self.add_fixed_in_frame_mobjects(bloch_title)
        self.play(Write(bloch_title), run_time=0.6)

        # Create spheres
        pure_sphere = Sphere(radius=1.2, resolution=(20, 20))
        pure_sphere.set_color(BLUE_E)
        pure_sphere.set_opacity(0.3)

        mixed_ball = Sphere(radius=0.7, resolution=(15, 15))
        mixed_ball.set_color(RED_E)
        mixed_ball.set_opacity(0.5)

        self.play(Create(pure_sphere), run_time=0.8)
        self.wait(0.5)
        self.play(Create(mixed_ball), run_time=0.8)

        # Labels
        pure_label = Text("Pure states:\nsurface", font_size=20, color=BLUE)
        pure_label.shift(LEFT * 2.5)
        self.add_fixed_in_frame_mobjects(pure_label)

        mixed_label = Text("Mixed states:\ninterior", font_size=20, color=RED)
        mixed_label.shift(RIGHT * 2.5)
        self.add_fixed_in_frame_mobjects(mixed_label)

        self.play(Write(pure_label), Write(mixed_label), run_time=0.8)
        self.wait(2)

        # Part 4: Decoherence
        self.play(
            FadeOut(bloch_title),
            FadeOut(pure_label),
            FadeOut(mixed_label),
            run_time=0.5,
        )

        deco_title = Text("Decoherence: Pure → Mixed", font_size=32, color=PURPLE)
        deco_title.shift(UP * 2.5)
        self.add_fixed_in_frame_mobjects(deco_title)
        self.play(Write(deco_title), run_time=0.6)

        # Show state vector shrinking
        state_vec = Arrow3D(
            start=[0, 0, 0], end=[0, 0, 1.2], color=YELLOW, thickness=0.06
        )

        self.play(Create(state_vec), run_time=0.5)

        # Animate decoherence step by step
        for i in range(3):
            scale = 1 - 0.35 * (i + 1)
            new_vec = Arrow3D(
                start=[0, 0, 0], end=[0, 0, 1.2 * scale], color=YELLOW, thickness=0.06
            )
            self.play(Transform(state_vec, new_vec), run_time=0.6)
            self.wait(0.5)  # Show each step

        # Master equation
        deco_eq = MathTex(
            r"\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \mathcal{L}_k[\rho]",
            font_size=28,
        )
        deco_eq.shift(DOWN * 3)
        self.add_fixed_in_frame_mobjects(deco_eq)

        # Labels for parts
        hamiltonian_label = Text("Unitary evolution", font_size=16, color=GREEN)
        hamiltonian_label.shift(DOWN * 3.6 + LEFT * 2)

        lindblad_label = Text("Dissipation", font_size=16, color=RED)
        lindblad_label.shift(DOWN * 3.6 + RIGHT * 2)

        self.add_fixed_in_frame_mobjects(hamiltonian_label, lindblad_label)

        self.play(Write(deco_eq), run_time=1.2)
        self.play(Write(hamiltonian_label), Write(lindblad_label), run_time=0.8)
        self.wait(2.5)  # Important equation

        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

    def unitary_evolution_hamiltonians(self):
        # Clear with smooth transition
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)

        # Title section
        title = Text("UNITARY EVOLUTION", font_size=40, color=YELLOW)
        title.to_edge(UP)
        underline = Line(
            title.get_left() + DOWN * 0.3, title.get_right() + DOWN * 0.3, color=YELLOW
        )

        self.play(Write(title), Create(underline), run_time=0.6)

        # Part 1: Schrödinger Equation
        schrodinger_title = Text(
            "Time-Dependent Schrödinger Equation", font_size=28, color=BLUE
        )
        schrodinger_title.shift(UP * 2)
        self.play(Write(schrodinger_title), run_time=0.6)

        schrodinger = MathTex(
            r"i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = H|\psi(t)\rangle",
            font_size=40,  # Larger for emphasis
        )
        schrodinger.shift(UP * 0.5)

        self.play(Write(schrodinger), run_time=1.2)
        self.wait(2)  # Fundamental equation

        # Part 2: Time Evolution Operator
        self.play(
            FadeOut(schrodinger_title), schrodinger.animate.shift(UP), run_time=0.5
        )

        evolution_title = Text(
            "Solution: Time Evolution Operator", font_size=28, color=GREEN
        )
        evolution_title.shift(DOWN * 0.5)
        self.play(Write(evolution_title), run_time=0.6)

        evolution_op = MathTex(r"|\psi(t)\rangle = U(t)|\psi(0)\rangle", font_size=36)
        evolution_op.shift(DOWN * 1.5)

        self.play(Write(evolution_op), run_time=0.8)
        self.wait(1)

        # Show U(t) explicitly
        u_explicit = MathTex(r"U(t) = e^{-iHt/\hbar}", font_size=36, color=GREEN)
        u_explicit.shift(DOWN * 2.5)

        self.play(Write(u_explicit), run_time=0.8)
        self.wait(1.5)

        # Unitary property
        unitary_prop = MathTex(
            r"U^\dagger U = UU^\dagger = I", font_size=32, color=YELLOW
        )
        unitary_prop.shift(DOWN * 3.5)

        self.play(Write(unitary_prop), run_time=0.8)
        self.wait(1.5)

        # Part 3: Gate Matrices - New screen
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in [title, underline]],
            run_time=0.5
        )

        gates_title = Text("Quantum Gate Matrices", font_size=32, color=PURPLE)
        gates_title.shift(UP * 2.5)
        self.play(Write(gates_title), run_time=0.5)

        # Pauli gates in organized grid
        pauli_label = Text("Pauli Gates:", font_size=24, color=GRAY)
        pauli_label.shift(UP * 1.5 + LEFT * 5)
        self.play(Write(pauli_label), run_time=0.4)

        # X gate
        x_matrix = MathTex(
            r"X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}", font_size=28
        )
        x_matrix.shift(LEFT * 4 + UP * 0.5)
        x_label = Text("Bit flip", font_size=18, color=BLUE)
        x_label.next_to(x_matrix, DOWN)

        self.play(Write(x_matrix), run_time=0.6)
        self.play(FadeIn(x_label), run_time=0.3)
        self.wait(0.8)

        # Y gate
        y_matrix = MathTex(
            r"Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}", font_size=28
        )
        y_matrix.shift(UP * 0.5)
        y_label = Text("Bit + phase flip", font_size=18, color=GREEN)
        y_label.next_to(y_matrix, DOWN)

        self.play(Write(y_matrix), run_time=0.6)
        self.play(FadeIn(y_label), run_time=0.3)
        self.wait(0.8)

        # Z gate
        z_matrix = MathTex(
            r"Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}", font_size=28
        )
        z_matrix.shift(RIGHT * 4 + UP * 0.5)
        z_label = Text("Phase flip", font_size=18, color=RED)
        z_label.next_to(z_matrix, DOWN)

        self.play(Write(z_matrix), run_time=0.6)
        self.play(FadeIn(z_label), run_time=0.3)
        self.wait(1)

        # Hadamard gate - special emphasis
        h_label = Text("Hadamard Gate:", font_size=24, color=GRAY)
        h_label.shift(DOWN * 1.2 + LEFT * 5)
        self.play(Write(h_label), run_time=0.4)

        h_matrix = MathTex(
            r"H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}",
            font_size=32,
        )
        h_matrix.shift(DOWN * 2)
        h_desc = Text("Creates superposition", font_size=20, color=ORANGE)
        h_desc.next_to(h_matrix, DOWN)

        self.play(Write(h_matrix), run_time=0.8)
        self.play(FadeIn(h_desc), run_time=0.4)
        self.wait(1.5)

        # CNOT gate
        cnot_label = Text("Two-Qubit Gate:", font_size=24, color=GRAY)
        cnot_label.shift(DOWN * 3.2 + LEFT * 5)
        self.play(Write(cnot_label), run_time=0.4)

        cnot_matrix = MathTex(
            r"\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}",
            font_size=24,
        )
        cnot_matrix.shift(DOWN * 4)

        self.play(Write(cnot_matrix), run_time=1)
        self.wait(2)

        # Universal gate set note
        universal_note = Text(
            "These gates form a universal set for quantum computation",
            font_size=22,
            color=YELLOW,
        )
        universal_note.shift(DOWN * 5)

        self.play(Write(universal_note), run_time=0.8)
        self.wait(2)

    def quantum_algorithms_depth(self):
        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)

        title = Text("QUANTUM ALGORITHMS", font_size=40, color=PURPLE)
        title.to_edge(UP)
        underline = Line(
            title.get_left() + DOWN * 0.3, title.get_right() + DOWN * 0.3, color=PURPLE
        )

        self.play(Write(title), Create(underline), run_time=0.6)

        # Part 1: Grover's Algorithm
        grover_title = Text("Grover's Search Algorithm", font_size=36, color=GREEN)
        grover_title.shift(UP * 2)
        self.play(Write(grover_title), run_time=0.6)
        self.wait(0.5)

        # Problem statement
        problem = Text(
            "Find marked item in unsorted database of N items", font_size=24, color=GRAY
        )
        problem.shift(UP * 1)
        self.play(Write(problem), run_time=0.8)
        self.wait(1)

        # Key components with clear layout
        components_title = Text("Key Components:", font_size=24)
        components_title.shift(LEFT * 4 + UP * 0.2)
        self.play(Write(components_title), run_time=0.4)

        # Oracle
        oracle_def = MathTex(r"O_f|x\rangle = (-1)^{f(x)}|x\rangle", font_size=28)
        oracle_def.shift(LEFT * 3 + DOWN * 0.5)
        oracle_label = Text("Oracle (marks solution)", font_size=18, color=BLUE)
        oracle_label.next_to(oracle_def, DOWN)

        self.play(Write(oracle_def), run_time=0.8)
        self.play(FadeIn(oracle_label), run_time=0.4)
        self.wait(1)

        # Diffusion operator
        diffusion_def = MathTex(r"D = 2|s\rangle\langle s| - I", font_size=28)
        diffusion_def.shift(RIGHT * 2 + DOWN * 0.5)

        uniform_state = MathTex(
            r"|s\rangle = \frac{1}{\sqrt{N}}\sum_{x}|x\rangle", font_size=22
        )
        uniform_state.shift(RIGHT * 2 + DOWN * 1.3)

        diff_label = Text("Diffusion (amplifies)", font_size=18, color=RED)
        diff_label.next_to(uniform_state, DOWN)

        self.play(Write(diffusion_def), run_time=0.8)
        self.play(Write(uniform_state), run_time=0.8)
        self.play(FadeIn(diff_label), run_time=0.4)
        self.wait(1.5)

        # Grover operator and iterations
        grover_op = MathTex(r"G = D \cdot O_f", font_size=32, color=YELLOW)
        grover_op.shift(DOWN * 2.5)

        iterations = MathTex(
            r"\text{Iterations: } k \approx \frac{\pi}{4}\sqrt{N}",
            font_size=28,
            color=GREEN,
        )
        iterations.shift(DOWN * 3.3)

        self.play(Write(grover_op), run_time=0.6)
        self.play(Write(iterations), run_time=0.8)
        self.wait(1.5)

        # Complexity comparison
        complexity_box = Rectangle(width=8, height=1.2, color=GOLD)
        complexity_box.shift(DOWN * 4.5)

        complexity_text = VGroup(
            Text("Classical: O(N)", font_size=24, color=RED),
            Text("→", font_size=28),
            Text("Quantum: O(√N)", font_size=24, color=GREEN),
        ).arrange(RIGHT, buff=0.5)
        complexity_text.move_to(complexity_box)

        self.play(Create(complexity_box), run_time=0.5)
        self.play(Write(complexity_text), run_time=0.8)
        self.wait(2)  # Key advantage

        # Part 2: Quantum Fourier Transform
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in [title, underline]],
            run_time=0.5
        )

        qft_title = Text("Quantum Fourier Transform", font_size=36, color=BLUE)
        qft_title.shift(UP * 2)
        self.play(Write(qft_title), run_time=0.6)
        self.wait(0.5)

        # Definition
        qft_def = MathTex(
            r"QFT|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle",
            font_size=32,
        )
        qft_def.shift(UP * 0.5)

        self.play(Write(qft_def), run_time=1.2)
        self.wait(2)

        # Matrix form (partial)
        qft_matrix_label = Text("Matrix representation:", font_size=22)
        qft_matrix_label.shift(DOWN * 0.5)

        qft_matrix = MathTex(
            r"F_N = \frac{1}{\sqrt{N}}\begin{pmatrix} "
            r"1 & 1 & 1 & \cdots \\ "
            r"1 & \omega & \omega^2 & \cdots \\ "
            r"1 & \omega^2 & \omega^4 & \cdots \\ "
            r"\vdots & \vdots & \vdots & \ddots "
            r"\end{pmatrix}",
            font_size=26,
        )
        qft_matrix.shift(DOWN * 2)

        omega_def = MathTex(r"\omega = e^{2\pi i/N}", font_size=22, color=BLUE)
        omega_def.shift(DOWN * 3.2)

        self.play(Write(qft_matrix_label), run_time=0.5)
        self.play(Write(qft_matrix), run_time=1.2)
        self.play(Write(omega_def), run_time=0.6)
        self.wait(2)

        # Applications
        applications = Text(
            "Key to: Shor's algorithm, Phase estimation, Order finding",
            font_size=22,
            color=YELLOW,
        )
        applications.shift(DOWN * 4.2)

        self.play(Write(applications), run_time=0.8)
        self.wait(2)

    def error_correction_topology(self):
        # Clear and 3D setup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        title = Text("QUANTUM ERROR CORRECTION", font_size=40, color=RED)
        title.to_edge(UP)
        underline = Line(
            title.get_left() + DOWN * 0.3, title.get_right() + DOWN * 0.3, color=RED
        )
        self.add_fixed_in_frame_mobjects(title, underline)
        self.play(Write(title), Create(underline), run_time=0.6)

        # Part 1: Types of Errors
        error_title = Text("Quantum Error Types", font_size=32, color=ORANGE)
        error_title.shift(UP * 2)
        self.add_fixed_in_frame_mobjects(error_title)
        self.play(Write(error_title), run_time=0.5)

        # Error table with clear layout
        errors = VGroup()

        # Bit flip
        bit_flip = MathTex(r"X: |0\rangle \leftrightarrow |1\rangle", font_size=28)
        bit_flip.shift(LEFT * 3 + UP * 0.5)
        bit_label = Text("Bit flip", font_size=20, color=BLUE)
        bit_label.next_to(bit_flip, LEFT)

        # Phase flip
        phase_flip = MathTex(r"Z: |+\rangle \leftrightarrow |-\rangle", font_size=28)
        phase_flip.shift(LEFT * 3 + DOWN * 0.5)
        phase_label = Text("Phase flip", font_size=20, color=GREEN)
        phase_label.next_to(phase_flip, LEFT)

        # Both
        both_flip = MathTex(r"Y = iXZ", font_size=28)
        both_flip.shift(LEFT * 3 + DOWN * 1.5)
        both_label = Text("Both", font_size=20, color=RED)
        both_label.next_to(both_flip, LEFT)

        self.add_fixed_in_frame_mobjects(
            bit_flip, bit_label, phase_flip, phase_label, both_flip, both_label
        )

        self.play(Write(bit_flip), Write(bit_label), run_time=0.8)
        self.wait(0.8)
        self.play(Write(phase_flip), Write(phase_label), run_time=0.8)
        self.wait(0.8)
        self.play(Write(both_flip), Write(both_label), run_time=0.8)
        self.wait(1.5)

        # Part 2: Simple Error Correction Code
        self.play(
            FadeOut(error_title),
            FadeOut(bit_flip),
            FadeOut(bit_label),
            FadeOut(phase_flip),
            FadeOut(phase_label),
            FadeOut(both_flip),
            FadeOut(both_label),
            run_time=0.5,
        )

        code_title = Text("3-Qubit Repetition Code", font_size=32, color=PURPLE)
        code_title.shift(UP * 2)
        self.add_fixed_in_frame_mobjects(code_title)
        self.play(Write(code_title), run_time=0.5)

        # Encoding
        encoding = MathTex(
            r"|0\rangle_L = |000\rangle, \quad |1\rangle_L = |111\rangle", font_size=32
        )
        encoding.shift(UP * 0.5)
        self.add_fixed_in_frame_mobjects(encoding)

        self.play(Write(encoding), run_time=1)
        self.wait(1.5)

        # Protection explanation
        protection = Text(
            "Protects against single bit flip errors", font_size=24, color=GREEN
        )
        protection.shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(protection)
        self.play(Write(protection), run_time=0.8)
        self.wait(1)

        # Part 3: Surface Code Visualization
        self.play(
            FadeOut(code_title), FadeOut(encoding), FadeOut(protection), run_time=0.5
        )

        surface_title = Text("Surface Code (2D Topology)", font_size=32, color=PURPLE)
        surface_title.shift(UP * 2.5)
        self.add_fixed_in_frame_mobjects(surface_title)
        self.play(Write(surface_title), run_time=0.6)

        # Create lattice with clear structure
        lattice = VGroup()
        connections = VGroup()

        # Create 4x4 grid
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    # Data qubit
                    qubit = Sphere(radius=0.2, resolution=(10, 10))
                    qubit.set_color(BLUE)
                    label = Text("D", font_size=12, color=WHITE)
                else:
                    # Syndrome qubit
                    qubit = Sphere(radius=0.2, resolution=(10, 10))
                    qubit.set_color(RED)
                    label = Text("S", font_size=12, color=WHITE)

                qubit.shift(RIGHT * (i - 1.5) + UP * (j - 1.5) + DOWN * 1.5)
                label.move_to(qubit.get_center())
                lattice.add(qubit)

        # Add connections
        for i in range(3):
            for j in range(4):
                # Horizontal
                line = Line(
                    RIGHT * (i - 1) + UP * (j - 1.5) + DOWN * 1.5,
                    RIGHT * (i) + UP * (j - 1.5) + DOWN * 1.5,
                    color=GRAY,
                    stroke_width=2,
                )
                connections.add(line)

        for i in range(4):
            for j in range(3):
                # Vertical
                line = Line(
                    RIGHT * (i - 1.5) + UP * (j - 1) + DOWN * 1.5,
                    RIGHT * (i - 1.5) + UP * (j) + DOWN * 1.5,
                    color=GRAY,
                    stroke_width=2,
                )
                connections.add(line)

        self.play(Create(connections), run_time=0.8)
        self.play(Create(lattice), run_time=1)

        # Legend
        legend = VGroup()
        data_sample = Sphere(radius=0.15, resolution=(8, 8), color=BLUE)
        data_sample.shift(DOWN * 3.5 + LEFT * 2)
        data_text = Text("Data qubits", font_size=18)
        data_text.next_to(data_sample, RIGHT)

        syndrome_sample = Sphere(radius=0.15, resolution=(8, 8), color=RED)
        syndrome_sample.shift(DOWN * 3.5 + RIGHT * 1)
        syndrome_text = Text("Syndrome qubits", font_size=18)
        syndrome_text.next_to(syndrome_sample, RIGHT)

        self.add_fixed_in_frame_mobjects(data_text, syndrome_text)
        self.play(
            Create(data_sample),
            Write(data_text),
            Create(syndrome_sample),
            Write(syndrome_text),
            run_time=0.8,
        )
        self.wait(2)

        # Threshold theorem
        threshold = MathTex(
            r"p < p_{th} \approx 10^{-2} \Rightarrow \text{Scalable QC}",
            font_size=28,
            color=GREEN,
        )
        threshold.shift(DOWN * 4.5)
        self.add_fixed_in_frame_mobjects(threshold)

        threshold_box = SurroundingRectangle(threshold, color=GREEN, buff=0.2)
        self.add_fixed_in_frame_mobjects(threshold_box)

        self.play(Write(threshold), Create(threshold_box), run_time=1)
        self.wait(2.5)  # Important result

        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

    def quantum_complexity_theory(self):
        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)

        title = Text("QUANTUM COMPLEXITY THEORY", font_size=40, color=GOLD)
        title.to_edge(UP)
        underline = Line(
            title.get_left() + DOWN * 0.3, title.get_right() + DOWN * 0.3, color=GOLD
        )

        self.play(Write(title), Create(underline), run_time=0.6)

        # Part 1: Complexity Classes
        classes_title = Text("Computational Complexity Classes", font_size=32)
        classes_title.shift(UP * 2)
        self.play(Write(classes_title), run_time=0.6)

        # Create nested diagram with labels
        diagram_center = LEFT * 3

        # PSPACE (outermost)
        pspace_circle = Circle(radius=2.2, color=RED)
        pspace_circle.shift(diagram_center)
        pspace_label = Text("PSPACE", font_size=20).shift(diagram_center + UP * 2)

        # BQP
        bqp_circle = Circle(radius=1.6, color=PURPLE)
        bqp_circle.shift(diagram_center)
        bqp_label = Text("BQP", font_size=20).shift(diagram_center + UP * 1.4)

        # BPP
        bpp_circle = Circle(radius=1.1, color=BLUE)
        bpp_circle.shift(diagram_center)
        bpp_label = Text("BPP", font_size=18).shift(diagram_center + UP * 0.9)

        # P (innermost)
        p_circle = Circle(radius=0.6, color=GREEN)
        p_circle.shift(diagram_center)
        p_label = Text("P", font_size=18).move_to(p_circle)

        self.play(Create(pspace_circle), Write(pspace_label), run_time=0.6)
        self.play(Create(bqp_circle), Write(bqp_label), run_time=0.6)
        self.play(Create(bpp_circle), Write(bpp_label), run_time=0.6)
        self.play(Create(p_circle), Write(p_label), run_time=0.6)
        self.wait(1)

        # Definitions
        def_title = Text("Key Definitions:", font_size=24)
        def_title.shift(RIGHT * 2 + UP * 1.5)
        self.play(Write(def_title), run_time=0.4)

        # BQP definition
        bqp_def = VGroup(
            Text("BQP:", font_size=22, color=PURPLE),
            MathTex(r"\Pr[M(x) = f(x)] \geq \frac{2}{3}", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT)
        bqp_def.shift(RIGHT * 2 + UP * 0.5)

        bqp_desc = Text(
            "Polynomial time\nquantum computation", font_size=18, color=GRAY
        )
        bqp_desc.next_to(bqp_def, DOWN, aligned_edge=LEFT)

        self.play(Write(bqp_def), run_time=0.8)
        self.play(FadeIn(bqp_desc), run_time=0.5)
        self.wait(1.5)

        # Key question
        question = Text("P ⊊ BQP ⊆ PSPACE", font_size=28, color=YELLOW)
        question.shift(RIGHT * 2 + DOWN * 1.5)
        question_box = SurroundingRectangle(question, color=YELLOW, buff=0.2)

        self.play(Write(question), Create(question_box), run_time=0.8)
        self.wait(1.5)

        # Part 2: Quantum Supremacy
        self.play(
            FadeOut(classes_title),
            FadeOut(def_title),
            FadeOut(bqp_def),
            FadeOut(bqp_desc),
            run_time=0.5,
        )

        supremacy_title = Text("Quantum Supremacy", font_size=36, color=ORANGE)
        supremacy_title.shift(UP * 2)
        self.play(Write(supremacy_title), run_time=0.6)

        supremacy_def = Text(
            "Quantum device solving problem infeasible for classical computers",
            font_size=24,
        )
        supremacy_def.shift(UP * 0.8)

        self.play(Write(supremacy_def), run_time=1)
        self.wait(1.5)

        # Examples
        examples_title = Text("Achieved Examples:", font_size=24)
        examples_title.shift(DOWN * 0.5)

        examples = (
            VGroup(
                Text("• Random circuit sampling (Google, 2019)", font_size=20),
                Text("• Boson sampling (USTC, 2020)", font_size=20),
                Text("• Gaussian boson sampling (Xanadu, 2020)", font_size=20),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .shift(DOWN * 1.5)
        )

        self.play(Write(examples_title), run_time=0.5)
        self.play(Write(examples), run_time=1.2)
        self.wait(2)

        # Part 3: NISQ Era
        self.play(
            FadeOut(supremacy_title),
            FadeOut(supremacy_def),
            FadeOut(examples_title),
            FadeOut(examples),
            run_time=0.5,
        )

        nisq_title = Text("NISQ Era", font_size=36, color=ORANGE)
        nisq_title.shift(RIGHT * 2 + UP * 0.5)

        nisq_full = Text("Noisy Intermediate-Scale Quantum", font_size=22, color=GRAY)
        nisq_full.next_to(nisq_title, DOWN)

        self.play(Write(nisq_title), run_time=0.5)
        self.play(Write(nisq_full), run_time=0.6)

        # Characteristics
        nisq_chars = (
            VGroup(
                Text("• 50-100 qubits", font_size=20),
                Text("• Limited coherence time", font_size=20),
                Text("• No error correction", font_size=20),
                Text("• Shallow circuits only", font_size=20),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .shift(RIGHT * 2 + DOWN * 1.5)
        )

        self.play(Write(nisq_chars), run_time=1)
        self.wait(2)

        # Future path
        future = Text(
            "Path Forward: Fault-Tolerant Quantum Computing", font_size=28, color=GREEN
        )
        future.shift(DOWN * 3.5)

        requirements = Text(
            "Requires: Better qubits, Error correction, Deeper circuits",
            font_size=20,
            color=GRAY,
        )
        requirements.shift(DOWN * 4.2)

        self.play(Write(future), run_time=0.8)
        self.play(FadeIn(requirements), run_time=0.6)
        self.wait(2.5)

        # End card
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)

        end_title = Text("QUANTUM COMPUTING", font_size=48, color=BLUE, weight=BOLD)
        end_subtitle = Text("The Mathematical Foundations", font_size=28, color=GRAY)
        end_group = VGroup(end_title, end_subtitle).arrange(DOWN, buff=0.4)

        self.play(Write(end_group), run_time=1)
        self.wait(2)
