import numpy as np
from manim import *


class QuantumComputingDemo(ThreeDScene):
    def construct(self):
        # Configure 3D scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Title
        title = Text("Quantum Computing", font_size=60, color=BLUE)
        subtitle = Text(
            "Visualizing Qubits and Quantum Gates", font_size=30, color=GRAY
        )
        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Classical vs Quantum
        self.classical_vs_quantum_simple()

        # Bloch sphere
        self.bloch_sphere_demo()

        # Quantum gates visualization
        self.quantum_gates_demo()

        # Entanglement
        self.entanglement_visual()

        # End screen
        self.end_screen()

    def classical_vs_quantum_simple(self):
        # Title
        section_title = Text("Classical Bit vs Quantum Qubit", font_size=40)
        self.play(Write(section_title))
        self.play(section_title.animate.to_edge(UP).scale(0.8))

        # Classical bit visualization
        bit_text = Text("Classical Bit:", font_size=30).shift(LEFT * 4 + UP)
        bit_0 = Circle(radius=0.8, color=WHITE, stroke_width=4).shift(LEFT * 5 + DOWN)
        bit_0_label = Text("0", font_size=40).move_to(bit_0)
        bit_1 = Circle(radius=0.8, color=WHITE, stroke_width=4).shift(LEFT * 3 + DOWN)
        bit_1_label = Text("1", font_size=40).move_to(bit_1)
        or_text = Text("OR", font_size=25).shift(LEFT * 4 + DOWN)

        self.play(Write(bit_text))
        self.play(Create(bit_0), Write(bit_0_label))
        self.play(Create(bit_1), Write(bit_1_label))
        self.play(Write(or_text))

        # Quantum qubit visualization
        qubit_text = Text("Quantum Qubit:", font_size=30).shift(RIGHT * 3 + UP)
        qubit_circle = Circle(radius=1.2, color=BLUE, fill_opacity=0.3).shift(
            RIGHT * 3 + DOWN
        )

        # Create superposition visual
        state_0 = Text("|0⟩", font_size=30, color=BLUE_C).shift(
            RIGHT * 3 + DOWN + UP * 0.5
        )
        state_1 = Text("|1⟩", font_size=30, color=RED_C).shift(
            RIGHT * 3 + DOWN + DOWN * 0.5
        )
        plus_sign = Text("+", font_size=25).shift(RIGHT * 3 + DOWN)

        alpha = Text("α", font_size=25, color=BLUE_C).next_to(state_0, LEFT)
        beta = Text("β", font_size=25, color=RED_C).next_to(state_1, LEFT)

        self.play(Write(qubit_text))
        self.play(Create(qubit_circle))
        self.play(
            Write(alpha), Write(state_0), Write(plus_sign), Write(beta), Write(state_1)
        )

        # Superposition label
        super_label = Text("SUPERPOSITION", font_size=25, color=GREEN)
        super_label.next_to(qubit_circle, DOWN, buff=0.5)
        self.play(FadeIn(super_label))

        self.wait(3)

        # Clear
        self.play(
            *[
                FadeOut(obj)
                for obj in [
                    section_title,
                    bit_text,
                    bit_0,
                    bit_0_label,
                    bit_1,
                    bit_1_label,
                    or_text,
                    qubit_text,
                    qubit_circle,
                    state_0,
                    state_1,
                    plus_sign,
                    alpha,
                    beta,
                    super_label,
                ]
            ]
        )

    def bloch_sphere_demo(self):
        # Move camera for 3D view
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)

        # Title
        title = Text("The Bloch Sphere", font_size=40)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Create Bloch sphere
        sphere = Sphere(radius=2, resolution=(20, 20))
        sphere.set_color(BLUE_E)
        sphere.set_opacity(0.2)

        # Coordinate axes
        x_axis = Arrow3D(start=[-2.5, 0, 0], end=[2.5, 0, 0], color=RED)
        y_axis = Arrow3D(start=[0, -2.5, 0], end=[0, 2.5, 0], color=GREEN)
        z_axis = Arrow3D(start=[0, 0, -2.5], end=[0, 0, 2.5], color=BLUE)

        # Axis labels
        x_label = Text("X", font_size=20, color=RED)
        x_label.move_to([2.8, 0, 0])
        y_label = Text("Y", font_size=20, color=GREEN)
        y_label.move_to([0, 2.8, 0])
        z_label = Text("Z", font_size=20, color=BLUE)
        z_label.move_to([0, 0, 2.8])

        self.play(Create(sphere))
        self.play(Create(x_axis), Create(y_axis), Create(z_axis))
        self.add_fixed_orientation_mobjects(x_label, y_label, z_label)
        self.play(Write(x_label), Write(y_label), Write(z_label))

        # |0⟩ and |1⟩ states
        zero_state = Text("|0⟩", font_size=30, color=BLUE)
        zero_state.move_to([0, 0, 2.5])
        one_state = Text("|1⟩", font_size=30, color=BLUE)
        one_state.move_to([0, 0, -2.5])

        self.add_fixed_orientation_mobjects(zero_state, one_state)
        self.play(Write(zero_state), Write(one_state))

        # State vector
        state_vector = Arrow3D(
            start=[0, 0, 0], end=[0, 0, 2], color=YELLOW, thickness=0.08
        )

        # State label
        psi_label = Text("|ψ⟩", font_size=25, color=YELLOW)
        psi_label.move_to([0.3, 0, 2.3])

        self.play(Create(state_vector))
        self.add_fixed_orientation_mobjects(psi_label)
        self.play(Write(psi_label))

        # Rotate state vector to show different states
        self.begin_ambient_camera_rotation(rate=0.1)

        # X rotation (bit flip)
        x_gate_text = Text("X Gate (Bit Flip)", font_size=30, color=RED)
        x_gate_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(x_gate_text)
        self.play(Write(x_gate_text))

        new_vector = Arrow3D(
            start=[0, 0, 0], end=[0, 0, -2], color=YELLOW, thickness=0.08
        )
        self.play(Transform(state_vector, new_vector), run_time=2)
        self.wait(1)
        self.play(FadeOut(x_gate_text))

        # Hadamard gate - superposition
        h_gate_text = Text("H Gate (Superposition)", font_size=30, color=GREEN)
        h_gate_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(h_gate_text)
        self.play(Write(h_gate_text))

        super_vector = Arrow3D(
            start=[0, 0, 0], end=[2, 0, 0], color=YELLOW, thickness=0.08
        )
        self.play(Transform(state_vector, super_vector), run_time=2)

        # Show |+⟩ state
        plus_state = Text("|+⟩", font_size=25, color=GREEN)
        plus_state.move_to([2.3, 0, 0])
        self.add_fixed_orientation_mobjects(plus_state)
        self.play(Write(plus_state))

        self.stop_ambient_camera_rotation()
        self.wait(2)

        # Store for later
        self.sphere = sphere
        self.axes = VGroup(x_axis, y_axis, z_axis)
        self.state_vector = state_vector

        # Clear labels
        self.play(
            FadeOut(title),
            FadeOut(h_gate_text),
            FadeOut(psi_label),
            FadeOut(plus_state),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(z_label),
            FadeOut(zero_state),
            FadeOut(one_state),
        )

    def quantum_gates_demo(self):
        # Reset camera
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)
        self.play(FadeOut(self.sphere), FadeOut(self.axes), FadeOut(self.state_vector))

        # Title
        title = Text("Quantum Gates", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Gate matrix representations
        gates_info = [
            ("X", "NOT Gate", "[[0, 1], [1, 0]]", BLUE),
            ("Z", "Phase Flip", "[[1, 0], [0, -1]]", RED),
            ("H", "Hadamard", "[[1, 1], [1, -1]]/√2", GREEN),
        ]

        gate_group = VGroup()
        for i, (name, desc, matrix, color) in enumerate(gates_info):
            # Gate box
            gate_rect = Rectangle(width=2, height=1.5, color=color, fill_opacity=0.3)
            gate_rect.shift(LEFT * 4 + RIGHT * i * 4 + DOWN * 0.5)

            # Gate name
            gate_name = Text(name, font_size=35, color=color)
            gate_name.move_to(gate_rect)

            # Description
            gate_desc = Text(desc, font_size=18)
            gate_desc.next_to(gate_rect, UP)

            # Matrix (simplified text representation)
            gate_matrix = Text(matrix, font_size=14)
            gate_matrix.next_to(gate_rect, DOWN)

            gate_group.add(gate_rect, gate_name, gate_desc, gate_matrix)

        self.play(Create(gate_group))
        self.wait(3)

        # Show gate action on qubit
        action_text = Text("Gates transform qubit states", font_size=30, color=YELLOW)
        action_text.shift(DOWN * 3)
        self.play(Write(action_text))

        self.wait(2)
        self.play(FadeOut(gate_group), FadeOut(title), FadeOut(action_text))

    def entanglement_visual(self):
        # Title
        title = Text("Quantum Entanglement", font_size=40, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title))

        # Two qubits
        qubit1 = Circle(radius=1, color=BLUE, stroke_width=4).shift(LEFT * 3)
        qubit1_label = Text("Qubit 1", font_size=25).next_to(qubit1, UP)
        qubit1_state = Text("|0⟩", font_size=30, color=BLUE).move_to(qubit1)

        qubit2 = Circle(radius=1, color=RED, stroke_width=4).shift(RIGHT * 3)
        qubit2_label = Text("Qubit 2", font_size=25).next_to(qubit2, UP)
        qubit2_state = Text("|0⟩", font_size=30, color=RED).move_to(qubit2)

        self.play(
            Create(qubit1),
            Create(qubit2),
            Write(qubit1_label),
            Write(qubit2_label),
            Write(qubit1_state),
            Write(qubit2_state),
        )

        # Initial state
        state_text = Text("Initial State: |00⟩", font_size=30)
        state_text.shift(UP * 2)
        self.play(Write(state_text))

        # Apply operations
        operation_text = Text("Apply H ⊗ I, then CNOT", font_size=25, color=YELLOW)
        operation_text.shift(DOWN * 1.5)
        self.play(Write(operation_text))
        self.wait(1)

        # Create entanglement - show connection
        connection = Line(
            qubit1.get_center(), qubit2.get_center(), color=PURPLE, stroke_width=8
        )
        connection.set_opacity(0)

        # Transform to Bell state
        bell_text = Text("Bell State: (|00⟩ + |11⟩)/√2", font_size=30, color=PURPLE)
        bell_text.move_to(state_text)

        # Update qubit visuals
        qubit1_super = Text("↑↓", font_size=30, color=PURPLE).move_to(qubit1)
        qubit2_super = Text("↑↓", font_size=30, color=PURPLE).move_to(qubit2)

        self.play(
            Transform(state_text, bell_text),
            Transform(qubit1_state, qubit1_super),
            Transform(qubit2_state, qubit2_super),
            connection.animate.set_opacity(0.8),
            qubit1.animate.set_stroke(PURPLE, width=6),
            qubit2.animate.set_stroke(PURPLE, width=6),
        )

        # Pulsing effect
        for _ in range(2):
            self.play(
                connection.animate.set_stroke_width(12).set_opacity(1),
                rate_func=there_and_back,
                run_time=0.8,
            )

        # Entangled label
        entangled_label = Text("ENTANGLED!", font_size=35, color=PURPLE)
        entangled_label.shift(DOWN * 3)
        self.play(Write(entangled_label))

        # Show correlation
        measure_text = Text(
            "Measuring one instantly affects the other!", font_size=25, color=GREEN
        )
        measure_text.next_to(entangled_label, DOWN)
        self.play(Write(measure_text))

        self.wait(3)

        # Clear
        self.play(
            *[
                FadeOut(obj)
                for obj in [
                    title,
                    qubit1,
                    qubit2,
                    qubit1_label,
                    qubit2_label,
                    qubit1_state,
                    qubit2_state,
                    state_text,
                    operation_text,
                    connection,
                    entangled_label,
                    measure_text,
                ]
            ]
        )

    def end_screen(self):
        # Title
        title = Text("Quantum Computing", font_size=60, color=BLUE)
        subtitle = Text("The Future of Computation", font_size=30, color=GRAY)
        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))

        # Key concepts
        concepts = VGroup(
            Text("• Superposition", font_size=25, color=GREEN),
            Text("• Entanglement", font_size=25, color=PURPLE),
            Text("• Quantum Gates", font_size=25, color=YELLOW),
            Text("• Quantum Algorithms", font_size=25, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        concepts.next_to(subtitle, DOWN, buff=1)

        self.play(Write(concepts))

        # Applications
        apps_text = Text("Applications:", font_size=28)
        apps_text.next_to(concepts, DOWN, buff=0.8)

        apps = Text(
            "Cryptography • Drug Discovery • AI • Optimization",
            font_size=22,
            color=GRAY,
        )
        apps.next_to(apps_text, DOWN)

        self.play(Write(apps_text))
        self.play(FadeIn(apps))

        self.wait(4)
