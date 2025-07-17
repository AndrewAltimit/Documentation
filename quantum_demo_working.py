import numpy as np
from manim import *


class QuantumDemo(Scene):
    def construct(self):
        # Fast title
        title = Text("Quantum Computing", font_size=48, color=BLUE)
        self.play(Write(title), run_time=0.5)
        self.play(FadeOut(title), run_time=0.3)

        # 1. Superposition
        self.show_superposition()

        # 2. Measurement
        self.show_measurement()

        # 3. Entanglement
        self.show_entanglement()

        # 4. Quantum advantage
        self.show_quantum_advantage()

        # End
        end_text = Text("Quantum Computing", font_size=40, color=BLUE)
        self.play(Write(end_text), run_time=0.5)
        self.wait(0.5)

    def show_superposition(self):
        # Section title
        title = Text("Superposition", font_size=36, color=GREEN)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Classical bit
        classical_text = Text("Classical Bit:", font_size=20)
        classical_text.shift(LEFT * 3 + UP * 2)

        bit_0 = Circle(radius=0.5, color=WHITE, fill_opacity=1)
        bit_0.shift(LEFT * 3.5 + UP * 0.5)
        bit_0_text = Text("0", font_size=24, color=BLACK)
        bit_0_text.move_to(bit_0)

        bit_1 = Circle(radius=0.5, color=BLACK, fill_opacity=1)
        bit_1.shift(LEFT * 2.5 + UP * 0.5)
        bit_1_text = Text("1", font_size=24, color=WHITE)
        bit_1_text.move_to(bit_1)

        self.play(
            Write(classical_text),
            Create(bit_0),
            Write(bit_0_text),
            Create(bit_1),
            Write(bit_1_text),
            run_time=0.5,
        )

        # Quantum qubit
        quantum_text = Text("Quantum Qubit:", font_size=20)
        quantum_text.shift(RIGHT * 2 + UP * 2)

        # Bloch sphere (simplified)
        sphere = Circle(radius=1, color=BLUE, stroke_width=3)
        sphere.shift(RIGHT * 2 + UP * 0.5)

        # State vector
        vector = Arrow(
            start=sphere.get_center(),
            end=sphere.get_center() + UP,
            color=YELLOW,
            buff=0,
            stroke_width=4,
        )

        # State equation
        state_eq = Text("α|0⟩ + β|1⟩", font_size=18, color=BLUE)
        state_eq.shift(RIGHT * 2 + DOWN * 1)

        self.play(
            Write(quantum_text),
            Create(sphere),
            Create(vector),
            Write(state_eq),
            run_time=0.5,
        )

        # Rotate vector to show superposition
        angles = [
            PI / 4,
            PI / 2,
            3 * PI / 4,
            PI,
            5 * PI / 4,
            3 * PI / 2,
            7 * PI / 4,
            2 * PI,
        ]
        for angle in angles:
            new_vector = Arrow(
                start=sphere.get_center(),
                end=sphere.get_center() + np.array([np.sin(angle), np.cos(angle), 0]),
                color=YELLOW,
                buff=0,
                stroke_width=4,
            )
            self.play(Transform(vector, new_vector), run_time=0.2)

        # Highlight
        super_text = Text("SUPERPOSITION!", font_size=24, color=GREEN)
        super_text.shift(DOWN * 2)
        self.play(Write(super_text), run_time=0.3)

        self.wait(0.3)

        # Clear
        self.play(
            FadeOut(title),
            FadeOut(classical_text),
            FadeOut(bit_0),
            FadeOut(bit_0_text),
            FadeOut(bit_1),
            FadeOut(bit_1_text),
            FadeOut(quantum_text),
            FadeOut(sphere),
            FadeOut(vector),
            FadeOut(state_eq),
            FadeOut(super_text),
            run_time=0.3,
        )

    def show_measurement(self):
        # Section title
        title = Text("Measurement", font_size=36, color=RED)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Bloch sphere
        sphere = Circle(radius=1.5, color=BLUE_E, stroke_width=2)

        # Axes
        x_axis = Line(LEFT * 2, RIGHT * 2, color=GRAY, stroke_width=1)
        y_axis = Line(DOWN * 2, UP * 2, color=GRAY, stroke_width=1)

        # States
        zero_state = Text("|0⟩", font_size=20, color=BLUE)
        zero_state.shift(UP * 2)
        one_state = Text("|1⟩", font_size=20, color=RED)
        one_state.shift(DOWN * 2)

        self.play(
            Create(sphere),
            Create(x_axis),
            Create(y_axis),
            Write(zero_state),
            Write(one_state),
            run_time=0.5,
        )

        # Superposition vector
        vector = Arrow(ORIGIN, UR, color=YELLOW, buff=0, stroke_width=5)
        psi_text = Text("|ψ⟩", font_size=18, color=YELLOW)
        psi_text.shift(UR * 1.3)

        self.play(Create(vector), Write(psi_text), run_time=0.3)

        # Measurement box
        measure_box = Rectangle(width=2, height=0.8, color=RED, fill_opacity=0.3)
        measure_box.shift(RIGHT * 3.5)
        measure_text = Text("MEASURE", font_size=16, color=RED)
        measure_text.move_to(measure_box)

        self.play(Create(measure_box), Write(measure_text), run_time=0.3)

        # Show collapse
        flash = Flash(vector, color=RED, flash_radius=0.8)

        collapsed_vector = Arrow(
            ORIGIN, DOWN * 1.5, color=YELLOW, buff=0, stroke_width=5
        )

        self.play(
            flash,
            Transform(vector, collapsed_vector),
            psi_text.animate.move_to(DOWN * 1.8 + RIGHT * 0.3),
            run_time=0.5,
        )

        # Result
        result_text = Text("Collapsed to |1⟩", font_size=20, color=RED)
        result_text.shift(DOWN * 3)
        self.play(Write(result_text), run_time=0.3)

        self.wait(0.3)

        # Clear
        self.play(
            FadeOut(title),
            FadeOut(sphere),
            FadeOut(x_axis),
            FadeOut(y_axis),
            FadeOut(zero_state),
            FadeOut(one_state),
            FadeOut(vector),
            FadeOut(psi_text),
            FadeOut(measure_box),
            FadeOut(measure_text),
            FadeOut(result_text),
            run_time=0.3,
        )

    def show_entanglement(self):
        # Section title
        title = Text("Entanglement", font_size=36, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Two qubits
        alice_circle = Circle(radius=0.7, color=BLUE, stroke_width=4)
        alice_circle.shift(LEFT * 2.5)
        alice_text = Text("Alice", font_size=18)
        alice_text.next_to(alice_circle, DOWN)

        bob_circle = Circle(radius=0.7, color=RED, stroke_width=4)
        bob_circle.shift(RIGHT * 2.5)
        bob_text = Text("Bob", font_size=18)
        bob_text.next_to(bob_circle, DOWN)

        self.play(
            Create(alice_circle),
            Write(alice_text),
            Create(bob_circle),
            Write(bob_text),
            run_time=0.5,
        )

        # Initial states
        alice_state = Arrow(
            alice_circle.get_center(),
            alice_circle.get_center() + UP * 0.7,
            color=YELLOW,
            buff=0,
            stroke_width=3,
        )
        bob_state = Arrow(
            bob_circle.get_center(),
            bob_circle.get_center() + UP * 0.7,
            color=YELLOW,
            buff=0,
            stroke_width=3,
        )

        self.play(Create(alice_state), Create(bob_state), run_time=0.3)

        # Bell state equation
        bell_eq = Text("(|00⟩ + |11⟩)/√2", font_size=20, color=PURPLE)
        bell_eq.shift(UP * 2)
        self.play(Write(bell_eq), run_time=0.3)

        # Create entanglement connection
        connection = DashedLine(
            alice_circle.get_center(),
            bob_circle.get_center(),
            color=PURPLE,
            stroke_width=5,
            dash_length=0.1,
        )

        self.play(
            Create(connection),
            alice_circle.animate.set_stroke(PURPLE, width=5),
            bob_circle.animate.set_stroke(PURPLE, width=5),
            run_time=0.5,
        )

        # Pulse effect
        pulse = Circle(radius=0.7, color=PURPLE, stroke_width=6)
        pulse.move_to(alice_circle)

        self.play(pulse.animate.scale(4).set_opacity(0), run_time=0.8)

        # Measure Alice
        measure_text = Text("Measure Alice", font_size=18, color=GREEN)
        measure_text.shift(DOWN * 2)
        self.play(Write(measure_text), run_time=0.3)

        # Alice collapses
        alice_collapsed = Arrow(
            alice_circle.get_center(),
            alice_circle.get_center() + DOWN * 0.7,
            color=YELLOW,
            buff=0,
            stroke_width=3,
        )

        self.play(
            Flash(alice_circle, color=GREEN),
            Transform(alice_state, alice_collapsed),
            run_time=0.4,
        )

        # Show correlation
        correlation_wave = Circle(radius=0.1, color=PURPLE, stroke_width=4)
        correlation_wave.move_to(alice_circle)

        self.play(
            correlation_wave.animate.move_to(bob_circle).scale(15).set_opacity(0),
            run_time=0.5,
        )

        # Bob collapses instantly
        bob_collapsed = Arrow(
            bob_circle.get_center(),
            bob_circle.get_center() + DOWN * 0.7,
            color=YELLOW,
            buff=0,
            stroke_width=3,
        )

        self.play(Transform(bob_state, bob_collapsed), run_time=0.3)

        # Spooky action
        spooky_text = Text("Spooky Action!", font_size=22, color=PURPLE)
        spooky_text.shift(DOWN * 3)
        self.play(Write(spooky_text), run_time=0.3)

        self.wait(0.5)

        # Clear
        self.play(
            FadeOut(title),
            FadeOut(alice_circle),
            FadeOut(alice_text),
            FadeOut(bob_circle),
            FadeOut(bob_text),
            FadeOut(alice_state),
            FadeOut(bob_state),
            FadeOut(bell_eq),
            FadeOut(connection),
            FadeOut(measure_text),
            FadeOut(spooky_text),
            run_time=0.3,
        )

    def show_quantum_advantage(self):
        # Section title
        title = Text("Quantum Advantage", font_size=36, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Classical vs Quantum
        classical_label = Text("Classical", font_size=24, color=RED)
        classical_label.shift(LEFT * 3 + UP * 2)

        quantum_label = Text("Quantum", font_size=24, color=BLUE)
        quantum_label.shift(RIGHT * 3 + UP * 2)

        self.play(Write(classical_label), Write(quantum_label), run_time=0.3)

        # Show scaling
        # 10 bits/qubits
        bits_10 = Text("10 bits:", font_size=18)
        bits_10.shift(LEFT * 4)
        classical_10 = Text("1 state", font_size=16, color=RED)
        classical_10.shift(LEFT * 2)
        quantum_10 = Text("1,024 states", font_size=16, color=BLUE)
        quantum_10.shift(RIGHT * 2)

        # 20 bits/qubits
        bits_20 = Text("20 bits:", font_size=18)
        bits_20.shift(LEFT * 4 + DOWN)
        classical_20 = Text("1 state", font_size=16, color=RED)
        classical_20.shift(LEFT * 2 + DOWN)
        quantum_20 = Text("1,048,576 states", font_size=16, color=BLUE)
        quantum_20.shift(RIGHT * 2 + DOWN)

        self.play(
            Write(bits_10),
            Write(classical_10),
            Write(quantum_10),
            Write(bits_20),
            Write(classical_20),
            Write(quantum_20),
            run_time=0.8,
        )

        # Exponential equation
        exp_eq = Text("2ⁿ states", font_size=28, color=GREEN)
        exp_eq.shift(DOWN * 2.5)

        self.play(Write(exp_eq), Flash(exp_eq, color=GREEN), run_time=0.5)

        self.wait(0.5)
