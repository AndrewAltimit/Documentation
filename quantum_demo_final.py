import numpy as np
from manim import *


class QuantumDemoFinal(Scene):
    def construct(self):
        # Quick title (0.5 seconds)
        title = Text("Quantum Computing", font_size=56, color=BLUE)
        self.add(title)
        self.wait(0.2)
        self.play(FadeOut(title), run_time=0.3)

        # 1. Qubit vs Bit with visual clarity
        self.qubit_vs_bit_visual()

        # 2. Superposition with measurement
        self.superposition_measurement()

        # 3. Entanglement with correlation
        self.entanglement_correlation()

        # 4. Quantum interference
        self.quantum_interference()

        # 5. Quantum advantage
        self.quantum_advantage_simple()

    def qubit_vs_bit_visual(self):
        # Fast title
        title = Text("Qubit vs Classical Bit", font_size=36)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Classical bit - binary states
        bit_label = Text("Classical Bit:", font_size=24).shift(LEFT * 4 + UP * 2)
        bit_0 = Circle(radius=0.7, color=WHITE, fill_opacity=1).shift(LEFT * 4)
        bit_0_text = Text("0", font_size=36, color=BLACK).move_to(bit_0)
        bit_state = Text("State: 0 OR 1", font_size=18).shift(LEFT * 4 + DOWN * 1.5)

        self.play(
            Write(bit_label),
            Create(bit_0),
            Write(bit_0_text),
            Write(bit_state),
            run_time=0.5,
        )

        # Flip bit animation
        bit_1 = Circle(radius=0.7, color=BLACK, fill_opacity=1).shift(LEFT * 4)
        bit_1_text = Text("1", font_size=36, color=WHITE).move_to(bit_1)

        for _ in range(2):
            self.play(
                Transform(bit_0, bit_1), Transform(bit_0_text, bit_1_text), run_time=0.3
            )
            self.play(
                Transform(
                    bit_0,
                    Circle(radius=0.7, color=WHITE, fill_opacity=1).shift(LEFT * 4),
                ),
                Transform(
                    bit_0_text, Text("0", font_size=36, color=BLACK).move_to(bit_0)
                ),
                run_time=0.3,
            )

        # Quantum qubit - continuous states
        qubit_label = Text("Quantum Qubit:", font_size=24).shift(RIGHT * 3 + UP * 2)

        # Bloch sphere
        sphere = Circle(radius=1.2, color=BLUE, stroke_width=3).shift(RIGHT * 3)

        # State vector
        vector = Arrow(
            sphere.get_center(),
            sphere.get_center() + np.array([0.85, 0.85, 0]),
            color=YELLOW,
            buff=0,
            stroke_width=5,
        )

        # State equation
        state_eq = Text("α|0⟩ + β|1⟩", font_size=20, color=BLUE).shift(
            RIGHT * 3 + DOWN * 1.5
        )

        # Visual probability indicators
        prob_text = Text("Probabilities:", font_size=16).shift(RIGHT * 3 + DOWN * 2.2)
        prob_0 = Text("|α|² = 50%", font_size=14, color=BLUE).shift(
            RIGHT * 2 + DOWN * 2.7
        )
        prob_1 = Text("|β|² = 50%", font_size=14, color=RED).shift(
            RIGHT * 4 + DOWN * 2.7
        )

        self.play(
            Write(qubit_label),
            Create(sphere),
            Create(vector),
            Write(state_eq),
            Write(prob_text),
            Write(prob_0),
            Write(prob_1),
            run_time=0.8,
        )

        # Animate continuous rotation
        for angle in np.linspace(0, 2 * PI, 20):
            new_end = sphere.get_center() + 1.2 * np.array(
                [np.sin(angle), np.cos(angle), 0]
            )
            vector.put_start_and_end_on(sphere.get_center(), new_end)

            # Update probability text
            p0 = int(np.cos(angle / 2) ** 2 * 100)
            p1 = 100 - p0
            prob_0.become(
                Text(f"|α|² = {p0}%", font_size=14, color=BLUE).shift(
                    RIGHT * 2 + DOWN * 2.7
                )
            )
            prob_1.become(
                Text(f"|β|² = {p1}%", font_size=14, color=RED).shift(
                    RIGHT * 4 + DOWN * 2.7
                )
            )

            self.wait(0.1)

        # Highlight superposition
        super_text = Text("SUPERPOSITION!", font_size=28, color=GREEN)
        super_text.shift(DOWN * 3.5)
        self.play(Write(super_text), Flash(sphere, color=GREEN), run_time=0.5)

        self.wait(0.3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.3)

    def superposition_measurement(self):
        title = Text("Superposition & Measurement", font_size=36)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Bloch sphere with states
        sphere = Circle(radius=2, color=BLUE_E, stroke_width=2)

        # Axes
        x_axis = Line(LEFT * 2.5, RIGHT * 2.5, color=GRAY)
        y_axis = Line(DOWN * 2.5, UP * 2.5, color=GRAY)

        # Key states
        zero = Text("|0⟩", font_size=24, color=BLUE).shift(UP * 2.5)
        one = Text("|1⟩", font_size=24, color=RED).shift(DOWN * 2.5)
        plus = Text("|+⟩", font_size=20, color=GREEN).shift(RIGHT * 2.5)
        minus = Text("|−⟩", font_size=20, color=PURPLE).shift(LEFT * 2.5)

        self.play(
            Create(sphere),
            Create(x_axis),
            Create(y_axis),
            Write(zero),
            Write(one),
            Write(plus),
            Write(minus),
            run_time=0.8,
        )

        # State vector
        vector = Arrow(ORIGIN, UP * 2, color=YELLOW, buff=0, stroke_width=6)
        psi_label = Text("|ψ⟩", font_size=20, color=YELLOW).shift(UP * 2 + RIGHT * 0.4)

        self.play(Create(vector), Write(psi_label), run_time=0.5)

        # Show superposition state
        super_eq = Text("|ψ⟩ = 1/√2(|0⟩ + |1⟩)", font_size=22, color=GREEN)
        super_eq.shift(DOWN * 3)

        # Rotate to superposition
        self.play(
            Rotate(vector, PI / 4, about_point=ORIGIN),
            psi_label.animate.shift(RIGHT * 1.4 + DOWN * 1.4),
            Write(super_eq),
            run_time=1,
        )

        # Measurement visualization
        measure_box = Rectangle(width=3, height=0.8, color=RED, fill_opacity=0.3)
        measure_box.shift(RIGHT * 4)
        measure_text = Text("MEASURE", font_size=20, color=RED).move_to(measure_box)

        self.play(Create(measure_box), Write(measure_text), run_time=0.5)

        # Show possible outcomes
        outcome_0 = Circle(radius=0.3, color=BLUE, fill_opacity=0.8).shift(
            RIGHT * 4 + UP * 1.5
        )
        outcome_0_text = Text("0", font_size=16).move_to(outcome_0)
        outcome_0_prob = Text("50%", font_size=14).next_to(outcome_0, RIGHT)

        outcome_1 = Circle(radius=0.3, color=RED, fill_opacity=0.8).shift(
            RIGHT * 4 + DOWN * 1.5
        )
        outcome_1_text = Text("1", font_size=16).move_to(outcome_1)
        outcome_1_prob = Text("50%", font_size=14).next_to(outcome_1, RIGHT)

        self.play(
            Create(outcome_0),
            Write(outcome_0_text),
            Write(outcome_0_prob),
            Create(outcome_1),
            Write(outcome_1_text),
            Write(outcome_1_prob),
            run_time=0.5,
        )

        # Collapse animation
        collapse_flash = Flash(vector, color=RED, flash_radius=1)

        self.play(
            collapse_flash,
            vector.animate.put_start_and_end_on(ORIGIN, DOWN * 2),
            psi_label.animate.move_to(DOWN * 2 + RIGHT * 0.4),
            super_eq.animate.become(
                Text("|ψ⟩ = |1⟩", font_size=22, color=RED).shift(DOWN * 3)
            ),
            outcome_1.animate.scale(1.5),
            outcome_1_prob.animate.become(
                Text("100%", font_size=14, color=RED).next_to(outcome_1, RIGHT)
            ),
            outcome_0.animate.fade_out(),
            outcome_0_text.animate.fade_out(),
            outcome_0_prob.animate.fade_out(),
            run_time=0.8,
        )

        collapse_text = Text("Wavefunction Collapsed!", font_size=24, color=YELLOW)
        collapse_text.shift(DOWN * 3.8)
        self.play(Write(collapse_text), run_time=0.5)

        self.wait(0.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.3)

    def entanglement_correlation(self):
        title = Text("Quantum Entanglement", font_size=36, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Two qubits
        alice_sphere = Circle(radius=1, color=BLUE, stroke_width=4).shift(LEFT * 3.5)
        alice_vector = Arrow(
            alice_sphere.get_center(),
            alice_sphere.get_center() + UP,
            color=YELLOW,
            buff=0,
            stroke_width=4,
        )
        alice_label = Text("Alice", font_size=22, color=BLUE).next_to(
            alice_sphere, DOWN
        )

        bob_sphere = Circle(radius=1, color=RED, stroke_width=4).shift(RIGHT * 3.5)
        bob_vector = Arrow(
            bob_sphere.get_center(),
            bob_sphere.get_center() + UP,
            color=YELLOW,
            buff=0,
            stroke_width=4,
        )
        bob_label = Text("Bob", font_size=22, color=RED).next_to(bob_sphere, DOWN)

        self.play(
            Create(alice_sphere),
            Create(bob_sphere),
            Create(alice_vector),
            Create(bob_vector),
            Write(alice_label),
            Write(bob_label),
            run_time=0.5,
        )

        # Initial state
        state_text = Text("Initial: |00⟩", font_size=24).shift(UP * 2)
        self.play(Write(state_text), run_time=0.3)

        # Create entanglement
        entangle_arrow = CurvedArrow(
            alice_sphere.get_top(), bob_sphere.get_top(), color=PURPLE, angle=TAU / 4
        )

        bell_state = Text("Bell State: (|00⟩ + |11⟩)/√2", font_size=24, color=PURPLE)
        bell_state.shift(UP * 2)

        # Entanglement visualization
        connection = DashedLine(
            alice_sphere.get_center(),
            bob_sphere.get_center(),
            color=PURPLE,
            stroke_width=6,
            dash_length=0.15,
        )

        self.play(
            Create(entangle_arrow), Transform(state_text, bell_state), run_time=0.5
        )

        self.play(
            FadeOut(entangle_arrow),
            Create(connection),
            alice_sphere.animate.set_stroke(PURPLE, width=6),
            bob_sphere.animate.set_stroke(PURPLE, width=6),
            run_time=0.5,
        )

        # Pulsing effect
        pulse_ring = Circle(radius=1, color=PURPLE, stroke_width=8)
        pulse_ring.move_to(alice_sphere.get_center())

        self.play(pulse_ring.animate.scale(4).fade_out(), run_time=0.8)

        # Measurement
        measure_text = Text("Measuring Alice...", font_size=20, color=GREEN)
        measure_text.shift(DOWN * 2.5)
        self.play(Write(measure_text), run_time=0.3)

        # Alice measurement
        self.play(
            Flash(alice_sphere, color=GREEN),
            alice_vector.animate.put_start_and_end_on(
                alice_sphere.get_center(), alice_sphere.get_center() + DOWN
            ),
            run_time=0.5,
        )

        alice_result = Text("|1⟩", font_size=20, color=BLUE)
        alice_result.next_to(alice_sphere, UP)
        self.play(Write(alice_result), run_time=0.3)

        # Instant correlation
        correlation_wave = Circle(radius=0.1, color=PURPLE, stroke_width=6)
        correlation_wave.move_to(alice_sphere.get_center())

        bob_instant = Text("Bob instantly becomes |1⟩!", font_size=20, color=RED)
        bob_instant.move_to(measure_text)

        self.play(
            Transform(measure_text, bob_instant),
            correlation_wave.animate.move_to(bob_sphere.get_center())
            .scale(15)
            .fade_out(),
            run_time=0.6,
        )

        self.play(
            bob_vector.animate.put_start_and_end_on(
                bob_sphere.get_center(), bob_sphere.get_center() + DOWN
            ),
            run_time=0.3,
        )

        bob_result = Text("|1⟩", font_size=20, color=RED)
        bob_result.next_to(bob_sphere, UP)
        self.play(Write(bob_result), run_time=0.3)

        # Spooky action
        spooky = Text(
            '"Spooky Action at a Distance"', font_size=26, color=PURPLE, slant=ITALIC
        )
        spooky.shift(DOWN * 3.5)
        einstein = Text("- Einstein", font_size=18, color=GRAY)
        einstein.next_to(spooky, DOWN)

        self.play(
            Write(spooky),
            Write(einstein),
            connection.animate.set_stroke_width(10),
            run_time=0.8,
        )

        self.wait(0.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.3)

    def quantum_interference(self):
        title = Text("Quantum Interference", font_size=36)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Double slit visualization
        barrier = Rectangle(width=0.3, height=4, color=GRAY, fill_opacity=1)

        # Slits
        slit1 = Rectangle(width=0.3, height=0.5, color=BLACK, fill_opacity=1).shift(UP)
        slit2 = Rectangle(width=0.3, height=0.5, color=BLACK, fill_opacity=1).shift(
            DOWN
        )

        self.play(Create(barrier), Create(slit1), Create(slit2), run_time=0.5)

        # Quantum paths
        source = Dot(LEFT * 3, color=YELLOW, radius=0.1)

        path1 = CurvedArrow(
            source.get_center(), RIGHT * 3 + UP * 0.5, color=GREEN, angle=TAU / 8
        )
        path1_label = Text("Path 1", font_size=14, color=GREEN)
        path1_label.next_to(path1, UP)

        path2 = CurvedArrow(
            source.get_center(), RIGHT * 3 + UP * 0.5, color=RED, angle=-TAU / 8
        )
        path2_label = Text("Path 2", font_size=14, color=RED)
        path2_label.next_to(path2, DOWN)

        self.play(
            Create(source),
            Create(path1),
            Write(path1_label),
            Create(path2),
            Write(path2_label),
            run_time=0.8,
        )

        # Interference pattern
        screen = Line(UP * 2, DOWN * 2, color=WHITE).shift(RIGHT * 3)
        self.play(Create(screen), run_time=0.3)

        # Create interference pattern
        pattern_points = []
        for i in range(9):
            y = -1.6 + i * 0.4
            intensity = np.cos(i * PI / 2) ** 2
            dot = Dot(RIGHT * 3 + UP * y, color=YELLOW, radius=0.05 + 0.1 * intensity)
            dot.set_opacity(intensity)
            pattern_points.append(dot)

        pattern_group = VGroup(*pattern_points)

        self.play(FadeIn(pattern_group), run_time=0.5)

        # Labels
        constructive = Text("Constructive", font_size=16, color=YELLOW)
        constructive.next_to(pattern_points[4], RIGHT)

        destructive = Text("Destructive", font_size=16, color=GRAY)
        destructive.next_to(pattern_points[2], RIGHT)

        self.play(Write(constructive), Write(destructive), run_time=0.5)

        # Key insight
        insight = Text(
            "Quantum particles interfere with themselves!", font_size=20, color=GREEN
        )
        insight.shift(DOWN * 3)
        self.play(Write(insight), run_time=0.5)

        self.wait(0.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.3)

    def quantum_advantage_simple(self):
        title = Text("Quantum Advantage", font_size=36, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.3)

        # Visual comparison
        classical_label = Text("Classical Computer", font_size=24, color=RED)
        classical_label.shift(LEFT * 3.5 + UP * 2)

        quantum_label = Text("Quantum Computer", font_size=24, color=BLUE)
        quantum_label.shift(RIGHT * 3.5 + UP * 2)

        self.play(Write(classical_label), Write(quantum_label), run_time=0.5)

        # Show scaling
        # Classical bits
        classical_bits = VGroup()
        for i in range(4):
            bit = Square(side_length=0.4, color=RED, fill_opacity=0.5)
            bit.shift(LEFT * 3.5 + LEFT * 0.6 * (i - 1.5))
            bit_text = Text(str(i % 2), font_size=14).move_to(bit)
            classical_bits.add(VGroup(bit, bit_text))

        classical_states = Text("4 bits = 1 state at a time", font_size=16, color=RED)
        classical_states.next_to(classical_bits, DOWN)

        # Quantum qubits
        quantum_qubits = VGroup()
        for i in range(4):
            qubit = Circle(radius=0.25, color=BLUE, fill_opacity=0.5)
            qubit.shift(RIGHT * 3.5 + RIGHT * 0.6 * (i - 1.5))
            qubit_text = Text("|ψ⟩", font_size=10).move_to(qubit)
            quantum_qubits.add(VGroup(qubit, qubit_text))

        quantum_states = Text("4 qubits = 16 states at once!", font_size=16, color=BLUE)
        quantum_states.next_to(quantum_qubits, DOWN)

        self.play(
            Create(classical_bits),
            Create(quantum_qubits),
            Write(classical_states),
            Write(quantum_states),
            run_time=0.8,
        )

        # Exponential scaling visualization
        scale_text = Text("Quantum Power = 2ⁿ", font_size=28, color=GREEN)
        scale_text.shift(DOWN * 1)

        examples = (
            VGroup(
                Text("10 qubits = 1,024 states", font_size=18),
                Text("20 qubits = 1,048,576 states", font_size=18),
                Text(
                    "50 qubits = 1,125,899,906,842,624 states!",
                    font_size=18,
                    color=YELLOW,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .shift(DOWN * 2)
        )

        self.play(Write(scale_text), run_time=0.5)
        self.play(Write(examples), run_time=0.8)

        # Applications
        apps_title = Text("Real World Applications:", font_size=22, color=GREEN)
        apps_title.shift(DOWN * 3.5)

        self.play(Write(apps_title), run_time=0.3)

        # Quick end
        self.wait(0.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.3)

        end_text = Text("Quantum Computing", font_size=48, color=BLUE)
        tagline = Text("The Future is Now", font_size=24, color=PURPLE)
        tagline.next_to(end_text, DOWN)

        self.play(Write(end_text), FadeIn(tagline), run_time=0.5)
        self.wait(0.5)
