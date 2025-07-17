import numpy as np
from manim import *


class QuantumDemoClean(Scene):
    def construct(self):
        # Ultra-fast title (0.3s total)
        title = Text("Quantum Computing", font_size=56, color=BLUE)
        self.add(title)
        self.wait(0.1)
        self.play(FadeOut(title), run_time=0.2)

        # Store title reference
        self.title = Text("", font_size=40)
        self.title.to_edge(UP)
        self.add(self.title)

        # Core concepts with improved timing
        self.qubit_superposition()  # 4s
        self.measurement_collapse()  # 3s
        self.entanglement_demo()  # 4s
        self.quantum_gates()  # 3s
        self.quantum_advantage()  # 3s

        # Quick end (0.5s)
        self.play(FadeOut(self.title), run_time=0.2)
        end = Text("Quantum Computing", font_size=48, color=BLUE)
        self.play(Write(end), run_time=0.3)
        self.wait(0.2)

    def qubit_superposition(self):
        # Title (0.3s)
        self.title.become(Text("Superposition", font_size=40, color=GREEN))
        self.title.to_edge(UP)

        # Classical bit vs Qubit comparison
        # Classical
        classical_group = VGroup()
        classical_label = Text("Classical Bit:", font_size=20).shift(LEFT * 4 + UP * 2)
        bit_0 = Circle(0.6, color=WHITE, fill_opacity=1).shift(LEFT * 4.5 + UP * 0.5)
        bit_0_text = Text("0", font_size=24, color=BLACK).move_to(bit_0)
        bit_1 = Circle(0.6, color=BLACK, fill_opacity=1).shift(LEFT * 3.5 + UP * 0.5)
        bit_1_text = Text("1", font_size=24, color=WHITE).move_to(bit_1)
        classical_group.add(classical_label, bit_0, bit_0_text, bit_1, bit_1_text)

        # Quantum
        quantum_group = VGroup()
        quantum_label = Text("Quantum Qubit:", font_size=20).shift(RIGHT * 2 + UP * 2)

        # Bloch sphere visualization
        sphere = Circle(1.2, color=BLUE, stroke_width=3).shift(RIGHT * 2 + UP * 0.5)

        # State vector that will rotate
        vector = Arrow(
            sphere.get_center(),
            sphere.get_center() + np.array([0, 1.2, 0]),
            color=YELLOW,
            buff=0,
            stroke_width=5,
        )

        # Probability visualization
        prob_bg = Rectangle(width=3, height=0.8, color=GRAY, fill_opacity=0.2)
        prob_bg.shift(RIGHT * 2 + DOWN * 1.5)

        prob_0_bar = Rectangle(width=1.4, height=0.6, color=BLUE, fill_opacity=0.8)
        prob_0_bar.shift(RIGHT * 1.3 + DOWN * 1.5)
        prob_0_text = Text("|0⟩: 100%", font_size=14).shift(RIGHT * 1.3 + DOWN * 2.2)

        prob_1_bar = Rectangle(width=0.01, height=0.6, color=RED, fill_opacity=0.8)
        prob_1_bar.shift(RIGHT * 2.7 + DOWN * 1.5)
        prob_1_text = Text("|1⟩: 0%", font_size=14).shift(RIGHT * 2.7 + DOWN * 2.2)

        quantum_group.add(
            quantum_label,
            sphere,
            vector,
            prob_bg,
            prob_0_bar,
            prob_1_bar,
            prob_0_text,
            prob_1_text,
        )

        # State equation
        state_eq = Text("α|0⟩ + β|1⟩", font_size=22, color=BLUE)
        state_eq.shift(DOWN * 3)

        # Show both (0.8s)
        self.play(
            Create(classical_group),
            Create(quantum_group),
            Write(state_eq),
            run_time=0.8,
        )

        # Animate qubit rotation with probability updates (2s)
        def update_quantum_state(mob, alpha):
            angle = alpha * PI
            # Update vector
            new_end = sphere.get_center() + 1.2 * np.array(
                [np.sin(angle), np.cos(angle), 0]
            )
            vector.put_start_and_end_on(sphere.get_center(), new_end)

            # Update probabilities
            p0 = np.cos(angle / 2) ** 2
            p1 = np.sin(angle / 2) ** 2

            prob_0_bar.set_width(max(0.01, p0 * 1.4))
            prob_0_bar.move_to(RIGHT * (0.6 + p0 * 0.7) + DOWN * 1.5)
            prob_0_text.become(
                Text(f"|0⟩: {int(p0*100)}%", font_size=14).shift(
                    RIGHT * 1.3 + DOWN * 2.2
                )
            )

            prob_1_bar.set_width(max(0.01, p1 * 1.4))
            prob_1_bar.move_to(RIGHT * (2 + p1 * 0.7) + DOWN * 1.5)
            prob_1_text.become(
                Text(f"|1⟩: {int(p1*100)}%", font_size=14).shift(
                    RIGHT * 2.7 + DOWN * 2.2
                )
            )

        self.play(
            UpdateFromAlphaFunc(
                VGroup(vector, prob_0_bar, prob_1_bar, prob_0_text, prob_1_text),
                update_quantum_state,
            ),
            run_time=2,
        )

        # Highlight superposition
        superposition_text = Text("SUPERPOSITION!", font_size=28, color=GREEN)
        superposition_text.shift(DOWN * 3.5)
        self.play(Write(superposition_text), Flash(sphere, color=GREEN), run_time=0.5)

        self.wait(0.2)
        self.play(
            FadeOut(
                Group(classical_group, quantum_group, state_eq, superposition_text)
            ),
            run_time=0.3,
        )

    def measurement_collapse(self):
        # Title
        self.title.become(Text("Measurement", font_size=40, color=RED))

        # Bloch sphere with superposition state
        sphere = Circle(2, color=BLUE_E, stroke_width=2)
        x_axis = Line(LEFT * 2.5, RIGHT * 2.5, color=GRAY, stroke_width=1)
        y_axis = Line(DOWN * 2.5, UP * 2.5, color=GRAY, stroke_width=1)

        # State labels
        zero = Text("|0⟩", font_size=20, color=BLUE).shift(UP * 2.3)
        one = Text("|1⟩", font_size=20, color=RED).shift(DOWN * 2.3)
        plus = Text("|+⟩", font_size=16, color=GREEN).shift(RIGHT * 2.3)

        # Superposition state vector
        vector = Arrow(ORIGIN, UR * 1.4, color=YELLOW, buff=0, stroke_width=6)
        psi = Text("|ψ⟩", font_size=18, color=YELLOW).shift(UR * 1.7)

        # Equation
        eq = Text("|ψ⟩ = 1/√2(|0⟩ + |1⟩)", font_size=20)
        eq.shift(DOWN * 3)

        self.play(
            Create(VGroup(sphere, x_axis, y_axis)),
            Write(VGroup(zero, one, plus)),
            Create(vector),
            Write(psi),
            Write(eq),
            run_time=0.8,
        )

        # Measurement device
        measure_box = Rectangle(width=2, height=1, color=RED, fill_opacity=0.3)
        measure_box.shift(RIGHT * 4)
        measure_text = Text("MEASURE", font_size=18, color=RED).move_to(measure_box)

        self.play(Create(measure_box), Write(measure_text), run_time=0.3)

        # Show measurement outcomes with probabilities
        outcome_group = VGroup()
        for i, (state, pos, prob) in enumerate(
            [("|0⟩", UP * 0.8, "50%"), ("|1⟩", DOWN * 0.8, "50%")]
        ):
            outcome = Circle(0.3, color=BLUE if i == 0 else RED, fill_opacity=0.7)
            outcome.shift(RIGHT * 4 + pos)
            outcome_text = Text(state, font_size=14).move_to(outcome)
            prob_text = Text(prob, font_size=12).next_to(outcome, RIGHT)
            outcome_group.add(outcome, outcome_text, prob_text)

        self.play(Create(outcome_group), run_time=0.5)

        # Collapse animation
        self.play(Flash(vector, color=RED, flash_radius=1), run_time=0.3)

        # Collapse to |1⟩
        collapsed_eq = Text("|ψ⟩ = |1⟩", font_size=20, color=RED).shift(DOWN * 3)

        self.play(
            vector.animate.put_start_and_end_on(ORIGIN, DOWN * 2),
            psi.animate.shift(DOWN * 3.4 + LEFT * 1.4),
            Transform(eq, collapsed_eq),
            outcome_group[3].animate.scale(1.5),  # Highlight |1⟩ outcome
            outcome_group[5].animate.become(
                Text("100%", font_size=12, color=RED).next_to(outcome_group[3], RIGHT)
            ),
            FadeOut(outcome_group[0:3]),  # Fade |0⟩ outcome
            run_time=0.8,
        )

        # Collapse text
        collapse_text = Text("Wavefunction Collapsed!", font_size=22, color=YELLOW)
        collapse_text.shift(DOWN * 3.5)
        self.play(Write(collapse_text), run_time=0.3)

        self.wait(0.3)
        self.play(
            FadeOut(
                Group(
                    sphere,
                    x_axis,
                    y_axis,
                    zero,
                    one,
                    plus,
                    vector,
                    psi,
                    eq,
                    measure_box,
                    measure_text,
                    outcome_group,
                    collapse_text,
                )
            ),
            run_time=0.3,
        )

    def entanglement_demo(self):
        # Title
        self.title.become(Text("Entanglement", font_size=40, color=PURPLE))

        # Two qubits
        alice = Circle(0.8, color=BLUE, stroke_width=4).shift(LEFT * 3)
        alice_v = Arrow(
            alice.get_center(),
            alice.get_center() + UP * 0.8,
            color=YELLOW,
            buff=0,
            stroke_width=3,
        )
        alice_label = Text("Alice", font_size=18).next_to(alice, DOWN)

        bob = Circle(0.8, color=RED, stroke_width=4).shift(RIGHT * 3)
        bob_v = Arrow(
            bob.get_center(),
            bob.get_center() + UP * 0.8,
            color=YELLOW,
            buff=0,
            stroke_width=3,
        )
        bob_label = Text("Bob", font_size=18).next_to(bob, DOWN)

        # Initial state
        state = Text("|ψ⟩ = |00⟩", font_size=20).shift(UP * 2)

        self.play(
            Create(VGroup(alice, bob, alice_v, bob_v)),
            Write(VGroup(alice_label, bob_label, state)),
            run_time=0.5,
        )

        # Create entanglement
        # Show operations
        h_gate = Square(0.4, color=GREEN).shift(LEFT * 1.5)
        h_text = Text("H", font_size=16).move_to(h_gate)
        cnot = VGroup(
            Dot(LEFT * 0.5, color=BLUE, radius=0.08),
            Circle(0.15, color=RED).shift(RIGHT * 0.5),
            Line(LEFT * 0.5, RIGHT * 0.5, stroke_width=2),
        )

        self.play(Create(VGroup(h_gate, h_text, cnot)), run_time=0.5)

        # Transform to Bell state
        bell_state = Text("|Φ⁺⟩ = (|00⟩ + |11⟩)/√2", font_size=20, color=PURPLE)
        bell_state.shift(UP * 2)

        # Entanglement visualization
        connection = DashedLine(
            alice.get_center(),
            bob.get_center(),
            color=PURPLE,
            stroke_width=6,
            dash_length=0.1,
        )

        self.play(
            Transform(state, bell_state),
            FadeOut(VGroup(h_gate, h_text, cnot)),
            Create(connection),
            alice.animate.set_stroke(PURPLE, 6),
            bob.animate.set_stroke(PURPLE, 6),
            run_time=0.8,
        )

        # Pulse effect
        pulse = Circle(0.8, color=PURPLE, stroke_width=8).move_to(alice)
        self.play(pulse.animate.scale(5).fade_out(), run_time=0.6)

        # Measurement
        measure_text = Text("Measuring Alice...", font_size=18, color=GREEN)
        measure_text.shift(DOWN * 2)
        self.play(Write(measure_text), run_time=0.3)

        # Alice collapses
        self.play(
            Flash(alice, color=GREEN),
            alice_v.animate.put_start_and_end_on(
                alice.get_center(), alice.get_center() + DOWN * 0.8
            ),
            run_time=0.4,
        )

        alice_result = Text("|1⟩", font_size=16, color=BLUE).next_to(alice, UP)
        self.play(Write(alice_result), run_time=0.2)

        # Instant correlation
        wave = Circle(0.1, color=PURPLE, stroke_width=6).move_to(alice)
        bob_text = Text("Bob instantly |1⟩!", font_size=18, color=RED)
        bob_text.move_to(measure_text)

        self.play(
            Transform(measure_text, bob_text),
            wave.animate.move_to(bob).scale(20).fade_out(),
            run_time=0.5,
        )

        self.play(
            bob_v.animate.put_start_and_end_on(
                bob.get_center(), bob.get_center() + DOWN * 0.8
            ),
            run_time=0.3,
        )

        bob_result = Text("|1⟩", font_size=16, color=RED).next_to(bob, UP)
        self.play(Write(bob_result), run_time=0.2)

        # Spooky action
        spooky = Text(
            '"Spooky Action at a Distance"', font_size=22, color=PURPLE, slant=ITALIC
        )
        spooky.shift(DOWN * 3)
        self.play(Write(spooky), run_time=0.5)

        self.wait(0.5)
        self.play(
            FadeOut(
                Group(
                    alice,
                    bob,
                    alice_v,
                    bob_v,
                    alice_label,
                    bob_label,
                    state,
                    connection,
                    alice_result,
                    bob_result,
                    measure_text,
                    spooky,
                    pulse,
                )
            ),
            run_time=0.3,
        )

    def quantum_gates(self):
        # Title
        self.title.become(Text("Quantum Gates", font_size=40, color=YELLOW))

        # Show key gates with visual effects
        gates = VGroup()

        # X gate (bit flip)
        x_gate = Square(0.8, color=BLUE, fill_opacity=0.3).shift(LEFT * 4)
        x_text = Text("X", font_size=28, color=BLUE).move_to(x_gate)
        x_desc = Text("Bit Flip", font_size=14).next_to(x_gate, DOWN)

        # Show effect
        x_before = Arrow(
            LEFT * 4 + LEFT * 1.5,
            LEFT * 4 + LEFT * 1.5 + UP * 0.6,
            color=YELLOW,
            buff=0,
        )
        x_after = Arrow(
            LEFT * 4 + RIGHT * 1.5,
            LEFT * 4 + RIGHT * 1.5 + DOWN * 0.6,
            color=YELLOW,
            buff=0,
        )

        # H gate (superposition)
        h_gate = Square(0.8, color=GREEN, fill_opacity=0.3)
        h_text = Text("H", font_size=28, color=GREEN).move_to(h_gate)
        h_desc = Text("Superposition", font_size=14).next_to(h_gate, DOWN)

        # Show effect
        h_before = Arrow(LEFT * 1.5, LEFT * 1.5 + UP * 0.6, color=YELLOW, buff=0)
        h_after = VGroup(
            Arrow(RIGHT * 1.5, RIGHT * 1.5 + UR * 0.6, color=YELLOW, buff=0),
            DashedLine(RIGHT * 1.5, RIGHT * 1.5 + DR * 0.6, color=YELLOW),
        )

        # Z gate (phase flip)
        z_gate = Square(0.8, color=RED, fill_opacity=0.3).shift(RIGHT * 4)
        z_text = Text("Z", font_size=28, color=RED).move_to(z_gate)
        z_desc = Text("Phase Flip", font_size=14).next_to(z_gate, DOWN)

        gates.add(
            x_gate,
            x_text,
            x_desc,
            x_before,
            x_after,
            h_gate,
            h_text,
            h_desc,
            h_before,
            h_after,
            z_gate,
            z_text,
            z_desc,
        )

        self.play(Create(gates), run_time=0.8)

        # Show quantum interference
        inter_title = Text("Quantum Interference", font_size=20, color=YELLOW)
        inter_title.shift(DOWN * 2.5)

        # Simple interference visualization
        path1 = CurvedArrow(LEFT * 3, RIGHT * 3, color=GREEN, angle=TAU / 8)
        path1.shift(DOWN * 3.2)
        path2 = CurvedArrow(LEFT * 3, RIGHT * 3, color=RED, angle=-TAU / 8)
        path2.shift(DOWN * 3.2)

        result = Dot(RIGHT * 3 + DOWN * 3.2, color=YELLOW, radius=0.15)

        self.play(
            Write(inter_title),
            Create(path1),
            Create(path2),
            FadeIn(result),
            run_time=0.8,
        )

        self.wait(0.5)
        self.play(
            FadeOut(Group(gates, inter_title, path1, path2, result)), run_time=0.3
        )

    def quantum_advantage(self):
        # Title
        self.title.become(Text("Quantum Advantage", font_size=40, color=GOLD))

        # Visual comparison
        classical = Text("Classical: n bits", font_size=20, color=RED).shift(
            LEFT * 3 + UP * 2
        )
        quantum = Text("Quantum: n qubits", font_size=20, color=BLUE).shift(
            RIGHT * 3 + UP * 2
        )

        self.play(Write(classical), Write(quantum), run_time=0.3)

        # Show exponential scaling
        examples = VGroup()
        for i, (n, classical_states, quantum_states) in enumerate(
            [(4, "1", "16"), (10, "1", "1,024"), (20, "1", "1,048,576")]
        ):
            example = VGroup(
                Text(f"{n}:", font_size=18).shift(LEFT * 4.5 + DOWN * (i * 0.8)),
                Text(f"{classical_states} state", font_size=16, color=RED).shift(
                    LEFT * 2 + DOWN * (i * 0.8)
                ),
                Text("vs", font_size=16).shift(DOWN * (i * 0.8)),
                Text(f"{quantum_states} states", font_size=16, color=BLUE).shift(
                    RIGHT * 2.5 + DOWN * (i * 0.8)
                ),
            )
            examples.add(example)

        self.play(Write(examples), run_time=0.8)

        # Exponential equation
        exp_eq = Text("Quantum Power = 2ⁿ", font_size=28, color=GREEN)
        exp_eq.shift(DOWN * 2.5)

        self.play(Write(exp_eq), Flash(exp_eq, color=GREEN), run_time=0.5)

        # Applications
        apps = Text(
            "Drug Discovery • Cryptography • AI • Optimization",
            font_size=18,
            color=GRAY,
        )
        apps.shift(DOWN * 3.5)

        self.play(Write(apps), run_time=0.5)

        self.wait(0.5)
