import numpy as np
from manim import *


class QuantumDemoImproved(Scene):
    def construct(self):
        # Quick title (0.5 seconds total)
        title = Text("Quantum Computing", font_size=56, color=BLUE)
        self.add(title)
        self.wait(0.3)
        self.play(FadeOut(title), run_time=0.2)

        # 1. Qubit vs Bit - Enhanced with probability visualization
        self.qubit_vs_bit_enhanced()

        # 2. Superposition - With live probability amplitudes
        self.superposition_enhanced()

        # 3. Entanglement - Show measurement correlation
        self.entanglement_enhanced()

        # 4. Quantum Gates & Interference
        self.quantum_gates_demo()

        # 5. Quantum Advantage visualization
        self.quantum_advantage()

    def qubit_vs_bit_enhanced(self):
        # Quick title
        title = Text("Qubit vs Classical Bit", font_size=36)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.5)

        # Classical bit - discrete states
        bit_group = VGroup()
        bit_0 = Circle(radius=0.6, color=WHITE, fill_opacity=0.8).shift(
            LEFT * 4 + UP * 0.5
        )
        bit_0_text = Text("0", font_size=30, color=BLACK).move_to(bit_0)
        bit_1 = Circle(radius=0.6, color=BLACK, fill_opacity=0.8).shift(
            LEFT * 4 + DOWN * 1.5
        )
        bit_1_text = Text("1", font_size=30, color=WHITE).move_to(bit_1)
        bit_label = Text("Classical Bit", font_size=20).shift(LEFT * 4 + DOWN * 2.5)

        # Show bit flipping
        self.play(
            Create(bit_0),
            Write(bit_0_text),
            Create(bit_1),
            Write(bit_1_text),
            Write(bit_label),
            run_time=0.8,
        )

        # Animate discrete switching
        for _ in range(2):
            self.play(
                bit_0.animate.set_fill(BLACK, 0.8),
                bit_0_text.animate.set_color(WHITE),
                bit_1.animate.set_fill(WHITE, 0.8),
                bit_1_text.animate.set_color(BLACK),
                run_time=0.3,
            )
            self.play(
                bit_0.animate.set_fill(WHITE, 0.8),
                bit_0_text.animate.set_color(BLACK),
                bit_1.animate.set_fill(BLACK, 0.8),
                bit_1_text.animate.set_color(WHITE),
                run_time=0.3,
            )

        # Qubit - continuous states with probability visualization
        qubit_group = VGroup()

        # Bloch sphere (2D projection)
        sphere = Circle(radius=1.2, color=BLUE, stroke_width=3).shift(RIGHT * 3)

        # Probability bars
        prob_0_bar = Rectangle(width=0.3, height=1.5, color=BLUE, fill_opacity=0.7)
        prob_0_bar.shift(RIGHT * 1.5 + UP * 0.5)
        prob_0_label = Text("|0⟩", font_size=16).next_to(prob_0_bar, DOWN)
        prob_0_value = DecimalNumber(0.5, num_decimal_places=2, font_size=14)
        prob_0_value.next_to(prob_0_bar, UP)

        prob_1_bar = Rectangle(width=0.3, height=1.5, color=RED, fill_opacity=0.7)
        prob_1_bar.shift(RIGHT * 4.5 + UP * 0.5)
        prob_1_label = Text("|1⟩", font_size=16).next_to(prob_1_bar, DOWN)
        prob_1_value = DecimalNumber(0.5, num_decimal_places=2, font_size=14)
        prob_1_value.next_to(prob_1_bar, UP)

        # State vector
        vector = Arrow(
            sphere.get_center(),
            sphere.get_center() + UP * 1.2,
            color=YELLOW,
            buff=0,
            stroke_width=4,
        )

        # Qubit equation
        qubit_eq = Text("α|0⟩ + β|1⟩", font_size=20).shift(RIGHT * 3 + DOWN * 2)
        qubit_label = Text("Quantum Qubit", font_size=20).shift(RIGHT * 3 + DOWN * 2.5)

        self.play(
            Create(sphere),
            Create(prob_0_bar),
            Create(prob_1_bar),
            Write(prob_0_label),
            Write(prob_1_label),
            Write(prob_0_value),
            Write(prob_1_value),
            Create(vector),
            Write(qubit_eq),
            Write(qubit_label),
            run_time=1,
        )

        # Animate continuous rotation with live probability updates
        def update_probs(mob, alpha):
            angle = alpha * 2 * PI
            # Update vector
            new_end = sphere.get_center() + np.array(
                [1.2 * np.sin(angle), 1.2 * np.cos(angle), 0]
            )
            vector.put_start_and_end_on(sphere.get_center(), new_end)

            # Update probabilities
            prob_0 = np.cos(angle / 2) ** 2
            prob_1 = np.sin(angle / 2) ** 2

            prob_0_bar.set_height(prob_0 * 2)
            prob_0_bar.move_to(RIGHT * 1.5 + UP * (prob_0 - 0.5))
            prob_0_value.set_value(prob_0)
            prob_0_value.next_to(prob_0_bar, UP)

            prob_1_bar.set_height(prob_1 * 2)
            prob_1_bar.move_to(RIGHT * 4.5 + UP * (prob_1 - 0.5))
            prob_1_value.set_value(prob_1)
            prob_1_value.next_to(prob_1_bar, UP)

        self.play(
            UpdateFromAlphaFunc(
                VGroup(vector, prob_0_bar, prob_1_bar, prob_0_value, prob_1_value),
                update_probs,
            ),
            run_time=3,
            rate_func=smooth,
        )

        # Highlight superposition
        super_text = Text("SUPERPOSITION", font_size=24, color=GREEN)
        super_text.shift(DOWN * 3.5)
        self.play(Write(super_text), run_time=0.5)

        self.wait(0.5)
        self.play(
            FadeOut(
                VGroup(
                    title,
                    bit_0,
                    bit_0_text,
                    bit_1,
                    bit_1_text,
                    bit_label,
                    sphere,
                    vector,
                    prob_0_bar,
                    prob_1_bar,
                    prob_0_label,
                    prob_1_label,
                    prob_0_value,
                    prob_1_value,
                    qubit_eq,
                    qubit_label,
                    super_text,
                )
            ),
            run_time=0.5,
        )

    def superposition_enhanced(self):
        # Title
        title = Text("Superposition & Measurement", font_size=36)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.5)

        # 3D Bloch sphere visualization
        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[-1.5, 1.5, 0.5],
            x_length=3,
            y_length=3,
            z_length=3,
        ).shift(LEFT * 3)

        # Sphere
        sphere = Circle(radius=1.2, color=BLUE_E, stroke_width=2).shift(LEFT * 3)

        # Key states
        zero = Text("|0⟩", font_size=20).shift(LEFT * 3 + UP * 1.5)
        one = Text("|1⟩", font_size=20).shift(LEFT * 3 + DOWN * 1.5)
        plus = Text("|+⟩", font_size=20).shift(LEFT * 1.5 + UP * 0.3)

        self.play(
            Create(axes),
            Create(sphere),
            Write(zero),
            Write(one),
            Write(plus),
            run_time=0.8,
        )

        # State vector with phase indicator
        vector = Arrow(
            LEFT * 3, LEFT * 3 + UP * 1.2, color=YELLOW, buff=0, stroke_width=5
        )
        phase_arc = Arc(radius=0.3, angle=0, color=GREEN).shift(LEFT * 3)

        # Probability distribution visualization
        prob_viz = VGroup()
        prob_title = Text("Measurement Probabilities", font_size=20).shift(
            RIGHT * 3 + UP * 2
        )

        # Create probability histogram
        prob_bars = VGroup()
        for i, (label, pos) in enumerate([("0", RIGHT * 2), ("1", RIGHT * 4)]):
            bar = Rectangle(
                width=0.8, height=0.01, color=BLUE if i == 0 else RED, fill_opacity=0.8
            ).shift(pos)
            bar_label = Text(f"|{label}⟩", font_size=16).next_to(bar, DOWN)
            bar_value = DecimalNumber(
                0, num_decimal_places=1, font_size=14, color=BLUE if i == 0 else RED
            )
            bar_value.next_to(bar, UP)
            prob_bars.add(VGroup(bar, bar_label, bar_value))

        self.play(
            Create(vector),
            Create(phase_arc),
            Write(prob_title),
            Create(prob_bars),
            run_time=0.8,
        )

        # Animate superposition with live updates
        def update_state(mob, alpha):
            t = alpha * 2 * PI
            # Rotate vector
            new_end = LEFT * 3 + np.array(
                [
                    1.2 * np.sin(t) * np.cos(0),
                    1.2 * np.sin(t) * np.sin(0),
                    1.2 * np.cos(t),
                ]
            )
            vector.put_start_and_end_on(LEFT * 3, new_end[:2] + [0])

            # Update phase arc
            phase_arc.become(Arc(radius=0.3, angle=t, color=GREEN).shift(LEFT * 3))

            # Update probabilities
            prob_0 = np.cos(t / 2) ** 2
            prob_1 = np.sin(t / 2) ** 2

            prob_bars[0][0].set_height(prob_0 * 2)
            prob_bars[0][0].move_to(RIGHT * 2 + UP * prob_0)
            prob_bars[0][2].set_value(prob_0 * 100)
            prob_bars[0][2].next_to(prob_bars[0][0], UP)

            prob_bars[1][0].set_height(prob_1 * 2)
            prob_bars[1][0].move_to(RIGHT * 4 + UP * prob_1)
            prob_bars[1][2].set_value(prob_1 * 100)
            prob_bars[1][2].next_to(prob_bars[1][0], UP)

        self.play(
            UpdateFromAlphaFunc(VGroup(vector, phase_arc, prob_bars), update_state),
            run_time=3,
            rate_func=smooth,
        )

        # Measurement collapse
        measure_text = Text("MEASURE!", font_size=28, color=RED).shift(DOWN * 2)
        self.play(
            Write(measure_text),
            Flash(vector, color=RED, flash_radius=0.5),
            run_time=0.5,
        )

        # Collapse animation
        self.play(
            vector.animate.put_start_and_end_on(LEFT * 3, LEFT * 3 + DOWN * 1.2),
            prob_bars[0][0].animate.set_height(0.01),
            prob_bars[0][2].animate.set_value(0),
            prob_bars[1][0].animate.set_height(2),
            prob_bars[1][2].animate.set_value(100),
            phase_arc.animate.fade_out(),
            run_time=0.8,
        )

        collapse_text = Text("Wavefunction Collapsed!", font_size=20, color=YELLOW)
        collapse_text.next_to(measure_text, DOWN)
        self.play(Write(collapse_text), run_time=0.5)

        self.wait(0.5)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=0.5)

    def entanglement_enhanced(self):
        # Title
        title = Text("Quantum Entanglement", font_size=36, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.5)

        # Two qubits with Bloch spheres
        q1_sphere = Circle(radius=0.8, color=BLUE, stroke_width=3).shift(LEFT * 4)
        q1_vector = Arrow(
            q1_sphere.get_center(),
            q1_sphere.get_center() + UP * 0.8,
            color=YELLOW,
            buff=0,
        )
        q1_label = Text("Alice", font_size=20).next_to(q1_sphere, DOWN)

        q2_sphere = Circle(radius=0.8, color=RED, stroke_width=3).shift(RIGHT * 4)
        q2_vector = Arrow(
            q2_sphere.get_center(),
            q2_sphere.get_center() + UP * 0.8,
            color=YELLOW,
            buff=0,
        )
        q2_label = Text("Bob", font_size=20).next_to(q2_sphere, DOWN)

        self.play(
            Create(q1_sphere),
            Create(q2_sphere),
            Create(q1_vector),
            Create(q2_vector),
            Write(q1_label),
            Write(q2_label),
            run_time=0.8,
        )

        # Initial state
        state_eq = Text("|ψ⟩ = |00⟩", font_size=24).shift(UP * 2)
        self.play(Write(state_eq), run_time=0.5)

        # Create entanglement
        entangle_text = Text("Creating Bell State", font_size=20, color=YELLOW)
        entangle_text.shift(DOWN * 2.5)
        self.play(Write(entangle_text), run_time=0.5)

        # Entanglement visualization - quantum circuit style
        h_gate = Square(side_length=0.4, color=GREEN).shift(LEFT * 2)
        h_label = Text("H", font_size=16).move_to(h_gate)

        cnot_control = Dot(LEFT * 0.5, color=BLUE)
        cnot_target = Circle(radius=0.2, color=RED).shift(RIGHT * 0.5)
        cnot_line = Line(cnot_control.get_center(), cnot_target.get_center())

        self.play(
            Create(h_gate),
            Write(h_label),
            Create(cnot_control),
            Create(cnot_target),
            Create(cnot_line),
            run_time=0.8,
        )

        # Transform to Bell state
        bell_state = Text("|Φ⁺⟩ = (|00⟩ + |11⟩)/√2", font_size=24, color=PURPLE)
        bell_state.move_to(state_eq)

        # Entanglement connection
        connection = DashedLine(
            q1_sphere.get_center(),
            q2_sphere.get_center(),
            color=PURPLE,
            stroke_width=4,
            dash_length=0.1,
        )

        # Create pulsing entanglement effect
        pulse_circles = VGroup()
        for i in range(3):
            circle = Circle(radius=0.1 + i * 0.3, color=PURPLE, stroke_width=2)
            circle.move_to(q1_sphere.get_center())
            circle.set_opacity(0)
            pulse_circles.add(circle)

        self.play(
            Transform(state_eq, bell_state),
            FadeOut(VGroup(h_gate, h_label, cnot_control, cnot_target, cnot_line)),
            Create(connection),
            FadeOut(entangle_text),
            run_time=0.8,
        )

        # Animate entanglement pulses
        self.play(
            *[
                circle.animate.move_to(q2_sphere.get_center()).set_opacity(0).scale(3)
                for circle in pulse_circles
            ],
            *[circle.animate.set_opacity(1).set_opacity(0) for circle in pulse_circles],
            lag_ratio=0.3,
            run_time=1.5,
        )

        # Measurement correlation demonstration
        measure_text = Text("Measuring Alice's qubit...", font_size=20, color=GREEN)
        measure_text.shift(DOWN * 2.5)
        self.play(Write(measure_text), run_time=0.5)

        # Alice's measurement
        self.play(
            Flash(q1_sphere, color=GREEN),
            q1_vector.animate.put_start_and_end_on(
                q1_sphere.get_center(), q1_sphere.get_center() + DOWN * 0.8
            ),
            run_time=0.5,
        )

        alice_result = Text("Result: |1⟩", font_size=18, color=BLUE)
        alice_result.next_to(q1_sphere, UP)
        self.play(Write(alice_result), run_time=0.3)

        # Instant correlation
        correlation_text = Text(
            "Bob's qubit instantly becomes |1⟩!", font_size=20, color=RED
        )
        correlation_text.move_to(measure_text)

        # Show correlation wave
        correlation_wave = Circle(radius=0.1, color=PURPLE, stroke_width=4)
        correlation_wave.move_to(q1_sphere.get_center())

        self.play(
            Transform(measure_text, correlation_text),
            correlation_wave.animate.move_to(q2_sphere.get_center())
            .scale(8)
            .fade_out(),
            run_time=0.8,
        )

        self.play(
            q2_vector.animate.put_start_and_end_on(
                q2_sphere.get_center(), q2_sphere.get_center() + DOWN * 0.8
            ),
            q2_sphere.animate.set_stroke(PURPLE, width=6),
            run_time=0.5,
        )

        bob_result = Text("Result: |1⟩", font_size=18, color=RED)
        bob_result.next_to(q2_sphere, UP)
        self.play(Write(bob_result), run_time=0.3)

        # Spooky action text
        spooky_text = Text(
            '"Spooky Action at a Distance"', font_size=24, color=PURPLE, style=ITALIC
        )
        spooky_text.shift(DOWN * 3.5)
        self.play(Write(spooky_text), run_time=0.8)

        self.wait(0.8)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=0.5)

    def quantum_gates_demo(self):
        # Title
        title = Text("Quantum Gates & Interference", font_size=36)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.5)

        # Show key gates with matrix and effect
        gates_data = [
            ("X", "[[0,1],[1,0]]", "Bit Flip", BLUE),
            ("H", "1/√2[[1,1],[1,-1]]", "Superposition", GREEN),
            ("Z", "[[1,0],[0,-1]]", "Phase Flip", RED),
        ]

        gate_visuals = VGroup()
        for i, (name, matrix, desc, color) in enumerate(gates_data):
            # Gate symbol
            gate_box = Square(side_length=0.8, color=color, fill_opacity=0.3)
            gate_box.shift(LEFT * 4 + RIGHT * i * 3 + UP)
            gate_name = Text(name, font_size=28, color=color).move_to(gate_box)
            gate_desc = Text(desc, font_size=14).next_to(gate_box, DOWN)

            # Mini Bloch sphere showing effect
            mini_sphere = Circle(radius=0.4, color=GRAY, stroke_width=1)
            mini_sphere.next_to(gate_box, DOWN, buff=0.8)

            gate_visual = VGroup(gate_box, gate_name, gate_desc, mini_sphere)
            gate_visuals.add(gate_visual)

        self.play(Create(gate_visuals), run_time=1)

        # Quantum interference visualization
        interference_title = Text("Quantum Interference", font_size=24, color=YELLOW)
        interference_title.shift(DOWN * 1.5)
        self.play(Write(interference_title), run_time=0.5)

        # Show interference pattern
        path1 = CurvedArrow(LEFT * 3, RIGHT * 3, color=GREEN, angle=TAU / 6).shift(
            DOWN * 2.5
        )
        path2 = CurvedArrow(LEFT * 3, RIGHT * 3, color=RED, angle=-TAU / 6).shift(
            DOWN * 2.5
        )

        path1_label = Text("Path 1", font_size=12, color=GREEN)
        path1_label.next_to(path1, UP)
        path2_label = Text("Path 2", font_size=12, color=RED)
        path2_label.next_to(path2, DOWN)

        self.play(
            Create(path1),
            Create(path2),
            Write(path1_label),
            Write(path2_label),
            run_time=0.8,
        )

        # Show constructive/destructive interference
        result_constructive = Dot(RIGHT * 3 + UP * 0.5, color=YELLOW, radius=0.15)
        result_destructive = Dot(RIGHT * 3 + DOWN * 0.5, color=GRAY, radius=0.15)

        const_label = Text("Constructive", font_size=12, color=YELLOW)
        const_label.next_to(result_constructive, RIGHT)
        dest_label = Text("Destructive", font_size=12, color=GRAY)
        dest_label.next_to(result_destructive, RIGHT)

        self.play(
            FadeIn(result_constructive),
            Write(const_label),
            FadeIn(result_destructive),
            Write(dest_label),
            run_time=0.8,
        )

        self.wait(0.5)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=0.5)

    def quantum_advantage(self):
        # Title
        title = Text("Quantum Advantage", font_size=36, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=0.5)

        # Classical vs Quantum scaling
        axes = Axes(
            x_range=[0, 20, 5],
            y_range=[0, 100, 20],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": True, "font_size": 16},
        ).shift(DOWN * 0.5)

        x_label = Text("Qubits", font_size=20).next_to(axes.x_axis, DOWN)
        y_label = Text("Power", font_size=20).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.8)

        # Exponential quantum curve
        quantum_curve = axes.plot(lambda x: 2 ** (x / 2), x_range=[0, 20], color=BLUE)
        quantum_label = Text("Quantum: 2ⁿ", font_size=20, color=BLUE)
        quantum_label.next_to(quantum_curve.point_from_proportion(0.7), UR)

        # Linear classical curve
        classical_curve = axes.plot(lambda x: 5 * x, x_range=[0, 20], color=RED)
        classical_label = Text("Classical: n", font_size=20, color=RED)
        classical_label.next_to(classical_curve.point_from_proportion(0.8), DOWN)

        self.play(
            Create(quantum_curve),
            Write(quantum_label),
            Create(classical_curve),
            Write(classical_label),
            run_time=1.2,
        )

        # Highlight exponential advantage
        advantage_point = Dot(axes.coords_to_point(10, 32), color=YELLOW, radius=0.1)
        advantage_text = Text("10 qubits = 1024 states!", font_size=18, color=YELLOW)
        advantage_text.next_to(advantage_point, UR)

        self.play(
            FadeIn(advantage_point),
            Write(advantage_text),
            Flash(advantage_point, color=YELLOW),
            run_time=0.8,
        )

        # Applications
        apps_text = Text("Quantum Applications:", font_size=24, color=GREEN)
        apps_text.shift(DOWN * 3)

        apps_list = Text(
            "Cryptography • Drug Discovery • Optimization • Machine Learning",
            font_size=18,
            color=GRAY,
        ).next_to(apps_text, DOWN)

        self.play(Write(apps_text), run_time=0.5)
        self.play(Write(apps_list), run_time=0.8)

        # Final message
        self.wait(1)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=0.5)

        end_text = Text("Quantum Computing", font_size=48, color=BLUE)
        end_subtitle = Text("The Future is Quantum", font_size=24, color=PURPLE)
        end_subtitle.next_to(end_text, DOWN)

        self.play(Write(end_text), FadeIn(end_subtitle), run_time=0.8)
        self.wait(1)
