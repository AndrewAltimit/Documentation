from manim import *
import numpy as np

class QuantumDemoFinal(ThreeDScene):
    def construct(self):
        # Slightly slower intro (0.5s)
        title = Text("QUANTUM COMPUTING", font_size=52, color=BLUE, weight=BOLD)
        self.add(title)
        self.play(title.animate.scale(0.1).set_opacity(0), run_time=0.5)
        
        # Core demonstrations with better pacing
        self.superposition_3d()           # 4.5s
        self.measurement_visualization()  # 4s
        self.entanglement_spooky()       # 4.5s
        self.quantum_gates_circuit()     # 4s
        self.quantum_supremacy()         # 3s
        
        # Fixed outro without overlap (0.8s)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.3)
        
        end = VGroup(
            Text("QUANTUM", font_size=44, color=BLUE, weight=BOLD),
            Text("COMPUTING", font_size=44, color=PURPLE, weight=BOLD)
        ).arrange(DOWN, buff=0.3)
        
        self.play(
            Write(end[0]),
            Write(end[1]),
            run_time=0.5
        )
        self.wait(0.5)
    
    def superposition_3d(self):
        # 3D camera setup
        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        
        # Title (0.3s)
        title = Text("SUPERPOSITION", font_size=32, color=GREEN)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title), run_time=0.3)
        
        # Classical bit visualization
        classical_label = Text("Classical Bit", font_size=18)
        classical_label.shift(LEFT * 4 + UP * 2)
        self.add_fixed_in_frame_mobjects(classical_label)
        
        # Binary states with flip animation
        bit_state = VGroup(
            Square(side_length=0.8, color=WHITE, fill_opacity=1),
            Text("0", font_size=24, color=BLACK)
        )
        bit_state.shift(LEFT * 4)
        
        self.add_fixed_in_frame_mobjects(bit_state)
        self.play(Create(bit_state[0]), Write(bit_state[1]), run_time=0.4)
        
        # Slower flips to show discrete states
        for _ in range(3):
            self.play(
                bit_state[0].animate.set_fill(BLACK, 1),
                bit_state[1].animate.set_color(WHITE).become(
                    Text("1", font_size=24, color=WHITE).shift(LEFT * 4)
                ),
                run_time=0.2
            )
            self.play(
                bit_state[0].animate.set_fill(WHITE, 1),
                bit_state[1].animate.set_color(BLACK).become(
                    Text("0", font_size=24, color=BLACK).shift(LEFT * 4)
                ),
                run_time=0.2
            )
        
        # Quantum qubit - 3D Bloch sphere
        quantum_label = Text("Quantum Qubit", font_size=18)
        quantum_label.shift(RIGHT * 3 + UP * 2)
        self.add_fixed_in_frame_mobjects(quantum_label)
        
        # Create 3D sphere
        sphere = Sphere(radius=1.2, resolution=(20, 20))
        sphere.set_color(BLUE_E)
        sphere.set_opacity(0.3)
        sphere.shift(RIGHT * 3)
        
        # Coordinate axes
        x_axis = Arrow3D(start=[2, 0, 0], end=[4.5, 0, 0], color=RED)
        y_axis = Arrow3D(start=[3, -1.5, 0], end=[3, 1.5, 0], color=GREEN)
        z_axis = Arrow3D(start=[3, 0, -1.5], end=[3, 0, 1.5], color=BLUE)
        
        # State labels
        zero_ket = Text("|0⟩", font_size=16, color=BLUE)
        zero_ket.move_to([3, 0, 1.7])
        one_ket = Text("|1⟩", font_size=16, color=RED)
        one_ket.move_to([3, 0, -1.7])
        
        # Quantum state vector
        state_vector = Arrow3D(
            start=[3, 0, 0],
            end=[3, 0, 1.2],
            color=YELLOW,
            thickness=0.04
        )
        
        # Probability amplitudes (live updating) - positioned to avoid overlap
        alpha_text = Text("α = 1.00", font_size=14, color=BLUE)
        beta_text = Text("β = 0.00", font_size=14, color=RED)
        alpha_text.shift(RIGHT * 3 + DOWN * 2.5)
        beta_text.shift(RIGHT * 3 + DOWN * 2.9)
        self.add_fixed_in_frame_mobjects(alpha_text, beta_text)
        
        self.play(
            Create(sphere),
            Create(x_axis), Create(y_axis), Create(z_axis),
            Write(classical_label),
            Write(quantum_label),
            run_time=0.6
        )
        
        self.add_fixed_orientation_mobjects(zero_ket, one_ket)
        self.play(
            Write(zero_ket), Write(one_ket),
            Create(state_vector),
            Write(alpha_text), Write(beta_text),
            run_time=0.4
        )
        
        # Animate smooth rotation with live probability updates
        self.begin_ambient_camera_rotation(rate=0.1)
        
        def update_state(mob, dt):
            # Rotate state vector
            mob.rotate(1.5 * dt, axis=np.array([1, 1, 0]), about_point=[3, 0, 0])
            
            # Calculate current angle from z-axis
            current_vec = mob.get_end() - mob.get_start()
            theta = np.arccos(current_vec[2] / np.linalg.norm(current_vec))
            
            # Update probability amplitudes
            alpha = np.cos(theta/2)
            beta = np.sin(theta/2)
            
            alpha_text.become(
                Text(f"α = {alpha:.2f}", font_size=14, color=BLUE)
                .shift(RIGHT * 3 + DOWN * 2.5)
            )
            beta_text.become(
                Text(f"β = {beta:.2f}", font_size=14, color=RED)
                .shift(RIGHT * 3 + DOWN * 2.9)
            )
        
        state_vector.add_updater(update_state)
        
        # Show superposition equation - positioned carefully
        super_eq = Text("|ψ⟩ = α|0⟩ + β|1⟩", font_size=20, color=YELLOW)
        super_eq.shift(DOWN * 3.5)
        self.add_fixed_in_frame_mobjects(super_eq)
        self.play(Write(super_eq), run_time=0.4)
        
        # Let it rotate for 2s
        self.wait(2)
        
        state_vector.remove_updater(update_state)
        self.stop_ambient_camera_rotation()
        
        # Clear scene
        self.play(
            FadeOut(sphere), FadeOut(x_axis), FadeOut(y_axis), FadeOut(z_axis),
            FadeOut(state_vector), FadeOut(zero_ket), FadeOut(one_ket),
            FadeOut(title), FadeOut(classical_label), FadeOut(quantum_label),
            FadeOut(bit_state), FadeOut(alpha_text), FadeOut(beta_text),
            FadeOut(super_eq),
            run_time=0.4
        )
        
        # Reset camera
        self.set_camera_orientation(phi=0, theta=-90*DEGREES)
    
    def measurement_visualization(self):
        # Title
        title = Text("MEASUREMENT", font_size=32, color=RED)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.3)
        
        # Quantum state visualization
        main_circle = Circle(radius=1.8, color=BLUE_E, stroke_width=2)
        
        # Create probability cloud
        prob_cloud = VGroup()
        for i in range(30):
            angle = i * TAU / 30
            radius = 1.8 * (0.7 + 0.3 * np.sin(3 * angle))
            dot = Dot(
                point=[radius * np.cos(angle), radius * np.sin(angle), 0],
                color=YELLOW,
                radius=0.08
            )
            dot.set_opacity(0.6)
            prob_cloud.add(dot)
        
        # State vector
        vector = Arrow(ORIGIN, UR * 1.3, color=YELLOW, buff=0, stroke_width=6)
        
        # Probability distribution bars - positioned on sides
        prob_bars = VGroup()
        bar_labels = VGroup()
        
        for i, (label, pos, color) in enumerate([
            ("|0⟩", LEFT * 4, BLUE),
            ("|1⟩", RIGHT * 4, RED)
        ]):
            bar_bg = Rectangle(width=1, height=2.5, color=GRAY, fill_opacity=0.2)
            bar_bg.shift(pos)
            
            bar = Rectangle(width=0.8, height=1.25, color=color, fill_opacity=0.8)
            bar.shift(pos)
            
            label_text = Text(label, font_size=20, color=color)
            label_text.next_to(bar_bg, DOWN)
            
            prob_text = Text("50%", font_size=16)
            prob_text.next_to(bar, UP)
            
            prob_bars.add(bar_bg, bar, prob_text)
            bar_labels.add(label_text)
        
        self.play(
            Create(main_circle),
            Create(prob_cloud),
            Create(vector),
            Create(prob_bars),
            Write(bar_labels),
            run_time=0.6
        )
        
        # Animate quantum uncertainty
        self.play(
            prob_cloud.animate.rotate(PI/4),
            vector.animate.rotate(PI/8, about_point=ORIGIN),
            run_time=0.6
        )
        
        # Measurement device
        measure_device = VGroup(
            Rectangle(width=1.5, height=1, color=RED, fill_opacity=0.3),
            Text("MEASURE", font_size=14, color=RED)
        )
        measure_device.arrange(DOWN, buff=0.1)
        measure_device.shift(UP * 2.5)
        
        self.play(
            Create(measure_device),
            Flash(main_circle, color=RED, flash_radius=2),
            run_time=0.5
        )
        
        # Collapse animation
        collapsed_vector = Arrow(ORIGIN, DOWN * 1.8, color=YELLOW, buff=0, stroke_width=6)
        
        # Update probabilities
        new_bars = VGroup()
        new_bars.add(
            Rectangle(width=1, height=2.5, color=GRAY, fill_opacity=0.2).shift(LEFT * 4),
            Rectangle(width=0.8, height=0.1, color=BLUE, fill_opacity=0.8).shift(LEFT * 4 + DOWN * 1.2),
            Text("0%", font_size=16).shift(LEFT * 4 + UP * 1.5),
            Rectangle(width=1, height=2.5, color=GRAY, fill_opacity=0.2).shift(RIGHT * 4),
            Rectangle(width=0.8, height=2.4, color=RED, fill_opacity=0.8).shift(RIGHT * 4 + UP * 0.2),
            Text("100%", font_size=16).shift(RIGHT * 4 + UP * 1.5)
        )
        
        self.play(
            Transform(prob_cloud, Dot(DOWN * 1.8, color=RED, radius=0.2)),
            Transform(vector, collapsed_vector),
            Transform(prob_bars, new_bars),
            run_time=0.8
        )
        
        # Wave function collapse text
        collapse_text = Text("WAVE FUNCTION COLLAPSED!", font_size=22, color=YELLOW)
        collapse_text.shift(DOWN * 3)
        self.play(Write(collapse_text), run_time=0.4)
        
        self.wait(0.4)
        
        # Clear
        self.play(
            FadeOut(VGroup(
                title, main_circle, prob_cloud, vector, prob_bars,
                bar_labels, measure_device, collapse_text
            )),
            run_time=0.4
        )
    
    def entanglement_spooky(self):
        # Title
        title = Text("ENTANGLEMENT", font_size=32, color=PURPLE)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.3)
        
        # Particle sources
        source = Dot(ORIGIN, color=YELLOW, radius=0.15)
        self.play(FadeIn(source), run_time=0.3)
        
        # Create entangled particles
        particle_a = VGroup(
            Circle(radius=0.6, color=BLUE, stroke_width=4),
            Arrow(ORIGIN, UP * 0.6, color=YELLOW, buff=0, stroke_width=3)
        )
        particle_a.shift(LEFT * 3)
        
        particle_b = VGroup(
            Circle(radius=0.6, color=RED, stroke_width=4),
            Arrow(ORIGIN, UP * 0.6, color=YELLOW, buff=0, stroke_width=3)
        )
        particle_b.shift(RIGHT * 3)
        
        # Labels
        alice_label = Text("Alice", font_size=16, color=BLUE)
        alice_label.next_to(particle_a, DOWN)
        bob_label = Text("Bob", font_size=16, color=RED)
        bob_label.next_to(particle_b, DOWN)
        
        # Animate particles shooting out from source
        self.play(
            particle_a.animate.shift(LEFT * 0),
            particle_b.animate.shift(RIGHT * 0),
            FadeIn(alice_label),
            FadeIn(bob_label),
            run_time=0.5
        )
        
        # Bell state
        bell_state = Text("|Φ⁺⟩ = (|00⟩ + |11⟩)/√2", font_size=20, color=PURPLE)
        bell_state.shift(UP * 2)
        self.play(Write(bell_state), run_time=0.4)
        
        # Quantum connection visualization
        connection = VGroup()
        for i in range(5):
            wave = ParametricFunction(
                lambda t: np.array([
                    -3 + 6 * t,
                    0.2 * np.sin(10 * t),
                    0
                ]),
                t_range=[0, 1],
                color=PURPLE,
                stroke_width=3
            )
            wave.set_opacity(0.5 - i * 0.1)
            connection.add(wave)
        
        self.play(Create(connection), run_time=0.5)
        
        # Continuous entanglement animation
        def pulse_connection(mob, dt):
            mob.rotate(0.5 * dt, about_point=ORIGIN)
            for i, wave in enumerate(mob):
                wave.set_opacity(0.5 * (1 + np.sin(mob.time * 3 + i)))
        
        connection.time = 0
        connection.add_updater(lambda m, dt: setattr(m, 'time', m.time + dt))
        connection.add_updater(pulse_connection)
        
        # Distance labels
        distance_arrow = DoubleArrow(LEFT * 3, RIGHT * 3, color=GRAY)
        distance_arrow.shift(DOWN * 2)
        distance_text = Text("ANY DISTANCE", font_size=14, color=GRAY)
        distance_text.next_to(distance_arrow, DOWN)
        
        self.play(
            Create(distance_arrow),
            Write(distance_text),
            run_time=0.4
        )
        
        self.wait(0.6)
        
        # Measurement on Alice
        measure_a = Text("MEASURE", font_size=14, color=GREEN)
        measure_a.next_to(particle_a, UP)
        
        self.play(
            Write(measure_a),
            Flash(particle_a[0], color=GREEN),
            run_time=0.4
        )
        
        # Alice collapses
        self.play(
            particle_a[1].animate.put_start_and_end_on(
                particle_a[0].get_center(),
                particle_a[0].get_center() + DOWN * 0.6
            ),
            run_time=0.4
        )
        
        # Instant correlation visualization
        correlation_pulse = Circle(radius=0.1, color=PURPLE, stroke_width=8)
        correlation_pulse.move_to(particle_a[0].get_center())
        
        self.play(
            correlation_pulse.animate.move_to(particle_b[0].get_center()).scale(20).set_opacity(0),
            run_time=0.5
        )
        
        # Bob instantly collapses
        self.play(
            particle_b[1].animate.put_start_and_end_on(
                particle_b[0].get_center(),
                particle_b[0].get_center() + DOWN * 0.6
            ),
            Flash(particle_b[0], color=PURPLE),
            run_time=0.3
        )
        
        # Results
        result_a = Text("↓", font_size=24, color=BLUE)
        result_a.next_to(particle_a[0], LEFT)
        result_b = Text("↓", font_size=24, color=RED)
        result_b.next_to(particle_b[0], RIGHT)
        
        self.play(
            Write(result_a),
            Write(result_b),
            run_time=0.3
        )
        
        # Einstein quote - positioned carefully
        spooky_text = Text("\"Spooky Action at a Distance\"", font_size=20, 
                          color=PURPLE, slant=ITALIC)
        einstein = Text("- Einstein", font_size=14, color=GRAY)
        quote_group = VGroup(spooky_text, einstein).arrange(DOWN, buff=0.2)
        quote_group.shift(DOWN * 3.2)
        
        self.play(
            Write(spooky_text),
            Write(einstein),
            run_time=0.6
        )
        
        connection.remove_updater(pulse_connection)
        
        self.wait(0.4)
        
        # Clear
        self.play(
            FadeOut(VGroup(
                title, source, particle_a, particle_b, alice_label, bob_label,
                bell_state, connection, distance_arrow, distance_text,
                measure_a, result_a, result_b, spooky_text, einstein
            )),
            run_time=0.4
        )
    
    def quantum_gates_circuit(self):
        # Title
        title = Text("QUANTUM GATES", font_size=32, color=YELLOW)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.3)
        
        # Quantum circuit visualization
        # Qubit lines
        qubit_lines = VGroup()
        for i in range(2):
            line = Line(LEFT * 5, RIGHT * 5, color=WHITE, stroke_width=2)
            line.shift(DOWN * i * 1.5)
            qubit_lines.add(line)
        
        # Initial states
        init_states = VGroup()
        for i, state in enumerate(["|0⟩", "|0⟩"]):
            text = Text(state, font_size=20)
            text.next_to(qubit_lines[i], LEFT)
            init_states.add(text)
        
        self.play(
            Create(qubit_lines),
            Write(init_states),
            run_time=0.4
        )
        
        # Gate sequence with animations
        gates = []
        
        # Hadamard gate
        h_gate = VGroup(
            Square(side_length=0.6, color=GREEN, fill_opacity=0.3),
            Text("H", font_size=20, color=GREEN)
        )
        h_gate[0].move_to(LEFT * 3 + UP * 0)
        h_gate[1].move_to(h_gate[0])
        
        self.play(Create(h_gate), run_time=0.3)
        
        # Show H gate effect
        h_effect = VGroup(
            Arrow(LEFT * 2.5, LEFT * 2, color=YELLOW, buff=0),
            Text("|+⟩", font_size=16, color=GREEN).shift(LEFT * 1.5)
        )
        self.play(Create(h_effect), run_time=0.4)
        
        # CNOT gate
        cnot_gate = VGroup(
            Dot(LEFT * 0.5, color=BLUE, radius=0.1),
            Circle(radius=0.3, color=RED).shift(LEFT * 0.5 + DOWN * 1.5),
            Line(LEFT * 0.5, LEFT * 0.5 + DOWN * 1.5, stroke_width=2)
        )
        cnot_gate.add(Cross(scale_factor=0.2).move_to(cnot_gate[1]))
        
        self.play(Create(cnot_gate), run_time=0.4)
        
        # Show entanglement creation
        entangle_effect = Text("ENTANGLED!", font_size=16, color=PURPLE)
        entangle_effect.shift(RIGHT * 0.5 + DOWN * 0.75)
        
        connection_effect = DashedLine(
            UP * 0, DOWN * 1.5,
            color=PURPLE, stroke_width=4
        ).shift(RIGHT * 1.5)
        
        self.play(
            Write(entangle_effect),
            Create(connection_effect),
            run_time=0.4
        )
        
        # Measurement
        measure_symbols = VGroup()
        for i in range(2):
            meter = VGroup(
                Arc(radius=0.3, angle=PI, color=ORANGE),
                Line(ORIGIN, UP * 0.3, color=ORANGE, stroke_width=3),
                Dot(ORIGIN, color=ORANGE, radius=0.05)
            )
            meter.shift(RIGHT * 3.5 + DOWN * i * 1.5)
            measure_symbols.add(meter)
        
        self.play(Create(measure_symbols), run_time=0.4)
        
        # Final states
        final_states = VGroup()
        for i, state in enumerate(["|0⟩", "|0⟩"]):
            text = Text(state, font_size=20, color=ORANGE)
            text.next_to(qubit_lines[i], RIGHT)
            final_states.add(text)
        
        self.play(Write(final_states), run_time=0.4)
        
        # Quantum advantage text
        advantage_text = Text("Parallel Processing Power!", font_size=20, color=GREEN)
        advantage_text.shift(DOWN * 3)
        self.play(Write(advantage_text), run_time=0.4)
        
        self.wait(0.6)
        
        # Clear
        self.play(
            FadeOut(VGroup(
                title, qubit_lines, init_states, h_gate, h_effect,
                cnot_gate, entangle_effect, connection_effect,
                measure_symbols, final_states, advantage_text
            )),
            run_time=0.4
        )
    
    def quantum_supremacy(self):
        # Title
        title = Text("QUANTUM SUPREMACY", font_size=32, color=GOLD)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.3)
        
        # Exponential scaling visualization
        axes = NumberPlane(
            x_range=[0, 10, 1],
            y_range=[0, 1000, 100],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": True}
        )
        axes.shift(DOWN * 0.5)
        
        x_label = Text("Qubits", font_size=16).next_to(axes.x_axis, DOWN)
        y_label = Text("States", font_size=16).next_to(axes.y_axis, LEFT)
        
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label),
            run_time=0.4
        )
        
        # Classical line (linear)
        classical_line = axes.plot(
            lambda x: 10 * x,
            x_range=[0, 10],
            color=RED
        )
        classical_label = Text("Classical", font_size=14, color=RED)
        classical_label.next_to(classical_line.point_from_proportion(0.8), UR)
        
        # Quantum curve (exponential)
        quantum_curve = axes.plot(
            lambda x: 2**x,
            x_range=[0, 10],
            color=BLUE
        )
        quantum_label = Text("Quantum", font_size=14, color=BLUE)
        quantum_label.next_to(quantum_curve.point_from_proportion(0.5), UR)
        
        self.play(
            Create(classical_line),
            Create(quantum_curve),
            Write(classical_label),
            Write(quantum_label),
            run_time=0.6
        )
        
        # Highlight exponential advantage
        points = VGroup()
        for x, y in [(5, 32), (10, 1024)]:
            dot = Dot(axes.coords_to_point(x, y), color=YELLOW, radius=0.08)
            label = Text(f"{y}", font_size=12, color=YELLOW)
            label.next_to(dot, UR)
            points.add(dot, label)
        
        self.play(Create(points), run_time=0.4)
        
        # Real applications - positioned carefully to avoid overlap
        apps_title = Text("APPLICATIONS", font_size=20, color=GREEN)
        apps_title.shift(DOWN * 2.8)
        
        apps = VGroup(
            Text("• Drug Discovery", font_size=14),
            Text("• Cryptography", font_size=14),
            Text("• AI/ML", font_size=14),
            Text("• Finance", font_size=14)
        ).arrange(RIGHT, buff=0.5).shift(DOWN * 3.3)
        
        self.play(
            Write(apps_title),
            Write(apps),
            run_time=0.6
        )
        
        self.wait(0.5)