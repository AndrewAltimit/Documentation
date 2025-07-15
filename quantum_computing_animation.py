from manim import *
import numpy as np

class QuantumComputingAnimation(ThreeDScene):
    def construct(self):
        # Configure 3D scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Title sequence
        self.intro_sequence()
        
        # Classical vs Quantum
        self.classical_vs_quantum()
        
        # Bloch sphere introduction
        self.bloch_sphere_intro()
        
        # Single qubit gates
        self.single_qubit_gates()
        
        # Superposition and measurement
        self.superposition_demo()
        
        # Two-qubit systems and entanglement
        self.entanglement_demo()
        
        # Deutsch's algorithm
        self.deutschs_algorithm()
        
        # Quantum advantage
        self.quantum_advantage()
    
    def intro_sequence(self):
        title = Text("Quantum Computing", font_size=60, color=BLUE)
        subtitle = Text("From Qubits to Algorithms", font_size=36, color=GRAY)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
    
    def classical_vs_quantum(self):
        # Section title
        section_title = Text("Classical Bit vs Quantum Qubit", font_size=48)
        self.play(Write(section_title))
        self.play(section_title.animate.to_edge(UP).scale(0.7))
        
        # Classical bit
        bit_label = Text("Classical Bit:", font_size=32).shift(LEFT * 4 + UP * 2)
        bit_0 = Circle(radius=0.5, color=WHITE).shift(LEFT * 5)
        bit_0_text = Text("0", font_size=28).move_to(bit_0)
        bit_1 = Circle(radius=0.5, color=WHITE).shift(LEFT * 3)
        bit_1_text = Text("1", font_size=28).move_to(bit_1)
        or_text = Text("OR", font_size=24).shift(LEFT * 4)
        
        self.play(Write(bit_label))
        self.play(Create(bit_0), Write(bit_0_text))
        self.play(Create(bit_1), Write(bit_1_text))
        self.play(Write(or_text))
        
        # Quantum qubit
        qubit_label = Text("Quantum Qubit:", font_size=32).shift(RIGHT * 3 + UP * 2)
        qubit_eq = MathTex(r"|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle", font_size=36)
        qubit_eq.shift(RIGHT * 3)
        constraint = MathTex(r"|\\alpha|^2 + |\\beta|^2 = 1", font_size=28, color=YELLOW)
        constraint.next_to(qubit_eq, DOWN)
        
        self.play(Write(qubit_label))
        self.play(Write(qubit_eq))
        self.play(Write(constraint))
        
        # Highlight superposition
        superposition_text = Text("SUPERPOSITION", font_size=24, color=GREEN)
        superposition_text.next_to(constraint, DOWN)
        self.play(FadeIn(superposition_text))
        
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(bit_label), FadeOut(bit_0), FadeOut(bit_0_text),
            FadeOut(bit_1), FadeOut(bit_1_text), FadeOut(or_text),
            FadeOut(qubit_label), FadeOut(qubit_eq), FadeOut(constraint),
            FadeOut(superposition_text), FadeOut(section_title)
        )
    
    def bloch_sphere_intro(self):
        # Move to 3D view
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)
        
        # Section title
        title = Text("The Bloch Sphere", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create Bloch sphere
        sphere = Sphere(radius=2, resolution=(30, 30))
        sphere.set_color(BLUE_E)
        sphere.set_opacity(0.3)
        
        # Axes
        x_axis = Arrow3D(start=[-2.5, 0, 0], end=[2.5, 0, 0], color=RED)
        y_axis = Arrow3D(start=[0, -2.5, 0], end=[0, 2.5, 0], color=GREEN)
        z_axis = Arrow3D(start=[0, 0, -2.5], end=[0, 0, 2.5], color=BLUE)
        
        # Labels
        zero_ket = MathTex(r"|0\\rangle", font_size=36)
        zero_ket.move_to([0, 0, 2.5])
        one_ket = MathTex(r"|1\\rangle", font_size=36)
        one_ket.move_to([0, 0, -2.5])
        
        self.play(Create(sphere))
        self.play(Create(x_axis), Create(y_axis), Create(z_axis))
        self.add_fixed_orientation_mobjects(zero_ket, one_ket)
        self.play(Write(zero_ket), Write(one_ket))
        
        # State vector
        state_vector = Arrow3D(
            start=[0, 0, 0],
            end=[1.4, 0, 1.4],
            color=YELLOW,
            thickness=0.05
        )
        
        # State label
        psi_label = MathTex(r"|\\psi\\rangle", font_size=32, color=YELLOW)
        psi_label.move_to([1.7, 0, 1.7])
        
        self.play(Create(state_vector))
        self.add_fixed_orientation_mobjects(psi_label)
        self.play(Write(psi_label))
        
        # Show decomposition
        decomp = MathTex(
            r"|\\psi\\rangle = \\cos(\\theta/2)|0\\rangle + e^{i\\phi}\\sin(\\theta/2)|1\\rangle",
            font_size=32
        )
        decomp.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(decomp)
        self.play(Write(decomp))
        
        # Animate rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        self.wait(2)
        
        # Store Bloch sphere elements for later use
        self.sphere = sphere
        self.state_vector = state_vector
        self.bloch_elements = VGroup(sphere, x_axis, y_axis, z_axis, zero_ket, one_ket)
        
        # Clear decomposition
        self.play(FadeOut(decomp), FadeOut(title), FadeOut(psi_label))
    
    def single_qubit_gates(self):
        # Section title
        title = Text("Single Qubit Gates", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Gate sequence: X, Y, Z, H
        gates_data = [
            ("X", "Pauli-X (NOT)", np.array([[0, 1], [1, 0]]), [np.pi, 0, 0]),
            ("Y", "Pauli-Y", np.array([[0, -1j], [1j, 0]]), [0, np.pi, 0]),
            ("Z", "Pauli-Z", np.array([[1, 0], [0, -1]]), [0, 0, np.pi]),
            ("H", "Hadamard", np.array([[1, 1], [1, -1]])/np.sqrt(2), [np.pi, 0, np.pi/2])
        ]
        
        for gate_name, gate_desc, gate_matrix, rotation in gates_data:
            # Show gate name and matrix
            gate_label = Text(f"{gate_desc} Gate", font_size=36)
            gate_label.next_to(title, DOWN)
            self.add_fixed_in_frame_mobjects(gate_label)
            
            # Display matrix
            if gate_name == "Y":
                matrix_tex = MathTex(
                    f"{gate_name} = \\begin{{pmatrix}} 0 & -i \\\\ i & 0 \\end{{pmatrix}}",
                    font_size=32
                )
            elif gate_name == "H":
                matrix_tex = MathTex(
                    f"{gate_name} = \\frac{{1}}{{\\sqrt{{2}}}} \\begin{{pmatrix}} 1 & 1 \\\\ 1 & -1 \\end{{pmatrix}}",
                    font_size=32
                )
            else:
                matrix_tex = MathTex(
                    f"{gate_name} = \\begin{{pmatrix}} {int(gate_matrix[0,0])} & {int(gate_matrix[0,1])} \\\\ {int(gate_matrix[1,0])} & {int(gate_matrix[1,1])} \\end{{pmatrix}}",
                    font_size=32
                )
            matrix_tex.to_edge(LEFT)
            self.add_fixed_in_frame_mobjects(matrix_tex)
            
            self.play(Write(gate_label), Write(matrix_tex))
            
            # Animate gate operation on Bloch sphere
            if gate_name == "H":
                # Special case for Hadamard - show |0> to superposition
                new_vector = Arrow3D(
                    start=[0, 0, 0],
                    end=[2, 0, 0],
                    color=YELLOW,
                    thickness=0.05
                )
                self.play(Transform(self.state_vector, new_vector), run_time=2)
                
                # Show superposition state
                super_state = MathTex(
                    r"|+\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}",
                    font_size=28,
                    color=GREEN
                )
                super_state.to_edge(RIGHT)
                self.add_fixed_in_frame_mobjects(super_state)
                self.play(Write(super_state))
                self.wait(1)
                self.play(FadeOut(super_state))
            else:
                # Rotate around appropriate axis
                self.play(
                    Rotate(self.state_vector, rotation[0], axis=RIGHT),
                    Rotate(self.state_vector, rotation[1], axis=UP),
                    Rotate(self.state_vector, rotation[2], axis=OUT),
                    run_time=2
                )
            
            self.wait(1)
            self.play(FadeOut(gate_label), FadeOut(matrix_tex))
        
        self.play(FadeOut(title))
    
    def superposition_demo(self):
        # Clear previous scene
        self.play(FadeOut(self.bloch_elements), FadeOut(self.state_vector))
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)
        
        # Title
        title = Text("Superposition and Measurement", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show superposition state
        super_eq = MathTex(
            r"|\\psi\\rangle = \\frac{1}{\\sqrt{2}}|0\\rangle + \\frac{1}{\\sqrt{2}}|1\\rangle",
            font_size=40
        )
        super_eq.shift(UP * 2)
        self.play(Write(super_eq))
        
        # Probability visualization
        prob_label = Text("Measurement Probabilities:", font_size=32)
        prob_label.shift(LEFT * 2)
        
        # Bar chart for probabilities
        bar_0 = Rectangle(width=1, height=2, color=BLUE, fill_opacity=0.7)
        bar_0.shift(LEFT * 2 + DOWN)
        bar_0_label = Text("0", font_size=24).next_to(bar_0, DOWN)
        bar_0_prob = Text("50%", font_size=20, color=BLUE).move_to(bar_0)
        
        bar_1 = Rectangle(width=1, height=2, color=RED, fill_opacity=0.7)
        bar_1.shift(RIGHT * 2 + DOWN)
        bar_1_label = Text("1", font_size=24).next_to(bar_1, DOWN)
        bar_1_prob = Text("50%", font_size=20, color=RED).move_to(bar_1)
        
        self.play(Write(prob_label))
        self.play(
            Create(bar_0), Create(bar_1),
            Write(bar_0_label), Write(bar_1_label),
            Write(bar_0_prob), Write(bar_1_prob)
        )
        
        # Measurement animation
        measure_text = Text("MEASURE", font_size=36, color=YELLOW)
        measure_text.shift(UP * 0.5)
        self.play(FadeIn(measure_text))
        self.play(Flash(measure_text, color=YELLOW, flash_radius=1))
        
        # Collapse to |0>
        collapsed_eq = MathTex(r"|\\psi\\rangle = |0\\rangle", font_size=40, color=BLUE)
        collapsed_eq.move_to(super_eq)
        
        self.play(
            Transform(super_eq, collapsed_eq),
            bar_0.animate.set_height(4),
            bar_1.animate.set_height(0.1),
            bar_0_prob.animate.become(Text("100%", font_size=20, color=BLUE).move_to(bar_0)),
            bar_1_prob.animate.become(Text("0%", font_size=20, color=RED).shift(RIGHT * 2 + DOWN * 0.5))
        )
        
        self.wait(2)
        
        # Clear scene
        self.play(
            FadeOut(title), FadeOut(super_eq), FadeOut(prob_label),
            FadeOut(bar_0), FadeOut(bar_1), FadeOut(bar_0_label),
            FadeOut(bar_1_label), FadeOut(bar_0_prob), FadeOut(bar_1_prob),
            FadeOut(measure_text)
        )
    
    def entanglement_demo(self):
        # Title
        title = Text("Quantum Entanglement", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create two qubits
        qubit1_label = Text("Qubit 1", font_size=24, color=BLUE).shift(LEFT * 4 + UP * 2)
        qubit2_label = Text("Qubit 2", font_size=24, color=RED).shift(RIGHT * 4 + UP * 2)
        
        qubit1 = Circle(radius=1, color=BLUE).shift(LEFT * 4)
        qubit2 = Circle(radius=1, color=RED).shift(RIGHT * 4)
        
        self.play(
            Write(qubit1_label), Write(qubit2_label),
            Create(qubit1), Create(qubit2)
        )
        
        # Initial state
        init_state = MathTex(r"|\\psi\\rangle = |00\\rangle", font_size=36)
        init_state.shift(UP * 0.5)
        self.play(Write(init_state))
        
        # Apply Hadamard to first qubit
        h_text = Text("Apply H to Qubit 1", font_size=28, color=YELLOW)
        h_text.shift(DOWN * 1.5)
        self.play(Write(h_text))
        
        # Show intermediate state
        inter_state = MathTex(
            r"|\\psi\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |10\\rangle)",
            font_size=32
        )
        inter_state.move_to(init_state)
        self.play(Transform(init_state, inter_state))
        self.wait(1)
        self.play(FadeOut(h_text))
        
        # Apply CNOT
        cnot_text = Text("Apply CNOT", font_size=28, color=YELLOW)
        cnot_text.shift(DOWN * 1.5)
        self.play(Write(cnot_text))
        
        # Create entanglement visualization - glowing connection
        connection = Line(
            qubit1.get_center(),
            qubit2.get_center(),
            color=PURPLE,
            stroke_width=8
        )
        connection.set_opacity(0)
        
        # Bell state
        bell_state = MathTex(
            r"|\\Phi^+\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)",
            font_size=32,
            color=PURPLE
        )
        bell_state.move_to(init_state)
        
        self.play(
            Transform(init_state, bell_state),
            connection.animate.set_opacity(0.8),
            qubit1.animate.set_stroke(PURPLE, width=4),
            qubit2.animate.set_stroke(PURPLE, width=4)
        )
        
        # Add pulsing effect to connection
        self.play(
            connection.animate.set_stroke_width(12).set_opacity(1),
            rate_func=there_and_back,
            run_time=1
        )
        
        # Show correlation
        entangled_text = Text("ENTANGLED!", font_size=36, color=PURPLE)
        entangled_text.shift(DOWN * 2.5)
        self.play(Write(entangled_text))
        
        self.wait(2)
        
        # Clear for next section
        self.play(
            FadeOut(title), FadeOut(init_state), FadeOut(qubit1_label),
            FadeOut(qubit2_label), FadeOut(qubit1), FadeOut(qubit2),
            FadeOut(connection), FadeOut(cnot_text), FadeOut(entangled_text)
        )
    
    def deutschs_algorithm(self):
        # Title
        title = Text("Deutsch's Algorithm", font_size=48)
        subtitle = Text("The First Quantum Algorithm", font_size=28, color=GRAY)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(subtitle))
        
        # Problem statement
        problem = Text("Is f(0) = f(1)?", font_size=36, color=YELLOW)
        problem.shift(UP * 2)
        classical = Text("Classical: 2 queries needed", font_size=28, color=RED)
        classical.next_to(problem, DOWN)
        quantum = Text("Quantum: 1 query needed!", font_size=28, color=GREEN)
        quantum.next_to(classical, DOWN)
        
        self.play(Write(problem))
        self.play(Write(classical))
        self.play(Write(quantum))
        self.wait(2)
        
        self.play(FadeOut(problem), FadeOut(classical), FadeOut(quantum))
        
        # Quantum circuit
        circuit_label = Text("Quantum Circuit:", font_size=32)
        circuit_label.shift(LEFT * 3 + UP * 2)
        self.play(Write(circuit_label))
        
        # Draw circuit elements
        # Qubit lines
        line1 = Line(LEFT * 5, RIGHT * 5, color=WHITE).shift(UP * 0.5)
        line2 = Line(LEFT * 5, RIGHT * 5, color=WHITE).shift(DOWN * 0.5)
        
        # Initial states
        q0_init = MathTex(r"|0\\rangle", font_size=24).next_to(line1, LEFT)
        q1_init = MathTex(r"|1\\rangle", font_size=24).next_to(line2, LEFT)
        
        self.play(Create(line1), Create(line2), Write(q0_init), Write(q1_init))
        
        # Hadamard gates
        h1_box = Square(side_length=0.5, color=BLUE).move_to(line1.get_start() + RIGHT * 2)
        h1_text = Text("H", font_size=20).move_to(h1_box)
        h2_box = Square(side_length=0.5, color=BLUE).move_to(line2.get_start() + RIGHT * 2)
        h2_text = Text("H", font_size=20).move_to(h2_box)
        
        self.play(Create(h1_box), Create(h2_box), Write(h1_text), Write(h2_text))
        
        # Oracle
        oracle_box = Rectangle(width=1, height=1.5, color=PURPLE)
        oracle_box.move_to([0, 0, 0])
        oracle_text = MathTex("U_f", font_size=24).move_to(oracle_box)
        
        self.play(Create(oracle_box), Write(oracle_text))
        
        # Final Hadamard
        h3_box = Square(side_length=0.5, color=BLUE).move_to(line1.get_end() + LEFT * 2)
        h3_text = Text("H", font_size=20).move_to(h3_box)
        
        self.play(Create(h3_box), Write(h3_text))
        
        # Measurement
        meter = Circle(radius=0.3, color=YELLOW).move_to(line1.get_end() + LEFT * 0.5)
        meter_needle = Line(meter.get_center(), meter.get_top(), color=RED, stroke_width=3)
        
        self.play(Create(meter), Create(meter_needle))
        
        # Show quantum interference
        interference_text = Text("Quantum Interference", font_size=32, color=GREEN)
        interference_text.shift(DOWN * 2.5)
        
        # Visualize interference
        const_path = CurvedArrow(
            start_point=h1_box.get_center(),
            end_point=h3_box.get_center(),
            color=GREEN,
            angle=TAU/4
        ).shift(UP * 0.8)
        const_label = Text("f constant", font_size=16, color=GREEN)
        const_label.next_to(const_path, UP)
        
        balanced_path = CurvedArrow(
            start_point=h1_box.get_center(),
            end_point=h3_box.get_center(),
            color=RED,
            angle=-TAU/4
        ).shift(DOWN * 0.3)
        balanced_label = Text("f balanced", font_size=16, color=RED)
        balanced_label.next_to(balanced_path, DOWN)
        
        self.play(Write(interference_text))
        self.play(Create(const_path), Create(balanced_path))
        self.play(Write(const_label), Write(balanced_label))
        
        # Show result
        result_text = Text("Measure 0 → f constant", font_size=24, color=GREEN)
        result_text2 = Text("Measure 1 → f balanced", font_size=24, color=RED)
        result_text.shift(DOWN * 3.5)
        result_text2.next_to(result_text, DOWN)
        
        self.play(Write(result_text), Write(result_text2))
        
        self.wait(3)
        
        # Clear scene
        all_objects = [
            title, circuit_label, line1, line2, q0_init, q1_init,
            h1_box, h1_text, h2_box, h2_text, oracle_box, oracle_text,
            h3_box, h3_text, meter, meter_needle, interference_text,
            const_path, const_label, balanced_path, balanced_label,
            result_text, result_text2
        ]
        self.play(*[FadeOut(obj) for obj in all_objects])
    
    def quantum_advantage(self):
        # Title
        title = Text("Quantum Advantage", font_size=48, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Exponential scaling visualization
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1000, 100],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True}
        )
        axes.shift(DOWN * 0.5)
        
        x_label = Text("Number of Qubits", font_size=20).next_to(axes.x_axis, DOWN)
        y_label = Text("Computational Power", font_size=20).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Exponential curve
        exp_curve = axes.plot(
            lambda x: 2**x,
            x_range=[0, 10],
            color=BLUE
        )
        exp_label = MathTex("2^n", font_size=24, color=BLUE)
        exp_label.next_to(exp_curve.point_from_proportion(0.8), UR)
        
        self.play(Create(exp_curve), Write(exp_label))
        
        # Highlight key points
        points_data = [
            (5, 32, "32 states"),
            (10, 1024, "1024 states"),
        ]
        
        for x, y, label in points_data:
            dot = Dot(axes.coords_to_point(x, y), color=YELLOW)
            text = Text(label, font_size=16, color=YELLOW)
            text.next_to(dot, UR)
            self.play(FadeIn(dot), Write(text))
        
        # Future text
        future_text = Text(
            "The Future of Computing",
            font_size=36,
            color=GREEN
        )
        future_text.shift(DOWN * 3.5)
        
        applications = Text(
            "Cryptography • Drug Discovery • AI • Materials Science",
            font_size=24,
            color=GRAY
        )
        applications.next_to(future_text, DOWN)
        
        self.play(Write(future_text))
        self.play(FadeIn(applications))
        
        self.wait(3)
        
        # End screen
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        end_text = Text("Quantum Computing", font_size=60, color=BLUE)
        end_subtitle = Text("The Next Frontier", font_size=36, color=GRAY)
        end_subtitle.next_to(end_text, DOWN)
        
        self.play(Write(end_text), FadeIn(end_subtitle))
        self.wait(2)