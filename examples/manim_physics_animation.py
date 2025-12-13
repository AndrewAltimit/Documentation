#!/usr/bin/env python3
"""
Manim Animation Examples for Physics Documentation
This creates beautiful mathematical animations similar to 3Blue1Brown style

Installation:
pip install manim

Usage:
manim -pql manim_physics_animation.py SimpleHarmonicMotion
manim -pqh manim_physics_animation.py WaveInterference
"""

import numpy as np
from manim import *


class SimpleHarmonicMotion(Scene):
    """Animates a mass-spring system and its corresponding sinusoidal motion"""

    def construct(self):
        # Title
        title = Text("Simple Harmonic Motion", font_size=48)
        subtitle = Text("Mass-Spring System", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        self.play(Write(title), Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Create spring and mass
        spring_start = np.array([-3, 2, 0])
        spring_end = np.array([0, 2, 0])

        # Wall
        wall = Line(start=spring_start + UP, end=spring_start + DOWN, stroke_width=8)

        # Mass
        mass = Square(side_length=0.8, color=BLUE, fill_opacity=0.7)
        mass.move_to(spring_end)
        mass_label = MathTex("m", color=WHITE).move_to(mass.get_center())

        # Spring (using parametric function)
        def spring_func(t):
            """Generate spring path"""
            coils = 8
            width = 0.3
            x = spring_start[0] + t * (spring_end[0] - spring_start[0])
            if 0.1 < t < 0.9:
                y = spring_start[1] + width * np.sin(coils * 2 * PI * t)
            else:
                y = spring_start[1]
            return np.array([x, y, 0])

        spring = ParametricFunction(
            spring_func, t_range=[0, 1], color=GRAY, stroke_width=3
        )

        # Equilibrium position marker
        eq_line = DashedLine(
            start=spring_end + 2 * UP,
            end=spring_end + 2 * DOWN,
            color=YELLOW,
            stroke_width=2,
        )
        eq_label = Text("Equilibrium", font_size=16, color=YELLOW)
        eq_label.next_to(eq_line, UP)

        # Group spring system
        spring_system = VGroup(wall, spring, mass, mass_label)

        # Create graph
        axes = Axes(
            x_range=[0, 4 * PI, PI],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=3,
            axis_config={"color": BLUE_C},
            tips=False,
        )
        axes.shift(DOWN * 2)

        x_label = MathTex("t", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("x(t)", font_size=24).next_to(axes.y_axis, UP)

        # Add to scene
        self.play(
            Create(wall),
            Create(spring),
            FadeIn(mass),
            Write(mass_label),
            Create(eq_line),
            Write(eq_label),
        )
        self.wait(0.5)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Animation parameters
        amplitude = 1.5
        omega = 1

        # Create sine curve
        sine_curve = axes.plot(
            lambda t: amplitude * np.sin(omega * t), color=RED, x_range=[0, 0.1]
        )

        # Create dot that follows the curve
        dot = Dot(color=RED)
        dot.move_to(axes.coords_to_point(0, 0))

        # Time tracking
        time_tracker = ValueTracker(0)

        # Update functions
        def update_mass(mob):
            t = time_tracker.get_value()
            displacement = amplitude * np.sin(omega * t)
            new_pos = spring_end + displacement * RIGHT
            mob.move_to(new_pos)

        def update_mass_label(mob):
            mob.move_to(mass.get_center())

        def update_spring(mob):
            t = time_tracker.get_value()
            displacement = amplitude * np.sin(omega * t)
            new_end = spring_end + displacement * RIGHT

            def new_spring_func(t):
                x = spring_start[0] + t * (new_end[0] - spring_start[0])
                if 0.1 < t < 0.9:
                    y = spring_start[1] + 0.3 * np.sin(8 * 2 * PI * t)
                else:
                    y = spring_start[1]
                return np.array([x, y, 0])

            mob.become(
                ParametricFunction(
                    new_spring_func, t_range=[0, 1], color=GRAY, stroke_width=3
                )
            )

        def update_curve(mob):
            t = time_tracker.get_value()
            new_curve = axes.plot(
                lambda x: amplitude * np.sin(omega * x), color=RED, x_range=[0, t]
            )
            mob.become(new_curve)

        def update_dot(mob):
            t = time_tracker.get_value()
            displacement = amplitude * np.sin(omega * t)
            mob.move_to(axes.coords_to_point(t, displacement))

        # Add updaters
        mass.add_updater(update_mass)
        mass_label.add_updater(update_mass_label)
        spring.add_updater(update_spring)
        sine_curve.add_updater(update_curve)
        dot.add_updater(update_dot)

        self.add(dot)

        # Animate
        self.play(time_tracker.animate.set_value(4 * PI), run_time=8, rate_func=linear)

        # Add equation
        equation = MathTex(
            r"x(t) = A\sin(\omega t)",
            r"\quad \text{where} \quad",
            r"A = 1.5, \omega = 1",
        )
        equation.scale(0.8)
        equation.to_edge(DOWN)
        equation.shift(DOWN * 0.5)

        self.play(Write(equation))
        self.wait(2)


class WaveInterference(Scene):
    """Demonstrates wave interference patterns"""

    def construct(self):
        # Title
        title = Text("Wave Interference", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Create two wave sources
        source1 = Dot(color=RED).shift(LEFT * 3)
        source2 = Dot(color=BLUE).shift(RIGHT * 3)

        source1_label = Text("S₁", font_size=20).next_to(source1, UP)
        source2_label = Text("S₂", font_size=20).next_to(source2, UP)

        self.play(
            FadeIn(source1), FadeIn(source2), Write(source1_label), Write(source2_label)
        )

        # Create waves
        num_circles = 15
        circles1 = []
        circles2 = []

        for i in range(num_circles):
            radius = 0.5 + i * 0.5
            circle1 = Circle(
                radius=radius, color=RED, stroke_opacity=0.5 - i * 0.03, stroke_width=2
            ).move_to(source1.get_center())

            circle2 = Circle(
                radius=radius, color=BLUE, stroke_opacity=0.5 - i * 0.03, stroke_width=2
            ).move_to(source2.get_center())

            circles1.append(circle1)
            circles2.append(circle2)

        # Animate wave propagation
        for i in range(num_circles):
            self.play(Create(circles1[i]), Create(circles2[i]), run_time=0.3)

        # Show interference pattern
        interference_text = Text(
            "Constructive and Destructive Interference", font_size=24
        ).to_edge(DOWN)

        # Mark constructive interference points
        const_points = [
            Dot(color=YELLOW, radius=0.1).move_to(ORIGIN),
            Dot(color=YELLOW, radius=0.1).move_to(UP * 2),
            Dot(color=YELLOW, radius=0.1).move_to(DOWN * 2),
        ]

        # Mark destructive interference points
        dest_points = [
            Dot(color=PURPLE, radius=0.1).move_to(UP + RIGHT * 0.5),
            Dot(color=PURPLE, radius=0.1).move_to(DOWN + RIGHT * 0.5),
            Dot(color=PURPLE, radius=0.1).move_to(UP + LEFT * 0.5),
            Dot(color=PURPLE, radius=0.1).move_to(DOWN + LEFT * 0.5),
        ]

        self.play(Write(interference_text))
        self.play(
            *[FadeIn(dot) for dot in const_points],
            *[FadeIn(dot) for dot in dest_points]
        )

        # Add legend
        legend = (
            VGroup(
                Text("Constructive", font_size=16, color=YELLOW),
                Text("Destructive", font_size=16, color=PURPLE),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UR)
        )

        self.play(Write(legend))
        self.wait(2)


class PendulumPhaseSpace(Scene):
    """Shows pendulum motion and its phase space representation"""

    def construct(self):
        # Title
        title = Text("Pendulum Phase Space", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Create pendulum
        pivot = Dot(color=WHITE).shift(UP * 2)
        bob = Circle(radius=0.3, color=BLUE, fill_opacity=0.8)
        initial_angle = PI / 6
        length = 2

        bob_pos = pivot.get_center() + length * np.array(
            [np.sin(initial_angle), -np.cos(initial_angle), 0]
        )
        bob.move_to(bob_pos)

        rod = Line(pivot.get_center(), bob.get_center(), color=GRAY)

        pendulum = VGroup(pivot, rod, bob)

        # Create phase space
        phase_axes = Axes(
            x_range=[-PI, PI, PI / 2],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": BLUE_C},
        ).shift(DOWN * 2 + RIGHT * 3)

        theta_label = MathTex(r"\theta", font_size=24).next_to(phase_axes.x_axis, RIGHT)
        omega_label = MathTex(r"\omega", font_size=24).next_to(phase_axes.y_axis, UP)

        self.play(
            Create(pendulum), Create(phase_axes), Write(theta_label), Write(omega_label)
        )

        # Time tracker
        time_tracker = ValueTracker(0)

        # Pendulum parameters
        g = 9.81
        L = 2
        damping = 0.1

        # Phase space trajectory
        trajectory = VMobject(color=RED)
        trajectory.set_points_as_corners([phase_axes.coords_to_point(initial_angle, 0)])

        # Current state dot
        phase_dot = Dot(color=RED).move_to(phase_axes.coords_to_point(initial_angle, 0))

        # Variables for integration
        theta = initial_angle
        omega = 0
        dt = 0.05

        def update_pendulum(mob, dt):
            nonlocal theta, omega

            # Physics integration (Euler method)
            alpha = -(g / L) * np.sin(theta) - damping * omega
            omega += alpha * dt
            theta += omega * dt

            # Update pendulum position
            new_bob_pos = pivot.get_center() + length * np.array(
                [np.sin(theta), -np.cos(theta), 0]
            )
            mob[2].move_to(new_bob_pos)  # bob
            mob[1].become(Line(pivot.get_center(), new_bob_pos, color=GRAY))  # rod

            # Update phase space
            phase_dot.move_to(phase_axes.coords_to_point(theta, omega))
            trajectory.add_points_as_corners([phase_axes.coords_to_point(theta, omega)])

        # Add updater
        pendulum.add_updater(update_pendulum)

        self.add(trajectory, phase_dot)
        self.wait(8)

        # Add explanation
        explanation = Text(
            "Phase space shows all possible states of the system", font_size=20
        ).to_edge(DOWN)

        self.play(Write(explanation))
        self.wait(2)


# Run with:
# manim -pql manim_physics_animation.py SimpleHarmonicMotion
# manim -pqh manim_physics_animation.py WaveInterference
# manim -pqh manim_physics_animation.py PendulumPhaseSpace
