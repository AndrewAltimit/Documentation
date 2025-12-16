from manim import *


class SimpleHarmonicMotion(Scene):
    def construct(self):
        # Title
        title = Text("Simple Harmonic Motion", font_size=48)
        subtitle = Text("Physics Animation Demo", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Create spring and mass system
        # Wall
        wall = Line(start=3 * LEFT + UP, end=3 * LEFT + DOWN, stroke_width=8)

        # Mass
        mass = Square(side_length=1, color=BLUE, fill_opacity=0.7)
        mass.move_to(ORIGIN)
        mass_label = MathTex("m", color=WHITE).move_to(mass.get_center())

        # Spring (simple zigzag)
        spring_points = []
        num_coils = 8
        for i in range(num_coils * 4):
            t = i / (num_coils * 4 - 1)
            x = -3 + 3 * t
            y = 0.3 * (-1) ** i if 0.1 < t < 0.9 else 0
            spring_points.append([x, y, 0])

        spring = VMobject()
        spring.set_points_smoothly(spring_points)
        spring.set_stroke(GRAY, width=3)

        # Group elements
        system = VGroup(wall, spring, mass, mass_label)
        system.shift(UP)

        # Add to scene
        self.play(Create(wall), Create(spring), FadeIn(mass), Write(mass_label))
        self.wait(0.5)

        # Create graph axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=3,
            axis_config={"color": BLUE_C},
        )
        axes.shift(DOWN * 2)

        # Labels
        x_label = Text("time", font_size=20).next_to(axes.x_axis, RIGHT)
        y_label = Text("position", font_size=20).next_to(axes.y_axis, UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Create sine curve
        amplitude = 1.5
        sine_curve = axes.plot(
            lambda t: amplitude * np.sin(t), x_range=[0, 10], color=RED
        )

        # Tracking dot
        dot = Dot(color=RED)
        dot.move_to(axes.coords_to_point(0, 0))

        self.play(Create(sine_curve), FadeIn(dot))

        # Animation with value tracker
        t_tracker = ValueTracker(0)

        def update_mass(mob):
            t = t_tracker.get_value()
            displacement = amplitude * np.sin(t)
            mob.move_to(UP + displacement * RIGHT)

        def update_spring(mob):
            t = t_tracker.get_value()
            displacement = amplitude * np.sin(t)
            # Recreate spring with new endpoint
            new_points = []
            for i in range(num_coils * 4):
                s = i / (num_coils * 4 - 1)
                x = -3 + (3 + displacement) * s
                y = 0.3 * (-1) ** i if 0.1 < s < 0.9 else 0
                new_points.append([x, y + 1, 0])
            mob.set_points_smoothly(new_points)

        def update_label(mob):
            mob.move_to(mass.get_center())

        def update_dot(mob):
            t = t_tracker.get_value()
            mob.move_to(axes.coords_to_point(t, amplitude * np.sin(t)))

        # Add updaters
        mass.add_updater(update_mass)
        spring.add_updater(update_spring)
        mass_label.add_updater(update_label)
        dot.add_updater(update_dot)

        # Animate
        self.play(t_tracker.animate.set_value(10), run_time=10, rate_func=linear)

        # Remove updaters
        mass.remove_updater(update_mass)
        spring.remove_updater(update_spring)
        mass_label.remove_updater(update_label)
        dot.remove_updater(update_dot)

        # Add equation
        equation = MathTex(r"x(t) = A\sin(\omega t)", font_size=36)
        equation.to_edge(DOWN)
        self.play(Write(equation))
        self.wait(2)
