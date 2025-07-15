from manim import *

class SimpleDemo(Scene):
    def construct(self):
        # Title
        title = Text("Simple Harmonic Motion", font_size=48, color=BLUE)
        subtitle = Text("Physics Animation Demo", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Create a circle that oscillates
        circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.7)
        circle.shift(LEFT * 3)
        
        # Create path
        path = Line(LEFT * 3, RIGHT * 3, color=GRAY)
        
        self.play(Create(path), Create(circle))
        
        # Create sine wave
        sine_curve = FunctionGraph(
            lambda x: np.sin(x),
            x_range=[-3, 3],
            color=RED
        ).shift(DOWN * 2)
        
        self.play(Create(sine_curve))
        
        # Oscillation
        self.play(
            circle.animate.shift(RIGHT * 6),
            rate_func=there_and_back,
            run_time=2
        )
        
        # Repeat with sine rate function
        def sine_rate(t):
            return (np.sin(2 * PI * t - PI/2) + 1) / 2
        
        for _ in range(3):
            self.play(
                circle.animate.shift(RIGHT * 6),
                rate_func=sine_rate,
                run_time=2
            )
            self.play(
                circle.animate.shift(LEFT * 6),
                rate_func=sine_rate,
                run_time=2
            )
        
        # End text
        end_text = Text("Created with Manim", font_size=36, color=GREEN)
        end_text.to_edge(DOWN)
        self.play(Write(end_text))
        self.wait(2)