#!/usr/bin/env python3
"""
Physics Animations using Matplotlib
Creates animated visualizations similar to Manim but using only matplotlib
"""

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# Set up the plotting style
plt.style.use("dark_background")


def create_simple_harmonic_motion():
    """Creates an animation of a mass-spring system"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Simple Harmonic Motion", fontsize=16, color="white")

    # Left subplot: Spring-mass system
    ax1.set_xlim(-4, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Mass-Spring System", fontsize=12)

    # Wall
    wall = Rectangle((-4, -1), 0.2, 2, facecolor="gray")
    ax1.add_patch(wall)

    # Mass
    mass_size = 0.6
    mass = FancyBboxPatch(
        (-0.3, -0.3),
        mass_size,
        mass_size,
        boxstyle="round,pad=0.1",
        facecolor="dodgerblue",
        edgecolor="white",
        linewidth=2,
    )
    ax1.add_patch(mass)

    # Mass label
    mass_text = ax1.text(
        0, 0, "m", ha="center", va="center", fontsize=14, color="white", weight="bold"
    )

    # Spring (will be updated in animation)
    spring_x = np.linspace(-3.8, 0, 20)
    spring_y = np.zeros_like(spring_x)
    (spring_line,) = ax1.plot(spring_x, spring_y, "gray", linewidth=2)

    # Equilibrium line
    ax1.axvline(x=0, color="yellow", linestyle="--", alpha=0.5)
    ax1.text(0, -1.5, "Equilibrium", ha="center", color="yellow", fontsize=10)

    # Right subplot: Position vs Time graph
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Position (m)", fontsize=12)
    ax2.set_title("Position vs Time", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="yellow", linestyle="--", alpha=0.5)

    # Initialize the position graph
    time_data = []
    position_data = []
    (position_line,) = ax2.plot([], [], "r-", linewidth=2)
    (current_dot,) = ax2.plot([], [], "ro", markersize=8)

    # Animation parameters
    amplitude = 1.5
    omega = 1.0

    def create_spring_shape(start_x, end_x, coils=8):
        """Create spring coordinates"""
        x = np.linspace(start_x, end_x, 100)
        y = np.zeros_like(x)

        # Create coils in the middle section
        coil_start = 0.1
        coil_end = 0.9
        mask = (x > start_x + coil_start * (end_x - start_x)) & (
            x < start_x + coil_end * (end_x - start_x)
        )

        x_normalized = (x[mask] - x[mask][0]) / (x[mask][-1] - x[mask][0])
        y[mask] = 0.2 * np.sin(2 * np.pi * coils * x_normalized)

        return x, y

    def animate(frame):
        t = frame * 0.05  # Time step

        # Calculate position
        x_pos = amplitude * np.sin(omega * t)

        # Update mass position
        mass.set_x(x_pos - mass_size / 2)
        mass_text.set_x(x_pos)

        # Update spring
        spring_x, spring_y = create_spring_shape(-3.8, x_pos - mass_size / 2)
        spring_line.set_data(spring_x, spring_y)

        # Update graph
        time_data.append(t)
        position_data.append(x_pos)

        # Keep only recent data
        if len(time_data) > 200:
            time_data.pop(0)
            position_data.pop(0)

        position_line.set_data(time_data, position_data)
        current_dot.set_data([t], [x_pos])

        # Adjust x-axis if needed
        if t > 10:
            ax2.set_xlim(t - 10, t)

        return mass, mass_text, spring_line, position_line, current_dot

    # Add equation
    equation_text = (
        r"$x(t) = A\sin(\omega t)$"
        + "\n"
        + r"$A = 1.5\,\mathrm{m}, \omega = 1\,\mathrm{rad/s}$"
    )
    fig.text(0.5, 0.02, equation_text, ha="center", fontsize=12, color="white")

    anim = animation.FuncAnimation(
        fig, animate, frames=400, interval=50, blit=True, repeat=True
    )

    return fig, anim


def create_wave_interference():
    """Creates an animation of wave interference from two sources"""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Wave Interference Pattern", fontsize=16, color="white")

    # Set up the axes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Create sources
    source1_pos = np.array([-2, 0])
    source2_pos = np.array([2, 0])

    source1 = Circle(source1_pos, 0.15, color="red", zorder=10)
    source2 = Circle(source2_pos, 0.15, color="blue", zorder=10)
    ax.add_patch(source1)
    ax.add_patch(source2)

    ax.text(
        source1_pos[0],
        source1_pos[1] - 0.5,
        "S₁",
        ha="center",
        color="red",
        fontsize=12,
        weight="bold",
    )
    ax.text(
        source2_pos[0],
        source2_pos[1] - 0.5,
        "S₂",
        ha="center",
        color="blue",
        fontsize=12,
        weight="bold",
    )

    # Create grid for wave calculation
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)

    # Wave parameters
    wavelength = 0.5
    k = 2 * np.pi / wavelength

    # Initialize wave field
    im = ax.imshow(
        np.zeros_like(X), extent=[-5, 5, -5, 5], cmap="RdBu", vmin=-2, vmax=2, alpha=0.8
    )

    # Add circles for wavefronts
    circles1 = []
    circles2 = []
    max_circles = 15

    def animate(frame):
        t = frame * 0.1

        # Calculate distances from sources
        r1 = np.sqrt((X - source1_pos[0]) ** 2 + (Y - source1_pos[1]) ** 2)
        r2 = np.sqrt((X - source2_pos[0]) ** 2 + (Y - source2_pos[1]) ** 2)

        # Calculate waves with attenuation
        wave1 = np.sin(k * r1 - 2 * np.pi * t) / (1 + 0.3 * r1)
        wave2 = np.sin(k * r2 - 2 * np.pi * t) / (1 + 0.3 * r2)

        # Superposition
        total_wave = wave1 + wave2

        im.set_array(total_wave)

        # Update wavefront circles
        # Remove old circles
        for circle in circles1 + circles2:
            circle.remove()
        circles1.clear()
        circles2.clear()

        # Add new circles
        for i in range(max_circles):
            radius = (t + i) * wavelength % (max_circles * wavelength)
            alpha = max(0, 1 - radius / (max_circles * wavelength))

            if alpha > 0.1:
                circle1 = Circle(
                    source1_pos,
                    radius,
                    fill=False,
                    edgecolor="red",
                    alpha=alpha * 0.3,
                    linewidth=1,
                )
                circle2 = Circle(
                    source2_pos,
                    radius,
                    fill=False,
                    edgecolor="blue",
                    alpha=alpha * 0.3,
                    linewidth=1,
                )
                ax.add_patch(circle1)
                ax.add_patch(circle2)
                circles1.append(circle1)
                circles2.append(circle2)

        return [im] + circles1 + circles2

    # Add legend
    ax.text(
        0,
        4.5,
        "Constructive Interference (Red)",
        ha="center",
        color="white",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="darkred", alpha=0.5),
    )
    ax.text(
        0,
        -4.5,
        "Destructive Interference (Blue)",
        ha="center",
        color="white",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="darkblue", alpha=0.5),
    )

    anim = animation.FuncAnimation(
        fig, animate, frames=200, interval=50, blit=True, repeat=True
    )

    return fig, anim


def create_pendulum_phase_space():
    """Creates an animation of a pendulum with its phase space diagram"""
    fig = plt.figure(figsize=(12, 6))

    # Create gridspec for better layout control
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle("Pendulum Motion and Phase Space", fontsize=16, color="white")

    # Left subplot: Pendulum
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2.5, 0.5)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Pendulum", fontsize=12)

    # Pendulum parameters
    L = 2.0  # Length
    g = 9.81  # Gravity
    damping = 0.1

    # Initial conditions
    theta0 = np.pi / 3
    omega0 = 0

    # Pendulum components
    pivot = Circle((0, 0), 0.05, color="white", zorder=5)
    ax1.add_patch(pivot)

    (rod,) = ax1.plot([], [], "gray", linewidth=3)
    bob = Circle((0, 0), 0.2, color="dodgerblue", zorder=4)
    ax1.add_patch(bob)

    # Trail
    trail_x, trail_y = [], []
    (trail_line,) = ax1.plot([], [], "c-", linewidth=1, alpha=0.5)

    # Right subplot: Phase space
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-4, 4)
    ax2.set_xlabel(r"$\theta$ (rad)", fontsize=12)
    ax2.set_ylabel(r"$\omega$ (rad/s)", fontsize=12)
    ax2.set_title("Phase Space", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Phase space trajectory
    phase_x, phase_y = [], []
    (phase_line,) = ax2.plot([], [], "r-", linewidth=2)
    (phase_dot,) = ax2.plot([], [], "ro", markersize=8)

    # Energy contours (for undamped pendulum)
    theta_range = np.linspace(-np.pi, np.pi, 100)
    omega_range = np.linspace(-4, 4, 100)
    Theta, Omega = np.meshgrid(theta_range, omega_range)

    # Energy function: E = 1/2 * L^2 * omega^2 + g*L*(1 - cos(theta))
    E = 0.5 * L**2 * Omega**2 + g * L * (1 - np.cos(Theta))

    contours = ax2.contour(
        Theta, Omega, E, levels=10, colors="gray", alpha=0.3, linewidths=1
    )

    # Time evolution
    dt = 0.05
    theta = theta0
    omega = omega0

    def animate(frame):
        nonlocal theta, omega

        # Physics integration (improved Euler method)
        # Angular acceleration
        alpha = -(g / L) * np.sin(theta) - damping * omega

        # Update state
        omega_new = omega + alpha * dt
        theta_new = theta + omega_new * dt

        omega = omega_new
        theta = theta_new

        # Pendulum position
        x = L * np.sin(theta)
        y = -L * np.cos(theta)

        # Update pendulum
        rod.set_data([0, x], [0, y])
        bob.center = (x, y)

        # Update trail
        trail_x.append(x)
        trail_y.append(y)
        if len(trail_x) > 100:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        # Update phase space
        # Wrap theta to [-pi, pi]
        theta_wrapped = np.arctan2(np.sin(theta), np.cos(theta))

        phase_x.append(theta_wrapped)
        phase_y.append(omega)
        if len(phase_x) > 500:
            phase_x.pop(0)
            phase_y.pop(0)

        phase_line.set_data(phase_x, phase_y)
        phase_dot.set_data([theta_wrapped], [omega])

        return rod, bob, trail_line, phase_line, phase_dot

    # Add text annotations
    ax2.text(
        0.02,
        0.98,
        "Energy contours",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=10,
        color="gray",
    )

    equation_text = r"$\ddot{\theta} + \frac{g}{L}\sin\theta + \gamma\dot{\theta} = 0$"
    fig.text(0.5, 0.02, equation_text, ha="center", fontsize=12, color="white")

    anim = animation.FuncAnimation(
        fig, animate, frames=800, interval=50, blit=True, repeat=True
    )

    return fig, anim


def save_all_animations():
    """Generate and save all animations"""
    print("Creating Simple Harmonic Motion animation...")
    fig1, anim1 = create_simple_harmonic_motion()
    writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    anim1.save("simple_harmonic_motion.mp4", writer=writer)
    plt.close(fig1)
    print("✓ Saved: simple_harmonic_motion.mp4")

    print("\nCreating Wave Interference animation...")
    fig2, anim2 = create_wave_interference()
    anim2.save("wave_interference.mp4", writer=writer)
    plt.close(fig2)
    print("✓ Saved: wave_interference.mp4")

    print("\nCreating Pendulum Phase Space animation...")
    fig3, anim3 = create_pendulum_phase_space()
    anim3.save("pendulum_phase_space.mp4", writer=writer)
    plt.close(fig3)
    print("✓ Saved: pendulum_phase_space.mp4")

    print("\nAll animations saved successfully!")


if __name__ == "__main__":
    # Check if we have ffmpeg
    try:
        import subprocess

        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        save_all_animations()
    except:
        print("FFmpeg not found. Showing animations in window instead...")
        print("Install ffmpeg to save as MP4 files.")

        # Show animations in matplotlib window
        fig1, anim1 = create_simple_harmonic_motion()
        fig2, anim2 = create_wave_interference()
        fig3, anim3 = create_pendulum_phase_space()

        plt.show()
