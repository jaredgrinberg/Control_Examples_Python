"""
Plotting utilities for trajectory analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def plot_trajectory(trajectory, controller_name, output_subdir=None, show=False):
    """
    Plot state, control, and phase portraits
    
    Args:
        trajectory: List of dicts with 'state', 'control', 'time' keys
        controller_name: Name for the plot title
        output_subdir: Subdirectory path to save plot (Path object)
        show: Whether to display plot interactively (default: False)
    """
    # Extract data
    times = [step['time'] for step in trajectory]
    states = np.array([step['state'] for step in trajectory])
    controls = np.array([step['control'] for step in trajectory])
    
    x = states[:, 0]
    theta = np.degrees(states[:, 1])  # Convert to degrees
    x_dot = states[:, 2]
    theta_dot = states[:, 3]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'{controller_name} Controller Performance', fontsize=16, fontweight='bold')
    
    # Position vs time
    axes[0, 0].plot(times, x, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Cart Position (m)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Cart Position')
    
    # Angle vs time
    axes[0, 1].plot(times, theta, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pole Angle (deg)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Pole Angle')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Velocities
    axes[1, 0].plot(times, x_dot, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Cart Velocity (m/s)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Cart Velocity')
    
    axes[1, 1].plot(times, theta_dot, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Pole Angular Velocity')
    
    # Control input
    axes[2, 0].plot(times, controls, 'g-', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Control Force (N)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title('Control Input')
    
    # Phase portrait (angle vs angular velocity)
    axes[2, 1].plot(theta, theta_dot, 'purple', linewidth=1.5, alpha=0.7)
    axes[2, 1].scatter(theta[0], theta_dot[0], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[2, 1].scatter(theta[-1], theta_dot[-1], c='red', s=100, marker='x', label='End', zorder=5)
    axes[2, 1].set_xlabel('Pole Angle (deg)')
    axes[2, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title('Phase Portrait')
    axes[2, 1].legend()
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_subdir:
        filepath = output_subdir / "plots.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)