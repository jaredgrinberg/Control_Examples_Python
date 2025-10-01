"""
Plotting utilities for 2-link arm torque tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_torque_tracking(trajectory, controller_name, output_subdir=None, show=False):
    """
    Plot torque tracking performance
    
    Args:
        trajectory: List of dicts with 'state', 'control', 'time' keys
        controller_name: Name for the plot title
        output_subdir: Subdirectory path to save plot (Path object)
        show: Whether to display plot interactively (default: False)
    """
    # Extract data
    times = [step['time'] for step in trajectory]
    measured = np.array([step['measured_torque'] for step in trajectory])
    desired = np.array([step['desired_torque'] for step in trajectory])
    controls = np.array([step['control'] for step in trajectory])
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'{controller_name} - Torque Tracking Performance', fontsize=16, fontweight='bold')
    
    # Shoulder torque tracking
    axes[0, 0].plot(times, desired[:, 0], 'r--', label='Desired', linewidth=2)
    axes[0, 0].plot(times, measured[:, 0], 'b-', label='Measured', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Torque (Nm)')
    axes[0, 0].set_title('Shoulder Joint Torque')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # Auto-scale with margin
    y_min, y_max = measured[:, 0].min(), measured[:, 0].max()
    y_range = y_max - y_min
    axes[0, 0].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Elbow torque tracking
    axes[0, 1].plot(times, desired[:, 1], 'r--', label='Desired', linewidth=2)
    axes[0, 1].plot(times, measured[:, 1], 'g-', label='Measured', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Torque (Nm)')
    axes[0, 1].set_title('Elbow Joint Torque')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    y_min, y_max = measured[:, 1].min(), measured[:, 1].max()
    y_range = y_max - y_min
    axes[0, 1].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Shoulder tracking error
    error_shoulder = desired[:, 0] - measured[:, 0]
    axes[1, 0].plot(times, error_shoulder, 'r-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (Nm)')
    axes[1, 0].set_title('Shoulder Tracking Error')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Elbow tracking error
    error_elbow = desired[:, 1] - measured[:, 1]
    axes[1, 1].plot(times, error_elbow, 'g-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (Nm)')
    axes[1, 1].set_title('Elbow Tracking Error')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Control signals
    axes[2, 0].plot(times, controls[:, 0], 'b-', linewidth=1.5)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Control (Nm)')
    axes[2, 0].set_title('Shoulder Control Signal')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(times, controls[:, 1], 'g-', linewidth=1.5)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Control (Nm)')
    axes[2, 1].set_title('Elbow Control Signal')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_subdir:
        filepath = output_subdir / "torque_tracking.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)