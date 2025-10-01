"""
Main script to run different controllers on cartpole
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from controllers import CartpoleLQR, CartpoleRL, CartpoleMPC
from utils import simulate_cartpole, replay_cartpole_in_mujoco, plot_trajectory


def create_output_dir(controller_name):
    """Create timestamped output directory for a run"""
    base_dir = Path("simulations")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{controller_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_lqr(duration=10.0, initial_angle=0.3, show_results=False):
    """Run LQR controller"""
    print("=" * 60)
    print("Running LQR Controller")
    print("=" * 60)
    
    output_dir = create_output_dir("LQR")
    print(f"Output directory: {output_dir}")
    
    controller = CartpoleLQR()
    initial_state = [0.0, initial_angle, 0.0, 0.0]
    
    print("\nComputing trajectory...")
    trajectory = simulate_cartpole(controller, initial_state, duration)
    
    print("\nSaving results...")
    plot_trajectory(trajectory, "LQR", output_subdir=output_dir, show=show_results)
    replay_cartpole_in_mujoco(trajectory, output_subdir=output_dir, show_viewer=show_results)
    
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def run_mpc(duration=10.0, initial_angle=1.3, show_results=False):
    """Run MPC controller"""
    print("=" * 60)
    print("Running MPC Controller")
    print("=" * 60)
    
    output_dir = create_output_dir("MPC")
    print(f"Output directory: {output_dir}")
    
    controller = CartpoleMPC(horizon=20)
    initial_state = [0.0, initial_angle, 0.0, 0.0]
    
    print("\nComputing trajectory...")
    trajectory = simulate_cartpole(controller, initial_state, duration)
    
    print("\nSaving results...")
    plot_trajectory(trajectory, "MPC", output_subdir=output_dir, show=show_results)
    replay_cartpole_in_mujoco(trajectory, output_subdir=output_dir, show_viewer=show_results)
    
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def run_rl(duration=10.0, show_results=False, train_steps=200000):
    """Run RL controller"""
    print("=" * 60)
    print("Running RL Controller")
    print("=" * 60)
    
    output_dir = create_output_dir("RL")
    print(f"Output directory: {output_dir}")
    
    # Use large angle model
    model_path = Path("trained_models/ppo_cartpole.zip")
    controller = CartpoleRL(model_path=model_path if model_path.exists() else None, 
                           train_steps=train_steps)
    
    print("\nComputing trajectory...")
    trajectory = simulate_cartpole(controller, None, duration)
    
    print("\nSaving results...")
    plot_trajectory(trajectory, "RL-PPO", output_subdir=output_dir, show=show_results)
    replay_cartpole_in_mujoco(trajectory, output_subdir=output_dir, show_viewer=show_results)
    
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    show_results = "--show" in sys.argv

    if len(sys.argv) > 1:
        controller = sys.argv[1]
        if controller == "lqr":
            run_lqr(show_results=show_results)
        elif controller == "mpc":
            run_mpc(show_results=show_results)
        elif controller == "rl":
            run_rl(show_results=show_results)
        else:
            print(f"Unknown controller: {controller}")
            print("Usage: python main_cartpole.py [lqr|mpc|rl] [--show]")
    else:
        run_lqr(show_results=show_results)