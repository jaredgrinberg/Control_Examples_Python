"""
Main script to run torque tracking controllers on 2-link arm
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from controllers import TorquePID, TorqueLQR, TorqueMPC, TorqueRL
from utils import simulate_arm_torque_tracking, replay_arm_in_mujoco, plot_torque_tracking


def create_output_dir(controller_name):
    """Create timestamped output directory"""
    base_dir = Path("simulations")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"Arm_{controller_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sinusoidal_reference(t):
    """Sinusoidal torque reference trajectory"""
    shoulder = 5.0 * np.sin(2 * np.pi * 0.5 * t)
    elbow = 3.0 * np.sin(2 * np.pi * 0.7 * t)
    return np.array([shoulder, elbow])


def aggressive_reference(t):
    """Aggressive torque reference that exceeds limits (for MPC testing)"""
    shoulder = 25.0 * np.sin(2 * np.pi * 0.5 * t)  # Exceeds +-20 Nm limit
    elbow = 18.0 * np.sin(2 * np.pi * 0.7 * t)     # Occasionally exceeds
    return np.array([shoulder, elbow])


def run_pid(duration=5.0, show_results=False):
    """Run PID torque tracking"""
    print("=" * 60)
    print("Running PID Torque Tracking")
    print("=" * 60)
    
    output_dir = create_output_dir("PID")
    print(f"Output directory: {output_dir}")
    
    controller = TorquePID(n_joints=2, kp=0.7, ki=0.2, kd=0.0)
    
    print("\nSimulating...")
    trajectory, model, data = simulate_arm_torque_tracking(
        controller=controller,
        model_path="environments/two_link_arm.xml",
        torque_references=sinusoidal_reference,
        duration=duration
    )
    
    print("\nSaving results...")
    plot_torque_tracking(trajectory, "PID", output_subdir=output_dir, show=show_results)
    replay_arm_in_mujoco(trajectory, "environments/two_link_arm.xml", 
                         output_subdir=output_dir, show_viewer=show_results)
    
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

def run_lqr(duration=5.0, show_results=False):
    """Run LQR torque tracking"""
    print("=" * 60)
    print("Running LQR Torque Tracking")
    print("=" * 60)
    
    output_dir = create_output_dir("LQR")
    print(f"Output directory: {output_dir}")
    
    controller = TorqueLQR(n_joints=2)
    
    print("\nSimulating...")
    trajectory, model, data = simulate_arm_torque_tracking(
        controller=controller,
        model_path="environments/two_link_arm.xml",
        torque_references=sinusoidal_reference,
        duration=duration
    )
    
    print("\nSaving results...")
    plot_torque_tracking(trajectory, "LQR", output_subdir=output_dir, show=show_results)
    replay_arm_in_mujoco(trajectory, "environments/two_link_arm.xml", 
                         output_subdir=output_dir, show_viewer=show_results)
    
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def run_mpc(duration=5.0, show_results=False):
    """Run MPC torque tracking with aggressive reference"""
    print("=" * 60)
    print("Running MPC Torque Tracking (Constrained)")
    print("=" * 60)

    output_dir = create_output_dir("MPC")
    print(f"Output directory: {output_dir}")

    controller = TorqueMPC(n_joints=2, horizon=20)

    print("\nSimulating...")
    print("NOTE: Using aggressive reference (±25/±18 Nm) that exceeds ±20 Nm limits")
    print("MPC should optimally handle saturation via prediction horizon")

    trajectory, model, data = simulate_arm_torque_tracking(
        controller=controller,
        model_path="environments/two_link_arm.xml",
        torque_references=aggressive_reference,
        duration=duration
    )

    print("\nSaving results...")
    plot_torque_tracking(trajectory, "MPC", output_subdir=output_dir, show=show_results)
    replay_arm_in_mujoco(trajectory, "environments/two_link_arm.xml",
                         output_subdir=output_dir, show_viewer=show_results)

    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def run_rl(duration=5.0, show_results=False, model_path=None, algorithm='PPO'):
    """Run RL torque tracking (trains if no model provided)"""
    print("=" * 60)
    print(f"Running {algorithm} Torque Tracking (with Disturbances)")
    print("=" * 60)

    output_dir = create_output_dir(f"RL_{algorithm}")
    print(f"Output directory: {output_dir}")

    # Auto-detect existing model if no path specified
    if model_path is None:
        auto_path = Path(f"trained_models/{algorithm.lower()}_arm_torque.zip")
        if auto_path.exists():
            print(f"Found existing model: {auto_path}")
            model_path = str(auto_path)

    # Create controller (will train if model doesn't exist)
    controller = TorqueRL(n_joints=2, model_path=model_path, train_steps=200000, algorithm=algorithm)

    print("\nSimulating...")
    print("NOTE: Regular sinusoidal reference (±5/±3 Nm)")
    print("RL was trained with random disturbances, testing without them here")

    trajectory, model, data = simulate_arm_torque_tracking(
        controller=controller,
        model_path="environments/two_link_arm.xml",
        torque_references=sinusoidal_reference,
        duration=duration
    )

    print("\nSaving results...")
    plot_torque_tracking(trajectory, "RL", output_subdir=output_dir, show=show_results)
    replay_arm_in_mujoco(trajectory, "environments/two_link_arm.xml",
                         output_subdir=output_dir, show_viewer=show_results)

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
        elif controller == "rl" or controller == "rl-ppo":
            run_rl(show_results=show_results, algorithm='PPO')
        elif controller == "rl-sac":
            run_rl(show_results=show_results, algorithm='SAC')
        elif controller == "pid":
            run_pid(show_results=show_results)
        else:
            print(f"Unknown controller: {controller}")
            print("Usage: python main_arm.py [pid|lqr|mpc|rl|rl-sac] [--show]")
    else:
        run_pid(show_results=show_results)