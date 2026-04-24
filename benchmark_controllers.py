"""
Benchmark all controllers and compute comparison metrics
"""

import numpy as np
from pathlib import Path
from controllers import TorquePID, TorqueLQR, TorqueMPC, TorqueRL
from utils import simulate_arm_torque_tracking


def sinusoidal_reference(t):
    """Sinusoidal torque reference trajectory"""
    shoulder = 5.0 * np.sin(2 * np.pi * 0.5 * t)
    elbow = 3.0 * np.sin(2 * np.pi * 0.7 * t)
    return np.array([shoulder, elbow])


def compute_metrics(trajectory):
    """Compute tracking performance metrics"""
    measured = np.array([step['measured_torque'] for step in trajectory])
    desired = np.array([step['desired_torque'] for step in trajectory])
    controls = np.array([step['control'] for step in trajectory])

    # Tracking errors
    errors = desired - measured

    # RMS tracking error (per joint)
    rms_error_shoulder = np.sqrt(np.mean(errors[:, 0]**2))
    rms_error_elbow = np.sqrt(np.mean(errors[:, 1]**2))
    rms_error_total = np.sqrt(np.mean(errors**2))

    # Max absolute error (per joint)
    max_error_shoulder = np.max(np.abs(errors[:, 0]))
    max_error_elbow = np.max(np.abs(errors[:, 1]))

    # Control effort (sum of squared controls)
    dt = trajectory[1]['time'] - trajectory[0]['time']
    control_effort = np.sum(controls**2) * dt

    # Average absolute control (per joint)
    avg_control_shoulder = np.mean(np.abs(controls[:, 0]))
    avg_control_elbow = np.mean(np.abs(controls[:, 1]))

    return {
        'rms_error_shoulder': rms_error_shoulder,
        'rms_error_elbow': rms_error_elbow,
        'rms_error_total': rms_error_total,
        'max_error_shoulder': max_error_shoulder,
        'max_error_elbow': max_error_elbow,
        'control_effort': control_effort,
        'avg_control_shoulder': avg_control_shoulder,
        'avg_control_elbow': avg_control_elbow,
    }


def run_benchmark(duration=10.0):
    """Run all controllers and compare performance"""
    model_path = "environments/two_link_arm.xml"
    results = {}

    print("=" * 70)
    print("BENCHMARKING ALL CONTROLLERS")
    print("=" * 70)

    # PID
    print("\n[1/5] Running PID...")
    controller = TorquePID(n_joints=2, kp=0.7, ki=0.2, kd=0.0)
    trajectory, _, _ = simulate_arm_torque_tracking(
        controller=controller,
        model_path=model_path,
        torque_references=sinusoidal_reference,
        duration=duration
    )
    results['PID'] = compute_metrics(trajectory)
    print(f"  RMS Error: {results['PID']['rms_error_total']:.4f} Nm")

    # LQR
    print("\n[2/5] Running LQR...")
    controller = TorqueLQR(n_joints=2)
    trajectory, _, _ = simulate_arm_torque_tracking(
        controller=controller,
        model_path=model_path,
        torque_references=sinusoidal_reference,
        duration=duration
    )
    results['LQR'] = compute_metrics(trajectory)
    print(f"  RMS Error: {results['LQR']['rms_error_total']:.4f} Nm")

    # MPC
    print("\n[3/5] Running MPC...")
    controller = TorqueMPC(n_joints=2, horizon=20)
    trajectory, _, _ = simulate_arm_torque_tracking(
        controller=controller,
        model_path=model_path,
        torque_references=sinusoidal_reference,
        duration=duration
    )
    results['MPC'] = compute_metrics(trajectory)
    print(f"  RMS Error: {results['MPC']['rms_error_total']:.4f} Nm")

    # PPO
    print("\n[4/5] Running RL (PPO)...")
    ppo_model = Path("trained_models/ppo_arm_torque.zip")
    if ppo_model.exists():
        controller = TorqueRL(n_joints=2, model_path=str(ppo_model), algorithm='PPO')
        trajectory, _, _ = simulate_arm_torque_tracking(
            controller=controller,
            model_path=model_path,
            torque_references=sinusoidal_reference,
            duration=duration
        )
        results['PPO'] = compute_metrics(trajectory)
        print(f"  RMS Error: {results['PPO']['rms_error_total']:.4f} Nm")
    else:
        print("  WARNING: PPO model not found, skipping...")

    # SAC
    print("\n[5/5] Running RL (SAC)...")
    sac_model = Path("trained_models/sac_arm_torque.zip")
    if sac_model.exists():
        controller = TorqueRL(n_joints=2, model_path=str(sac_model), algorithm='SAC')
        trajectory, _, _ = simulate_arm_torque_tracking(
            controller=controller,
            model_path=model_path,
            torque_references=sinusoidal_reference,
            duration=duration
        )
        results['SAC'] = compute_metrics(trajectory)
        print(f"  RMS Error: {results['SAC']['rms_error_total']:.4f} Nm")
    else:
        print("  WARNING: SAC model not found, skipping...")

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\nRMS Tracking Error (Nm):")
    print(f"{'Controller':<12} {'Shoulder':>10} {'Elbow':>10} {'Total':>10}")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['rms_error_shoulder']:>10.4f} "
              f"{metrics['rms_error_elbow']:>10.4f} {metrics['rms_error_total']:>10.4f}")

    print("\nMax Absolute Error (Nm):")
    print(f"{'Controller':<12} {'Shoulder':>10} {'Elbow':>10}")
    print("-" * 35)
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['max_error_shoulder']:>10.4f} "
              f"{metrics['max_error_elbow']:>10.4f}")

    print("\nControl Effort:")
    print(f"{'Controller':<12} {'Total Effort':>15} {'Avg Shoulder':>15} {'Avg Elbow':>15}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['control_effort']:>15.2f} "
              f"{metrics['avg_control_shoulder']:>15.4f} {metrics['avg_control_elbow']:>15.4f}")

    # Save results to file
    output_file = Path("controller_comparison_results.txt")
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CONTROLLER COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("RMS Tracking Error (Nm):\n")
        f.write(f"{'Controller':<12} {'Shoulder':>10} {'Elbow':>10} {'Total':>10}\n")
        f.write("-" * 45 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<12} {metrics['rms_error_shoulder']:>10.4f} "
                   f"{metrics['rms_error_elbow']:>10.4f} {metrics['rms_error_total']:>10.4f}\n")

        f.write("\nMax Absolute Error (Nm):\n")
        f.write(f"{'Controller':<12} {'Shoulder':>10} {'Elbow':>10}\n")
        f.write("-" * 35 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<12} {metrics['max_error_shoulder']:>10.4f} "
                   f"{metrics['max_error_elbow']:>10.4f}\n")

        f.write("\nControl Effort:\n")
        f.write(f"{'Controller':<12} {'Total Effort':>15} {'Avg Shoulder':>15} {'Avg Elbow':>15}\n")
        f.write("-" * 60 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<12} {metrics['control_effort']:>15.2f} "
                   f"{metrics['avg_control_shoulder']:>15.4f} {metrics['avg_control_elbow']:>15.4f}\n")

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_benchmark(duration=10.0)
