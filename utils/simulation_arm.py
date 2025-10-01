"""
Simulation utilities for 2-link arm torque tracking
"""

import numpy as np
import mujoco


def simulate_arm_torque_tracking(controller, model_path, torque_references, duration, dt=0.002):
    """
    Simulate 2-link arm with torque tracking controller
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    num_steps = int(duration / dt)
    trajectory = []
    
    controller.reset()

    print(f"\nSimulating {controller.name} for {duration}s...")

    for step in range(num_steps):
        t = step * dt

        # Get desired torques from reference
        desired_torque = torque_references(t)

        # Compute control (using previous applied torque as measurement)
        measured_torque = data.ctrl[:2].copy()

        # Handle different controller types
        if hasattr(controller, 'horizon'):
            # MPC: needs future desired torques
            horizon = controller.horizon
            future_desired = np.array([torque_references(t + k*dt) for k in range(horizon)])
            control = controller.control(measured_torque, desired_torque,
                                        gravity_comp=None, future_desired=future_desired)
        elif controller.name == "Torque-RL":
            # RL: needs joint states
            joint_pos = data.qpos[:2].copy()
            joint_vel = data.qvel[:2].copy()
            control = controller.control(measured_torque, desired_torque,
                                        joint_pos, joint_vel, gravity_comp=None)
        else:
            # PID/LQR: standard interface
            control = controller.control(measured_torque, desired_torque, gravity_comp=None)

        # Use the clipped control as the "measured" torque for feedback
        # This represents what actually gets applied to the system
        actual_applied = np.clip(control, -20.0, 20.0)
        data.ctrl[:] = actual_applied

        # Step simulation
        mujoco.mj_step(model, data)

        # Store data
        trajectory.append({
            'time': t,
            'joint_pos': data.qpos.copy(),
            'joint_vel': data.qvel.copy(),
            'measured_torque': measured_torque.copy(),
            'desired_torque': desired_torque.copy(),
            'control': actual_applied.copy()
        })

        # Print progress
        if step % 500 == 0:
            print(f"t={t:.2f}s: q=[{data.qpos[0]:.2f}, {data.qpos[1]:.2f}] shoulder={measured_torque[0]:.2f}Nm (des={desired_torque[0]:.2f}), elbow={measured_torque[1]:.2f}Nm (des={desired_torque[1]:.2f})")
    
    print(f"Simulation complete. {len(trajectory)} steps")
    return trajectory, model, data