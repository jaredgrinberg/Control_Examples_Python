"""
Gymnasium environment for training RL torque tracking controller

Features:
- MuJoCo 2-link arm simulation
- Sinusoidal torque reference tracking
- Random external disturbances (forces on end-effector)
- Reward based on tracking error and control effort
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from pathlib import Path


class ArmTorqueTrackingEnv(gym.Env):
    """
    Environment for learning torque tracking with disturbance rejection

    State: [desired_torque (2), measured_torque (2), error (2), integral (2),
            joint_pos (2), joint_vel (2), error_history (10)]
    Action: Motor commands [-20, 20] Nm (2,)
    Reward: -||error||^2 - lambda*||action - desired||^2
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 model_path="environments/two_link_arm.xml",
                 max_episode_steps=2500,  # 5 seconds at dt=0.002
                 disturbance_prob=0.1,     # 10% chance per step
                 disturbance_magnitude=10.0,  # Max 10N force
                 disturbance_duration=100,    # ~200ms
                 track_error_weight=1.0,
                 control_effort_weight=0.01):

        super().__init__()

        # Load MuJoCo model
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # Episode settings
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Disturbance settings
        self.disturbance_prob = disturbance_prob
        self.disturbance_magnitude = disturbance_magnitude
        self.disturbance_duration = disturbance_duration
        self.disturbance_remaining = 0
        self.current_disturbance = np.zeros(3)  # [fx, fy, fz]

        # Reward weights
        self.track_error_weight = track_error_weight
        self.control_effort_weight = control_effort_weight

        # State tracking
        self.integral = np.zeros(2)
        self.error_history = np.zeros((5, 2))  # Last 5 errors
        self.previous_control = np.zeros(2)

        # Reference trajectory parameters (sinusoidal)
        self.ref_freq_shoulder = 0.5  # Hz
        self.ref_freq_elbow = 0.7     # Hz
        self.ref_amp_shoulder = 5.0   # Nm
        self.ref_amp_elbow = 3.0      # Nm

        # Action space: joint torques [-20, 20] Nm
        self.action_space = spaces.Box(
            low=-20.0, high=20.0, shape=(2,), dtype=np.float32
        )

        # Observation space: [desired(2), measured(2), error(2), integral(2),
        #                     pos(2), vel(2), history(10)] = 22 dimensions
        obs_dim = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        print(f"ArmTorqueTrackingEnv initialized")
        print(f"  Episode length: {max_episode_steps} steps ({max_episode_steps*self.dt:.1f}s)")
        print(f"  Disturbance: {disturbance_prob*100:.0f}% chance, {disturbance_magnitude}N max")

    def _get_desired_torque(self, t):
        """Generate sinusoidal reference torque"""
        shoulder = self.ref_amp_shoulder * np.sin(2 * np.pi * self.ref_freq_shoulder * t)
        elbow = self.ref_amp_elbow * np.sin(2 * np.pi * self.ref_freq_elbow * t)
        return np.array([shoulder, elbow], dtype=np.float32)

    def _apply_disturbance(self):
        """Apply random external force to end-effector"""
        # Check if we should start a new disturbance
        if self.disturbance_remaining <= 0 and np.random.rand() < self.disturbance_prob:
            # Start new disturbance
            self.disturbance_remaining = self.disturbance_duration

            # Random force direction (mostly horizontal)
            angle = np.random.uniform(0, 2*np.pi)
            magnitude = np.random.uniform(0, self.disturbance_magnitude)
            self.current_disturbance = np.array([
                magnitude * np.cos(angle),  # fx
                magnitude * np.sin(angle),  # fy
                0.0                         # fz (no vertical)
            ])

        # Apply current disturbance if active
        if self.disturbance_remaining > 0:
            # Apply force to end-effector body
            end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
            self.data.xfrc_applied[end_effector_id, :3] = self.current_disturbance
            self.disturbance_remaining -= 1
        else:
            # No disturbance
            self.data.xfrc_applied[:] = 0.0

    def _get_obs(self):
        """Get current observation"""
        t = self.current_step * self.dt
        desired_torque = self._get_desired_torque(t)
        measured_torque = self.previous_control  # What was applied
        error = desired_torque - measured_torque

        obs = np.concatenate([
            desired_torque,              # 2
            measured_torque,             # 2
            error,                       # 2
            self.integral,               # 2
            self.data.qpos[:2],          # 2 (joint positions)
            self.data.qvel[:2],          # 2 (joint velocities)
            self.error_history.flatten() # 10 (5 steps × 2 joints)
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, error, action, desired_torque):
        """Compute reward based on tracking error and control effort"""
        # Tracking error penalty
        tracking_penalty = -self.track_error_weight * np.sum(error**2)

        # Control effort penalty (deviation from desired)
        effort_penalty = -self.control_effort_weight * np.sum((action - desired_torque)**2)

        reward = tracking_penalty + effort_penalty

        return reward

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Small random initial position
        self.data.qpos[:2] = np.random.uniform(-0.1, 0.1, size=2)
        self.data.qvel[:2] = 0.0

        # Reset state tracking
        self.current_step = 0
        self.integral = np.zeros(2)
        self.error_history = np.zeros((5, 2))
        self.previous_control = np.zeros(2)
        self.disturbance_remaining = 0
        self.current_disturbance = np.zeros(3)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        """Take one step in the environment"""
        # Clip action to limits
        action = np.clip(action, -20.0, 20.0)

        # Apply control
        self.data.ctrl[:2] = action
        self.previous_control = action.copy()

        # Apply random disturbance
        self._apply_disturbance()

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get new observation
        t = self.current_step * self.dt
        desired_torque = self._get_desired_torque(t)
        measured_torque = action  # What we applied
        error = desired_torque - measured_torque

        # Update integral and history
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup

        self.error_history = np.roll(self.error_history, shift=-1, axis=0)
        self.error_history[-1] = error

        # Compute reward
        reward = self._compute_reward(error, action, desired_torque)

        # Check termination
        self.current_step += 1
        terminated = False  # No early termination for now
        truncated = self.current_step >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            'error': error,
            'desired_torque': desired_torque,
            'measured_torque': measured_torque,
            'disturbance_active': self.disturbance_remaining > 0
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment """
        # Can implement MuJoCo viewer here, TODO for later
        pass

    def close(self):
        """Clean up resources"""
        pass


if __name__ == "__main__":
    # Test the environment
    print("Testing ArmTorqueTrackingEnv...")

    env = ArmTorqueTrackingEnv()

    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, error={info['error']}, disturb={info['disturbance_active']}")

        if terminated or truncated:
            break

    print("Environment test complete!")
