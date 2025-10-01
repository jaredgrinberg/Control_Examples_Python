"""
RL Controller for joint torque tracking with disturbance rejection
"""

import numpy as np
from pathlib import Path
from .base import BaseController
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
sys.path.append('.')
from environments.arm_torque_env import ArmTorqueTrackingEnv


class TorqueRL(BaseController):
    """
    RL controller for torque tracking with disturbance rejection

    Args:
        n_joints: Number of joints (default: 2)
        model_path: Path to load existing model (default: None = auto-detect or train)
        train_steps: Training steps if training new model (default: 200000)
        algorithm: 'PPO' or 'SAC' (default: 'PPO')

    Models saved to:
        - Final: trained_models/{algorithm}_arm_torque.zip
        - Checkpoints: trained_models/checkpoints/{algorithm}_arm_torque_XXXXX_steps.zip

    To resume from checkpoint:
        controller = TorqueRL(model_path='trained_models/checkpoints/ppo_arm_torque_150000_steps.zip')
    """

    def __init__(self, n_joints=2, model_path=None, train_steps=200000, algorithm='PPO'):
        super().__init__("Torque-RL")

        self.n_joints = n_joints
        self.train_steps = train_steps
        self.algorithm = algorithm.upper()
        self.model = None
        self.model_path = model_path

        # State tracking
        self.error_history = np.zeros((5, n_joints))
        self.integral = np.zeros(n_joints)
        self.dt = 0.002

        # Load or train model
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            # Detect if it's a checkpoint for resuming training
            if 'checkpoint' in str(model_path) or '_steps.zip' in str(model_path):
                print("  -> This is a checkpoint. You can continue training from here.")
            if self.algorithm == 'SAC':
                self.model = SAC.load(model_path)
            else:
                self.model = PPO.load(model_path)
        else:
            print(f"Training new {self.algorithm} model for {train_steps} steps...")
            self._train_model()

    def _train_model(self):
        """Train RL agent (PPO or SAC) for torque tracking with disturbances"""

        # Create training environment
        env = ArmTorqueTrackingEnv(
            max_episode_steps=2500,
            disturbance_prob=0.1,
            disturbance_magnitude=10.0
        )

        # Create save directory and checkpoint callback
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=str(checkpoint_dir),
            name_prefix=f"{self.algorithm.lower()}_arm_torque"
        )

        # Create model based on algorithm
        if self.algorithm == 'SAC':
            self.model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                tensorboard_log=str(save_dir / "tensorboard")
            )
        else:  # PPO
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log=str(save_dir / "tensorboard")
            )

        # Train with checkpoints
        print(f"Training {self.algorithm} for torque tracking with {self.train_steps} steps...")
        print(f"Checkpoints will be saved every 50k steps to: {checkpoint_dir}")
        print(f"Tensorboard logs: {save_dir / 'tensorboard'}")
        print(f"Run 'tensorboard --logdir {save_dir / 'tensorboard'}' to view training progress")

        self.model.learn(
            total_timesteps=self.train_steps,
            callback=checkpoint_callback,
            progress_bar=True
        )

        # Save final model
        save_path = save_dir / f"{self.algorithm.lower()}_arm_torque.zip"
        self.model.save(save_path)
        print(f"\nFinal model saved to {save_path}")

        env.close()

    def reset(self):
        """Reset controller state"""
        self.error_history = np.zeros((5, self.n_joints))
        self.integral = np.zeros(self.n_joints)

    def control(self, measured_torque, desired_torque, joint_pos, joint_vel,
                gravity_comp=None):
        """
        Compute control using trained RL policy

        Args:
            measured_torque: Current measured torques [n_joints]
            desired_torque: Current desired torques [n_joints]
            joint_pos: Joint positions (for state)
            joint_vel: Joint velocities (for state)
            gravity_comp: Not used

        Returns:
            control: RL policy output [n_joints]
        """

        # Build observation matching the environment
        error = desired_torque - measured_torque
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10, 10)

        # Update history
        self.error_history = np.roll(self.error_history, shift=-1, axis=0)
        self.error_history[-1] = error

        # Observation: [desired, measured, error, integral, pos, vel, history]
        obs = np.concatenate([
            desired_torque,
            measured_torque,
            error,
            self.integral,
            joint_pos,
            joint_vel,
            self.error_history.flatten()
        ]).astype(np.float32)

        # Get action from policy
        action, _ = self.model.predict(obs, deterministic=True)

        return action