"""
RL Controller for Cartpole using Stable-Baselines3
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
from .base import BaseController


class CartpoleRL(BaseController):
    """RL controller using PPO from Stable-Baselines3"""
    
    def __init__(self, model_path=None, train_steps=200000):
        super().__init__("RL-PPO")
        
        self.train_steps = train_steps
        self.model = None
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            print(f"Loading trained model from {model_path}")
            self.model = PPO.load(model_path)
        else:
            print(f"Training new PPO model for {train_steps} steps...")
            self._train_model()
    
    def _train_model(self):
        """Train PPO agent starting from larger angles"""
        
        def make_large_angle_env():
            """Wrapper that starts from larger initial angles"""
            env = gym.make("InvertedPendulum-v5")
            
            class LargeAngleWrapper(gym.Wrapper):
                def reset(self, **kwargs):
                    obs, info = self.env.reset(**kwargs)
                    # Start from larger angles: uniformly sample from -1.3 to 1.3 rad (~75 degrees)
                    angle = np.random.uniform(-1.3, 1.3)
                    self.env.unwrapped.set_state(
                        np.array([0.0, angle]),
                        np.array([0.0, 0.0])
                    )
                    return self.env.unwrapped._get_obs(), info
            
            return LargeAngleWrapper(env)
        
        env = make_vec_env(make_large_angle_env, n_envs=4)
        
        # Create PPO model
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
        )
        
        # Train
        print("Training PPO agent from large initial angles (~75 degrees)...")
        self.model.learn(total_timesteps=self.train_steps)
        
        # Save model
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "ppo_cartpole.zip"
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        env.close()
    
    def control(self, state, target=None):
        """Compute control using trained RL policy"""
        action, _ = self.model.predict(state, deterministic=True)
        return action[0]
    
    def simulate_step(self, state, u):
        """For RL, we don't use custom dynamics"""
        raise NotImplementedError("RL uses gym environment directly")