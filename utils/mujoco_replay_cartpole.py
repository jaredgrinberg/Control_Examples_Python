"""
MuJoCo visualization and video recording
"""

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import imageio
from pathlib import Path
from datetime import datetime
import time


def replay_cartpole_in_mujoco(trajectory, dt=0.02, save_video=True, camera_config=None, output_subdir=None, show_viewer=False):
    """
    Replay trajectory in MuJoCo viewer with video recording
    
    Args:
        trajectory: List of dicts with 'state' key
        dt: Time step for playback
        save_video: Whether to save video
        camera_config: Dict with camera settings (optional)
        output_subdir: Subdirectory path to save video (Path object)
        show_viewer: Whether to show interactive viewer (default: False, just saves)
    """
    env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
    env.reset()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Setup video
    writer = None
    if save_video and output_subdir:
        filepath = output_subdir / "simulation.mp4"
        writer = imageio.get_writer(str(filepath), fps=int(1/dt))
    
    # Default camera settings
    if camera_config is None:
        camera_config = {
            'azimuth': 90,
            'elevation': 0,
            'distance': 3.0,
            'lookat': [0, 0, 0.5]
        }
    
    if show_viewer:
        # Interactive viewer with real-time playback
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = camera_config['azimuth']
            viewer.cam.elevation = camera_config['elevation']
            viewer.cam.distance = camera_config['distance']
            viewer.cam.lookat[:] = camera_config['lookat']
            
            for step_data in trajectory:
                state = step_data['state']
                
                data.qpos[0] = state[0]
                data.qpos[1] = state[1]
                data.qvel[0] = state[2]
                data.qvel[1] = state[3]
                
                mujoco.mj_forward(model, data)
                
                frame = env.render()
                if writer:
                    writer.append_data(frame)
                viewer.sync()
                
                time.sleep(dt)
    else:
        # Headless: just save video without interactive viewer
        for step_data in trajectory:
            state = step_data['state']
            
            data.qpos[0] = state[0]
            data.qpos[1] = state[1]
            data.qvel[0] = state[2]
            data.qvel[1] = state[3]
            
            mujoco.mj_forward(model, data)
            
            frame = env.render()
            if writer:
                writer.append_data(frame)
    
    if writer:
        writer.close()
        print(f"Video saved: {filepath}")
    
    env.close()