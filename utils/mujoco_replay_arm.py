"""
MuJoCo visualization and video recording for 2-link arm
"""

import mujoco
import mujoco.viewer
import numpy as np
import imageio
from pathlib import Path
import time


def replay_arm_in_mujoco(trajectory, model_path, output_subdir=None, save_video=True, show_viewer=False, dt=0.002):
    """
    Replay arm trajectory in MuJoCo viewer with video recording
    
    Args:
        trajectory: List of dicts from simulate_arm_torque_tracking
        model_path: Path to MuJoCo XML model
        output_subdir: Where to save video
        save_video: Whether to save video
        show_viewer: Whether to show interactive viewer
        dt: Playback timestep
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Setup video
    writer = None
    if save_video and output_subdir:
        filepath = output_subdir / "arm_simulation.mp4"
        writer = imageio.get_writer(str(filepath), fps=int(1/dt))
    
    # Setup renderer for video
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    if show_viewer:
        # Interactive viewer with real-time playback
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = [0.3, 0, 1.5]
            
            for step_data in trajectory:
                data.qpos[:] = step_data['joint_pos']
                data.qvel[:] = step_data['joint_vel']
                data.ctrl[:] = step_data['control']
                
                mujoco.mj_forward(model, data)
                
                if writer:
                    renderer.update_scene(data)
                    frame = renderer.render()
                    writer.append_data(frame)
                
                viewer.sync()
                time.sleep(dt)
    else:
        # Headless: just save video without interactive viewer
        for step_data in trajectory:
            data.qpos[:] = step_data['joint_pos']
            data.qvel[:] = step_data['joint_vel']
            data.ctrl[:] = step_data['control']
            
            mujoco.mj_forward(model, data)
            
            if writer:
                renderer.update_scene(data)
                frame = renderer.render()
                writer.append_data(frame)
    
    if writer:
        writer.close()
        print(f"Video saved: {filepath}")