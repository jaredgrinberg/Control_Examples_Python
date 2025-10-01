"""
Simulation utilities
"""

import numpy as np
import gymnasium as gym


def simulate_cartpole(controller, initial_state, duration, dt=0.02):
    """
    Simulate cartpole with a given controller
    
    Args:
        controller: Controller object with control() method
        initial_state: Initial state [x, theta, x_dot, theta_dot]
        duration: Simulation time in seconds
        dt: Time step
        
    Returns:
        trajectory: List of dicts with 'state', 'control', 'time'
    """
    num_steps = int(duration / dt)
    trajectory = []
    
    # Check if this is an RL controller
    if controller.name == "RL-PPO":
        # Use gym environment directly for RL
        env = gym.make("InvertedPendulum-v5")
        state, _ = env.reset()  # Start from environment's default
        
        # DO NOT set custom initial state for RL; The agent was trained on random resets, not specific states
        
        controller.reset()
        
        print(f"Starting RL simulation from default state: {state}")
        print(f"Initial angle: {np.degrees(state[1]):.1f} degrees")
        
        for t in range(num_steps):
            # Compute control
            u = controller.control(state)
            
            # Store
            trajectory.append({
                'state': state.copy(),
                'control': u,
                'time': t * dt
            })
            
            # Step environment
            state, reward, terminated, truncated, _ = env.step([u])
            
            if terminated or truncated:
                print(f"Episode terminated at step {t}, time={t*dt:.2f}s")
                print(f"Final state: {state}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                break
            
            # Print progress
            if t % 50 == 0:
                x, theta, dx, dtheta = state
                print(f"t={t*dt:.2f}s: x={x:.3f}m, theta={np.degrees(theta):.1f}deg, u={u:.2f}N")
        
        print(f"Total trajectory length: {len(trajectory)} steps")
        env.close()
    else:
        # Use custom dynamics for LQR/MPC
        state = np.array(initial_state)
        controller.reset()
        
        for t in range(num_steps):
            # Compute control
            u = controller.control(state)
            
            # Store
            trajectory.append({
                'state': state.copy(),
                'control': u,
                'time': t * dt
            })
            
            # Simulate
            if hasattr(controller, 'simulate_step'):
                state = controller.simulate_step(state, u)
            else:
                state = simple_cartpole_step(state, u, dt)
            
            # Print progress
            if t % 50 == 0:
                x, theta, dx, dtheta = state
                print(f"t={t*dt:.2f}s: x={x:.3f}m, theta={np.degrees(theta):.1f}deg, u={u:.2f}N")
    
    return trajectory


def simple_cartpole_step(state, u, dt, m_c=1.0, m_p=0.1, l=0.5, g=9.81):
    """Simple Euler integration for cartpole"""
    x, theta, dx, dtheta = state
    
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    
    ddx = (u + m_p * sin_th * (l * dtheta**2 + g * cos_th)) / (m_c + m_p * sin_th**2)
    ddtheta = (-u * cos_th - m_p * l * dtheta**2 * cos_th * sin_th - (m_c + m_p) * g * sin_th) / (l * (m_c + m_p * sin_th**2))
    
    return state + dt * np.array([dx, dtheta, ddx, ddtheta])