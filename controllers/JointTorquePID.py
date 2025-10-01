"""
PID Controller for joint torque tracking
"""

import numpy as np
from .base import BaseController


class TorquePID(BaseController):
    """PID controller for tracking desired joint torques"""
    
    def __init__(self, n_joints=2, kp=0.5, ki=0.1, kd=0.05, dt=0.002):
        super().__init__("Torque-PID")

        self.n_joints = n_joints
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        # Per-joint PID state
        self.integral = np.zeros(n_joints)
        self.prev_error = np.zeros(n_joints)
        self.step_count = 0

        print(f"Torque PID Controller initialized")
        print(f"  Joints: {n_joints}")
        print(f"  Gains: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def reset(self):
        self.integral = np.zeros(self.n_joints)
        self.prev_error = np.zeros(self.n_joints)
        self.step_count = 0
    
    def control(self, measured_torque, desired_torque, gravity_comp=None):
        """
        Compute motor commands to track desired torque

        Args:
            measured_torque: Current joint torques
            desired_torque: Desired joint torques
            gravity_comp: Feedforward gravity compensation torques

        Returns:
            motor_commands: Motor torque commands
        """
        error = desired_torque - measured_torque

        # Derivative on measurement to avoid derivative kick
        derivative = -(measured_torque - self.prev_error) / self.dt
        self.prev_error = measured_torque.copy()

        # PID feedback correction
        feedback = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Feedforward + feedback for direct torque control
        control = desired_torque + feedback

        # Add gravity compensation if provided
        if gravity_comp is not None:
            control += gravity_comp

        # Anti-windup: only integrate when not saturated
        if np.all(np.abs(control) < 19.5):
            self.integral += error * self.dt
            self.integral = np.clip(self.integral, -10.0, 10.0)

        return control