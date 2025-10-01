"""
LQR Controller for joint torque tracking
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from .base import BaseController


class TorqueLQR(BaseController):
    """LQR for torque tracking with integral action"""

    def __init__(self, n_joints=2, dt=0.002):
        super().__init__("Torque-LQR")

        self.n_joints = n_joints
        self.dt = dt
        self.step_count = 0

        # For direct torque control, we use simple proportional-integral (PI) gains
        # LQR formulation: state = [error, integral_error]
        # Dynamics: e[k+1] = e[k] + dt*(u_des - u_applied)
        #           int[k+1] = int[k] + dt*e[k]

        A = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [self.dt, 0, 1, 0],
            [0, self.dt, 0, 1]
        ])

        B = np.array([
            [self.dt, 0],
            [0, self.dt],
            [0, 0],
            [0, 0]
        ])

        self.A_d = A
        self.B_d = B

        # Cost matrices - very low proportional gain for direct torque control
        Q = np.diag([0.1, 0.1, 0.05, 0.05])
        R = np.diag([1.0, 1.0])

        # Solve discrete LQR
        P = solve_discrete_are(self.A_d, self.B_d, Q, R)
        self.K = np.linalg.inv(R + self.B_d.T @ P @ self.B_d) @ (self.B_d.T @ P @ self.A_d)

        self.integral = np.zeros(n_joints)

        print(f"Torque LQR Controller initialized")
        print(f"  Gain K: {self.K}")

    def reset(self):
        self.integral = np.zeros(self.n_joints)
        self.step_count = 0

    def control(self, measured_torque, desired_torque, gravity_comp=None):
        error = desired_torque - measured_torque

        state = np.concatenate([error, self.integral])
        feedback = self.K @ state  # Positive feedback to correct error

        # Feedforward + feedback
        control = desired_torque + feedback


        # Anti-windup: only integrate if control is not saturated
        if np.all(np.abs(control) < 19.5):  # Not near saturation limits
            self.integral += error * self.dt
            self.integral = np.clip(self.integral, -10.0, 10.0)

        return control