"""
MPC Controller for joint torque tracking with constraints
"""

import numpy as np
import cvxpy as cp
from .base import BaseController


class TorqueMPC(BaseController):
    """MPC for torque tracking with input constraints"""

    def __init__(self, n_joints=2, dt=0.002, horizon=15):
        super().__init__("Torque-MPC")

        self.n_joints = n_joints
        self.dt = dt
        self.horizon = horizon  # Prediction horizon steps

        # Torque limits (will be the main constraint to handle)
        self.torque_min = -20.0
        self.torque_max = 20.0

        # Cost weights
        self.Q = 100.0  # Tracking error weight
        self.R = 1.0    # Control effort weight

        # State: just the error (simple model for torque tracking)
        # Dynamics: error[k+1] = error[k] + dt*(desired[k+1] - desired[k]) - dt*u[k]
        # Since we have direct torque control, state propagation is simple

        print(f"Torque MPC Controller initialized")
        print(f"  Horizon: {horizon} steps ({horizon*dt*1000:.1f} ms)")
        print(f"  Torque limits: [{self.torque_min}, {self.torque_max}] Nm")

    def reset(self):
        """Reset controller state"""
        pass

    def control(self, measured_torque, desired_torque, gravity_comp=None,
                future_desired=None):
        """
        Compute optimal control using MPC

        Args:
            measured_torque: Current measured torques [n_joints]
            desired_torque: Current desired torques [n_joints]
            gravity_comp: Not used for torque tracking
            future_desired: Future desired torques [horizon, n_joints]
                           If None, assumes constant desired torque

        Returns:
            control: Optimal control for current timestep [n_joints]
        """

        # If no future trajectory provided, assume constant
        if future_desired is None:
            future_desired = np.tile(desired_torque, (self.horizon, 1))

        # Formulate QP problem using cvxpy:
        # min sum Q*(tau_des[k] - tau[k])^2 + R*u[k]^2
        # s.t. tau[k+1] = u[k]  (direct torque control)
        # torque_min ≤ u[k] =< torque_max

        # Decision variables: control sequence over horizon
        u = cp.Variable((self.horizon, self.n_joints))

        # Cost function
        cost = 0
        tau = measured_torque  # Starting state

        for k in range(self.horizon):
            # Predicted torque (direct control)
            tau_next = u[k, :]

            # Tracking error cost
            error = future_desired[k, :] - tau_next
            cost += self.Q * cp.sum_squares(error)

            # Control effort cost
            cost += self.R * cp.sum_squares(u[k, :])

            tau = tau_next

        # Constraints
        constraints = [
            u >= self.torque_min,
            u <= self.torque_max
        ]

        # Solve optimization
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if problem.status == cp.OPTIMAL:
                # Return first control in sequence (receding horizon)
                control = u.value[0, :]
            else:
                print(f"MPC solve failed: {problem.status}, using feedforward")
                control = np.clip(desired_torque, self.torque_min, self.torque_max)

        except Exception as e:
            print(f"MPC exception: {e}, using feedforward")
            control = np.clip(desired_torque, self.torque_min, self.torque_max)

        return control