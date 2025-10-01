"""
MPC Controller for Cartpole
"""

import numpy as np
from scipy.optimize import minimize
from .base import BaseController


class CartpoleMPC(BaseController):
    """Model Predictive Control for cartpole"""
    
    def __init__(self, horizon=20, dt=0.02, m_cart=1.0, m_pole=0.1, length=0.5):
        super().__init__("MPC")
        
        self.horizon = horizon  # Prediction horizon
        self.dt = dt
        self.m_c = m_cart
        self.m_p = m_pole
        self.l = length
        self.g = 9.81
        
        # Cost weights
        self.Q = np.diag([10.0, 100.0, 1.0, 1.0])  # State cost
        self.R = 0.01  # Control cost
        
        print(f"MPC Controller initialized")
        print(f"  Horizon: {horizon} steps")
        print(f"  State weights: {np.diag(self.Q)}")
        print(f"  Control weight: {self.R}")
    
    def _dynamics(self, state, u):
        """Cartpole dynamics"""
        x, theta, dx, dtheta = state
        mc, mp, l, g = self.m_c, self.m_p, self.l, self.g
        
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        
        ddx = (u + mp * sin_th * (l * dtheta**2 + g * cos_th)) / (mc + mp * sin_th**2)
        ddtheta = (-u * cos_th - mp * l * dtheta**2 * cos_th * sin_th - (mc + mp) * g * sin_th) / (l * (mc + mp * sin_th**2))
        
        return np.array([dx, dtheta, ddx, ddtheta])
    
    def _integrate(self, state, u):
        """RK4 integration"""
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self._dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self._dynamics(state + self.dt * k3, u)
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _cost_function(self, u_sequence, current_state, target_state):
        """
        Cost function for optimization
        
        Args:
            u_sequence: Control sequence over horizon
            current_state: Current state
            target_state: Desired state
        """
        state = current_state.copy()
        total_cost = 0.0
        
        # Simulate forward over horizon
        for i in range(self.horizon):
            u = u_sequence[i]
            
            # State cost
            error = state - target_state
            state_cost = error.T @ self.Q @ error
            
            # Control cost
            control_cost = self.R * u**2
            
            total_cost += state_cost + control_cost
            
            # Propagate dynamics
            state = self._integrate(state, u)
        
        # Terminal cost
        error = state - target_state
        total_cost += error.T @ self.Q @ error
        
        return total_cost
    
    def control(self, state, target=None):
        """
        Compute MPC control
        
        Args:
            state: Current state [x, theta, x_dot, theta_dot]
            target: Desired state (default: upright at origin)
        """
        if target is None:
            target = np.zeros(4)
        
        # Initial guess: zero control
        u0 = np.zeros(self.horizon)
        
        # Bounds on control: -100 to 100 N
        bounds = [(-100, 100) for _ in range(self.horizon)]
        
        # Optimize
        result = minimize(
            self._cost_function,
            u0,
            args=(state, target),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )
        
        # Return first control action
        return result.x[0]
    
    def simulate_step(self, state, u):
        """Simulate one step using RK4"""
        return self._integrate(state, u)