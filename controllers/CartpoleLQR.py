"""
LQR Controller for Cartpole
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from .base import BaseController


class CartpoleLQR(BaseController):
    """LQR controller for cartpole stabilization"""
    
    def __init__(self, m_cart=1.0, m_pole=0.1, length=0.5, dt=0.02, Q=None, R=None):
        super().__init__("LQR")
        
        self.m_c = m_cart
        self.m_p = m_pole
        self.l = length
        self.g = 9.81
        self.dt = dt
        
        # Default cost matrices if not provided
        if Q is None:
            Q = np.diag([1.0, 10.0, 1.0, 1.0])
        if R is None:
            R = np.array([[0.01]])
        
        # Linearize around upright equilibrium
        A_cont, B_cont = self._linearize_dynamics()
        
        # Discretize
        self.A = np.eye(4) + dt * A_cont
        self.B = dt * B_cont
        
        # Compute LQR gain
        P = solve_discrete_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        
        print(f"LQR Controller initialized")
        print(f"  Gain K: {self.K.flatten()}")
    
    def _linearize_dynamics(self):
        """Linearize cartpole dynamics around upright using finite differences"""
        eps = 1e-5
        x_eq = np.zeros(4)
        
        A_cont = np.zeros((4, 4))
        for i in range(4):
            x_pert = x_eq.copy()
            x_pert[i] += eps
            f1 = self._dynamics(x_pert, 0)
            f2 = self._dynamics(x_eq, 0)
            A_cont[:, i] = (f1 - f2) / eps
        
        f1 = self._dynamics(x_eq, eps)
        f2 = self._dynamics(x_eq, 0)
        B_cont = ((f1 - f2) / eps).reshape(-1, 1)
        
        return A_cont, B_cont
    
    def _dynamics(self, state, u):
        """Cartpole dynamics"""
        x, theta, dx, dtheta = state
        mc, mp, l, g = self.m_c, self.m_p, self.l, self.g
        
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        
        ddx = (u + mp * sin_th * (l * dtheta**2 + g * cos_th)) / (mc + mp * sin_th**2)
        ddtheta = (-u * cos_th - mp * l * dtheta**2 * cos_th * sin_th - (mc + mp) * g * sin_th) / (l * (mc + mp * sin_th**2))
        
        return np.array([dx, dtheta, ddx, ddtheta])
    
    def control(self, state, target=None):
        """Compute LQR control"""
        if target is None:
            target = np.zeros(4)
        
        error = target - state
        u = self.K @ error
        return u[0]
    
    def simulate_step(self, state, u):
        """Simulate one step using RK4"""
        dt = self.dt
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5 * dt * k1, u)
        k3 = self._dynamics(state + 0.5 * dt * k2, u)
        k4 = self._dynamics(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)