import numpy as np
import matplotlib.pyplot as plt
import torch
from pink_noise.pink_noise_generator import generate_pink_noise_fft

class StephanModelEq5:
    def __init__(self, V0=0.04, theta0=40.3, epsilon=1, k=0.64, E0=0.4, TE=0.04,
                 r0=25.0, gamma=0.32, tau=1.0, alpha=0.32, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize parameters as tensors
        self.V0 = torch.tensor(V0, device=self.device, dtype=torch.float32)
        self.theta0 = torch.tensor(theta0, device=self.device, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, device=self.device, dtype=torch.float32)
        self.k = torch.tensor(k, device=self.device, dtype=torch.float32)
        self.E0 = torch.tensor(E0, device=self.device, dtype=torch.float32)
        self.TE = torch.tensor(TE, device=self.device, dtype=torch.float32)
        self.r0 = torch.tensor(r0, device=self.device, dtype=torch.float32)
        self.gamma = torch.tensor(gamma, device=self.device, dtype=torch.float32)
        self.tau = torch.tensor(tau, device=self.device, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)

        # Derived parameters
        self.k1 = 4.3 * self.theta0 * self.E0 * self.TE
        self.k2 = self.epsilon * self.r0 * self.E0 * self.TE
        self.k3 = 1 - self.epsilon

    def odes(self, state, u):
        """
        Compute derivatives for all state variables.
        state: Tensor [s, f, v, q]
        u: External input (pink noise)
        Returns: Derivative of state variables [ds, df, dv, dq]
        """
        s, f, v, q = state
        ds = -self.k * s - self.gamma * (f - 1) + u
        df = s
        dv = (f - v ** (1 / self.alpha)) / self.tau
        dq = (f * (1 - (1 - self.E0) ** (1 / f)) / self.E0 - (q * v ** (1 / self.alpha))) / self.tau
        return torch.stack([ds, df, dv, dq])

    def y(self, q, v):
        """
        Compute the BOLD signal.
        """
        return self.V0 * (self.k1 * (1 - q) + self.k2 * (1 - q / v) + self.k3 * (1 - v))


# RK4 Solver
def rk4_step(func, x, dt, *args):
    k1 = func(x, *args)
    k2 = func(x + dt * k1 / 2, *args)
    k3 = func(x + dt * k2 / 2, *args)
    k4 = func(x + dt * k3, *args)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)



