import numpy as np
import torch

from pink_noise.pink_noise_generator import generate_pink_noise_fft


import torch

class PDCMMODEL:
    def __init__(self, w=1.0, sigma=0.5, mu=0.4, lamb=0.2, c=1,
                 varphi=0.6, phi=1.5, chi=0.6, mtt=2.0, tau=4.0, alpha=0.32,
                 E0=0.4, V0=0.04, epsilon=1, theta0=40.3, r0=15.0, TE=0.04,
                 ignore_range=False, cross_valid=False, device=None):
        """
        Initializes the BOLD model with default parameters, validating ranges if specified.
        All parameters are stored as attributes.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.w = torch.tensor(w, device=self.device, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, device=self.device, dtype=torch.float32) #
        self.mu = torch.tensor(mu, device=self.device, dtype=torch.float32)
        self.lamb = torch.tensor(lamb, device=self.device, dtype=torch.float32)
        self.c = torch.tensor(c, device=self.device, dtype=torch.float32)
        self.phi = torch.tensor(phi, device=self.device, dtype=torch.float32)
        self.varphi = torch.tensor(varphi, device=self.device, dtype=torch.float32)
        self.chi = torch.tensor(chi, device=self.device, dtype=torch.float32)
        self.mtt = torch.tensor(mtt, device=self.device, dtype=torch.float32)
        self.tau = torch.tensor(tau, device=self.device, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
        self.E0 = torch.tensor(E0, device=self.device, dtype=torch.float32)
        self.V0 = torch.tensor(V0, device=self.device, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, device=self.device, dtype=torch.float32)
        self.theta0 = torch.tensor(theta0, device=self.device, dtype=torch.float32)
        self.r0 = torch.tensor(r0, device=self.device, dtype=torch.float32)
        self.TE = torch.tensor(TE, device=self.device, dtype=torch.float32)

        if not ignore_range:
            self._validate_params(cross_valid)

    def _validate_params(self, cross_valid):
        # Define acceptable ranges for some parameters.
        if not (0.1 <= self.sigma <= 1.5):
            raise ValueError(f"sigma out of range: (0.1, 1.5)")
        if not (0 <= self.mu <= 1.5) and not (cross_valid):
            raise ValueError(f"mu out of range: (0, 1.5)")
        if not (0 <= self.lamb <= 0.3):
            raise ValueError(f"lamb out of range: (0, 0.3)")
        if not (1 <= self.mtt <= 5):
            raise ValueError(f"mtt out of range: (1, 5)")
        if not (0 <= self.tau <= 30):
            raise ValueError(f"tau out of range: (0, 30)")
        if not (0.1291 <= self.epsilon <= 1.312):
            raise ValueError(f"epsilon out of range: (0.1291, 1.312)")
        if not (0.03 <= self.TE <= 0.045):
            raise ValueError(f"TE out of range: (0.03, 0.045)")

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                # If the parameter is already a torch.Tensor, assign it directly
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(self.device))
                else:
                    setattr(self, key, torch.tensor(value, device=self.device, dtype=torch.float32))
            else:
                raise KeyError(f"Invalid parameter: {key}")

    def to(self, device):
        """Moves all parameter tensors to the specified device."""
        self.device = device
        self.w = self.w.to(device)
        self.sigma = self.sigma.to(device)
        self.mu = self.mu.to(device)
        self.lamb = self.lamb.to(device)
        self.c = self.c.to(device)
        self.phi = self.phi.to(device)
        self.varphi = self.varphi.to(device)
        self.chi = self.chi.to(device)
        self.mtt = self.mtt.to(device)
        self.tau = self.tau.to(device)
        self.alpha = self.alpha.to(device)
        self.E0 = self.E0.to(device)
        self.V0 = self.V0.to(device)
        self.epsilon = self.epsilon.to(device)
        self.theta0 = self.theta0.to(device)
        self.r0 = self.r0.to(device)
        self.TE = self.TE.to(device)
        return self

    def sti_u(self, t):
        """
        Returns a rectangular stimulus function u(t) that is 1 between t=1 and t=w+1.
        """
        # Use the attribute directly.
        t_tensor = torch.tensor(t, dtype=torch.float32)
        condition = (t_tensor >= 1) & (t_tensor <= (self.w + 1))
        return torch.where(condition, torch.tensor(1.0), torch.tensor(0.0))

    def sti_pink(self, n=10000, T=0.01, alpha=1.0, beta=1.0, scaling_factor=0.005, normalize=True):
        """
        Generates pink noise stimulus using an FFT-based method.
        """
        from pink_noise.pink_noise_generator import generate_pink_noise_fft
        pink_noise, _ = generate_pink_noise_fft(
            n=n, T=T, alpha=alpha, beta=beta, scaling_factor=scaling_factor, normalize=normalize
        )
        return pink_noise

    def dxE(self, u, xE, xI):
        """
        Computes the rate of change of the excitatory state.
        """
        return -self.sigma * xE - self.mu * xI + self.c * u

    def dxI(self, xE, xI):
        """
        Computes the rate of change of the inhibitory state.
        """
        return self.lamb * (xE - xI)

    def da(self, a, xE):
        """
        Computes the rate of change of the autoregulatory signal.
        """
        return -self.varphi * a + xE

    def df(self, a, f):
        """
        Computes the rate of change of blood flow.
        """
        return self.phi * a - self.chi * (f - 1)

    def dv(self, f, fout):
        """
        Computes the rate of change of blood volume.
        """
        return (f - fout) / self.mtt

    def dq(self, f, E, fout, q, v):
        """
        Computes the rate of change of deoxyhemoglobin content.
        """
        return (f * (E / self.E0) - fout * (q / v)) / self.mtt

    def E(self, f):
        return 1 - (1 - self.E0) ** (1 / f)

    def fout(self, v, f=1.0, couple=False):
        if couple:
            return v ** (1 / self.alpha)
        return (1 / (self.mtt + self.tau)) * (self.mtt * v ** (1 / self.alpha) + self.tau * f)

    def y(self, q, v):
        """
        Computes the BOLD signal.
        """
        k1 = 4.3 * self.theta0 * self.E0 * self.TE
        k2 = self.epsilon * self.r0 * self.E0 * self.TE
        k3 = 1 - self.epsilon
        return self.V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

