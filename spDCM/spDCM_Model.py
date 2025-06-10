import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

class spDCM:
    """
    Spectral Dynamic Causal Modeling (spDCM) implementation for hemodynamic modeling in fMRI.

    Attributes:
        V0 (torch.Tensor): Resting blood volume fraction.
        theta0 (torch.Tensor): Frequency-dependent weighting factor.
        phi (torch.Tensor): Scaling parameter for signal intensity.
        chi (torch.Tensor): Rate constant for autoregulatory feedback.
        epsilon (torch.Tensor): Ratio of intra-/extravascular signal contributions.
        varphi (torch.Tensor): Rate constant for flow-inducing signal decay.
        E0 (torch.Tensor): Resting oxygen extraction fraction.
        TE (torch.Tensor): Echo time (s).
        r0 (torch.Tensor): Slope of the intravascular relaxation rate.
        mtt (torch.Tensor): Mean transit time.
        tau (torch.Tensor): Time constant of autoregulatory feedback.
        alpha (torch.Tensor): Grubb's exponent.
        A (torch.Tensor): Effective connectivity matrix (complex-valued).
        device (str): Device used for computation ('cuda' or 'cpu').
        k1, k2, k3 (torch.Tensor): Model-specific constants derived from physiological parameters.
    """
    def __init__(self, V0=0.04, theta0=40.3, phi=1.5, chi=0.6, epsilon=1, varphi=0.6, E0=0.4, TE=0.04,
                 r0=15.0, mtt=2.0, tau=4, alpha=0.32, A=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.V0 = torch.tensor(V0, device=self.device, dtype=torch.float32)
        self.theta0 = torch.tensor(theta0, device=self.device, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, device=self.device, dtype=torch.float32)
        self.varphi = torch.tensor(varphi, device=self.device, dtype=torch.float32)
        self.E0 = torch.tensor(E0, device=self.device, dtype=torch.float32)
        self.TE = torch.tensor(TE, device=self.device, dtype=torch.float32)
        self.r0 = torch.tensor(r0, device=self.device, dtype=torch.float32)
        self.mtt = torch.tensor(mtt, device=self.device, dtype=torch.float32)
        self.tau = torch.tensor(tau, device=self.device, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
        self.chi = torch.tensor(chi, device=self.device, dtype=torch.float32)
        self.phi = torch.tensor(phi, device=self.device, dtype=torch.float32)

        self.k1 = torch.tensor(4.3 * theta0 * E0 * TE, device=self.device, dtype=torch.float32)
        self.k2 = torch.tensor(epsilon * r0 * E0 * TE, device=self.device, dtype=torch.float32)
        self.k3 = torch.tensor(1 - epsilon, device=self.device, dtype=torch.float32)

        self.A = torch.tensor(A, dtype=torch.complex64, device=self.device)

    def transfer_function(self, omega):
        """
        Compute the frequency-domain hemodynamic transfer function.

        Args:
            omega (float or torch.Tensor): Angular frequency (rad/s).

        Returns:
            torch.Tensor: Complex transfer function H(ω).
        """
        i = torch.tensor(1j, dtype=torch.complex64, device=self.device)
        s = i * omega

        numerator = self.V0 * self.phi * (
            (self.E0 - 1) * torch.log(1 - self.E0) * (self.k1 + self.k2) *
            (self.alpha * s * (self.mtt + self.tau) + 1)
            - self.alpha * self.E0 * (self.k1 + self.k3) * (s * self.tau + 1)
        )

        denominator = (
            self.E0 *
            (s + self.varphi) * (s + self.chi) * (s * self.tau + 1) *
            (self.alpha * s * (self.mtt + self.tau) + 1)
        )

        return numerator.to(torch.complex64) / denominator.to(torch.complex64)

    def impulse_response(self, t):
        """
        Compute the time-domain impulse response using the inverse Fourier transform of the transfer function.

        Args:
            t (np.ndarray): Time vector (seconds).

        Returns:
            np.ndarray: Impulse response in time domain.
        """
        n = len(t)
        freqs = np.fft.fftfreq(n, d=(t[1] - t[0]))  # Frequency bins
        omega = 2 * np.pi * freqs  # Angular frequencies

        # Compute transfer function at each frequency
        H_w = torch.tensor([self.transfer_function(w) for w in omega], dtype=torch.complex128)

        # Compute impulse response via inverse FFT
        impulse_response = np.fft.ifft(H_w.cpu().numpy()).real
        return impulse_response

def compute_csd(hrf_model, frequencies, alpha_v, beta_v, alpha_e, beta_e, A, num_regions):
    """
    Compute the Cross-Spectral Density (CSD) matrix for a network of brain regions.

    Args:
        hrf_model (spDCM): Instance of the spDCM model for transfer function computation.
        frequencies (torch.Tensor): 1D tensor of angular frequencies (rad/s).
        alpha_v (torch.Tensor or float): Power of endogenous fluctuations (neuronal).
        beta_v (torch.Tensor or float): Spectral exponent for neuronal fluctuations.
        alpha_e (torch.Tensor or float): Power of measurement noise.
        beta_e (torch.Tensor or float): Spectral exponent for measurement noise.
        A (torch.Tensor): Effective connectivity matrix (complex-valued).
        num_regions (int): Number of brain regions (nodes in the network).

    Returns:
        torch.Tensor: CSD matrix with shape (num_regions**2, len(frequencies)), complex-valued.
    """
    I = torch.eye(num_regions, dtype=torch.complex64, device=frequencies.device)

    def g(omega, alphas, betas):
        # Convert to 1D tensors if scalar
        if not torch.is_tensor(alphas):
            alphas = torch.tensor([alphas], dtype=torch.float32, device=frequencies.device)
        elif alphas.dim() == 0:
            alphas = alphas.unsqueeze(0)

        if not torch.is_tensor(betas):
            betas = torch.tensor([betas], dtype=torch.float32, device=frequencies.device)
        elif betas.dim() == 0:
            betas = betas.unsqueeze(0)

        return torch.diag(alphas * (omega ** (-betas))).to(torch.complex64)

    csd_data = torch.empty((num_regions ** 2, len(frequencies)), dtype=torch.complex64, device=frequencies.device)

    for index, omega in enumerate(frequencies):
        hrf = hrf_model.transfer_function(omega) * I
        G_v = g(omega, alpha_v, beta_v)
        G_e = g(omega, alpha_e, beta_e)
        X = (1j * omega * I - A).to(torch.complex64)
        C = torch.linalg.solve(X, hrf)
        csd_values = (C @ G_v @ torch.conj(C).T + G_e).reshape(-1)
        csd_data[:, index] = csd_values

    return csd_data
