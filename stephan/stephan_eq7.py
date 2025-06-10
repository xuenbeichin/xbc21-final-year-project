import torch
import numpy as np
import matplotlib.pyplot as plt

import torch

class StephanHRFEq7:
    def __init__(self, V0=0.04, theta0=40.3, epsilon=1, k=0.64, E0=0.4, TE=0.04,
                 r0=25.0, gamma=0.32, mtt=2.0, alpha=0.32, A=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert all attributes to tensors
        self.V0 = torch.tensor(V0, device=self.device, dtype=torch.float32)
        self.theta0 = torch.tensor(theta0, device=self.device, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, device=self.device, dtype=torch.float32)
        self.k = torch.tensor(k, device=self.device, dtype=torch.float32)
        self.E0 = torch.tensor(E0, device=self.device, dtype=torch.float32)
        self.TE = torch.tensor(TE, device=self.device, dtype=torch.float32)
        self.r0 = torch.tensor(r0, device=self.device, dtype=torch.float32)
        self.gamma = torch.tensor(gamma, device=self.device, dtype=torch.float32)
        self.mtt = torch.tensor(mtt, device=self.device, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)

        # Derived tensors
        self.k1 = torch.tensor(4.3 * theta0 * E0 * TE, device=self.device, dtype=torch.float32)
        self.k2 = torch.tensor(epsilon * r0 * E0 * TE, device=self.device, dtype=torch.float32)
        self.k3 = torch.tensor(1 - epsilon, device=self.device, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.complex128, device=self.device) if A is not None else None

    def transfer_function(self, omega):
        s = 1j * omega

        numerator = self.V0 * (
            (self.E0 - 1) * torch.log(1 - self.E0) * (self.k1 + self.k2) *
            (self.alpha * s * self.mtt + 1)
            - self.alpha * self.E0 * (self.k1 + self.k3) * (s * self.mtt + 1)
        )

        denominator = (
            self.E0 *
            (s * self.mtt + 1) * (self.gamma + s * (self.k + s)) * (self.alpha * s * self.mtt + 1)
        )

        return numerator / denominator


    def forward_linearized(self, freqs, alpha_v=1.0, beta_v=1.0, alpha_e=1.0, beta_e=1.0):
        # Number of regions (n x n matrix size from A)
        n = self.A.shape[0]

        # List to store the computed CSD for each frequency
        G_y_w_matrices = []

        # Identity matrix of size n x n
        I = torch.eye(n, dtype=torch.complex128).to(self.device)

        # Loop through each frequency to compute the corresponding G_y_w matrix
        for freq in freqs:
            omega = 2 * np.pi * freq  # Convert frequency to angular frequency

            # Compute the transfer function for the given frequency
            H_w = self.transfer_function(omega)
            H_w_matrix = H_w * I  # Create a diagonal matrix with H_w on the diagonals

            # Compute state noise and observation noise
            G_v = alpha_v * (omega) ** (-beta_v)
            G_e = alpha_e * (omega ) ** (-beta_e)

            # Create diagonal matrices for state noise and observation noise
            G_v_matrix = G_v * I  # Diagonal matrix for state noise
            G_e_matrix = G_e * I  # Diagonal matrix for observation noise

            # Compute inverse terms
            term_1 = torch.linalg.inv(1j * omega * I - self.A)  # (jωI - A)^(-1)
            term_2 = torch.linalg.inv(-1j * omega * I - self.A.T)  # (-jωI - A^T)^(-1)

            # Compute the CSD matrix
            G_y_w = (
                    H_w_matrix @ term_1 @ G_v_matrix @ term_2 @ H_w_matrix.T.conj()
                    + G_e_matrix
            )

            G_y_w_matrices.append(G_y_w)

        # Return the list of G_y_w matrices for all frequencies
        return G_y_w_matrices

    def extract_psd(self, G_y_w_matrices):
        PSD_values = [torch.abs(torch.diag(G_y)) ** 2 for G_y in G_y_w_matrices]
        return PSD_values

    def extract_csd(self, G_y_w_matrices):

        CSD_values = [G_y - torch.diag_embed(torch.diag(G_y)) for G_y in G_y_w_matrices]
        return CSD_values

    def extract_asd(self, G_y_w_matrices):
        ASD_values = [torch.sqrt(torch.abs(torch.diag(G_y))) for G_y in G_y_w_matrices]
        return ASD_values

    def impulse_response(self, t):
        """Compute the impulse response using the inverse Fourier transform."""
        n = len(t)
        freqs = np.fft.fftfreq(n, d=(t[1] - t[0]))  # Frequency bins
        omega = 2 * np.pi * freqs  # Angular frequencies

        # Compute transfer function at each frequency
        H_w = torch.tensor([self.transfer_function(w) for w in omega], dtype=torch.complex128)

        # Compute impulse response via inverse FFT
        impulse_response = np.fft.ifft(H_w.cpu().numpy()).real
        return impulse_response

