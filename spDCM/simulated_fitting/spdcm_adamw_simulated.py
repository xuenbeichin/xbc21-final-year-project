import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from helper_functions import plot_psd_comparison, plot_psd_loss_curve, normalise_data
from spDCM.spDCM_Model import compute_csd, spDCM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class spDCMTrainableAll(nn.Module, spDCM):
    """
    A fully trainable implementation of the spDCM model for PSD fitting tasks.
    """

    def __init__(self, A, phi, varphi, chi, mtt, tau, alpha_v, beta_v, alpha_e, beta_e, alpha, E0, **kwargs):
        super().__init__()
        spDCM.__init__(self, A=A, **kwargs)
        device = self.device

        A_real_raw = torch.tensor([[[-A[0][0].real]]], dtype=torch.float32, device=device)
        A_imag_tensor = torch.tensor([[[A[0][0].imag]]], dtype=torch.float32, device=device)
        A_complex = torch.complex(-torch.nn.functional.softplus(A_real_raw), A_imag_tensor)
        self.A = nn.Parameter(A_complex.to(device))

        def positive_tensor(x):
            return torch.nn.functional.softplus(torch.tensor([x], dtype=torch.float32, device=device))

        self.varphi = nn.Parameter(torch.tensor([varphi], device=device))
        self.phi = nn.Parameter(torch.tensor([phi], device=device))
        self.chi = nn.Parameter(torch.tensor([chi], device=device))
        self.mtt = nn.Parameter(torch.tensor([mtt], device=device))
        self.tau = nn.Parameter(torch.tensor([tau], device=device))
        self.alpha_v = nn.Parameter(positive_tensor(alpha_v))
        self.beta_v = nn.Parameter(positive_tensor(beta_v))
        self.alpha_e = nn.Parameter(positive_tensor(alpha_e))
        self.beta_e = nn.Parameter(positive_tensor(beta_e))
        self.alpha = nn.Parameter(torch.tensor([alpha], device=device))
        self.E0 = nn.Parameter(torch.tensor([E0], device=device))

        self.k1 = torch.tensor(4.3 * self.theta0 * self.E0.item() * self.TE, device=device)
        self.k2 = torch.tensor(self.epsilon * self.r0 * self.E0.item() * self.TE, device=device)
        self.k3 = torch.tensor(1 - self.epsilon, device=device)


class spDCMTrainer:
    """
    Handles the generation of synthetic PSD data, model training, and visualization.
    """

    def __init__(self, device):
        self.device = device
        self.freqs_tensor, self.psd_tensor = self.generate_synthetic_data()
        self.losses = []

    def generate_synthetic_data(self):
        """
        Generates synthetic PSD data using ground truth parameters with added noise.
        """
        print("Generating synthetic CSD data...")

        true_params = {
            'A': [[complex(-0.2, 0.05)]],
            'varphi': 0.6, 'phi': 1.5, 'chi': 0.6, 'mtt': 2.0, 'tau': 4.0,
            'alpha_v': 0.5, 'beta_v': 0.5, 'alpha_e': 0.5, 'beta_e': 0.5,
            'alpha': 0.3, 'E0': 0.4,
            'theta0': 40.3, 'TE': 0.04, 'r0': 15.0, 'epsilon': 0.5
        }

        model_gt = spDCMTrainableAll(**true_params).to(self.device)
        freqs = torch.linspace(0.01, 0.1, 50).to(self.device)

        with torch.no_grad():
            psd = compute_csd(model_gt, freqs, model_gt.alpha_v, model_gt.beta_v,
                              model_gt.alpha_e, model_gt.beta_e, model_gt.A, num_regions=1).squeeze()

        noise_level = 0.05
        psd_noisy = psd + noise_level * (torch.randn_like(psd.real) + 1j * torch.randn_like(psd.imag))
        return freqs, psd_noisy.to(torch.complex64)

    def build_model(self):
        """
        Instantiates the spDCM model with suboptimal initial parameters.
        """
        init_params = {
            'A': [[complex(-0.25, 0.01)]],
            'varphi': 0.5, 'phi': 1.4, 'chi': 0.55, 'mtt': 2.5, 'tau': 4.2,
            'alpha_v': 0.3, 'beta_v': 0.3, 'alpha_e': 0.3, 'beta_e': 0.3,
            'alpha': 0.2, 'E0': 0.35,
            'theta0': 40.3, 'TE': 0.04, 'r0': 15.0, 'epsilon': 0.5
        }

        self.model = spDCMTrainableAll(**init_params).to(self.device)

    def train_model(self, num_epochs=1000):
        """
        Trains the spDCM model using AdamW and tracks the loss curve.
        """
        print("Training with AdamW...")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred = compute_csd(self.model, self.freqs_tensor, self.model.alpha_v, self.model.beta_v,
                               self.model.alpha_e, self.model.beta_e, self.model.A, num_regions=1).squeeze()

            loss = torch.mean((pred.real - self.psd_tensor.real) ** 2) + \
                   torch.mean((pred.imag - self.psd_tensor.imag) ** 2)

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.model.alpha_v.clamp_(min=1e-6)
                self.model.beta_v.clamp_(min=1e-6)
                self.model.alpha_e.clamp_(min=1e-6)
                self.model.beta_e.clamp_(min=1e-6)

            self.losses.append(loss.item())

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch} | Loss: {loss.item():.6f}")

    def plot_results(self):
        """
        Plots the PSD fit and the loss curve, and prints CSD evaluation metrics.
        """
        with torch.no_grad():
            pred = compute_csd(self.model, self.freqs_tensor, self.model.alpha_v, self.model.beta_v,
                               self.model.alpha_e, self.model.beta_e, self.model.A,
                               num_regions=1).squeeze().cpu().numpy()

        psd_true = self.psd_tensor.cpu().numpy()
        psd_norm = normalise_data(psd_true)
        pred_norm = normalise_data(pred)

        # Compute evaluation metrics
        raw_metrics = CSDMetricsCalculatorTorch(psd_true, pred).evaluate()
        norm_metrics = CSDMetricsCalculatorTorch(psd_norm, pred_norm).evaluate()

        # Print metrics
        print("\nRaw CSD Metrics:")
        for k, v in raw_metrics.items():
            print(f"  {k}: {v:.6f}")

        print("\nNormalised CSD Metrics:")
        for k, v in norm_metrics.items():
            print(f"  {k}: {v:.6f}")

        # Plot PSD comparison
        plot_psd_comparison(
            self.freqs_tensor.cpu().numpy(),
            psd_true,
            pred,
            title="PSD Comparison",
            region1_index=0,
            log_scale=False
        )
        plt.show()

        # Plot loss curve
        plot_psd_loss_curve(
            self.losses,
            title="AdamW Loss Curve",
            region1_index=0,
            label="Loss"
        )
        plt.show()


if __name__ == '__main__':
    trainer = spDCMTrainer(device)
    trainer.build_model()
    trainer.train_model(num_epochs=1000)
    trainer.plot_results()
