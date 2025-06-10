import json
import time
import torch
import torch.nn as nn
import os
from itertools import product
from matplotlib import pyplot as plt

from helper_functions import filter_frequencies, plot_psd_comparison, plot_psd_loss_curve, normalise_data
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from spDCM.spDCM_Model import compute_csd, spDCM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class spDCMTrainableAll(nn.Module, spDCM):
    """
    A trainable spDCM model for simulation with all major parameters exposed for optimization.
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
    Simulates data, performs parameter search using a grid, trains the model, and plots the results.
    """

    def __init__(self, subject_id, region_index, exp, device, simulate=True):
        self.subject_id = subject_id
        self.region_index = region_index
        self.exp = exp
        self.padded_id = f"{subject_id:03d}"
        self.device = device
        self.simulate = simulate
        self.load_data()
        self.model = None
        self.losses = []
        self.best_loss = float('inf')
        self.best_params = {}

    def load_data(self):
        """
        Simulates PSD data using ground truth parameters and adds noise.
        """
        if self.simulate:
            print(f"Simulating data for Subject {self.padded_id}, ROI {self.region_index}, {self.exp}")
            true_params = {
                'A': [[complex(-0.2, 0.05)]],
                'varphi': 0.6, 'phi': 1.5, 'chi': 0.6, 'mtt': 2.0, 'tau': 4.0,
                'alpha_v': 0.5, 'beta_v': 0.5, 'alpha_e': 0.5, 'beta_e': 0.5,
                'alpha': 0.3, 'E0': 0.4,
                'theta0': 40.3, 'TE': 0.04, 'r0': 15.0, 'epsilon': 0.5
            }
            model_gt = spDCMTrainableAll(**true_params).to(self.device)

            self.freqs = torch.linspace(0.01, 0.1, 50).to(self.device)
            self.freqs_tensor = self.freqs.clone()

            with torch.no_grad():
                psd_sim = compute_csd(model_gt, self.freqs_tensor, model_gt.alpha_v, model_gt.beta_v,
                                      model_gt.alpha_e, model_gt.beta_e, model_gt.A, num_regions=1).squeeze()

            noise_level = 0.05
            psd_sim_noisy = psd_sim + noise_level * (
                torch.randn_like(psd_sim.real) + 1j * torch.randn_like(psd_sim.imag)
            )

            self.psd = psd_sim_noisy
            self.psd_tensor = self.psd.to(torch.complex64)
        else:
            raise NotImplementedError("Real data loading is disabled for simulation mode.")

    def run_grid_search(self):
        """
        Manually evaluates combinations of parameter values to find the best fit using mean squared error.
        """
        print("Running grid search...")
        self.time_start = time.time()

        # Parameter ranges for brute-force search
        A_real_vals = [-0.3, -0.2]
        A_imag_vals = [0.01, 0.05]
        varphi_vals = [0.5]
        phi_vals = [1.5]
        chi_vals = [0.5]
        mtt_vals = [2.0]
        tau_vals = [4.0]
        alpha_v_vals = [0.4]
        beta_v_vals = [0.4]
        alpha_e_vals = [0.4]
        beta_e_vals = [0.4]
        alpha_vals = [0.3]
        E0_vals = [0.4]

        # Full parameter grid
        grid = list(product(A_real_vals, A_imag_vals, varphi_vals, phi_vals, chi_vals,
                            mtt_vals, tau_vals, alpha_v_vals, beta_v_vals,
                            alpha_e_vals, beta_e_vals, alpha_vals, E0_vals))

        best_loss = float("inf")
        best_params = None

        for values in grid:
            param_names = ['A_real', 'A_imag', 'varphi', 'phi', 'chi', 'mtt', 'tau',
                           'alpha_v', 'beta_v', 'alpha_e', 'beta_e', 'alpha', 'E0']
            params = dict(zip(param_names, values))

            model = spDCMTrainableAll(
                A=[[complex(params['A_real'], params['A_imag'])]],
                varphi=params['varphi'], phi=params['phi'], chi=params['chi'],
                mtt=params['mtt'], tau=params['tau'],
                alpha_v=params['alpha_v'], beta_v=params['beta_v'],
                alpha_e=params['alpha_e'], beta_e=params['beta_e'],
                alpha=params['alpha'], E0=params['E0'],
                theta0=40.3, TE=0.04, r0=15.0, epsilon=0.5
            ).to(self.device)

            with torch.no_grad():
                pred = compute_csd(model, self.freqs_tensor, model.alpha_v, model.beta_v,
                                   model.alpha_e, model.beta_e, model.A, num_regions=1).squeeze()
                loss = torch.mean((pred.real - self.psd_tensor.real) ** 2) + \
                       torch.mean((pred.imag - self.psd_tensor.imag) ** 2)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params

        self.best_loss = best_loss
        self.optuna_params = best_params
        print("Best loss:", best_loss)
        return best_params

    def build_model(self, best_params):
        """
        Initializes the trainable model with the best grid search parameters.
        """
        self.model = spDCMTrainableAll(
            A=[[complex(best_params['A_real'], best_params['A_imag'])]],
            varphi=best_params['varphi'], phi=best_params['phi'], chi=best_params['chi'],
            mtt=best_params['mtt'], tau=best_params['tau'],
            alpha_v=best_params['alpha_v'], beta_v=best_params['beta_v'],
            alpha_e=best_params['alpha_e'], beta_e=best_params['beta_e'],
            alpha=best_params['alpha'], E0=best_params['E0'],
            theta0=40.3, TE=0.04, r0=15.0, epsilon=0.5
        ).to(self.device)

    def train_model(self, num_epochs=1600):
        """
        Optimizes the model parameters using AdamW and tracks training loss.
        """
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
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()

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


def run_spdcm_for_subjects(subject_ids, region_indices, exp_list, num_epochs=500):
    """
    Run grid search and training for all subjects and regions specified.
    """
    for exp in exp_list:
        for subject_id in subject_ids:
            padded_id = f"{subject_id:03d}"
            print(f"\nRunning Subject {padded_id}, Condition: {exp}")
            for region_index in region_indices:
                trainer = spDCMTrainer(subject_id, region_index, exp, device, simulate=True)
                try:
                    best_params = trainer.run_grid_search()
                    trainer.build_model(best_params)
                    trainer.train_model(num_epochs=num_epochs)
                    trainer.plot_results()
                except Exception as e:
                    print(f"Failed for Subject {padded_id}, ROI {region_index}, {exp}: {str(e)}")


if __name__ == '__main__':
    exp_list = ["SIM"]
    selected_subjects = [0]
    selected_rois = [0]

    run_spdcm_for_subjects(
        subject_ids=selected_subjects,
        region_indices=selected_rois,
        exp_list=exp_list,
        num_epochs=1
    )
