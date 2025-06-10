import json
import time
import torch
import torch.nn as nn
import os
import optuna
from matplotlib import pyplot as plt

from helper_functions import load_single_time_series, filter_frequencies, plot_psd_comparison, plot_psd_loss_curve, normalise_data, normalise_complex_psd
from scipy_to_torch import torch_csd
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from spDCM.spDCM_Model import compute_csd, spDCM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class spDCMTrainableAll(nn.Module, spDCM):
    """
    Custom PyTorch module that integrates trainable physiological parameters
    into the spDCM model for learning via gradient-based optimization.
    """

    def __init__(self, A, phi, varphi, chi, mtt, tau, alpha_v, beta_v, alpha_e, beta_e, alpha, E0, **kwargs):
        super().__init__()
        spDCM.__init__(self, A=A, **kwargs)
        device = self.device

        # Initialize connectivity matrix A as a trainable complex parameter
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

        # Constants used for computing the BOLD signal
        self.k1 = torch.tensor(4.3 * self.theta0 * self.E0.item() * self.TE, device=device)
        self.k2 = torch.tensor(self.epsilon * self.r0 * self.E0.item() * self.TE, device=device)
        self.k3 = torch.tensor(1 - self.epsilon, device=device)


class spDCMTrainer:
    """
    Handles data loading, model optimization, training, and result saving
    for one subject, region, and experiment condition.
    """

    def __init__(self, subject_id, region_index, exp, device):
        self.subject_id = subject_id
        self.region_index = region_index
        self.exp = exp
        self.padded_id = f"{subject_id:03d}"
        self.device = device
        self.load_data()
        self.model = None
        self.losses = []
        self.best_loss = float('inf')
        self.best_params = {}

    def load_data(self):
        """
        Load time series and compute its cross-spectral density.
        """
        path = f'/Users/xuenbei/Desktop/finalyearproject/time_series/sub-{self.padded_id}-{self.exp}-ROI{self.region_index}.txt'
        ts = load_single_time_series(path)
        ts = torch.tensor(ts, dtype=torch.float32, device=self.device)
        ts = ts / ts.std()
        freqs, psd = torch_csd(ts, ts, fs=0.5, nperseg=128, nfft=512)
        self.freqs, self.psd = filter_frequencies(freqs, psd, min_freq=0.01, max_freq=0.1)
        self.freqs_tensor = torch.tensor(self.freqs, dtype=torch.float32, device=self.device)
        self.psd_tensor = torch.tensor(self.psd, dtype=torch.complex64, device=self.device)

    def optuna_objective(self, trial):
        """
        Objective function used by Optuna to evaluate each parameter set.
        """
        A_val = complex(
            trial.suggest_float("A_real", -0.4, -1e-4),
            trial.suggest_float("A_imag", -0.1, 0.05)
        )
        params = {
            'A': [[A_val]],
            'varphi': trial.suggest_float("varphi", 0.4, 0.8),
            'phi': trial.suggest_float("phi", 1.0, 2.0),
            'chi': trial.suggest_float("chi", 0.4, 0.8),
            'mtt': trial.suggest_float("mtt", 1.0, 5.0),
            'tau': trial.suggest_float("tau", 3.0, 5.0),
            'alpha_v': trial.suggest_float("alpha_v", 1e-4, 1.0),
            'beta_v': trial.suggest_float("beta_v", 1e-4, 1.0),
            'alpha_e': trial.suggest_float("alpha_e", 1e-4, 1.0),
            'beta_e': trial.suggest_float("beta_e", 1e-4, 1.0),
            'alpha': trial.suggest_float("alpha", 0.1, 0.5),
            'E0': trial.suggest_float("E0", 0.2, 0.6)
        }
        model = spDCMTrainableAll(**params, theta0=40.3, TE=0.04, r0=15.0, epsilon=0.5).to(self.device)
        pred = compute_csd(model, self.freqs_tensor, model.alpha_v, model.beta_v,
                           model.alpha_e, model.beta_e, model.A, num_regions=1).squeeze()
        loss = torch.mean((pred.real - self.psd_tensor.real) ** 2) + \
               torch.mean((pred.imag - self.psd_tensor.imag) ** 2)
        return loss.item()

    def run_optuna(self, n_trials=100):
        """
        Run Optuna hyperparameter search to find best-fitting model parameters.
        """
        self.time_start = time.time()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.optuna_objective, n_trials=n_trials)
        self.optuna_params = study.best_params
        return self.optuna_params

    def build_model(self, best_params):
        """
        Initialize the model using the best parameters found by Optuna.
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
        Train the model using AdamW optimizer and exponential learning rate scheduler.
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

            # Clamp parameters to avoid invalid values
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

    def save_outputs(self):
        """
        Save training results, model parameters, evaluation metrics, and plots.
        """
        base_dir = "/Users/xuenbei/Desktop/finalyearproject/spDCM/fitted_data"
        subject_dir = os.path.join(base_dir, f"sub-{self.padded_id}")
        plots_dir = os.path.join(subject_dir, "plots")
        os.makedirs(subject_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        prefix = f"roi-{self.region_index}-{self.exp}"

        model = self.model
        pred = compute_csd(model, self.freqs_tensor, model.alpha_v, model.beta_v,
                           model.alpha_e, model.beta_e, model.A, num_regions=1).squeeze().detach().cpu().numpy()
        psd_norm = normalise_data(self.psd)
        pred_norm = normalise_data(pred)

        plot_psd_comparison(self.freqs, self.psd, pred, "PSD Comparison", self.region_index, False)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_psd_raw.png"))
        plt.clf()

        plot_psd_comparison(self.freqs, self.psd, pred, "PSD Comparison (log)", self.region_index, True)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_psd_log.png"))
        plt.clf()

        plot_psd_loss_curve(self.losses, f"Loss Curve - Subject {self.padded_id}, ROI {self.region_index}, {self.exp}",
                            "AdamW Loss", self.region_index)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_loss_curve.png"))
        plt.clf()

        raw_metrics = CSDMetricsCalculatorTorch(self.psd, pred).evaluate()
        norm_metrics = CSDMetricsCalculatorTorch(psd_norm, pred_norm).evaluate()

        A_np = model.A.detach().cpu().numpy()
        A_serialized = [[[float(elem.real), float(elem.imag)] for elem in row] for row in A_np]

        final_parameters = {
            "A": A_serialized,
            "varphi": float(model.varphi.item()),
            "phi": float(model.phi.item()),
            "chi": float(model.chi.item()),
            "mtt": float(model.mtt.item()),
            "tau": float(model.tau.item()),
            "alpha_v": float(model.alpha_v.item()),
            "beta_v": float(model.beta_v.item()),
            "alpha_e": float(model.alpha_e.item()),
            "beta_e": float(model.beta_e.item()),
            "alpha": float(model.alpha.item()),
            "E0": float(model.E0.item())
        }

        params = {
            "subject_id": int(self.subject_id),
            "region": int(self.region_index),
            "experiment": self.exp,
            "fit_time": float(time.time() - self.time_start),
            "final_loss": float(self.best_loss),
            "final_parameters": final_parameters
        }

        metrics = {
            "subject_id": int(self.subject_id),
            "region": int(self.region_index),
            "experiment": self.exp,
            "raw_csd_metrics": {k: float(v) for k, v in raw_metrics.items()},
            "normalised_csd_metrics": {k: float(v) for k, v in norm_metrics.items()}
        }

        with open(os.path.join(subject_dir, f"{prefix}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(os.path.join(subject_dir, f"{prefix}_results.json"), "w") as f:
            json.dump(params, f, indent=4)

        with open(os.path.join(subject_dir, f"{prefix}_loss.txt"), "w") as f:
            for i, l in enumerate(self.losses):
                f.write(f"Epoch {i:04d}: {l:.6f}\n")

        torch.save(self.model.state_dict(), os.path.join(subject_dir, f"{prefix}_model.pth"))

        print(f"Saved outputs for Subject {self.padded_id}, ROI {self.region_index}, {self.exp}")


def run_spdcm_for_subjects(subject_ids, region_indices, exp_list, n_trials=1000, num_epochs=1500):
    """
    Iterate over all subjects, ROIs, and experiment conditions,
    running training and saving results for each combination.
    """
    for exp in exp_list:
        for subject_id in subject_ids:
            padded_id = f"{subject_id:03d}"
            print(f"\nRunning Subject {padded_id}, Condition: {exp}")
            for region_index in region_indices:
                trainer = spDCMTrainer(subject_id, region_index, exp, device)
                try:
                    best_params = trainer.run_optuna(n_trials=n_trials)
                    trainer.build_model(best_params)
                    trainer.train_model(num_epochs=num_epochs)
                    trainer.save_outputs()
                except Exception as e:
                    print(f"Failed for Subject {padded_id}, ROI {region_index}, {exp}: {str(e)}")


if __name__ == '__main__':
    exp_list = ["PLCB", "LSD"]
    selected_subjects = [1, 2, 3, 4, 6, 10, 11, 12, 19, 20]
    selected_rois = list(range(100))

    run_spdcm_for_subjects(
        subject_ids=selected_subjects,
        region_indices=selected_rois,
        exp_list=exp_list,
        n_trials=1000,
        num_epochs=1500
    )
