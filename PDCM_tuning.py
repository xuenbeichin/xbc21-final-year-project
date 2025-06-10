import os
import json
import time
import torch
import torch.nn as nn
import optuna
from matplotlib import pyplot as plt

from PDCM.PDCMBOLDModel import PDCMMODEL
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from helper_functions import load_single_time_series, filter_frequencies, plot_psd_comparison, \
    plot_psd_loss_curve, normalise_complex_psd
from scipy_to_torch import torch_csd
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def complex_mse_loss(output, target, fs=0.5):
    """
    Compute the mean squared error between the complex-valued CSDs of the output and target BOLD signals.

    Parameters:
        output (torch.Tensor): Simulated BOLD signal
        target (torch.Tensor): Ground truth BOLD signal
        fs (float): Sampling frequency

    Returns:
        torch.Tensor: Complex-valued MSE loss
    """
    output = output / output.std()
    f1, csdx = torch_csd(output, output, fs=fs, nperseg=128, nfft=512)
    _, cdx_filtered = filter_frequencies(f1, csdx, 0.01, 0.1)

    f2, csdy = torch_csd(target, target, fs=fs, nperseg=128, nfft=512)
    _, csdy_filtered = filter_frequencies(f2, csdy, 0.01, 0.1)

    mse_real = torch.mean((cdx_filtered.real - csdy_filtered.real) ** 2)
    mse_imag = torch.mean((cdx_filtered.imag - csdy_filtered.imag) ** 2)
    return mse_real + mse_imag


class TrainableBOLDParams(nn.Module):
    """
    A PyTorch module representing trainable physiological parameters of the BOLD model.
    """

    def __init__(self, init_values, device):
        super().__init__()
        self.device = device
        for key, value in init_values.items():
            setattr(self, key, nn.Parameter(torch.tensor(value, device=device)))

    def get_model(self):
        """
        Construct a BOLD model from the current trainable parameters.

        Returns:
            PDCMMODEL: Configured model instance
        """
        return PDCMMODEL(
            ignore_range=True,
            phi=self.phi.item(), varphi=self.varphi.item(), chi=self.chi.item(),
            mtt=self.mtt.item(), tau=self.tau.item(), sigma=self.sigma.item(),
            mu=self.mu.item(), lamb=self.lamb.item(), alpha=self.alpha.item(), E0=self.E0.item(),
            device=self.device
        )


class PDCMTrainer:
    """
    Trainer class for simulating and optimizing a subject's region-specific BOLD signal using PDCM.
    """

    def __init__(self, subject_id, region_index, exp, device):
        self.subject_id = subject_id
        self.region_index = region_index
        self.exp = exp
        self.padded_id = f"{subject_id:03d}"
        self.device = device
        self.fs = 0.5
        self.h = 0.01

        self.real_bold_tensor = self.load_time_series()
        self.t_sim = torch.arange(0, len(self.real_bold_tensor) * 2, self.h, device=device)

        self.model = None
        self.losses = []
        self.best_loss = float('inf')
        self.best_optuna_params = {}

    def load_time_series(self):
        """
        Load and normalise the time series data for the specified subject, region, and experiment.

        Returns:
            torch.Tensor: Normalised BOLD signal
        """
        path = f'/Users/xuenbei/Desktop/finalyearproject/time_series/sub-{self.padded_id}-{self.exp}-ROI{self.region_index}.txt'
        ts = load_single_time_series(path)
        ts = torch.tensor(ts, dtype=torch.float32, device=self.device)
        return ts / ts.std()

    def optuna_objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning.

        Parameters:
            trial (optuna.Trial): Optuna trial object

        Returns:
            float: Loss value to be minimized
        """
        init = {
            'phi': trial.suggest_float("phi", 1.0, 2.0),
            'varphi': trial.suggest_float("varphi", 0.1, 1.0),
            'chi': trial.suggest_float("chi", 0.1, 1.0),
            'mtt': trial.suggest_float("tMTT", 0.5, 5.5),
            'tau': trial.suggest_float("tau", 0.0, 32.0),
            'sigma': trial.suggest_float("sigma", 0.5, 2.0),
            'mu': trial.suggest_float("mu", 0.0, 2.0),
            'lamb': trial.suggest_float("lamb", 0.0, 0.35),
            'alpha_v': trial.suggest_float("alpha_v", 0.1, 20.0),
            'beta_v': trial.suggest_float("beta_v", 0.1, 5.0),
            'alpha_e': trial.suggest_float("alpha_e", 0.1, 20.0),
            'beta_e': trial.suggest_float("beta_e", 0.1, 5.0),
            'alpha': trial.suggest_float("alpha", 0.1, 0.5),
            'E0': trial.suggest_float("E0", 0.2, 0.6)
        }

        model = TrainableBOLDParams(init, self.device)
        bold_model = model.get_model()
        _, yhat = simulate_bold_euler_maruyama(
            model=bold_model, time=self.t_sim, h=self.h,
            alpha_v=model.alpha_v.item(), beta_v=model.beta_v.item(),
            alpha_e=model.alpha_e.item(), beta_e=model.beta_e.item(),
            desired_TR=2.0, add_state_noise=True, add_obs_noise=True
        )
        loss = complex_mse_loss(yhat, self.real_bold_tensor, fs=self.fs)
        return loss.item()

    def run_optuna(self, n_trials=100):
        """
        Run Optuna to find optimal model parameters.

        Parameters:
            n_trials (int): Number of trials for hyperparameter optimization

        Returns:
            dict: Best found parameters
        """
        self.start_time = time.time()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.optuna_objective, n_trials=n_trials)
        self.best_optuna_params = study.best_params
        return self.best_optuna_params

    def train(self, best_params, num_epochs=1600):
        """
        Train the model using the best parameters from Optuna.

        Parameters:
            best_params (dict): Optimized parameters
            num_epochs (int): Number of training epochs
        """
        self.model = TrainableBOLDParams(best_params, self.device).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            bold_model = self.model.get_model()
            _, yhat = simulate_bold_euler_maruyama(
                model=bold_model, time=self.t_sim, h=self.h,
                alpha_v=self.model.alpha_v.item(), beta_v=self.model.beta_v.item(),
                alpha_e=self.model.alpha_e.item(), beta_e=self.model.beta_e.item(),
                desired_TR=2.0, add_state_noise=True, add_obs_noise=True
            )
            yhat = yhat / yhat.std()
            loss = complex_mse_loss(yhat, self.real_bold_tensor, fs=self.fs)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            self.losses.append(loss_val)
            self.best_loss = min(self.best_loss, loss_val)

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(f"[Epoch {epoch}] Loss: {loss_val:.6f}")

        self.final_yhat = yhat.detach().cpu().numpy()

    def save_results(self):
        """
        Save evaluation metrics, PSD plots, and trained parameters to disk.
        """
        base_dir = "/PDCM/tuning_all_params/trial_data/"
        subject_dir = os.path.join(base_dir, f"sub-{self.padded_id}")
        plots_dir = os.path.join(subject_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        prefix = f"roi-{self.region_index}-{self.exp}"

        bold_true = self.real_bold_tensor.detach().cpu().numpy()
        bold_pred = self.final_yhat

        # Compute filtered CSDs
        f1, csd_true = torch_csd(torch.tensor(bold_true), torch.tensor(bold_true), fs=self.fs)
        _, csd_true = filter_frequencies(f1, csd_true, 0.01, 0.1)

        f2, csd_pred = torch_csd(torch.tensor(bold_pred), torch.tensor(bold_pred), fs=self.fs)
        _, csd_pred = filter_frequencies(f2, csd_pred, 0.01, 0.1)

        # Plot PSD comparisons
        plot_psd_comparison(f1, csd_true, csd_pred, f"Recovered vs Real PSD", self.region_index, False)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_psd_raw.png"))
        plt.clf()

        plot_psd_comparison(f1, csd_true, csd_pred, f"Recovered vs Real PSD (log)", self.region_index, True)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_psd_log.png"))
        plt.clf()

        # Plot loss curve
        plot_psd_loss_curve(self.losses, f"Loss Curve - Subject {self.padded_id}, ROI {self.region_index}, {self.exp}",
                            "AdamW Loss", self.region_index)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_loss_curve.png"))
        plt.clf()

        # Compute evaluation metrics
        csd_norm_true = normalise_complex_psd(csd_true)
        csd_norm_pred = normalise_complex_psd(csd_pred)

        raw_metrics = CSDMetricsCalculatorTorch(csd_true, csd_pred).evaluate()
        norm_metrics = CSDMetricsCalculatorTorch(csd_norm_true, csd_norm_pred).evaluate()

        metrics = {
            "subject_id": self.subject_id,
            "region": self.region_index,
            "experiment": self.exp,
            "raw_csd_metrics": raw_metrics,
            "normalised_csd_metrics": norm_metrics
        }

        trained_params = {k: float(v.item()) for k, v in self.model.state_dict().items()}
        results = {
            "subject_id": self.subject_id,
            "region": self.region_index,
            "experiment": self.exp,
            "fit_time": time.time() - self.start_time,
            "final_loss": self.best_loss,
            "final_parameters": trained_params
        }

        # Save to disk
        with open(os.path.join(subject_dir, f"{prefix}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(os.path.join(subject_dir, f"{prefix}_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(subject_dir, f"{prefix}_loss.txt"), "w") as f:
            for i, l in enumerate(self.losses):
                f.write(f"Epoch {i:04d}: {l:.6f}\n")


def run_pdcm_for_subjects(subject_ids, region_indices, exp_list, n_trials=100, num_epochs=1):
    """
    Run the full optimization and training pipeline across a set of subjects, ROIs, and experiments.

    Parameters:
        subject_ids (list): List of subject IDs
        region_indices (list): List of brain ROI indices
        exp_list (list): List of experimental conditions
        n_trials (int): Number of Optuna optimization trials
        num_epochs (int): Number of epochs to train the final model
    """
    for subject_id in subject_ids:
        for region_index in region_indices:
            for exp in exp_list:
                print(f"\nRunning P-DCM for Subject {subject_id:03d}, ROI {region_index}, Condition: {exp}")
                try:
                    trainer = PDCMTrainer(subject_id, region_index, exp, device)
                    best_init = trainer.run_optuna(n_trials=n_trials)
                    trainer.train(best_init, num_epochs=num_epochs)
                    trainer.save_results()
                except Exception as e:
                    print(f"Failed for Subject {subject_id:03d}, ROI {region_index}, Exp {exp}: {str(e)}")


if __name__ == '__main__':
    subject_ids = [1, 2, 3, 4, 6, 10, 11, 12, 19, 20]
    selected_rois = list(range(100))
    experiments = ["PLCB", "LSD"]

    run_pdcm_for_subjects(
        subject_ids=subject_ids,
        region_indices=selected_rois,
        exp_list=experiments,
        n_trials=50,
        num_epochs=1
    )
