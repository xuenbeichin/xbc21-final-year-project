import os
import json
import time
import torch
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from PDCM.PDCMBOLDModel import PDCMMODEL
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from helper_functions import load_single_time_series, filter_frequencies, plot_psd_comparison, normalise_complex_psd
from scipy_to_torch import torch_csd
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch

# Default parameter values used for simulating BOLD signals
DEFAULT_PARAMS = {
    'phi': 1.5, 'varphi': 0.6, 'chi': 0.6, 'mtt': 2.0, 'tau': 4.0,
    'sigma': 0.5, 'mu': 0.4, 'lamb': 0.2, 'alpha_v': 1.0, 'beta_v': 1.0,
    'alpha_e': 1.0, 'beta_e': 1.0, 'alpha': 0.32, 'E0': 0.4
}


def complex_mse_loss(output, target, fs=0.5):
    """
    Compute the mean squared error between the complex power spectral densities (CSDs)
    of two time series using Welch's method.

    Parameters:
        output (torch.Tensor): Simulated BOLD signal
        target (torch.Tensor): Real BOLD signal
        fs (float): Sampling frequency

    Returns:
        torch.Tensor: Complex MSE loss value
    """
    output = output / output.std()
    f1, csdx = torch_csd(output, output, fs=fs, nperseg=128, nfft=512)
    f1, cdx_filtered = filter_frequencies(f1, csdx, 0.01, 0.1)

    f2, csdy = torch_csd(target, target, fs=fs, nperseg=128, nfft=512)
    f2, csdy_filtered = filter_frequencies(f2, csdy, 0.01, 0.1)

    return torch.mean((cdx_filtered.real - csdy_filtered.real) ** 2 +
                      (cdx_filtered.imag - csdy_filtered.imag) ** 2)


class PDCMTrainer:
    """
    Trainer class for simulating and evaluating the PDCM BOLD model.
    """

    def __init__(self, subject_id, region_index, exp, device):
        self.subject_id = subject_id
        self.region_index = region_index
        self.exp = exp
        self.device = device
        self.padded_id = f"{subject_id:03d}"
        self.fs = 0.5  # Sampling frequency
        self.h = 0.01  # Integration step size
        self.start_time = time.time()

        # Load and se the real BOLD time series
        self.real_bold_tensor = self.load_time_series()

        # Simulation time vector
        self.t_sim = torch.arange(0, len(self.real_bold_tensor) * 2, self.h, device=device)

    def load_time_series(self):
        """
        Load and standardize the BOLD signal for the current subject and region.

        Returns:
            torch.Tensor: Normalised BOLD time series
        """
        path = f"/Users/xuenbei/Desktop/finalyearproject/time_series/sub-{self.padded_id}-{self.exp}-ROI{self.region_index}.txt"
        ts = load_single_time_series(path)
        ts = torch.tensor(ts, dtype=torch.float32, device=self.device)
        return ts / ts.std()

    def run_default_model(self):
        """
        Simulate the BOLD model using predefined default parameters.
        """
        p = DEFAULT_PARAMS
        model = PDCMMODEL(
            ignore_range=True,
            phi=p['phi'], varphi=p['varphi'], chi=p['chi'], mtt=p['mtt'], tau=p['tau'],
            sigma=p['sigma'], mu=p['mu'], lamb=p['lamb'], alpha=p['alpha'], E0=p['E0'],
            device=self.device
        )
        _, yhat = simulate_bold_euler_maruyama(
            model=model, time=self.t_sim, h=self.h,
            alpha_v=p['alpha_v'], beta_v=p['beta_v'],
            alpha_e=p['alpha_e'], beta_e=p['beta_e'],
            desired_TR=2.0, add_state_noise=True, add_obs_noise=True
        )
        self.final_yhat = yhat.detach().cpu().numpy()
        self.final_params = p

    def save_results(self):
        """
        Save the simulated results, power spectral density plots, and evaluation metrics.
        """
        base_dir = "/Users/xuenbei/Desktop/finalyearproject/PDCM/fitted_data/"
        subject_dir = os.path.join(base_dir, f"sub-{self.padded_id}")
        plots_dir = os.path.join(subject_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        prefix = f"roi-{self.region_index}-{self.exp}"
        result_file = os.path.join(subject_dir, f"{prefix}_baseline_results.json")
        metric_file = os.path.join(subject_dir, f"{prefix}_baseline_metrics.json")

        bold_true = self.real_bold_tensor.detach().cpu().numpy()
        bold_pred = self.final_yhat

        # Compute and filter CSDs
        f1, csd_true = torch_csd(torch.tensor(bold_true), torch.tensor(bold_true), fs=self.fs)
        f1, csd_true = filter_frequencies(f1, csd_true, 0.01, 0.1)

        f2, csd_pred = torch_csd(torch.tensor(bold_pred), torch.tensor(bold_pred), fs=self.fs)
        f2, csd_pred = filter_frequencies(f2, csd_pred, 0.01, 0.1)

        # Generate and save plots
        plot_psd_comparison(f1, csd_true, csd_pred, "PSD Comparison", self.region_index, log_scale=False)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_baseline_psd_raw.png"))
        plt.clf()

        plot_psd_comparison(f1, csd_true, csd_pred, "PSD Comparison (log)", self.region_index, log_scale=True)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_baseline_psd_log.png"))
        plt.clf()

        # Calculate metrics
        csd_norm_true = normalise_complex_psd(csd_true)
        csd_norm_pred = normalise_complex_psd(csd_pred)
        raw_metrics = CSDMetricsCalculatorTorch(csd_true, csd_pred).evaluate()
        norm_metrics = CSDMetricsCalculatorTorch(csd_norm_true, csd_norm_pred).evaluate()

        with open(metric_file, "w") as f:
            json.dump({
                "subject_id": self.subject_id,
                "region": self.region_index,
                "experiment": self.exp,
                "raw_csd_metrics": raw_metrics,
                "normalised_csd_metrics": norm_metrics
            }, f, indent=4)

        with open(result_file, "w") as f:
            json.dump({
                "subject_id": self.subject_id,
                "region": self.region_index,
                "experiment": self.exp,
                "fit_time": time.time() - self.start_time,
                "final_loss": float(complex_mse_loss(torch.tensor(bold_pred), torch.tensor(bold_true), fs=self.fs)),
                "final_parameters": self.final_params
            }, f, indent=4)


def run_single_job(subject_id, region_index, exp):
    """
    Run the simulation and evaluation for a single subject, ROI, and experiment
    if results do not already exist.

    Parameters:
        subject_id (int): Subject identifier
        region_index (int): Brain region index
        exp (str): Experimental condition
    """
    padded_id = f"{subject_id:03d}"
    result_file = f"/Users/xuenbei/Desktop/finalyearproject/PDCM/fitted_data/sub-{padded_id}/roi-{region_index}-{exp}_baseline_results.json"
    if os.path.exists(result_file):
        print(f"Skipping Subject {padded_id} ROI {region_index} {exp}")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = PDCMTrainer(subject_id, region_index, exp, device)
        trainer.run_default_model()
        trainer.save_results()
        print(f"Finished Subject {padded_id} ROI {region_index} {exp}")
    except Exception as e:
        print(f"Failed Subject {padded_id} ROI {region_index} {exp}: {e}")


def run_pdcm_for_subjects(subject_ids, region_indices, exp_list, max_workers=None):
    """
    Run PDCM model training and evaluation across multiple subjects and ROIs using parallel processing.

    Parameters:
        subject_ids (list[int]): List of subject identifiers
        region_indices (list[int]): List of ROI indices
        exp_list (list[str]): List of experimental conditions
        max_workers (int, optional): Number of parallel workers to use
    """
    tasks = [(s, r, e) for s in subject_ids for r in region_indices for e in exp_list]
    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))

    print(f"Running {len(tasks)} jobs using {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_job, *t) for t in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    selected_subjects = [1, 2, 3, 4, 6, 10, 11, 12, 19, 20]
    selected_rois = list(range(100))
    experiments = ["PLCB", "LSD"]
    run_pdcm_for_subjects(selected_subjects, selected_rois, experiments, max_workers=8)
