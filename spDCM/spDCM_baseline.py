import json
import time
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt

from helper_functions import load_single_time_series, filter_frequencies, plot_psd_comparison, normalise_data
from scipy_to_torch import torch_csd
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from spDCM.spDCM_Model import compute_csd, spDCM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class spDCMTrainableAll(nn.Module, spDCM):
    def __init__(self, A, phi, varphi, chi, mtt, tau, alpha_v, beta_v, alpha_e, beta_e, alpha, E0, **kwargs):
        nn.Module.__init__(self)
        spDCM.__init__(self, A=A, **kwargs)
        device = self.device

        self.A = nn.Parameter(torch.tensor(A, device=device, dtype=torch.complex64, requires_grad=False))
        self.varphi = nn.Parameter(torch.tensor([varphi], device=device), requires_grad=False)
        self.phi = nn.Parameter(torch.tensor([phi], device=device), requires_grad=False)
        self.chi = nn.Parameter(torch.tensor([chi], device=device), requires_grad=False)
        self.mtt = nn.Parameter(torch.tensor([mtt], device=device), requires_grad=False)
        self.tau = nn.Parameter(torch.tensor([tau], device=device), requires_grad=False)
        self.alpha_v = nn.Parameter(torch.tensor([alpha_v], device=device), requires_grad=False)
        self.beta_v = nn.Parameter(torch.tensor([beta_v], device=device), requires_grad=False)
        self.alpha_e = nn.Parameter(torch.tensor([alpha_e], device=device), requires_grad=False)
        self.beta_e = nn.Parameter(torch.tensor([beta_e], device=device), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor([alpha], device=device), requires_grad=False)
        self.E0 = nn.Parameter(torch.tensor([E0], device=device), requires_grad=False)

        self.k1 = torch.tensor(4.3 * self.theta0 * self.E0.item() * self.TE, device=device)
        self.k2 = torch.tensor(self.epsilon * self.r0 * self.E0.item() * self.TE, device=device)
        self.k3 = torch.tensor(1 - self.epsilon, device=device)


class spDCMTrainer:
    def __init__(self, subject_id, region_index, exp, device):
        self.subject_id = subject_id
        self.region_index = region_index
        self.exp = exp
        self.padded_id = f"{subject_id:03d}"
        self.device = device
        self.load_data()
        self.model = None

    def load_data(self):
        path = f'/Users/xuenbei/Desktop/finalyearproject/time_series/sub-{self.padded_id}-{self.exp}-ROI{self.region_index}.txt'
        ts = load_single_time_series(path)
        ts = torch.tensor(ts, dtype=torch.float32, device=self.device)
        ts = ts / ts.std()
        freqs, psd = torch_csd(ts, ts, fs=0.5, nperseg=128, nfft=512)
        self.freqs, self.psd = filter_frequencies(freqs, psd, min_freq=0.01, max_freq=0.1)
        self.freqs_tensor = torch.tensor(self.freqs, dtype=torch.float32, device=self.device)
        self.psd_tensor = torch.tensor(self.psd, dtype=torch.complex64, device=self.device)

    def build_model(self, params):
        self.model = spDCMTrainableAll(**params, theta0=40.3, TE=0.04, r0=15.0, epsilon=0.5).to(self.device)

    def save_outputs(self):
        base_dir = "/Users/xuenbei/Desktop/finalyearproject/spDCM/spdcm_fitted"
        subject_dir = os.path.join(base_dir, f"sub-{self.padded_id}")
        os.makedirs(subject_dir, exist_ok=True)
        plots_dir = os.path.join(subject_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        prefix = f"roi-{self.region_index}-{self.exp}"

        pred = compute_csd(self.model, self.freqs_tensor, self.model.alpha_v, self.model.beta_v,
                           self.model.alpha_e, self.model.beta_e, self.model.A, num_regions=1).squeeze().detach().cpu().numpy()
        psd_norm = normalise_data(self.psd)
        pred_norm = normalise_data(pred)

        # Plot raw PSD comparison
        plot_psd_comparison(self.freqs, self.psd, pred,
                            f"Recovered vs Real PSD (Subject {self.padded_id} - ROI {self.region_index} - {self.exp})",
                            self.region_index, log_scale=False)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_baseline_psd_raw.png"))
        plt.clf()

        # Plot log PSD comparison
        plot_psd_comparison(self.freqs, self.psd, pred,
                            f"Recovered vs Real PSD (Subject {self.padded_id} - ROI {self.region_index} - {self.exp})",
                            self.region_index, log_scale=True)
        plt.savefig(os.path.join(plots_dir, f"{prefix}_baseline_psd_log.png"))
        plt.clf()

        raw_metrics = CSDMetricsCalculatorTorch(self.psd, pred).evaluate()
        norm_metrics = CSDMetricsCalculatorTorch(psd_norm, pred_norm).evaluate()

        A_np = self.model.A.detach().cpu().numpy()
        A_serialized = [[[float(elem.real), float(elem.imag)] for elem in row] for row in A_np]

        final_parameters = {
            "A": A_serialized,
            "varphi": float(self.model.varphi.item()),
            "phi": float(self.model.phi.item()),
            "chi": float(self.model.chi.item()),
            "mtt": float(self.model.mtt.item()),
            "tau": float(self.model.tau.item()),
            "alpha_v": float(self.model.alpha_v.item()),
            "beta_v": float(self.model.beta_v.item()),
            "alpha_e": float(self.model.alpha_e.item()),
            "beta_e": float(self.model.beta_e.item()),
            "alpha": float(self.model.alpha.item()),
            "E0": float(self.model.E0.item())
        }

        params = {
            "subject_id": int(self.subject_id),
            "region": int(self.region_index),
            "experiment": self.exp,
            "final_parameters": final_parameters
        }

        metrics = {
            "subject_id": int(self.subject_id),
            "region": int(self.region_index),
            "experiment": self.exp,
            "raw_csd_metrics": {k: float(v) for k, v in raw_metrics.items()},
            "normalised_csd_metrics": {k: float(v) for k, v in norm_metrics.items()}
        }

        with open(os.path.join(subject_dir, f"{prefix}_baseline_results.json"), "w") as f:
            json.dump(params, f, indent=4)

        with open(os.path.join(subject_dir, f"{prefix}_baseline_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved outputs for Subject {self.padded_id}, ROI {self.region_index}, {self.exp}")

def run_spdcm_for_subjects_default(subject_ids, region_indices, exp_list):
    for exp in exp_list:
        for subject_id in subject_ids:
            padded_id = f"{subject_id:03d}"
            print(f"\nRunning Subject {padded_id}, Condition: {exp}")
            for region_index in region_indices:
                try:
                    prefix = f"roi-{region_index}-{exp}"
                    subject_dir = f"/Users/xuenbei/Desktop/finalyearproject/spDCM/spdcm_fitted/sub-{padded_id}"
                    result_file = os.path.join(subject_dir, f"{prefix}_baseline_results.json")

                    if os.path.exists(result_file):
                        print(f"Skipping Subject {padded_id}, ROI {region_index}, {exp} — results already exist.")
                        continue

                    trainer = spDCMTrainer(subject_id, region_index, exp, device)

                    # Default parameters
                    default_params = {
                        "A": [[complex(-0.1, 0.01)]],
                        "varphi": 0.6,
                        "phi": 1.5,
                        "chi": 0.6,
                        "mtt": 2.0,
                        "tau": 4.0,
                        "alpha_v": 1.0,
                        "beta_v": 1.0,
                        "alpha_e": 1.0,
                        "beta_e": 1.0,
                        "alpha": 0.32,
                        "E0": 0.4
                    }

                    trainer.time_start = time.time()
                    trainer.build_model(default_params)
                    trainer.save_outputs()

                except Exception as e:
                    print(f"Failed for Subject {padded_id}, ROI {region_index}, {exp}: {str(e)}")


if __name__ == '__main__':
    exp_list = ["PLCB", "LSD"]
    selected_subjects = [1, 2, 3, 4, 6, 10, 11, 12, 19, 20]
    selected_rois = list(range(100))

    run_spdcm_for_subjects_default(
        subject_ids=selected_subjects,
        region_indices=selected_rois,
        exp_list=exp_list
    )
