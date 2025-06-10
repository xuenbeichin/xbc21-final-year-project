import json
import torch
from itertools import product

from CSD_metrics.CSD_metrics import *
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from helper_functions import normalise_data, filter_frequencies, plot_psd_comparison, load_single_time_series, \
    plot_psd_loss_curve
from scipy_to_torch import torch_csd
from PDCM.PDCMBOLDModel import PDCMMODEL


class GridSearchOptimisationHemodynamic:
    def __init__(self, t, h, time_series, device,
                 grid_sigma, grid_phi, grid_varphi, grid_chi, grid_lamb, grid_mu, grid_tMTT,
                 # New grids for noise parameters:
                 grid_alpha_v, grid_beta_v, grid_alpha_e, grid_beta_e,
                 desired_TR=2.0, fs=0.5, nfft=256, nperseg=128):
        """
        Initializes the grid search optimizer for hemodynamic parameters as well as the noise parameters.

        Args:
            t (np.array or torch.Tensor): Time vector.
            h (float): Integration time step.
            time_series (np.array or torch.Tensor): Real fMRI time series.
            device (str): 'cuda' or 'cpu'.
            grid_sigma, grid_phi, grid_varphi, grid_chi, grid_lamb, grid_mu, grid_tMTT (iterables): Candidate values.
            grid_alpha_v, grid_beta_v, grid_alpha_e, grid_beta_e (iterables): Candidate noise parameter values.
            desired_TR (float): Effective repetition time for simulation.
            fs (float): Sampling frequency for PSD computation.
            nfft (int): nfft parameter for torch_welch.
            nperseg (int): nperseg parameter for torch_welch.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.h = h
        self.t = (torch.tensor(t, dtype=torch.float32, device=self.device)
                  if not torch.is_tensor(t) else t.to(self.device))
        self.time_series = (torch.tensor(time_series, dtype=torch.float32, device=self.device)
                            if not torch.is_tensor(time_series) else time_series.to(self.device))
        self.desired_TR = desired_TR
        self.fs = fs
        self.nfft = nfft
        self.nperseg = nperseg

        self.grid_sigma = grid_sigma
        self.grid_phi = grid_phi
        self.grid_varphi = grid_varphi
        self.grid_chi = grid_chi
        self.grid_lamb = grid_lamb
        self.grid_mu = grid_mu
        self.grid_tMTT = grid_tMTT

        self.grid_alpha_v = grid_alpha_v
        self.grid_beta_v = grid_beta_v
        self.grid_alpha_e = grid_alpha_e
        self.grid_beta_e = grid_beta_e

        # Pre-compute the real PSD for comparison.
        real_freqs, real_psd = torch_csd(self.time_series, self.time_series, fs = 0.5, nperseg = 128, nfft = 512)
        real_freqs, real_psd = filter_frequencies(real_freqs, real_psd, min_freq=0.01, max_freq=0.08)
        self.real_psd = normalise_data(real_psd)
        self.real_freqs = real_freqs

    def _complex_mse_loss(self, predicted, target):
        # Compute mean-squared error for complex-valued tensors.
        predicted = predicted.to(torch.complex64)
        predicted = normalise_data(predicted)
        target = normalise_data(predicted)
        real_loss = torch.mean((predicted.real - target.real) ** 2)
        imag_loss = torch.mean((predicted.imag - target.imag) ** 2)
        return real_loss + imag_loss

    def run_search(self):
        best_loss = float('inf')
        best_params = None
        torch.manual_seed(42)
        np.random.seed(42)

        # Loop over all candidate combinations including noise parameters.
        for sigma_val, phi_val, varphi_val, chi_val, lamb_val, mu_val, tMTT_val, \
                alpha_v_val, beta_v_val, alpha_e_val, beta_e_val in product(
            self.grid_sigma, self.grid_phi, self.grid_varphi, self.grid_chi,
            self.grid_lamb, self.grid_mu, self.grid_tMTT,
            self.grid_alpha_v, self.grid_beta_v, self.grid_alpha_e, self.grid_beta_e
        ):
            # Create tensors for each candidate parameter.
            sigma = torch.tensor(sigma_val, dtype=torch.float32, device=self.device)
            phi = torch.tensor(phi_val, dtype=torch.float32, device=self.device)
            varphi = torch.tensor(varphi_val, dtype=torch.float32, device=self.device)
            chi = torch.tensor(chi_val, dtype=torch.float32, device=self.device)
            lamb = torch.tensor(lamb_val, dtype=torch.float32, device=self.device)
            mu = torch.tensor(mu_val, dtype=torch.float32, device=self.device)
            tMTT = torch.tensor(tMTT_val, dtype=torch.float32, device=self.device)

            alpha_v = torch.tensor(alpha_v_val, dtype=torch.float32, device=self.device)
            beta_v = torch.tensor(beta_v_val, dtype=torch.float32, device=self.device)
            alpha_e = torch.tensor(alpha_e_val, dtype=torch.float32, device=self.device)
            beta_e = torch.tensor(beta_e_val, dtype=torch.float32, device=self.device)

            # Create a new model instance and set its hemodynamic parameters.
            model = PDCMMODEL().to(self.device)
            model.set_params(sigma=sigma, phi=phi, varphi=varphi, chi=chi, lamb=lamb, mu=mu, tMTT=tMTT)

            # Run the simulation with the candidate noise parameters.
            sim_time, sim_bold = simulate_bold_euler_maruyama(
                model, self.t, self.h,
                alpha_v=alpha_v,
                beta_v=beta_v,
                alpha_e=alpha_e,
                beta_e=beta_e,
                desired_TR=self.desired_TR,
                add_obs_noise=True, add_state_noise=True
            )

            # Compute the simulated PSD.
            sim_freqs, sim_psd = torch_csd(sim_bold, sim_bold, fs = 0.5, nperseg = 128, nfft = 512)
            sim_freqs, sim_psd = filter_frequencies(sim_freqs, sim_psd, min_freq=0.01, max_freq=0.1)
            sim_psd = normalise_data(sim_psd)

            # Compute the loss between the simulated and real PSD.
            loss = self._complex_mse_loss(sim_psd, self.real_psd)
            loss_val = loss.item()

            if loss_val < best_loss:
                best_loss = loss_val
                best_params = {
                    "sigma": sigma_val,
                    "phi": phi_val,
                    "varphi": varphi_val,
                    "chi": chi_val,
                    "lamb": lamb_val,
                    "mu": mu_val,
                    "tMTT": tMTT_val,
                    "alpha_v": alpha_v_val,
                    "beta_v": beta_v_val,
                    "alpha_e": alpha_e_val,
                    "beta_e": beta_e_val
                }
                print(f"New best: {best_params} with loss: {best_loss:.6f}")

        print("Grid search complete.")
        return best_params, best_loss

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    region1_index = 52  # e.g., ROI39
    region1_file = f'/Users/xuenbei/Desktop/finalyearproject/extract_time_series/time_series/sub-002-PLCB-ROI{region1_index}.txt'
    time_series = load_single_time_series(region1_file)
    time_series = torch.tensor(time_series, dtype=torch.float32, device=device)

    h = 0.1
    t = np.arange(0, 434, h)
    t = torch.tensor(t, dtype=torch.float32, device=device)

    # Define grids for hemodynamic parameters.
    grid_sigma = np.linspace(0.5, 1.5, 1)
    grid_phi = np.linspace(0.6, 2.0, 1)
    grid_varphi = np.linspace(1.5, 1.8, 1)
    grid_chi = np.linspace(0.6, 0.8, 1)
    grid_lamb = np.linspace(0.2, 0.3, 1)
    grid_mu = np.linspace(0.5, 1.5, 1)
    grid_tMTT = np.linspace(2.0, 5.0, 1)

    # Define grids for noise parameters.
    grid_alpha_v = np.linspace(0.5, 10, 10)
    grid_beta_v = np.linspace(0.5, 10, 10)
    grid_alpha_e = np.linspace(0.5, 10, 10)
    grid_beta_e = np.linspace(0.5, 10, 10)

    grid_search = GridSearchOptimisationHemodynamic(
        t=t, h=h, time_series=time_series, device=device,
        grid_sigma=grid_sigma, grid_phi=grid_phi, grid_varphi=grid_varphi, grid_chi=grid_chi,
        grid_lamb=grid_lamb, grid_mu=grid_mu, grid_tMTT=grid_tMTT,
        grid_alpha_v=grid_alpha_v, grid_beta_v=grid_beta_v, grid_alpha_e=grid_alpha_e, grid_beta_e=grid_beta_e,
        desired_TR=2.0, fs=0.5, nperseg=128, nfft = 512
    )

    optimised_params, grid_loss = grid_search.run_search()

    # Create a model instance using the optimised parameters.
    model_grid_final = PDCMMODEL().to(device)
    model_grid_final.set_params(sigma=optimised_params["sigma"], phi=optimised_params["phi"],
                                varphi=optimised_params["varphi"],
                                chi=optimised_params["chi"], lamb=optimised_params["lamb"],
                                mu=optimised_params["mu"], tMTT=optimised_params["tMTT"])

    down_time, final_signal_grid = simulate_bold_euler_maruyama(
        model_grid_final, t, h,
        alpha_v=torch.tensor(optimised_params["alpha_v"], device=device),
        beta_v=torch.tensor(optimised_params["beta_v"], device=device),
        alpha_e=torch.tensor(optimised_params["alpha_e"], device=device),
        beta_e=torch.tensor(optimised_params["beta_e"], device=device),
        desired_TR=2.0,  # Effective TR = 2 s
        add_obs_noise=True,
        add_state_noise=True
    )

    final_signal_grid = torch.tensor(final_signal_grid, dtype=torch.float32, device=device)
    final_freqs_grid, final_psd_grid = torch_csd(final_signal_grid, final_signal_grid, fs=0.5, nperseg = 64)
    final_freqs_grid, final_psd_grid = filter_frequencies(final_freqs_grid, final_psd_grid, min_freq=0.01,
                                                          max_freq=0.1)
    final_psd_grid = normalise_data(final_psd_grid)

    real_freqs, real_psd = torch_csd(time_series, time_series, fs = 0.5, nperseg = 64)
    real_psd = normalise_data(real_psd)
    real_freqs, real_psd = filter_frequencies(real_freqs, real_psd, min_freq=0.01, max_freq=0.1)

    plot_psd_loss_curve([grid_loss], 'Loss Curve - Grid Search', 'Grid Search Loss', region1_index)
    plot_psd_comparison(real_freqs, real_psd, final_psd_grid,
                        'True vs. Recovered PSD (Grid Search)', region1_index)

    print("\nBest Parameters from Grid Search:")
    for k, v in optimised_params.items():
        print(f"{k}: {v}")
    print(f"\nFinal grid search loss: {grid_loss:.6f}")

    # Save the best parameters and loss.
    with open("grid_best_params.json", "w") as f:
        json.dump(optimised_params, f, indent=4)
