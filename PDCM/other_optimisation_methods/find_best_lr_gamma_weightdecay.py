from torch import optim
import torch
import numpy as np
from PDCM.PDCMBOLDModel import PDCMMODEL
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from euler_maruyama import simulate_bold_euler_maruyama
from scipy_to_torch import torch_csd
from helper_functions import load_single_time_series, filter_frequencies, normalise_data, plot_psd_loss_curve, plot_psd_comparison, plot_parameter_error

#  Full Optimisation Class
class FullPDCMOptimisation:
    """
    Jointly optimizes the noise parameters (α_v, β_v, α_e, β_e) and
    the neurovascular parameters so that the PSD of the simulated
    BOLD signal matches a target PSD provided externally.
    """
    def __init__(self, t, h, real_psd, real_freqs, device,
                 num_epochs=100, lr=0.01, weight_decay = 1e-4, gamma = 0.99, fs=0.5,
                 initial_alpha_v=1.0, initial_beta_v=1.0,
                 initial_alpha_e=1.0, initial_beta_e=1.0,
                 initial_phi=1.0, initial_varphi=0.6, initial_chi=0.6,
                 initial_tMTT=2.0, initial_tau = 4.0,
                 initial_lamb=0.2, initial_sigma=0.5, initial_mu=0.4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.lr = lr
        self.fs = fs
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.h = h
        self.t = (torch.tensor(t, dtype=torch.float32, device=self.device)
                  if not torch.is_tensor(t) else t.to(self.device))

        # Store the externally computed target PSD and frequencies.
        self.real_psd = real_psd
        self.real_freqs = real_freqs


        # Noise parameters (learnable)
        self.alpha_v = torch.tensor(initial_alpha_v, dtype=torch.float32, requires_grad=True, device=self.device)
        self.beta_v = torch.tensor(initial_beta_v, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_e = torch.tensor(initial_alpha_e, dtype=torch.float32, requires_grad=True, device=self.device)
        self.beta_e = torch.tensor(initial_beta_e, dtype=torch.float32, requires_grad=True, device=self.device)

        # Neurovascular parameters (learnable)
        self.phi = torch.tensor(initial_phi, dtype=torch.float32, requires_grad=True, device=self.device)
        self.varphi = torch.tensor(initial_varphi, dtype=torch.float32, requires_grad=True, device=self.device)
        self.chi = torch.tensor(initial_chi, dtype=torch.float32, requires_grad=True, device=self.device)
        self.tMTT = torch.tensor(initial_tMTT, dtype=torch.float32, requires_grad=True, device=self.device)
        self.tau = torch.tensor(initial_tau, dtype=torch.float32, requires_grad=True, device=self.device)
        self.lamb = torch.tensor(initial_lamb, dtype=torch.float32, requires_grad=True, device=self.device)
        self.sigma = torch.tensor(initial_sigma, dtype=torch.float32, requires_grad=True, device=self.device)
        self.mu = torch.tensor(initial_mu, dtype=torch.float32, requires_grad=True, device=self.device)

        # Create optimizer for all 11 parameters.
        self.optimizer = optim.AdamW(
            [self.alpha_v, self.beta_v, self.alpha_e, self.beta_e,
             self.phi, self.varphi, self.chi, self.tMTT, self.lamb, self.sigma, self.mu],
            lr=self.lr, weight_decay = self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

        # Initialize history storage.
        self.loss_history = []
        self.param_history = {
            "alpha_v": [],
            "beta_v": [],
            "alpha_e": [],
            "beta_e": [],
            "phi": [],
            "varphi": [],
            "chi": [],
            "tMTT": [],
            "lamb": [],
            "sigma": [],
            "mu": []
        }

    def _complex_mse_loss(self, predicted, target):
        # Compute MSE over real and imaginary parts.
        predicted = predicted.to(torch.complex64)
        target = target.to(torch.complex64)
        #predicted = normalise_data(predicted)
        #target = normalise_data(target)
        real_loss = torch.mean((predicted.real - target.real) ** 2)
        imag_loss = torch.mean((predicted.imag - target.imag) ** 2)
        return real_loss + imag_loss

    def run_optimisation(self):
        best_loss = float('inf')
        best_params = {}

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            model = PDCMMODEL().to(self.device)
            model.set_params(
                phi=self.phi,
                varphi=self.varphi,
                chi=self.chi,
                tMTT=self.tMTT,
                tau = self.tau,
                lamb=self.lamb,
                sigma=self.sigma,
                mu=self.mu
            )
            sim_time, sim_bold = simulate_bold_euler_maruyama(
                model, self.t, self.h,
                alpha_v=self.alpha_v,
                beta_v=self.beta_v,
                alpha_e=self.alpha_e,
                beta_e=self.beta_e,
                desired_TR=2.0,
                add_obs_noise=True,
                add_state_noise=True,
            )
            sim_freqs, sim_psd = torch_csd(sim_bold, sim_bold, fs=self.fs, nperseg=64, nfft = 64)
            sim_freqs, sim_psd = filter_frequencies(sim_freqs, sim_psd, min_freq=0.01, max_freq=0.1)

            loss = self._complex_mse_loss(sim_psd, self.real_psd)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [self.alpha_v, self.beta_v, self.alpha_e, self.beta_e,
                 self.phi, self.varphi, self.chi, self.tMTT, self.lamb, self.sigma, self.mu],
                max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()

            self.loss_history.append(loss.item())
            self.param_history["alpha_v"].append(self.alpha_v.detach().cpu().item())
            self.param_history["beta_v"].append(self.beta_v.detach().cpu().item())
            self.param_history["alpha_e"].append(self.alpha_e.detach().cpu().item())
            self.param_history["beta_e"].append(self.beta_e.detach().cpu().item())
            self.param_history["phi"].append(self.phi.detach().cpu().item())
            self.param_history["varphi"].append(self.varphi.detach().cpu().item())
            self.param_history["chi"].append(self.chi.detach().cpu().item())
            self.param_history["tMTT"].append(self.tMTT.detach().cpu().item())
            self.param_history["lamb"].append(self.lamb.detach().cpu().item())
            self.param_history["sigma"].append(self.sigma.detach().cpu().item())
            self.param_history["mu"].append(self.mu.detach().cpu().item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {
                    "alpha_v": self.alpha_v.item(),
                    "beta_v": self.beta_v.item(),
                    "alpha_e": self.alpha_e.item(),
                    "beta_e": self.beta_e.item(),
                    "phi": self.phi.item(),
                    "varphi": self.varphi.item(),
                    "chi": self.chi.item(),
                    "tMTT": self.tMTT.item(),
                    "tau": self.tau.item(),
                    "lamb": self.lamb.item(),
                    "sigma": self.sigma.item(),
                    "mu": self.mu.item()
                }

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.6f}")

        return best_params, self.loss_history, self.param_history

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    region1_index = 0  # Example ROI index.
    exp = "LSD"
    region1_file = f'/Users/xuenbei/Desktop/finalyearproject/time_series/sub-002-{exp}-ROI{region1_index}.txt'
    time_series = load_single_time_series(region1_file)
    time_series = torch.tensor(time_series, dtype=torch.float32, device=device)


    TR = 2.0
    h = 0.01
    T = 434
    # Create a simulation time vector that matches the target time series length.
    t =  np.arange(0, T, h)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)

    # Compute target PSD externally using torch_csd.
    real_freqs, real_psd = torch_csd(time_series, time_series, fs=0.5, nperseg=128, nfft = 512)
    real_freqs, real_psd = filter_frequencies(real_freqs, real_psd, min_freq=0.01, max_freq=0.1)
    real_psd_tensor = torch.tensor(real_psd, dtype=torch.complex64, device = device)
    real_freqs_tensor = torch.tensor(real_freqs, dtype=torch.float32, device=device)
    # Define grid search values
    lr_list = [1e-3, 5e-3, 1e-2]
    wd_list = [0, 1e-4, 1e-3]
    gamma_list = [0.95, 0.98, 0.99]

    results = []

    for lr in lr_list:
        for wd in wd_list:
            for gamma in gamma_list:
                print(f"\n🔧 Testing: LR={lr}, Weight Decay={wd}, Gamma={gamma}")

                # Instantiate and run the full optimization
                full_optimizer = FullPDCMOptimisation(
                    t=t_tensor, h=h,
                    real_psd=real_psd_tensor, real_freqs=real_freqs_tensor,
                    device=device, num_epochs=10,
                    lr=lr, weight_decay=wd, gamma=gamma
                )

                best_params, loss_history, param_history = full_optimizer.run_optimisation()
                final_loss = loss_history[-1]

                results.append({
                    "lr": lr,
                    "weight_decay": wd,
                    "gamma": gamma,
                    "final_loss": final_loss,
                    "best_params": best_params
                })

                print(f"Done: Final Loss = {final_loss:.6f}")
