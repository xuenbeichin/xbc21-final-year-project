import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from PDCM.PDCMBOLDModel import PDCMMODEL
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from helper_functions import plot_psd_comparison, plot_psd_loss_curve, filter_frequencies, normalise_complex_psd
from scipy_to_torch import torch_csd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def complex_mse_loss(output, target, fs=0.5):
    output = output / output.std()
    target = target / target.std()

    f1, csd_x = torch_csd(output, output, fs=fs)
    _, csd_x = filter_frequencies(f1, csd_x, 0.01, 0.1)

    f2, csd_y = torch_csd(target, target, fs=fs)
    _, csd_y = filter_frequencies(f2, csd_y, 0.01, 0.1)

    return torch.mean((csd_x.real - csd_y.real) ** 2) + torch.mean((csd_x.imag - csd_y.imag) ** 2)


class TrainableBOLDParams(nn.Module):
    def __init__(self, init, device):
        super().__init__()
        self.device = device
        for k, v in init.items():
            setattr(self, k, nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device)))

    def get_model(self):
        return PDCMMODEL(
            phi=self.phi.item(), varphi=self.varphi.item(), chi=self.chi.item(),
            mtt=self.mtt.item(), tau=self.tau.item(), sigma=self.sigma.item(),
            mu=self.mu.item(), lamb=self.lamb.item(), alpha=self.alpha.item(), E0=self.E0.item(),
            ignore_range=True, device=self.device
        )


class PDCMAdamWTrainer:
    def __init__(self, device, h=0.01):
        self.device = device
        self.h = h
        self.fs = 0.5
        self.t_sim = torch.arange(0, 800, self.h, device=device)
        self.true_params = {
            'phi': 1.5, 'varphi': 0.6, 'chi': 0.6, 'mtt': 2.0, 'tau': 4.0,
            'sigma': 1.0, 'mu': 1.0, 'lamb': 0.1,
            'alpha': 0.3, 'E0': 0.4,
            'alpha_v': 0.5, 'beta_v': 0.5, 'alpha_e': 0.5, 'beta_e': 0.5
        }

        self.real_bold = self.simulate_bold(self.true_params)

    def simulate_bold(self, params):
        model = PDCMMODEL(
            phi=params['phi'], varphi=params['varphi'], chi=params['chi'],
            mtt=params['mtt'], tau=params['tau'], sigma=params['sigma'],
            mu=params['mu'], lamb=params['lamb'], alpha=params['alpha'], E0=params['E0'],
            ignore_range=True, device=self.device
        )

        _, bold = simulate_bold_euler_maruyama(
            model=model, time=self.t_sim, h=self.h,
            alpha_v=params['alpha_v'], beta_v=params['beta_v'],
            alpha_e=params['alpha_e'], beta_e=params['beta_e'],
            desired_TR=2.0, add_state_noise=True, add_obs_noise=True
        )
        return bold

    def train(self, num_epochs=1000):
        init_guess = {
            'phi': 1.2, 'varphi': 0.5, 'chi': 0.5, 'mtt': 2.5, 'tau': 5.0,
            'sigma': 0.8, 'mu': 0.8, 'lamb': 0.05,
            'alpha': 0.25, 'E0': 0.35,
            'alpha_v': 0.4, 'beta_v': 0.4, 'alpha_e': 0.4, 'beta_e': 0.4
        }

        model = TrainableBOLDParams(init_guess, self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        self.losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            bold_model = model.get_model()
            _, yhat = simulate_bold_euler_maruyama(
                model=bold_model, time=self.t_sim, h=self.h,
                alpha_v=model.alpha_v.item(), beta_v=model.beta_v.item(),
                alpha_e=model.alpha_e.item(), beta_e=model.beta_e.item(),
                desired_TR=2.0, add_state_noise=True, add_obs_noise=True
            )

            loss = complex_mse_loss(yhat, self.real_bold, fs=self.fs)
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.losses.append(loss.item())
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

        self.model = model
        self.final_yhat = yhat.detach().cpu()
        self.real_bold = self.real_bold.cpu()

    def plot_results(self):
        f1, csd_true = torch_csd(self.real_bold, self.real_bold, fs=self.fs)
        f2, csd_pred = torch_csd(self.final_yhat, self.final_yhat, fs=self.fs)
        f1, csd_true = filter_frequencies(f1, csd_true, 0.01, 0.1)
        f2, csd_pred = filter_frequencies(f2, csd_pred, 0.01, 0.1)

        csd_norm_true = normalise_complex_psd(csd_true)
        csd_norm_pred = normalise_complex_psd(csd_pred)

        raw_metrics = CSDMetricsCalculatorTorch(csd_true, csd_pred).evaluate()
        norm_metrics = CSDMetricsCalculatorTorch(csd_norm_true, csd_norm_pred).evaluate()

        print("Raw CSD Metrics:")
        for k, v in raw_metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nNormalised CSD Metrics:")
        for k, v in norm_metrics.items():
            print(f"  {k}: {v:.4f}")

        plot_psd_comparison(f1, csd_true, csd_pred, title="Simulated vs Fitted PSD", region1_index=0, log_scale=False)
        plt.show()

        plot_psd_loss_curve(self.losses, title="Training Loss (AdamW)", label="Loss", region1_index=0)
        plt.show()


if __name__ == "__main__":
    trainer = PDCMAdamWTrainer(device)
    trainer.train(num_epochs=1)
    trainer.plot_results()
