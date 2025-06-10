import time
import torch
import torch.nn as nn
import optuna
from matplotlib import pyplot as plt

from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from PDCM.PDCMBOLDModel import PDCMMODEL
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from scipy_to_torch import torch_csd
from helper_functions import filter_frequencies, plot_psd_comparison, plot_psd_loss_curve, normalise_complex_psd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def complex_mse_loss(x, y, fs=0.5):
    x = x / x.std()
    y = y / y.std()
    f1, csd_x = torch_csd(x, x, fs=fs)
    f2, csd_y = torch_csd(y, y, fs=fs)
    _, csd_x = filter_frequencies(f1, csd_x, 0.01, 0.1)
    _, csd_y = filter_frequencies(f2, csd_y, 0.01, 0.1)
    return torch.mean((csd_x.real - csd_y.real) ** 2) + torch.mean((csd_x.imag - csd_y.imag) ** 2)


class TrainableBOLDParams(nn.Module):
    def __init__(self, init_dict, device):
        super().__init__()
        self.device = device
        for k, v in init_dict.items():
            setattr(self, k, nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device)))

    def get_model(self):
        return PDCMMODEL(
            phi=self.phi.item(), varphi=self.varphi.item(), chi=self.chi.item(),
            mtt=self.mtt.item(), tau=self.tau.item(), sigma=self.sigma.item(),
            mu=self.mu.item(), lamb=self.lamb.item(), alpha=self.alpha.item(), E0=self.E0.item(),
            ignore_range=True, device=self.device
        )


class PDCMOptunaAdamWTrainer:
    def __init__(self, device, h=0.01):
        self.device = device
        self.fs = 0.5
        self.h = h
        self.t_sim = torch.arange(0, 800, h, device=device)

        # Ground truth parameters for simulation
        self.true_params = {
            'phi': 1.5, 'varphi': 0.6, 'chi': 0.6, 'mtt': 2.0, 'tau': 4.0,
            'sigma': 1.0, 'mu': 1.0, 'lamb': 0.1, 'alpha': 0.3, 'E0': 0.4,
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
            model, self.t_sim, h=self.h,
            alpha_v=params['alpha_v'], beta_v=params['beta_v'],
            alpha_e=params['alpha_e'], beta_e=params['beta_e'],
            desired_TR=2.0, add_state_noise=True, add_obs_noise=True
        )
        return bold

    def optuna_objective(self, trial):
        init = {
            'phi': trial.suggest_float("phi", 1.0, 2.0),
            'varphi': trial.suggest_float("varphi", 0.1, 1.0),
            'chi': trial.suggest_float("chi", 0.1, 1.0),
            'mtt': trial.suggest_float("mtt", 0.5, 5.0),
            'tau': trial.suggest_float("tau", 1.0, 10.0),
            'sigma': trial.suggest_float("sigma", 0.5, 2.0),
            'mu': trial.suggest_float("mu", 0.0, 2.0),
            'lamb': trial.suggest_float("lamb", 0.0, 0.5),
            'alpha': trial.suggest_float("alpha", 0.1, 0.5),
            'E0': trial.suggest_float("E0", 0.2, 0.6),
            'alpha_v': trial.suggest_float("alpha_v", 0.1, 1.0),
            'beta_v': trial.suggest_float("beta_v", 0.1, 1.0),
            'alpha_e': trial.suggest_float("alpha_e", 0.1, 1.0),
            'beta_e': trial.suggest_float("beta_e", 0.1, 1.0)
        }

        model = PDCMMODEL(
            phi=init['phi'], varphi=init['varphi'], chi=init['chi'],
            mtt=init['mtt'], tau=init['tau'], sigma=init['sigma'],
            mu=init['mu'], lamb=init['lamb'], alpha=init['alpha'], E0=init['E0'],
            ignore_range=True, device=self.device
        )
        _, yhat = simulate_bold_euler_maruyama(
            model=model, time=self.t_sim, h=self.h,
            alpha_v=init['alpha_v'], beta_v=init['beta_v'],
            alpha_e=init['alpha_e'], beta_e=init['beta_e'],
            desired_TR=2.0, add_state_noise=True, add_obs_noise=True
        )
        loss = complex_mse_loss(yhat, self.real_bold, fs=self.fs)
        return loss.item()

    def run_optuna(self, n_trials=50):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.optuna_objective, n_trials=n_trials)
        return study.best_params

    def train_with_adamw(self, init_params, num_epochs=1000):
        self.model = TrainableBOLDParams(init_params, self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        self.losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            bold_model = self.model.get_model()
            _, yhat = simulate_bold_euler_maruyama(
                model=bold_model, time=self.t_sim, h=self.h,
                alpha_v=self.model.alpha_v.item(), beta_v=self.model.beta_v.item(),
                alpha_e=self.model.alpha_e.item(), beta_e=self.model.beta_e.item(),
                desired_TR=2.0, add_state_noise=True, add_obs_noise=True
            )
            loss = complex_mse_loss(yhat, self.real_bold, fs=self.fs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.losses.append(loss.item())
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

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

        plot_psd_comparison(f1, csd_true, csd_pred, title="PSD Comparison", region1_index=0, log_scale=False)
        plt.show()

        plot_psd_loss_curve(self.losses, title="AdamW Loss Curve", label="Loss", region1_index=0)
        plt.show()


if __name__ == "__main__":
    trainer = PDCMOptunaAdamWTrainer(device)
    best_init = trainer.run_optuna(n_trials=50)

    trainer.train_with_adamw(best_init, num_epochs=1000)

    trainer.plot_results()
