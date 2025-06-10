import torch
from itertools import product

from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch
from PDCM.PDCMBOLDModel import PDCMMODEL
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from helper_functions import plot_psd_comparison, plot_psd_loss_curve, filter_frequencies, normalise_complex_psd
from scipy_to_torch import torch_csd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def complex_mse_loss(output, target, fs=0.5):
    output = output / output.std()
    target = target / target.std()
    f1, csd_x = torch_csd(output, output, fs=fs)
    _, csd_x = filter_frequencies(f1, csd_x, 0.01, 0.1)
    f2, csd_y = torch_csd(target, target, fs=fs)
    _, csd_y = filter_frequencies(f2, csd_y, 0.01, 0.1)
    return torch.mean((csd_x.real - csd_y.real) ** 2) + torch.mean((csd_x.imag - csd_y.imag) ** 2)


class PDCMGridSearch:
    def __init__(self, device, h=0.01):
        self.device = device
        self.h = h
        self.fs = 0.5
        self.t_sim = torch.arange(0, 800, h, device=device)
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
            mu=params['mu'], lamb=params['lamb'],
            alpha=params['alpha'], E0=params['E0'],
            ignore_range=True, device=self.device
        )
        _, bold = simulate_bold_euler_maruyama(
            model=model, time=self.t_sim, h=self.h,
            alpha_v=params['alpha_v'], beta_v=params['beta_v'],
            alpha_e=params['alpha_e'], beta_e=params['beta_e'],
            desired_TR=2.0, add_state_noise=True, add_obs_noise=True
        )
        return bold

    def run_grid_search(self):
        print("Running full-parameter grid search...")

        # Keep 2 values per parameter for feasibility (2^14 = 16,384)
        grid_values = {
            'phi': [1.3, 1.5],
            'varphi': [0.5, 0.6],
            'chi': [0.5, 0.6],
            'mtt': [1.8, 2.0],
            'tau': [3.5, 4.0],
            'sigma': [0.8, 1.0],
            'mu': [0.8, 1.0],
            'lamb': [0.05, 0.1],
            'alpha': [0.25, 0.3],
            'E0': [0.35, 0.4],
            'alpha_v': [0.4, 0.5],
            'beta_v': [0.4, 0.5],
            'alpha_e': [0.4, 0.5],
            'beta_e': [0.4, 0.5]
        }

        param_names = list(grid_values.keys())
        param_grid = list(product(*[grid_values[k] for k in param_names]))

        best_loss = float('inf')
        best_params = None
        best_bold = None

        for i, values in enumerate(param_grid):
            trial_params = dict(zip(param_names, values))
            try:
                yhat = self.simulate_bold(trial_params)
                loss = complex_mse_loss(yhat, self.real_bold, fs=self.fs)
                print(f"[{i + 1}/{len(param_grid)}] Loss: {loss.item():.6f}")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = trial_params.copy()
                    best_bold = yhat
            except Exception as e:
                print(f"Simulation failed at index {i}: {str(e)}")

        self.best_params = best_params
        self.final_yhat = best_bold.detach().cpu()
        self.real_bold = self.real_bold.cpu()

        print("\nBest Loss:", best_loss)
        print("Best Params:", best_params)

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

        plot_psd_comparison(f1, csd_true, csd_pred, title="Best PSD Fit (Grid Search)", region1_index=0, log_scale=False)
        plt.show()


if __name__ == '__main__':
    trainer = PDCMGridSearch(device)
    trainer.run_grid_search()
    trainer.plot_results()
