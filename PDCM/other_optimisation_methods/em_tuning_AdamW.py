import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import json
from PDCM.PDCMBOLDModel import PDCMMODEL
from PDCM.euler_maruyama import simulate_bold_euler_maruyama
from helper_functions import (
    load_single_time_series,
    filter_frequencies,
    plot_psd_comparison
)
import time

from scipy_to_torch import torch_csd
from CSD_metrics.CSD_metrics import CSDMetricsCalculatorTorch

class PDCMBOLDFitter(nn.Module):
    """
    A PyTorch neural module to fit the parameters of a PDCM BOLD model.
    It contains learnable parameters and performs simulation of BOLD signals
    using the Euler-Maruyama method.
    """

    def __init__(self, device=None):
        """
        Initialize the PDCM BOLD fitter with default trainable parameters.

        Args:
            device (str, optional): Device for computation (e.g., 'cuda', 'cpu').
        """
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PDCMMODEL(device=self.device)

        self.sigma = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.tensor(0.4))
        self.lamb = nn.Parameter(torch.tensor(0.2))
        self.phi = nn.Parameter(torch.tensor(1.5))
        self.chi = nn.Parameter(torch.tensor(0.6))
        self.varphi = nn.Parameter(torch.tensor(0.6))
        self.tMTT = nn.Parameter(torch.tensor(2.0))
        self.tau = nn.Parameter(torch.tensor(4.0))
        self.alpha_v = nn.Parameter(torch.tensor(0.8))
        self.beta_v = nn.Parameter(torch.tensor(3.0))
        self.alpha_e = nn.Parameter(torch.tensor(13.0))
        self.beta_e = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        """
        Simulates the BOLD signal using the current model parameters.

        Returns:
            Tensor: Simulated BOLD time series.
        """
        self.model.set_params(
            sigma=self.sigma, mu=self.mu, lamb=self.lamb,
            phi=self.phi, chi=self.chi, varphi=self.varphi,
            mtt=self.mtt, tau=self.tau
        )

        T, h, TR = 434, 0.01, 2.0
        time = torch.arange(0, T, h, device=self.device)

        _, bold = simulate_bold_euler_maruyama(
            model=self.model, time=time, h=h,
            alpha_v=self.alpha_v, beta_v=self.beta_v,
            alpha_e=self.alpha_e, beta_e=self.beta_e,
            desired_TR=TR,
            add_state_noise=True, add_obs_noise=True
        )
        return bold


def complex_mse_loss(pred_bold, target_csd, freqs_obs):
    """
    Computes the complex MSE loss between predicted and target CSDs.

    Args:
        pred_bold (Tensor): Predicted BOLD signal.
        target_csd (Tensor): Target cross-spectral density (CSD).
        freqs_obs (Tensor): Frequency vector corresponding to target CSD.

    Returns:
        Tensor: Scalar loss value.
    """
    pred_bold = pred_bold / torch.std(pred_bold)
    f_pred, csd_pred = torch_csd(pred_bold, pred_bold, fs=0.5, nperseg=128, nfft=512)

    f_pred, csd_pred = filter_frequencies(f_pred, csd_pred, 0.01, 0.1)
    f_obs, target_csd = filter_frequencies(freqs_obs, target_csd, 0.01, 0.1)

    #csd_pred = csd_pred/csd_pred.std()
    #target_csd = target_csd/target_csd.std()

    loss_real = torch.mean((csd_pred.real - target_csd.real) ** 2)
    loss_imag = torch.mean((csd_pred.imag - target_csd.imag) ** 2)
    return loss_real + loss_imag


def train_model(observed_csd, f_ref, epochs=50, lr=1e-4, wd=1e-3, gamma=0.99, device='cuda'):
    """
    Trains a single PDCM BOLD model on the observed spectral data.

    Args:
        observed_csd (Tensor): Empirical cross-spectral density.
        f_ref (Tensor): Frequency vector.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        wd (float): Weight decay.
        gamma (float): LR scheduler decay factor.
        device (str): Device for computation.

    Returns:
        model (nn.Module): Trained model.
        losses (list): Loss values across epochs.
    """
    model = PDCMBOLDFitter(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    losses = []
    for epoch in range(epochs):
        print(f"[Epoch {epoch}] ", end="")
        pred_bold = model()
        loss = complex_mse_loss(pred_bold, observed_csd, f_ref)
        print(f"Loss: {loss.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return model, losses

def train_multiple_initializations(
    observed_csd,
    f_ref,
    num_models=5,
    num_epochs=40,
    lr=1e-3,
    wd=1e-4,
    gamma=0.9,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Trains multiple instances of the PDCM BOLD fitter with different initializations
    and returns the best-performing model based on final loss.

    Args:
        observed_csd (Tensor): Empirical cross-spectral density.
        f_ref (Tensor): Corresponding frequency vector.
        num_models (int): Number of random initializations.
        num_epochs (int): Training epochs per model.
        lr (float): Learning rate.
        wd (float): Weight decay.
        gamma (float): Exponential LR decay factor.
        device (str): Device string for PyTorch.

    Returns:
        best_model (nn.Module): Trained model with lowest final loss.
        all_losses (list): List of loss trajectories for each model.
    """
    all_losses = []
    best_model = None
    best_final_loss = float('inf')

    for run in range(num_models):
        print(f"\n Training model {run + 1}/{num_models}")
        model = PDCMBOLDFitter(device=device).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        run_losses = []
        for epoch in range(num_epochs):
            model.train()
            pred_bold = model()

            loss = complex_mse_loss(pred_bold, observed_csd, f_ref)
            run_losses.append(loss.item())

            print(f"    Epoch {epoch} | Loss: {loss.item():.6f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        all_losses.append(run_losses)

        final_loss = run_losses[-1]
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_model = model
            print(f"  New best model (loss: {final_loss:.6f})")

    print("\n Best Model Parameters:")
    print_model_params(best_model)

    return best_model, all_losses


def train_with_random_noise_hyperparams(observed_csd, f_ref, epochs=10, device='cuda'):
    """
    Trains a model with random α/β noise parameters.

    Args:
        observed_csd (Tensor): Empirical CSD.
        f_ref (Tensor): Frequencies of the observed CSD.
        epochs (int): Number of training epochs.
        device (str): Computation device.

    Returns:
        model (nn.Module): Trained model.
        losses (list): List of losses per epoch.
    """    # Randomly generate values in reasonable physiological/log-spaced ranges
    alpha_v = random.uniform(0.1, 5.0)
    beta_v  = random.uniform(0.1, 5.0)
    alpha_e = random.uniform(0.1, 5.0)
    beta_e  = random.uniform(0.1, 5.0)

    print(f"\n Training with Random α/β:")
    print(f"alpha_v = {alpha_v:.3f}, beta_v = {beta_v:.3f}")
    print(f"alpha_e = {alpha_e:.3f}, beta_e = {beta_e:.3f}")

    model = PDCMBOLDFitter(device=device).to(device)

    # Manually set the randomized α/β values
    with torch.no_grad():
        model.alpha_v.fill_(alpha_v)
        model.beta_v.fill_(beta_v)
        model.alpha_e.fill_(alpha_e)
        model.beta_e.fill_(beta_e)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    losses = []

    for epoch in range(epochs):
        model.train()
        pred_bold = model()
        loss = complex_mse_loss(pred_bold, observed_csd, f_ref)

        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    print_model_params(model, "Final Parameters (Random α/β Init)")

    return model, losses

import random

def train_random_noise_combinations(observed_csd, f_ref, num_trials=4, epochs=5, device='cuda'):
    """
    Trains multiple models with randomly sampled α/β combinations and selects the best.

    Args:
        observed_csd (Tensor): Empirical CSD.
        f_ref (Tensor): Frequency vector.
        num_trials (int): Number of random hyperparameter trials.
        epochs (int): Epochs per trial.
        device (str): Computation device.

    Returns:
        best_model (nn.Module): Best-performing model.
        results (list): List of loss and parameter tuples.
        losses (list): Per-epoch losses for each trial.
    """
    best_model = None
    best_loss = float('inf')
    best_config = None

    results = []
    losses = []

    for trial in range(num_trials):
        print(f"\n Trial {trial + 1}/{num_trials}")

        # Randomize alphas and betas
        alpha_v = random.uniform(0.1, 25.0)
        beta_v = random.uniform(0.05, 2.0)
        alpha_e = random.uniform(0.1, 25.0)
        beta_e = random.uniform(0.05, 2.0)

        # Initialize model and overwrite those params
        model = PDCMBOLDFitter(device=device).to(device)
        with torch.no_grad():
            model.alpha_v.fill_(alpha_v)
            model.beta_v.fill_(beta_v)
            model.alpha_e.fill_(alpha_e)
            model.beta_e.fill_(beta_e)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        trial_losses = []

        for epoch in range(epochs):
            model.train()
            pred_bold = model()
            loss = complex_mse_loss(pred_bold, observed_csd, f_ref)
            trial_losses.append(loss.item())
            print(f"   Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_loss = trial_losses[-1]
        results.append((final_loss, (alpha_v, beta_v, alpha_e, beta_e)))
        losses.append(trial_losses)

        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            best_config = (alpha_v, beta_v, alpha_e, beta_e)

    print("\n Best Noise Configuration:")
    print(f"alpha_v = {best_config[0]:.4f}, beta_v = {best_config[1]:.4f}")
    print(f"alpha_e = {best_config[2]:.4f}, beta_e = {best_config[3]:.4f}")
    print(f"Final Loss = {best_loss:.6f}")

    return best_model, results, losses



def print_model_params(model: nn.Module, title="Learned Parameters"):
    """
    Prints all trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The trained model.
        title (str): Optional title for the parameter section.
    """
    print(f"\n {title}")
    for name, param in model.named_parameters():
        print(f"{name:10s} = {param.item():.4f}")

def save_loss_json(losses, path="losses.json"):
    """
    Saves a list of loss values to a JSON file.

    Args:
        losses (list): List of loss values.
        path (str): Output file path.
    """
    with open(path, "w") as f:
        json.dump(losses, f)

def torch_rand(min, max, device=None):
    """
    Samples a uniform random value from [min, max) on a given device.

    Args:
        min (float): Lower bound.
        max (float): Upper bound.
        device (str): Torch device.

    Returns:
        Tensor: A random scalar tensor.
    """
    return torch.rand((), device=device) * (max - min) + min


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and Normalise Data
    roi = 52
    exp_tag = "PLCB"
    sub = "002"
    path = f'/Users/xuenbei/Desktop/finalyearproject/time_series/sub-{sub}-{exp_tag}-ROI{roi}.txt'

    time_series = load_single_time_series(path)
    observed_bold = torch.tensor(time_series, dtype=torch.float32, device=device)
    observed_bold = observed_bold / torch.std(observed_bold)

    TR = 2.0  # seconds
    # Create time axis for the real data.
    time_axis = np.arange(0, len(time_series) * TR, TR)

    # Compute Ground Truth Spectral Data
    f_ref, csd_empirical = torch_csd(observed_bold, observed_bold, fs=0.5, nperseg=128, nfft=512)
    f_ref, csd_empirical = filter_frequencies(f_ref, csd_empirical, 0.01, 0.1)

    start_time = time.time()

    # Train
    #model, losses = train_model(csd_empirical, f_ref, epochs=1, device=device)
    #model, all_losses = train_multiple_initializations(csd_empirical, f_ref, num_epochs = 1)
    model, noise_trials, losses = train_random_noise_combinations(csd_empirical, f_ref, num_trials=500, epochs=1, device=device)

    #print("\nAll Trial Results:")
    for i, (loss, (a_v, b_v, a_e, b_e)) in enumerate(noise_trials):
        print(f"Trial {i + 1}: Loss={loss:.6f} | alpha_v={a_v:.2f}, beta_v={b_v:.2f}, alpha_e={a_e:.2f}, beta_e={b_e:.2f}")

    # Get loss curve from best run
    #best_index = np.argmin([losses[-1] for losses in all_losses])
    #losses = all_losses[best_index]

    # Show Final Parameters
    #print_model_params(model, "Best Parameters")

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n Training completed in {elapsed:.2f} seconds.")

    # Simulate and Compare CSD
    sim_bold = model()
    sim_bold = sim_bold / sim_bold.std()

    # Plotting the simulated and real BOLD signals
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, sim_bold.detach().numpy(), label='Simulated BOLD Signal')
    plt.plot(time_axis, observed_bold.detach().numpy(), label=f'Real BOLD Signal - ROI {roi}')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal')
    plt.title('Simulated Downsampled BOLD Signal vs Real Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    f_sim, csd_sim = torch_csd(sim_bold, sim_bold, fs=0.5, nperseg=128, nfft=512)
    f_sim, csd_sim = filter_frequencies(f_sim.cpu(), csd_sim.cpu(), 0.01, 0.1)

    # Ensure complex format
    csd_empirical = csd_empirical.to(torch.complex64)
    csd_sim = csd_sim.to(torch.complex64)

    # Plot comparisons
    for scale in [True, False]:
        plot_psd_comparison(f_sim, csd_empirical, csd_sim, f"PSD Comparison", roi, log_scale=scale)
        plot_psd_comparison(f_sim, csd_empirical/csd_empirical.std(), csd_sim/csd_sim.std(),
                            f"PSD Normalised Comparison", roi, log_scale=scale)

    #plot_psd_loss_curve(losses, 'Loss Curve - Best of Multiple Inits', 'Loss', region_index)

    # Plot Loss Curve
    #plot_psd_loss_curve(losses, 'Loss Curve', 'Loss', roi)

    # Evaluation
    evaluator1 = CSDMetricsCalculatorTorch(csd_empirical, csd_sim)
    metrics1 = evaluator1.evaluate()

    # Evaluation
    evaluator = CSDMetricsCalculatorTorch(csd_empirical/csd_empirical.std(), csd_sim/csd_sim.std())
    metrics = evaluator.evaluate()
