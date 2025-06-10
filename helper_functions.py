
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer
from scipy_to_torch import torch_csd, torch_psd




def load_time_series(region1_file, region2_file):
    time_series_region1 = np.loadtxt(region1_file, delimiter=',')
    time_series_region2 = np.loadtxt(region2_file, delimiter=',')
    return time_series_region1, time_series_region2

def load_single_time_series(region_file):
    time_series_region = np.loadtxt(region_file, delimiter=',')
    return time_series_region


def filter_frequencies(frequencies, csd_values, min_freq, max_freq):
    # If both inputs are torch tensors, keep them as tensors.
    if isinstance(frequencies, torch.Tensor) and isinstance(csd_values, torch.Tensor):
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        return frequencies[mask], csd_values[mask]
    else:
        # Otherwise, convert to numpy if necessary
        if isinstance(frequencies, torch.Tensor):
            frequencies = frequencies.detach().cpu().numpy()
        if isinstance(csd_values, torch.Tensor):
            csd_values = csd_values.detach().cpu().numpy()
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        return frequencies[mask], csd_values[mask]



def moving_average(x, window_size=3):
    return torch.conv1d(x.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, window_size) / window_size, padding='same').squeeze()

def make_json_serializable(obj):
    """Converts NumPy arrays, PyTorch tensors, and complex numbers to JSON-serializable formats."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, torch.Tensor):
        return make_json_serializable(obj.cpu().numpy())  # Convert PyTorch tensor to NumPy
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}  # Store complex numbers as JSON-friendly dict
    else:
        return obj


import numpy as np
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer


def compute_spectrum(time_series, TR=2,
                            freq_low=0.01, freq_high=0.08,
                            filter_method='original',
                            spectral_method='welch',
                            smooth=False, smoothing_window=5):
    """
    Computes the spectrum of a time series using nitime, with an option to smooth the PSD.

    Parameters:
      time_series (np.array): The input data. Can be 1D or 2D (n_channels x n_samples).
      TR (float): Sampling interval (seconds).
      freq_low (float): Lower bound of frequency mask (Hz).
      freq_high (float): Upper bound of frequency mask (Hz).
      filter_method (str): Which filtered time series to use.
             Options: 'original', 'boxcar', 'fir', 'iir', 'fourier'.
      spectral_method (str): Which spectral estimator to use.
             Options: 'welch', 'fft', 'periodogram', 'multitaper'.
      smooth (bool): If True, smooth the computed power spectrum.
      smoothing_window (int): The window size for smoothing (number of frequency bins).

    Returns:
      freq_masked (np.array): Frequencies in the specified band.
      power_masked (np.array): (Smoothed) power corresponding to those frequencies.
    """
    # Ensure time_series is 2D: (n_channels x n_samples)
    if time_series.ndim == 1:
        time_series = time_series.reshape(1, -1)

    # Create the base TimeSeries object
    T = TimeSeries(time_series, sampling_interval=TR)

    # Decide which time series to use based on the filter_method
    if filter_method.lower() == 'original':
        T_used = T
    else:
        # Initialize FilterAnalyzer with the desired frequency bounds.
        F = FilterAnalyzer(T, lb=freq_low, ub=freq_high)
        if filter_method.lower() == 'boxcar':
            data = F.filtered_boxcar.data
        elif filter_method.lower() == 'fir':
            data = F.fir.data
        elif filter_method.lower() == 'iir':
            data = F.iir.data
        elif filter_method.lower() == 'fourier':
            data = F.filtered_fourier.data
        else:
            raise ValueError("Unknown filter_method. Use 'original', 'boxcar', 'fir', 'iir', or 'fourier'.")

        # Ensure the filtered data is 2D:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        T_used = TimeSeries(data, sampling_interval=TR)

    # Compute the spectrum from the chosen TimeSeries
    S = SpectralAnalyzer(T_used)

    # Depending on the spectral_method, extract frequency and power arrays.
    if spectral_method.lower() == 'welch':
        freq = np.squeeze(S.psd[0])
        power = np.squeeze(S.psd[1])
    elif spectral_method.lower() == 'fft':
        freq = np.squeeze(S.spectrum_fourier[0])
        power = np.abs(np.squeeze(S.spectrum_fourier[1][0]))
    elif spectral_method.lower() == 'periodogram':
        freq = np.squeeze(S.periodogram[0])
        power = np.squeeze(S.periodogram[1][0])
    elif spectral_method.lower() == 'multitaper':
        freq = np.squeeze(S.spectrum_multi_taper[0])
        power = np.squeeze(S.spectrum_multi_taper[1][0])
    else:
        raise ValueError("Unknown spectral_method. Choose 'welch', 'fft', 'periodogram', or 'multitaper'.")

    # Create a frequency mask and apply it
    mask = (freq >= freq_low) & (freq <= freq_high)
    freq_masked = freq[mask]
    power_masked = power[mask]

    # Optional smoothing of the PSD: apply a uniform moving average filter.
    if smooth:
        # Using SciPy's uniform_filter1d to smooth the power spectrum.
        power_masked = uniform_filter1d(power_masked, size=smoothing_window)

    return freq_masked, power_masked


def csd_choice(time_series1, time_series2, method='csd', TR = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(time_series1, np.ndarray):
        time_series1 = torch.tensor(time_series1, dtype=torch.float32, device=device)
    if isinstance(time_series2, np.ndarray):
        time_series2 = torch.tensor(time_series2, dtype=torch.float32, device=device)

    if method == 'csd':
        fs = 1/TR
        frequencies, csd_values = torch_csd(time_series1, time_series2, fs, nperseg=min(128, len(time_series1)))

    elif method == 'fft':
        fs = 1/TR
        n = len(time_series1)

        # Step 1: Compute FFT of both signals
        fft1 = np.fft.fft(time_series1)
        fft2 = np.fft.fft(time_series2)

        # Step 2: Compute two-sided CSD and normalise
        csd_two_sided = (1 / (fs * n)) * (fft1 * np.conj(fft2))

        # Step 3: Select one-sided CSD
        csd_values = csd_two_sided[:n // 2 + 1]
        frequencies = np.fft.rfftfreq(n, d= TR)

        # Step 4: Double values except DC and Nyquist.
        # - The DC component (0 Hz) represents the mean value of the signal and does not have a symmetric counterpart
        # in the negative frequencies, so it should not be doubled.
        # - The Nyquist frequency lies at the boundary between the positive and
        # negative frequency ranges, so it also does not have a symmetric counterpart and should not be doubled.

        csd_values[1:-1] *= 2

    else:
        raise ValueError("Invalid method. Choose 'scipy' or 'fft'.")

    return frequencies, csd_values


def psd_choice(time_series, method='welch', TR=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(time_series, np.ndarray):
        time_series = torch.tensor(time_series, dtype=torch.float32, device=device)

    if method == 'welch':
        fs = 1 / TR
        nperseg = 512  # Use a larger segment length
        noverlap = int(512 * 0.75)  # 75% overlap
        torch_psd(time_series, fs = fs)


    elif method == 'fft':
        fs = 1 / TR
        n = len(time_series)
        # Compute FFT of the time series
        fft_vals = np.fft.fft(time_series)
        # Compute two-sided PSD (|FFT|^2 normalised)
        psd_two_sided = (1 / (fs * n)) * (np.abs(fft_vals) ** 2)
        # Select the one-sided PSD and corresponding frequencies
        psd_values = psd_two_sided[:n // 2 + 1]
        frequencies = np.fft.rfftfreq(n, d=TR)
        # Double the power for all frequencies except DC and Nyquist
        psd_values[1:-1] *= 2
    else:
        raise ValueError("Invalid method. Choose 'welch' or 'fft'.")

    return frequencies, psd_values


# --- Plotting functions ---
def plot_loss_curve(losses, title, label, region1_index, region2_index):
    label = f'{label} (ROI{region1_index}-ROI{region2_index})'
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label=label)
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_psd_loss_curve(losses, title, label, region1_index):
    label = f'{label} (ROI {region1_index})'
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label=label)
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    #plt.show()


def plot_csd_comparison(frequencies, true_csd, recovered_csd, title, region1_index, region2_index, log_scale = True):
    label_true = f'True CSD (ROI{region1_index}-ROI {region2_index})'
    label_recovered = f'Recovered CSD (ROI{region1_index}-ROI {region2_index})'

    # Ensure true_csd and recovered_csd are torch tensors of complex type.
    if not torch.is_tensor(true_csd):
        true_csd = torch.tensor(true_csd, dtype=torch.complex64)
    if not torch.is_tensor(recovered_csd):
        recovered_csd = torch.tensor(recovered_csd, dtype=torch.complex64)

    plt.figure(figsize=(12, 6))

    if not log_scale:
        plt.plot(
            frequencies,
            torch.sqrt(true_csd.real ** 2 + true_csd.imag ** 2).detach().cpu().numpy(),
            label=label_true, color='b'
        )
        plt.plot(
            frequencies,
            torch.sqrt(recovered_csd.real ** 2 + recovered_csd.imag ** 2).detach().cpu().numpy(),
            label=label_recovered, color='r', linestyle='--'
        )
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('CSD Magnitude')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    else:

        plt.semilogy(
            frequencies,
            torch.sqrt(true_csd.real ** 2 + true_csd.imag ** 2).detach().cpu().numpy(),
            label=label_true, color='b'
        )
        plt.semilogy(
            frequencies,
            torch.sqrt(recovered_csd.real ** 2 + recovered_csd.imag ** 2).detach().cpu().numpy(),
            label=label_recovered, color='r', linestyle='--'
        )
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('CSD Magnitude')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_psd_comparison(frequencies, true_psd, recovered_psd, title, region1_index, log_scale=True, use_magnitude=True):
    if not torch.is_tensor(true_psd):
        true_psd = torch.tensor(true_psd, dtype=torch.complex64)
    if not torch.is_tensor(recovered_psd):
        recovered_psd = torch.tensor(recovered_psd, dtype=torch.complex64)

    if use_magnitude:
        y_true = torch.sqrt(true_psd.real**2 + true_psd.imag**2).cpu().numpy()
        y_pred = torch.sqrt(recovered_psd.real**2 + recovered_psd.imag**2).cpu().numpy()
    else:
        y_true = true_psd.real.cpu().numpy()
        y_pred = recovered_psd.real.cpu().numpy()

    plt.figure(figsize=(8, 6))
    if log_scale:
        plt.semilogy(frequencies, y_true, label='True PSD')
        plt.semilogy(frequencies, y_pred, label='Recovered PSD', linestyle='--')
    else:
        plt.plot(frequencies, y_true, label='True PSD')
        plt.plot(frequencies, y_pred, label='Recovered PSD', linestyle='--')

    # Labels
    plt.xlabel("Frequency (Hz)", fontsize=18)
    plt.ylabel("PSD", fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()




def normalise_data(x, eps = 1e-6):
    std = x.std()
    if std == 0:
        return x
    return x / (x.std() + eps)

def normalise_complex_psd(psd):
    mag = torch.sqrt(psd.real ** 2 + psd.imag ** 2)
    return psd / mag.std()


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve_with_std(losses, title, label, region1_index, region2_index):
    label = f'{label} (ROI{region1_index}-ROI{region2_index})'
    plt.figure(figsize=(12, 6))

    # Convert losses to a numpy array
    losses = np.array(losses)

    # Check if losses are 1D or 2D
    if losses.ndim == 1:
        mean_loss = losses
        std_loss = np.zeros_like(losses)  # No std deviation if we only have one set of losses
    else:
        # Compute the mean and std across the multiple runs (axis=0 means across epochs)
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)

    # Plot the mean loss
    plt.plot(mean_loss, label=label, color='b')

    # Fill the area between the mean - std and mean + std (shading the std region)
    plt.fill_between(range(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, color='b', alpha=0.3)

    # Adding labels, title, and legend
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_parameter_error(history, true_value, param_name):
    """
    Plots the error (difference) between the parameter history and the ground truth.

    Args:
        history (list or np.array): History of the parameter values over epochs.
        true_value (float or complex): Ground-truth parameter value.
        param_name (str): Name of the parameter.
    """
    epochs = np.arange(len(history))
    history = np.array(history)
    # For complex parameters, compare real and imaginary parts separately
    if np.iscomplexobj(true_value):
        error_real = np.abs(history.real - true_value.real)
        error_imag = np.abs(history.imag - true_value.imag)
        error = np.sqrt(error_real ** 2 + error_imag ** 2)  # overall error
    else:
        error = np.abs(history - true_value)

    # If you have multiple runs, you can compute mean and std along axis=0.
    # Here we assume a single run, so mean=history and std=0.
    mean_error = error
    std_error = np.zeros_like(mean_error)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, mean_error, label=f'{param_name} error', linewidth=2)
    plt.fill_between(epochs, mean_error - std_error, mean_error + std_error, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Curve')
    plt.title(f'{param_name} Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
