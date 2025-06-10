
import torch

import torch
from spectrum import pmtm


def get_window_torch(window, nperseg):
    """Returns the requested window function or applies a custom one."""
    if isinstance(window, str):
        if window.lower() in ["hann", "hanning"]:
            return torch.hann_window(nperseg, periodic=True)
        elif window.lower() == "hamming":
            return torch.hamming_window(nperseg, periodic=True)
        elif window.lower() == "blackman":
            return torch.blackman_window(nperseg, periodic=True)
        elif window.lower() == "bartlett":
            return torch.bartlett_window(nperseg)
        else:
            raise ValueError(f"Unsupported window type: {window}")
    elif callable(window):  # Allow a user-defined function
        return window(nperseg)
    elif isinstance(window, (tuple, list, torch.Tensor)):
        window = torch.as_tensor(window, dtype=torch.float32)
        if window.shape[0] != nperseg:
            raise ValueError("Window length must match nperseg")
        return window
    else:
        raise ValueError("Window must be a string, function, or a 1D array")


def detrend_torch(x, method="constant"):
    """Detrending function for PyTorch tensors."""
    if method is None or method == "none":
        return x
    elif method == "constant":
        return x - x.mean()
    elif method == "linear":
        N = x.shape[-1]
        X = torch.stack([torch.linspace(0, 1, N, device=x.device), torch.ones(N, device=x.device)], dim=0)
        beta = torch.linalg.lstsq(X.T, x.T).solution  # Solve least squares
        trend = (X.T @ beta).T  # Compute trend
        return x - trend
    elif method == "polynomial":
        order = 2  # Polynomial order
        N = x.shape[-1]
        X = torch.stack([torch.linspace(0, 1, N, device=x.device) ** i for i in range(order + 1)], dim=0)
        beta = torch.linalg.lstsq(X.T, x.T).solution
        trend = (X.T @ beta).T
        return x - trend
    else:
        raise ValueError("Unsupported detrend method")

def torch_psd(x, fs=1.0, window='hann', nperseg=256, noverlap=None,
              nfft=None, detrend="constant", return_onesided=True, scaling='density',
              average='mean', freq_range=None):
    """
    Compute the Power Spectral Density (PSD) using Welch's method in PyTorch,
    returning the power spectrum.

    Parameters:
        x (torch.Tensor): Input signal.
        fs (float): Sampling frequency.
        window (str, tuple, or torch.Tensor): Window function.
        nperseg (int): Segment length.
        noverlap (int or None): Overlapping samples. Defaults to nperseg//2.
        nfft (int or None): FFT length.
        detrend (str or function): Detrending method.
        return_onesided (bool): Whether to return a one-sided spectrum.
        scaling (str): 'density' (dividing by (fs*power of window)) or 'spectrum' (dividing by square of sum of window).
        average (str): Method for averaging ('mean' or 'median').
        freq_range (tuple): Frequency range (min, max) to keep.

    Returns:
        f (torch.Tensor): Frequency bins.
        Pxx (torch.Tensor): Power spectral density (PSD) (real-valued).
    """
    # Ensure x is a float tensor on its device
    x = torch.as_tensor(x, dtype=torch.float32, device=x.device)

    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap >= nperseg:
        raise ValueError("`noverlap` must be smaller than `nperseg`")

    shape = x.shape[0]
    if shape < nperseg:
        nperseg = shape
        noverlap = nperseg // 2

    if nfft is None:
        nfft = nperseg

    # Choose FFT functions based on onesided/twosided spectrum
    if return_onesided:
        fft_func = torch.fft.rfft
        freq_func = torch.fft.rfftfreq
    else:
        fft_func = torch.fft.fft
        freq_func = torch.fft.fftfreq

    step = nperseg - noverlap
    win = get_window_torch(window, nperseg).to(x.device)

    # Determine scaling factor based on the chosen scaling type
    if scaling == 'density':
        scale_factor = fs * torch.sum(win ** 2)
    elif scaling == 'spectrum':
        scale_factor = (torch.sum(win)) ** 2
    else:
        raise ValueError("Invalid scaling option: choose 'density' or 'spectrum'")

    psd_list = []
    # Loop over segments
    for i in range((shape - noverlap) // step):
        start = i * step
        end = start + nperseg
        seg = detrend_torch(x[start:end], detrend) * win
        fft_seg = fft_func(seg, n=nfft)
        # Compute the periodogram for this segment: square magnitude and scale.
        psd_seg = (torch.abs(fft_seg) ** 2) / scale_factor
        psd_list.append(psd_seg)

    # Average across segments
    if average == 'median':
        Pxx = torch.median(torch.stack(psd_list), dim=0).values
    else:
        Pxx = torch.stack(psd_list).mean(dim=0)

    f = freq_func(nfft, d=1 / fs)

    # Optionally filter frequency range
    if freq_range is not None:
        fmin, fmax = freq_range
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        Pxx = Pxx[mask]

    return f, Pxx


def torch_csd(x, y, fs=1.0, window='hann', nperseg=256, noverlap=None,
              nfft=None, detrend="constant", return_onesided=True, scaling='density',
              average='mean', freq_range=None):
    """
    Compute the Cross-Spectral Density (CSD) using Welch's method in PyTorch,
    returning the complex-valued spectrum (preserving both amplitude and phase).

    Parameters:
        x (torch.Tensor): First input signal.
        y (torch.Tensor): Second input signal (same length as x).
        fs (float): Sampling frequency.
        window (str, tuple, or torch.Tensor): Window function.
        nperseg (int): Segment length.
        noverlap (int or None): Overlapping samples. Defaults to nperseg//2.
        nfft (int or None): FFT length.
        detrend (str or function): Detrending method.
        return_onesided (bool): Whether to return a one-sided spectrum.
        scaling (str): 'density' (dividing by (fs*power of window)) or 'spectrum' (dividing by square of sum of window).
        average (str): Method for averaging ('mean' or 'median').
        freq_range (tuple): Frequency range (min, max) to keep.

    Returns:
        f (torch.Tensor): Frequency bins.
        Pxy (torch.Tensor): Cross-power spectral density (complex-valued).
    """
    # Ensure signals are float tensors on the same device
    x = torch.as_tensor(x, dtype=torch.float32, device=x.device)
    y = torch.as_tensor(y, dtype=torch.float32, device=x.device)

    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap >= nperseg:
        raise ValueError("`noverlap` must be smaller than `nperseg`")

    shape = x.shape[0]
    if shape < nperseg:
        nperseg = shape
        noverlap = nperseg // 2

    if nfft is None:
        nfft = nperseg

    # Choose FFT functions based on onesided/twosided spectrum
    if return_onesided:
        fft_func = torch.fft.rfft
        freq_func = torch.fft.rfftfreq
    else:
        fft_func = torch.fft.fft
        freq_func = torch.fft.fftfreq

    step = nperseg - noverlap
    win = get_window_torch(window, nperseg).to(x.device)

    # Determine scaling factor
    if scaling == 'density':
        scale_factor = fs * torch.sum(win ** 2)
    elif scaling == 'spectrum':
        scale_factor = (torch.sum(win)) ** 2
    else:
        raise ValueError("Invalid scaling option: choose 'density' or 'spectrum'")

    csd_list = []
    for i in range((shape - noverlap) // step):
        start = i * step
        end = start + nperseg

        seg_x = detrend_torch(x[start:end], detrend) * win
        seg_y = detrend_torch(y[start:end], detrend) * win

        fft_x = fft_func(seg_x, n=nfft)
        fft_y = fft_func(seg_y, n=nfft)

        csd = fft_x * torch.conj(fft_y) / scale_factor
        csd_list.append(csd)

    # Average across segments
    if average == 'median':
        Pxy = torch.median(torch.stack(csd_list), dim=0).values
    else:
        Pxy = torch.stack(csd_list).mean(dim=0)

    f = freq_func(nfft, d=1 / fs)

    # Optionally filter frequency range
    if freq_range is not None:
        fmin, fmax = freq_range
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        Pxy = Pxy[mask]

    return f, Pxy



