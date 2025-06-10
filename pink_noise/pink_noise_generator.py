
import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.fft as fft
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
When you increase the scaling factor, you're making the noise much larger,
 which can overwhelm the model's dynamics. In our simulation, the pink noise is meant to add small 
 fluctuations that mimic resting-state variability. If the noise becomes too large, 
 it can drive the model into a region where its output saturates or becomes unstable, 
 effectively flattening or "washing out" the BOLD signal. 
 In simple terms, too much noise can drown out the signal you’re trying to observe.
"""


def generate_pink_noise_fft(n, T, alpha, beta, scaling_factor=0.005, normalise=True, device=None):
    """
    Generates noise with a power spectral density proportional to 1/f^beta using an FFT-based method.
    This version produces a spectral amplitude ~ f^(-beta/2) (so the power scales as 1/f^beta) and uses
    torch.fft.irfft to form the time-domain signal. It handles negative alpha by flipping the polarity.

    Parameters:
        n (int): Number of samples.
        T (float): Sampling interval (1/sampling rate).
        alpha (float): Overall scaling factor. If negative, the noise polarity is flipped.
        beta (float): Exponent for the spectral density (e.g. 1 for pink noise). Negative values are supported.
        scaling_factor (float): Scaling factor applied to the final noise signal.
        normalise (bool): If True, the time-domain noise is normalised to unit standard deviation.
        device (str): The device on which to perform the computation ('cpu' or 'cuda').

    Returns:
        noise_time_domain (torch.Tensor): The generated noise signal.
        C_k_complex (torch.Tensor): The complex spectral coefficients for the positive frequencies.
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

    # Number of positive frequency coefficients (including DC and Nyquist if n is even)
    k_max = n // 2

    # Build frequency vector for indices 1...k_max (we handle DC separately)
    # f_k is defined so that f = (1/(n*T)), (2/(n*T)), ..., (k_max/(n*T))
    f_k = torch.linspace(1, k_max, steps=k_max, device=device) * (1.0 / (n * T))

    # Determine scaling factor for amplitude.
    # To avoid taking sqrt of a negative number, branch on alpha.
    if alpha < 0:
        scale = math.sqrt(-alpha * n * T)
        sign_alpha = -1
    else:
        scale = math.sqrt(alpha * n * T)
        sign_alpha = 1

    # Initialize an array for spectral magnitudes (length k_max+1, including DC)
    C_k_magnitude = torch.zeros(k_max + 1, device=device)
    # Set DC component to zero (no offset)
    C_k_magnitude[0] = 0
    # For indices 1...k_max: the amplitude is proportional to f^(-beta/2)
    # Multiplying by sign_alpha ensures that a negative alpha flips the polarity.
    C_k_magnitude[1:] = sign_alpha * scale * (f_k ** (-beta / 2))

    # Now create the complex spectral coefficients.
    # We'll generate random phases for frequencies 1 to k_max (excluding DC)
    # For real-valued signals, the Nyquist frequency (if n is even) must be purely real.
    C_k_real = torch.zeros(k_max + 1, device=device)
    C_k_imag = torch.zeros(k_max + 1, device=device)

    # Number of random phases:
    num_random = k_max - 1 if n % 2 == 0 else k_max

    # Generate random phases
    phi = torch.rand(num_random, device=device) * 2 * math.pi

    # Assign random phases for indices 1...num_random
    C_k_real[1:1 + num_random] = C_k_magnitude[1:1 + num_random] * torch.cos(phi)
    C_k_imag[1:1 + num_random] = C_k_magnitude[1:1 + num_random] * torch.sin(phi)

    # If n is even, the Nyquist frequency (index k_max) must be purely real.
    if n % 2 == 0:
        C_k_real[k_max] = C_k_magnitude[k_max]
        C_k_imag[k_max] = 0.0

    # Combine into a complex tensor for positive frequencies
    C_k_complex = torch.complex(C_k_real, C_k_imag)

    # Compute the time-domain signal via the inverse real FFT
    noise_time_domain = torch.fft.irfft(C_k_complex, n=n)

    # Optionally normalise the time-domain signal to unit standard deviation
    if normalise:
        noise_time_domain = noise_time_domain / torch.std(noise_time_domain)

    # Apply final scaling factor
    noise_time_domain = noise_time_domain * scaling_factor

    return noise_time_domain, C_k_complex

import torch
import numpy as np

import torch
import numpy as np


import torch
import math

import torch
import math


import torch

def generate_coloured_noise(
    n: int,
    T: float,
    alpha: float,
    beta: float,
    scaling_factor: float = 0.005,
    normalise: bool = True,
    device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 1/f^beta (“power‑law”) noise via direct FFT.

    Parameters
    ----------
    n             Number of time‑domain samples.
    T             Sample interval (so sampling rate is 1/T).
    alpha         Overall spectral scaling (power amplitude).
    beta          Exponent (1 for pink, 2 for brown, etc).
    scaling_factor  Final multiply on the time series.
    normalise     If True, force the raw noise to σ=1 before scaling.
    device        'cpu' or 'cuda'.

    Returns
    -------
    noise, Ck     Tuple of:
                  * noise:    (n,) real tensor = irfft(Ck) normalised & scaled
                  * Ck:       (n//2+1,) complex spectral coefficients
    """
    # move to the right device and dtype
    dev = torch.device(device)
    # 1) build the vector of positive frequencies: [0, 1/(nT), 2/(nT), …, floor(n/2)/(nT)]
    f = torch.fft.rfftfreq(n, d=T, device=dev)

    # 2) power‑law amplitude ~ alpha * f^(–beta/2); enforce zero DC (f=0) amplitude
    S = torch.zeros_like(f)
    positive = f > 0
    S[positive] = alpha * (f[positive] ** (-beta / 2.0))

    # 3) random phases in [0,2π) for each positive frequency
    phi = torch.rand_like(f) * 2 * torch.pi
    real = S * torch.cos(phi)
    imag = S * torch.sin(phi)

    # 4) pack into complex spectrum
    Ck = torch.complex(real, imag)

    # 5) inverse FFT → real time‑series
    noise = torch.fft.irfft(Ck, n=n, dim=0)

    # 6) normalise and scale
    if normalise:
        noise = noise / noise.std()
    noise = noise * scaling_factor

    return noise, Ck

import torch
import torch.fft as fft
import numpy as np
import math

def generate_coloured_noise1(n, T, alpha, beta,
                           scaling_factor=0.005,
                           normalise=True,
                           device=None):
    """
    Generates 1/f^beta noise by sampling Gaussian spectral coefficients.

    Args:
        n (int): number of output samples
        T (float): sample interval (1/sampling rate)
        alpha (float): overall PSD scaling (so S(f)=alpha * f^{-beta})
        beta  (float): exponent for the PSD (e.g. beta=1 → pink noise)
        scaling_factor (float): final time‐domain gain
        normalise (bool): if True, renormalise output to unit std
        device (str|torch.device): 'cpu' or 'cuda'
    Returns:
        noise (torch.Tensor, shape [n]): the time‐domain noise × scaling_factor
        Ck  (torch.Tensor, shape [n//2+1], dtype complex): the positive‐freq. FFT bins
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # number of positive bins (including DC, maybe Nyquist)
    k_max = n//2

    # build the frequencies: f_k = k / (n*T) for k=0…k_max
    f = torch.arange(0, k_max+1, device=device, dtype=torch.float64) / (n*T)
    f[0] = 1e-12  # avoid division by zero at DC

    # power spectral density S(f) = alpha * f^{-beta}
    S = alpha * f.pow(-beta)

    # variance of each complex bin for real-valued signal:
    #  Var[Re]=Var[Im]=½ S(f)
    sigma = (0.5 * S).sqrt()

    # sample real & imag parts
    real = torch.randn(k_max+1, device=device) * sigma
    imag = torch.randn(k_max+1, device=device) * sigma
    # enforce purely real DC and (if n even) Nyquist
    real[0] = 0.0
    imag[0] = 0.0
    if n % 2 == 0:
        real[k_max] = torch.randn(1, device=device) * sigma[k_max]
        imag[k_max] = 0.0

    Ck = torch.complex(real, imag)

    # inverse real FFT to get time domain
    noise_td = fft.irfft(Ck, n=n)

    if normalise:
        noise_td = noise_td / noise_td.std()

    noise_td = noise_td * scaling_factor

    return noise_td, Ck





def generate_coloured_noise_psd(
        n: int,
        T: float,
        alpha: float,
        beta: float,
        scaling_factor: float = 0.005,
        normalise = True,
        device: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates 1/f^beta noise with correct PSD scaling.

    Ensures that when scaling_factor=1.0, the PSD follows S(f) = alpha * f^(-beta).

    Args:
        n (int): Number of time-domain samples.
        T (float): Sample interval (1/sampling rate).
        alpha (float): PSD amplitude scaling (S(f) = alpha * f^(-beta)).
        beta (float): Spectral exponent (1=pink, 2=brown, etc.).
        scaling_factor (float): Optional scaling of the output (default=1.0).
        device (str): 'cpu' or 'cuda'.

    Returns:
        noise (torch.Tensor): Time-domain noise (n samples).
        Ck (torch.Tensor): Complex FFT coefficients (n//2 + 1 bins).
    """
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Number of positive frequency bins (including DC and Nyquist if n is even)
    k_max = n // 2

    # Frequency vector: f_k = k / (n*T) for k = 0, ..., k_max
    f = torch.arange(0, k_max + 1, device=device, dtype=torch.float64) / (n * T)
    f[0] = 1e-12  # Avoid division by zero (DC component)

    # Target PSD: S(f) = alpha * f^(-beta)
    S = alpha * f.pow(-beta)

    # Variance per FFT bin (real & imaginary parts each have variance S(f)/2)
    sigma = torch.sqrt(S / 2.0)

    # Sample real and imaginary parts with correct variance
    real = torch.randn(k_max + 1, device=device) * sigma
    imag = torch.randn(k_max + 1, device=device) * sigma

    # Enforce real-only DC and Nyquist (if n is even)
    real[0] = 0.0  # Remove DC offset (optional)
    imag[0] = 0.0
    if n % 2 == 0:  # Nyquist frequency must be real
        imag[k_max] = 0.0

    # Combine into complex spectrum
    Ck = torch.complex(real, imag)

    # Inverse FFT to get time-domain noise
    noise = fft.irfft(Ck, n=n)

    if normalise:
        noise = noise / noise.std()

    noise = noise * scaling_factor

    return noise, Ck
