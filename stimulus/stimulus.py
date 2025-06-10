import torch
import numpy as np

from pink_noise.pink_noise_generator import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Stimulus:
    @staticmethod
    def rectangular(total_time, dt, w=1.0):
        # Generate time steps as a tensor
        time_steps = torch.arange(0, total_time, dt)
        return torch.where((time_steps >= 1) & (time_steps <= (1 + w)), torch.tensor(1.0), torch.tensor(0.0))

    @staticmethod
    def sinusoidal(t, freq=1.0, amplitude=1.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        return amplitude * torch.sin(2 * torch.pi * freq * t)

    @staticmethod
    def step(t, start=1.0, amplitude=1.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        return torch.where(t >= start, torch.tensor([amplitude]), torch.tensor([0.0]))

    @staticmethod
    def triangle(t, period=10.0, amplitude=1.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        fractional_part = (t % period) / period
        return amplitude * (2 * torch.abs(fractional_part - 0.5))

    @staticmethod
    def exponential_decay(t, start=1.0, tau=5.0, amplitude=1.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        return torch.where(
            t >= start,
            amplitude * torch.exp(-(t - start) / tau),
            torch.tensor([0.0])
        )

    @staticmethod
    def boxcar(t, block_duration=10.0, period=20.0, amplitude=1.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        return torch.where(
            ((t % period) < block_duration),
            torch.tensor([amplitude]),
            torch.tensor([0.0])
        )

    @staticmethod
    def gamma_variate(t, alpha=6.0, beta=1.0, scale=1.0, onset=0.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        t_shifted = t - onset
        gamma = torch.where(
            t_shifted > 0,
            scale * (t_shifted ** (alpha - 1)) * torch.exp(-t_shifted / beta),
            torch.tensor([0.0])
        )
        return gamma

    @staticmethod
    def impulse(t, timepoint=1.0, amplitude=1.0):
        t = torch.tensor(t) if not torch.is_tensor(t) else t
        return torch.where(
            torch.abs(t - timepoint) < 1e-3,  # Relaxed precision
            torch.tensor([amplitude]),
            torch.tensor([0.0])
        )

    @staticmethod
    def pink_noise(n, T, alpha, beta, scaling_factor = 0.005, normalise=True):
        pink_noise, _ = generate_pink_noise_fft(n=n, T=T, alpha=alpha, beta=beta, scaling_factor=scaling_factor,
                                            normalise=normalise)
        return pink_noise, _

    @staticmethod
    def gaussian_noise(length, mean=0.0, std=1.0):
        return torch.tensor(np.random.normal(mean, std, length), dtype=torch.float32)

    @staticmethod
    def sinusoidal_stimulus(t, frequency=1.0, amplitude=1.0, dt=0.01):
        time = torch.arange(0, len(t) * dt, dt)
        return amplitude * torch.sin(2 * torch.pi * frequency * time)
