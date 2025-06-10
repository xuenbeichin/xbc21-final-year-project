import numpy as np
import torch
import math

from matplotlib import pyplot as plt

from PDCM.PDCMBOLDModel import PDCMMODEL
from helper_functions import plot_psd_comparison, filter_frequencies
from pink_noise.pink_noise_generator import generate_coloured_noise_psd
from scipy_to_torch import torch_csd


def euler_maruyama_solver(
        func,
        t,
        h,
        var_init,
        indicator,
        alpha_v,
        beta_v,
        alpha_e,
        beta_e,
        add_state_noise=False,
        add_obs_noise=False,
        add_stimulus=True,
        select=False
):
    """
    Simulate system dynamics using first-order Euler integration with optional pink noise.
    This version runs at full resolution (no internal downsampling).

    Args:
        func: Model instance (e.g., PDCMMODEL) with differential and algebraic equations.
        t: Time vector.
        h: Integration time step.
        var_init: Tuple with initial state values.
        indicator: Use CBF-CBV coupling if True.
        alpha_v, beta_v: Pink noise parameters for state noise.
        alpha_e, beta_e: Pink noise parameters for observation noise.
        add_state_noise: Add noise to state updates.
        add_obs_noise: Add noise to observation output.
        add_stimulus: Include stimulus input.
        select: If True, return only downsampled output. (Method 2 uses select=False)

    Returns:
        If select is False, returns a tuple with:
          (u_tensor, xE_tensor, xI_tensor, a_tensor, f_tensor, v_tensor,
           q_tensor, fout_tensor, E_tensor, y_tensor, t_tensor)
        Otherwise, returns (t_list, y_tensor) (but here, select should be False for full resolution).
    """

    device = func.device if hasattr(func, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Unpack initial states:
    xE_val, xI_val, a_val, f_val, v_val, q_val, fout_val, E_val, y_val = var_init

    t_list = [t[0]]
    u_list = []
    xE_list = [xE_val]
    xI_list = [xI_val]
    a_list = [a_val]
    f_list = [f_val]
    v_list = [v_val]
    q_list = [q_val]
    fout_list = [fout_val]
    E_list = [E_val]
    if not torch.is_tensor(y_val):
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_list = [y_val.to(device).detach().requires_grad_(True)]

    # Generate pink noise for state and observation if needed.
    if add_state_noise:
        state_noise, _ = generate_coloured_noise_psd(len(t), t[-1], alpha=alpha_v, beta=beta_v, device=device)
    else:
        state_noise = torch.zeros(len(t), device=device)

    if add_obs_noise:
        obs_noise, _ = generate_coloured_noise_psd(len(t), t[-1], alpha=alpha_e, beta=beta_e,
                                                device=device)
    else:
        obs_noise = torch.zeros(len(t), device=device)

    # Integration loop
    for i in range(len(t) - 1):
        # Get stimulus: if add_stimulus is True, use func.sti_u; otherwise create a zero tensor.
        if add_stimulus:
            u_val = func.sti_u(t[i])
            if not torch.is_tensor(u_val):
                u_val = torch.tensor(u_val, dtype=torch.float32, device=device)
        else:
            u_val = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Compute noise increment for the excitatory state.
        dw = state_noise[i] * math.sqrt(h) if add_state_noise else 0.0

        # Euler updates for differential states.
        xE_next = xE_val + h * func.dxE(u=u_val, xE=xE_val, xI=xI_val) + dw
        xI_next = xI_val + h * func.dxI(xE=xE_val, xI=xI_val)
        a_next = a_val + h * func.da(a=a_val, xE=xE_val)
        f_next = f_val + h * func.df(a=a_val, f=f_val)
        v_next = v_val + h * func.dv(f=f_val, fout=fout_val)
        q_next = q_val + h * func.dq(f=f_val, E=E_val, fout=fout_val, q=q_val, v=v_val)

        # Update algebraic variables and compute observation with noise.
        fout_next = func.fout(v=v_next, couple=True) if indicator else func.fout(v=v_next, f=f_next, couple=False)
        E_next = func.E(f=f_next)
        y_next = func.y(q=q_next, v=v_next) + obs_noise[i]
        if not torch.is_tensor(y_next):
            y_next = torch.tensor(y_next, dtype=torch.float32, device=device, requires_grad=True)

        # Append every computed value.
        t_list.append(t[i + 1])
        u_list.append(u_val)
        xE_list.append(xE_next)
        xI_list.append(xI_next)
        a_list.append(a_next)
        f_list.append(f_next)
        v_list.append(v_next)
        q_list.append(q_next)
        fout_list.append(fout_next)
        E_list.append(E_next)
        y_list.append(y_next)

        # Update variables for next iteration.
        xE_val, xI_val = xE_next, xI_next
        a_val, f_val = a_next, f_next
        v_val, q_val = v_next, q_next
        fout_val, E_val = fout_next, E_next

    # Append final stimulus value.
    if add_stimulus:
        final_u = func.sti_u(t[-1])
        if not torch.is_tensor(final_u):
            final_u = torch.tensor(final_u, dtype=torch.float32, device=device)
    else:
        final_u = torch.tensor(0.0, dtype=torch.float32, device=device)
    u_list.append(final_u)

    # Return full-resolution outputs.
    if select:
        return t_list, torch.stack(y_list)
    else:
        return (
            torch.stack(u_list), torch.stack(xE_list), torch.stack(xI_list),
            torch.stack(a_list), torch.stack(f_list), torch.stack(v_list),
            torch.stack(q_list), torch.stack(fout_list), torch.stack(E_list),
            torch.stack(y_list), torch.tensor(t_list, dtype=torch.float32, device=device)
        )


def simulate_bold_euler_maruyama(model, time, h,
                                 alpha_v, beta_v, alpha_e, beta_e,
                                 desired_TR=2.0,
                                 add_state_noise=True, add_obs_noise=True):
    """
    Simulates BOLD dynamics using the Euler–Maruyama method with high temporal resolution,
    then downsamples the signal to match a desired repetition time (TR).

    Args:
        model (PDCMMODEL): Instance of the PDCM hemodynamic model.
        time (torch.Tensor): Time vector for the high-resolution simulation.
        h (float): Integration step size.
        alpha_v (float): Amplitude of endogenous neuronal fluctuations.
        beta_v (float): Spectral exponent of endogenous fluctuations.
        alpha_e (float): Amplitude of measurement noise.
        beta_e (float): Spectral exponent of measurement noise.
        desired_TR (float): Desired TR for the final BOLD signal (in seconds).
        add_state_noise (bool): Whether to include state (system) noise.
        add_obs_noise (bool): Whether to include observation noise.

    Returns:
        tuple: (downsampled_time, downsampled_bold_signal)
            - downsampled_time (torch.Tensor): Time vector after downsampling.
            - downsampled_bold_signal (torch.Tensor): BOLD signal downsampled to the desired TR.
    """

    device = model.device if hasattr(model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    # Set initial conditions.
    var_init = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(0.4, device=device),
        torch.tensor(0.0, device=device)
    )

    # Run full-resolution simulation (select=False).
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        y_full,
        t_full
    ) = euler_maruyama_solver(
        func=model,
        t=time,
        h=h,
        var_init=var_init,
        indicator=False,
        alpha_v=alpha_v,
        beta_v=beta_v,
        alpha_e=alpha_e,
        beta_e=beta_e,
        add_stimulus=False,
        add_state_noise=add_state_noise,
        add_obs_noise=add_obs_noise,
        select=False
    )

    # Compute the downsampling factor.
    downsample_factor = int(desired_TR / h)
    t_down = t_full[::downsample_factor]
    y_down = y_full[::downsample_factor]

    return t_down, y_down

def simulate_bold(
    alpha_v=1.0, beta_v=1.0,
    alpha_e=1.0, beta_e=1.0,
    add_state_noise=True,
    add_obs_noise=True,
    add_stimulus=False,
    length=None,
    desired_TR=2.0,
    **kwargs
):
    """
    Simulates a synthetic BOLD signal using a PDCM model with Euler–Maruyama integration
    and returns a downsampled version according to the desired TR.

    Args:
        alpha_v (float): Amplitude of endogenous neuronal fluctuations.
        beta_v (float): Spectral exponent of endogenous fluctuations.
        alpha_e (float): Amplitude of measurement noise.
        beta_e (float): Spectral exponent of measurement noise.
        add_state_noise (bool): Whether to include state (system) noise.
        add_obs_noise (bool): Whether to include observation noise.
        add_stimulus (bool): Whether to simulate an external stimulus input.
        length (float): Length of simulation (in seconds). Must be specified.
        desired_TR (float): Desired temporal resolution (TR) for output BOLD signal.
        **kwargs: Additional parameters passed to the `PDCMMODEL.set_params()` function.

    Returns:
        tuple: (downsampled_time, downsampled_bold_signal)
            - downsampled_time (torch.Tensor): Time vector at desired TR.
            - downsampled_bold_signal (torch.Tensor): Downsampled BOLD signal.
    """

    torch.manual_seed(42)
    np.random.seed(42)

    model = PDCMMODEL()
    model.set_params(**kwargs)
    device = model.device

    h = 0.01  # Integration step size
    if length is None:
        raise ValueError("You must specify 'length' in seconds.")

    # Create high-res time vector
    num_steps = int(length / h)
    t = torch.arange(0, num_steps * h, h, device=device)

    # Initial state
    var_init = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(0.4, device=device),
        torch.tensor(0.0, device=device)
    )
    t_list, bold_signal = euler_maruyama_solver(
        func=model,
        t=t,
        h=h,
        var_init=var_init,
        indicator=False,
        alpha_v=alpha_v,
        beta_v=beta_v,
        alpha_e=alpha_e,
        beta_e=beta_e,
        add_stimulus=add_stimulus,
        add_state_noise=add_state_noise,
        add_obs_noise=add_obs_noise,
        select=True
    )

    # Downsample to match desired TR
    downsample_factor = int(desired_TR / h)
    bold_downsampled = bold_signal[::downsample_factor]

    # Clamp to expected number of points: int(length / TR)
    expected_len = int(length / desired_TR)
    bold_downsampled = bold_downsampled[:expected_len]

    t_tensor = torch.tensor(t_list, dtype=torch.float32, device=bold_signal.device)
    t_downsampled = t_tensor[::downsample_factor][:expected_len]

    return t_downsampled, bold_downsampled


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PDCMMODEL(device=device)
    model.set_params(sigma=0.5, mu=0.4, lamb=0.2, phi=1.5, chi=0.6, varphi=0.6, mtt=2.0, tau=4.0)

    # Simulation parameters
    T = 434  # Total simulation time in seconds
    h = 0.01
    time = torch.arange(0, T, h, device=device)

    # Run simulation and downsampling (using post-simulation downsampling, Method 2).
    t_down, bold_down = simulate_bold_euler_maruyama(
        model=model,
        time=time,
        h=h,
        alpha_v=1.0, beta_v=1.0,
        alpha_e=1.0, beta_e=1.0,
        desired_TR=2.0,
        add_state_noise=True,
        add_obs_noise=True
    )

    print("Downsampled Time Shape:", t_down.shape)
    print("Downsampled BOLD Shape:", bold_down.shape)

    # Detach the tensors and convert to NumPy arrays for plotting.
    t_down_np = t_down.detach().cpu().numpy() if isinstance(t_down, torch.Tensor) else np.array(t_down)
    bold_down_np = bold_down.detach().cpu().numpy() if isinstance(bold_down, torch.Tensor) else np.array(bold_down)

    #  Load real BOLD signal for comparison 
    roi_index = 52
    exp = "PLCB"
    file_path = f'/Users/xuenbei/Desktop/finalyearproject/time_series/sub-002-{exp}-ROI{roi_index}.txt'
    time_series = np.loadtxt(file_path, delimiter=',')
    TR = 2.0  # seconds
    # Create time axis for the real data.
    time_axis = np.arange(0, len(time_series) * TR, TR)
    print("Total Duration (s):", len(time_series) * TR)

    #  Plotting the simulated and real BOLD signals
    plt.figure(figsize=(12, 6))
    plt.plot(t_down_np, bold_down_np/bold_down_np.std(), label='Simulated BOLD Signal')
    plt.plot(time_axis, time_series/time_series.std(), label=f'Real BOLD Signal - ROI {roi_index}')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal')
    plt.title('Simulated Downsampled BOLD Signal vs Real Data')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Normalise signals
    bold_sim_norm = bold_down_np / bold_down_np.std()
    bold_real_norm = time_series / time_series.std()

    bold_sim_tensor = torch.tensor(bold_sim_norm, dtype=torch.float32, device=device)
    f_sim, psd_sim = torch_csd(bold_sim_tensor, bold_sim_tensor, fs=0.5, nperseg=128, nfft=512)
    f_sim, psd_sim = filter_frequencies(f_sim, psd_sim, 0.01, 0.1)

    bold_real_tensor = torch.tensor(bold_real_norm, dtype=torch.float32, device=device)
    f_real, psd_real = torch_csd(bold_real_tensor, bold_real_tensor, fs=0.5, nperseg=128, nfft=512)
    f_real, psd_real = filter_frequencies(f_real, psd_real, 0.01, 0.1)

    psd_sim = psd_sim.to(torch.complex64)
    psd_real = psd_real.to(torch.complex64)

    plot_psd_comparison(f_real, psd_real, psd_sim, f"PSD Comparison", roi_index, log_scale=True)
    plot_psd_comparison(f_sim, psd_real, psd_sim, f"PSD Comparison", roi_index, log_scale=False)
