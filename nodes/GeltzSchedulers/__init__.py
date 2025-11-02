# Rewind: Log-linear with sinusoidal warp (amp=0.15); non-uniform spacing for ODE steps with optional stall kicks.
# River: Tailor-made for rectified flow models. Step uniformly in alpha, then map to sigma via sigma = sqrt(1/α^2 - 1).     
# Power: Power-law interpolation (rho=7.0); concentrates steps where noise changes matter most for improved quality.
# SNR Uniform: Steps uniformly in log SNR and maps back to σ to produce a stable k-diffusion sigma schedule.    

import torch
import math
import sys
import comfy.samplers
import comfy.k_diffusion.sampling as k_diffusion_sampling
from comfy.samplers import KSampler

def rewind_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    warp_amp = 0.15
    u = torch.linspace(0, 1, num_steps)
    u = torch.clamp(u + warp_amp * torch.sin(2 * math.pi * u), 0.0, 1.0)
    log_s = (1 - u) * math.log(sigma_max) + u * math.log(sigma_min)
    sigmas = torch.exp(log_s)
    sigmas, _ = torch.sort(sigmas, descending=True)
    sigmas[-1] = max(sigmas[-1].item(), sigma_min)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    sigmas = torch.clamp(sigmas, max=sigma_max)
    sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=sigma_max, neginf=0.0)
    return sigmas

def river_sigmas(model_sampling, steps, **kwargs):
    import torch, math
    num_steps = int(steps)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    if num_steps < 1:
        return torch.zeros(1)

    sigma_min = max(sigma_min, 1e-12)
    sigma_max = max(sigma_max, sigma_min)

    def alpha_of_sigma(s):
        return 1.0 / math.sqrt(1.0 + s * s)

    alpha_min = alpha_of_sigma(sigma_max)
    alpha_max = alpha_of_sigma(sigma_min)

    u = torch.linspace(0.0, 1.0, num_steps)

    # minimal low-noise ramp: pushes mass toward small σ without changing endpoints
    r = float(kwargs.get("low_noise_ramp", 0.20))  # 0 disables bias when set to 0
    if r > 0:
        u = 1.0 - torch.pow(1.0 - u, 1.0 + r)

    alpha = alpha_min + u * (alpha_max - alpha_min)
    sigmas = torch.sqrt(torch.clamp(1.0 / (alpha * alpha) - 1.0, min=0.0))

    sigmas, _ = torch.sort(sigmas, descending=True)
    sigmas[-1] = max(sigmas[-1].item(), sigma_min)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    sigmas = torch.clamp(sigmas, max=sigma_max)
    sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=sigma_max, neginf=0.0)
    return sigmas

def power_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    rho = 7.0
    if num_steps < 1:
        return torch.zeros(1)
    sigma_min = max(sigma_min, 1e-10)
    sigma_max = max(sigma_max, sigma_min)
    t = torch.linspace(0, 1, num_steps)
    min_pow = sigma_min ** (1/rho)
    max_pow = sigma_max ** (1/rho)
    sigmas = (max_pow + t * (min_pow - max_pow)) ** rho
    sigmas = torch.clamp(sigmas, min=0.0)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=sigma_max, neginf=0.0)
    return sigmas

def snr_uniform_sigmas(model_sampling, steps, *, device=None, out_dtype=torch.float32, sigma_min=None, sigma_max=None):
    n = int(steps)
    if n < 1:
        return torch.zeros(1, dtype=out_dtype, device=device or "cpu")
    
    smin = float(sigma_min if sigma_min is not None else getattr(model_sampling, "sigma_min", 0.0))
    smax = float(sigma_max if sigma_max is not None else getattr(model_sampling, "sigma_max", 1.0))
    smin = max(smin, 1e-6)
    smax = max(smax, smin)

    work_dev = device or "cpu"
    wdtype = torch.float64

    sigma_min_t = torch.tensor(smin, dtype=wdtype, device=work_dev)
    sigma_max_t = torch.tensor(smax, dtype=wdtype, device=work_dev)

    def lam(s):
        return -0.5 * torch.log1p(s * s) - torch.log(s)

    lam_min = lam(sigma_max_t)
    lam_max = lam(sigma_min_t)

    lam_u = torch.linspace(lam_min, lam_max, n, dtype=wdtype, device=work_dev)
    t = torch.exp(-2.0 * lam_u)
    t = torch.clamp(t, min=0.0, max=1e50)

    sigma_sq = 0.5 * (-1.0 + torch.sqrt(1.0 + 4.0 * t))
    sigmas = torch.sqrt(torch.clamp(sigma_sq, min=0.0))
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=wdtype, device=work_dev)], dim=0)

    return sigmas.to(dtype=out_dtype)

k_diffusion_sampling.get_sigmas_rewind = rewind_sigmas
k_diffusion_sampling.get_sigmas_rewind = river_sigmas
k_diffusion_sampling.get_sigmas_power = power_sigmas
k_diffusion_sampling.get_sigmas_snr_uniform = snr_uniform_sigmas

if hasattr(KSampler, 'SCHEDULERS') and isinstance(KSampler.SCHEDULERS, list):
    for name in ["rewind", "river", "power", "snr_uniform"]:
        if name not in KSampler.SCHEDULERS:
            KSampler.SCHEDULERS.append(name)

original_calculate = comfy.samplers.calculate_sigmas

def patched_calculate(model_sampling, scheduler_name, steps):
    scheduler_map = {
        'rewind': rewind_sigmas,
        'river': river_sigmas,
        'power': power_sigmas,
        'snr_uniform': snr_uniform_sigmas
    }
    if scheduler_name in scheduler_map:
        return scheduler_map[scheduler_name](model_sampling, steps)
    return original_calculate(model_sampling, scheduler_name, steps)

comfy.samplers.calculate_sigmas = patched_calculate

NODE_CLASS_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS']



