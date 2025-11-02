# Rewind: Log-linear with sinusoidal warp (amp=0.15); non-uniform spacing for ODE steps with optional stall kicks.
# River: Tailor-made for rectified flow models. Step uniformly in alpha, then map to sigma via sigma = sqrt(1/α^2 - 1).     
# Power: Power-law interpolation (rho=7.0); concentrates steps where noise changes matter most for improved quality.
# SNR Uniform: Steps uniformly in log SNR and maps back to σ to produce a stable k-diffusion sigma schedule.    
# Momentum: Velocity-based spacing: accelerates through stable regions, decelerates at critical transitions.

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

    # ramp
    r = float(kwargs.get("low_noise_ramp", 0.30))
    if r > 0:
        u = 1.0 - torch.pow(1.0 - u, 1.0 + r)

    # calculate sigmas
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

def snr_uniform_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    
    if num_steps < 1:
        return torch.zeros(1)
    
    sigma_min = max(sigma_min, 1e-6)
    sigma_max = max(sigma_max, sigma_min)
    
    # log-SNR: λ(σ) = -0.5*log(1+σ²) - log(σ)
    def log_snr(s):
        return -0.5 * torch.log1p(s * s) - torch.log(s)
    
    lam_min = log_snr(torch.tensor(sigma_max))
    lam_max = log_snr(torch.tensor(sigma_min))
    
    u = torch.linspace(0.0, 1.0, num_steps)
    
    # apply ramp like river
    r = float(kwargs.get("low_noise_ramp", 0.30))
    if r > 0:
        u = 1.0 - torch.pow(1.0 - u, 1.0 + r)
    
    lam_u = lam_min + u * (lam_max - lam_min)
    t = torch.exp(-2.0 * lam_u)
    
    # Solve σ² from t = exp(-2λ) = σ²(1+σ²)
    sigma_sq = 0.5 * (-1.0 + torch.sqrt(1.0 + 4.0 * t))
    sigmas = torch.sqrt(torch.clamp(sigma_sq, min=0.0))
    
    sigmas, _ = torch.sort(sigmas, descending=True)
    sigmas[-1] = max(sigmas[-1].item(), sigma_min)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    sigmas = torch.clamp(sigmas, max=sigma_max)
    sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=sigma_max, neginf=0.0)
    return sigmas

def momentum_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)
    
    accel = float(kwargs.get("accel", 1.8))
    damping = float(kwargs.get("damping", 0.3))
    inertia = float(kwargs.get("inertia", 0.82))
    
    append_zero = True
    if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
        num_steps += 1
        append_zero = False
    
    # Simulate momentum-based traversal with velocity persistence
    u = torch.zeros(num_steps, device=s.sigma_max.device)
    velocity = 0.0
    position = 0.0
    
    for i in range(num_steps):
        # Acceleration varies by position (slower at ends)
        force = accel * (1.0 - damping * abs(2.0 * position - 1.0))
        velocity = inertia * velocity + (1.0 - inertia) * force / num_steps
        position += velocity / num_steps
        position = min(position, 1.0)
        u[i] = position
    
    # Normalize to [0, 1]
    u = u / u[-1] if u[-1] > 0 else u
    
    timesteps = start + u * (end - start)
    
    sigmas = torch.tensor([float(s.sigma(ts)) for ts in timesteps])
    if append_zero:
        sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas

k_diffusion_sampling.get_sigmas_rewind = rewind_sigmas
k_diffusion_sampling.get_sigmas_rewind = river_sigmas
k_diffusion_sampling.get_sigmas_power = power_sigmas
k_diffusion_sampling.get_sigmas_snr_uniform = snr_uniform_sigmas
k_diffusion_sampling.get_sigmas_momentum_uniform = momentum_sigmas

if hasattr(KSampler, 'SCHEDULERS') and isinstance(KSampler.SCHEDULERS, list):
    for name in ["rewind", "river", "power", "snr_uniform", "momentum"]:
        if name not in KSampler.SCHEDULERS:
            KSampler.SCHEDULERS.append(name)

original_calculate = comfy.samplers.calculate_sigmas

def patched_calculate(model_sampling, scheduler_name, steps):
    scheduler_map = {
        'rewind': rewind_sigmas,
        'river': river_sigmas,
        'power': power_sigmas,
        'snr_uniform': snr_uniform_sigmas,
        'momentum': momentum_sigmas
    }
    if scheduler_name in scheduler_map:
        return scheduler_map[scheduler_name](model_sampling, steps)
    return original_calculate(model_sampling, scheduler_name, steps)

comfy.samplers.calculate_sigmas = patched_calculate

NODE_CLASS_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS']



















