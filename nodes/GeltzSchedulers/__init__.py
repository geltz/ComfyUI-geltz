# Momentum: Prediction agnostic. Velocity-based spacing: accelerates through stable regions, decelerates at critical transitions.
# River: Designed for rectified flow. Step uniformly in alpha, then map to sigma via sigma = sqrt(1/Î±^2 - 1).
# Slide: S-curve spacing. Smooth transition between momentum's brightness and river's darkness.
# Line: Perceptual linear. Uniform steps in log-space for consistent perceptual density.

import torch
import math
import sys
import comfy.samplers
import comfy.k_diffusion.sampling as k_diffusion_sampling
from comfy.samplers import KSampler

def momentum_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)
    
    accel = float(kwargs.get("accel", 1.8))
    damping = float(kwargs.get("damping", 0.3))
    inertia = float(kwargs.get("inertia", 0.9))
    
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

def slide_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)
    
    steepness = float(kwargs.get("steepness", 5.0))
    
    append_zero = True
    if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
        num_steps += 1
        append_zero = False
    
    # S-curve using sigmoid function
    t = torch.linspace(0, 1, num_steps, device=s.sigma_max.device)
    # Center sigmoid at 0.5, map to [0, 1]
    u = 1.0 / (1.0 + torch.exp(-steepness * (t - 0.5)))
    # Normalize to exact [0, 1] range
    u = (u - u[0]) / (u[-1] - u[0])
    
    timesteps = start + u * (end - start)
    
    sigmas = torch.tensor([float(s.sigma(ts)) for ts in timesteps])
    if append_zero:
        sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas

def line_sigmas(model_sampling, steps, **kwargs):
    num_steps = int(steps)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    
    if num_steps < 1:
        return torch.zeros(1)
    
    sigma_min = max(sigma_min, 1e-12)
    sigma_max = max(sigma_max, sigma_min)
    
    # Linear in log-space for perceptual uniformity
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    
    log_sigmas = torch.linspace(log_max, log_min, num_steps)
    sigmas = torch.exp(log_sigmas)
    
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas

k_diffusion_sampling.get_sigmas_momentum = momentum_sigmas
k_diffusion_sampling.get_sigmas_river = river_sigmas
k_diffusion_sampling.get_sigmas_slide = slide_sigmas
k_diffusion_sampling.get_sigmas_line = line_sigmas

if hasattr(KSampler, 'SCHEDULERS') and isinstance(KSampler.SCHEDULERS, list):
    for name in ["momentum", "river", "slide", "line"]:
        if name not in KSampler.SCHEDULERS:
            KSampler.SCHEDULERS.append(name)

original_calculate = comfy.samplers.calculate_sigmas

def patched_calculate(model_sampling, scheduler_name, steps):
    scheduler_map = {
        'momentum': momentum_sigmas,
        'river': river_sigmas,
        'slide': slide_sigmas,
        'line': line_sigmas
    }
    if scheduler_name in scheduler_map:
        return scheduler_map[scheduler_name](model_sampling, steps)
    return original_calculate(model_sampling, scheduler_name, steps)

comfy.samplers.calculate_sigmas = patched_calculate

NODE_CLASS_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS']
