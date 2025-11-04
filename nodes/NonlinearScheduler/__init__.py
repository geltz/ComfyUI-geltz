import torch
import math
import comfy.samplers
import comfy.k_diffusion.sampling as k_diffusion_sampling
from comfy.samplers import KSampler

def nonlinear_scheduler(model_sampling, steps):
    """
    Aggressive early denoising.
    """
    # keep these as tensor-like, don't cast to float
    sigma_max = model_sampling.sigma_max
    sigma_min = model_sampling.sigma_min

    # convert to timestep space for proper interpolation
    ts_max = model_sampling.timestep(sigma_max)
    ts_min = model_sampling.timestep(sigma_min)

    rho = 2.5
    gamma = 2.0

    timesteps = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        t_warped = t ** gamma
        ts = (ts_max ** (1 / rho) * (1 - t_warped) + ts_min ** (1 / rho) * t_warped) ** rho
        timesteps.append(ts)

    sigmas = []
    for ts in timesteps:
        sigma = model_sampling.sigma(ts)
        # keep as tensor-like, don't cast to float
        sigmas.append(sigma)

    # ensure endpoint as tensor
    sigmas.append(torch.tensor(0.0, device=sigmas[0].device, dtype=sigmas[0].dtype))

    # stack into a single tensor like other schedulers
    return torch.stack(sigmas)

k_diffusion_sampling.get_sigmas_nonlinear = nonlinear_scheduler

if hasattr(KSampler, 'SCHEDULERS') and isinstance(KSampler.SCHEDULERS, list):
    if "nonlinear" not in KSampler.SCHEDULERS:
        KSampler.SCHEDULERS.append("nonlinear")

original_calculate = comfy.samplers.calculate_sigmas

def patched_calculate(model_sampling, scheduler_name, steps):
    if scheduler_name == 'nonlinear':
        return nonlinear_scheduler(model_sampling, steps)
    return original_calculate(model_sampling, scheduler_name, steps)

comfy.samplers.calculate_sigmas = patched_calculate

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
