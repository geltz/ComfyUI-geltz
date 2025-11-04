import torch
import math
import comfy.samplers
import comfy.k_diffusion.sampling as k_diffusion_sampling
from comfy.samplers import KSampler

def cosine_scheduler(model_sampling, steps):
    """
    Smooth cosine progression for denoising.
    """
    # keep these as tensor-like
    sigma_max = model_sampling.sigma_max
    sigma_min = model_sampling.sigma_min

    # go to timestep space
    ts_max = model_sampling.timestep(sigma_max)
    ts_min = model_sampling.timestep(sigma_min)

    timesteps = []
    for i in range(steps):
        # linear phase
        t = i / max(steps - 1, 1)
        # cosine ease: 0 -> 1 smoothly
        t_warped = 0.5 * (1.0 - math.cos(math.pi * t))
        # interpolate in the same "power" domain as upstream does
        # but since we're already in timestep space we can stay simple
        ts = ts_max * (1.0 - t_warped) + ts_min * t_warped
        timesteps.append(ts)

    sigmas = []
    for ts in timesteps:
        sigma = model_sampling.sigma(ts)
        sigmas.append(sigma)

    # ensure endpoint as tensor
    sigmas.append(torch.tensor(0.0, device=sigmas[0].device, dtype=sigmas[0].dtype))

    return torch.stack(sigmas)

# expose to k-diffusion-like namespace
k_diffusion_sampling.get_sigmas_cosine = cosine_scheduler

# register with KSampler if available
if hasattr(KSampler, 'SCHEDULERS') and isinstance(KSampler.SCHEDULERS, list):
    if "cosine" not in KSampler.SCHEDULERS:
        KSampler.SCHEDULERS.append("cosine")

# preserve original calculate, then patch
original_calculate = comfy.samplers.calculate_sigmas

def patched_calculate(model_sampling, scheduler_name, steps):
    if scheduler_name == 'cosine':
        return cosine_scheduler(model_sampling, steps)
    # keep other schedulers working
    return original_calculate(model_sampling, scheduler_name, steps)

comfy.samplers.calculate_sigmas = patched_calculate

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
