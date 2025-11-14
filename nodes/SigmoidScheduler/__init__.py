import torch
import math
import comfy.samplers
import comfy.k_diffusion.sampling as k_diffusion_sampling
from comfy.samplers import KSampler

def sigmoid_scheduler(model_sampling, steps, steepness=6.0, center=0.5):
    """
    Configurable parameters.
    """
    sigma_max = model_sampling.sigma_max
    sigma_min = model_sampling.sigma_min
    
    # precompute normalization constants more efficiently
    sig_min = 1.0 / (1.0 + math.exp(steepness * (0.5 - center)))
    sig_max = 1.0 / (1.0 + math.exp(steepness * (-center)))
    norm_denom = sig_max - sig_min
    
    # vectorized computation
    t_linear = torch.linspace(1, 0, steps, device=sigma_max.device)  # reversed
    
    # sigmoid warping with better numerical stability
    t_warped = 1.0 / (1.0 + torch.exp(-steepness * (t_linear - center)))
    t_warped = (t_warped - sig_min) / norm_denom
    
    # clamp to avoid numerical issues
    t_warped = torch.clamp(t_warped, 0.0, 1.0)
    
    # interpolate timesteps (reversed for denoising)
    ts_max = model_sampling.timestep(sigma_max)
    ts_min = model_sampling.timestep(sigma_min)
    timesteps = ts_max * t_warped + ts_min * (1.0 - t_warped)
    
    # convert to sigmas
    sigmas = model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, torch.tensor([0.0], device=sigmas.device)])
    
    return sigmas

# expose to k-diffusion-like namespace
k_diffusion_sampling.get_sigmas_sigmoid = sigmoid_scheduler

# register with KSampler if available
if hasattr(KSampler, 'SCHEDULERS') and isinstance(KSampler.SCHEDULERS, list):
    if "sigmoid" not in KSampler.SCHEDULERS:
        KSampler.SCHEDULERS.append("sigmoid")

# preserve original calculate, then patch
original_calculate = comfy.samplers.calculate_sigmas

def patched_calculate(model_sampling, scheduler_name, steps):
    if scheduler_name == 'sigmoid':
        return sigmoid_scheduler(model_sampling, steps)
    # keep other schedulers working
    return original_calculate(model_sampling, scheduler_name, steps)

comfy.samplers.calculate_sigmas = patched_calculate

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
