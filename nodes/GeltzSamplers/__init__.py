import torch
from tqdm.auto import trange
from comfy.k_diffusion import sampling as ksampling
from comfy.samplers import KSampler

# clamp sigmas

def to_d(x, sigma, denoised):
    eps = torch.finfo(x.dtype).eps if torch.is_floating_point(x) else 1e-8
    return (x - denoised) / sigma.clamp_min(eps)

# ----------------- sampler definitions -----------------

@torch.no_grad()
def sample_ralston(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Third-order Ralston method with optimal error coefficients."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        
        # Stage 1
        denoised = model(x, sigma * s_in, **extra_args)
        d1 = to_d(x, sigma, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'denoised': denoised})
        
        # Stage 2
        x_2 = x + d1 * dt * 0.5
        sigma_2 = sigma + dt * 0.5
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        d2 = to_d(x_2, sigma_2, denoised_2)
        
        # Stage 3
        x_3 = x + d2 * dt * 0.75
        sigma_3 = sigma + dt * 0.75
        denoised_3 = model(x_3, sigma_3 * s_in, **extra_args)
        d3 = to_d(x_3, sigma_3, denoised_3)
        
        # Ralston's optimal weights
        x = x + dt * (2/9 * d1 + 3/9 * d2 + 4/9 * d3)
    
    return x

@torch.no_grad()
def sample_bogacki(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Third-order Bogacki-Shampine method."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        
        # Stage 1
        denoised = model(x, sigma * s_in, **extra_args)
        d1 = to_d(x, sigma, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'denoised': denoised})
        
        # Stage 2
        x_2 = x + d1 * dt * 0.5
        sigma_2 = sigma + dt * 0.5
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        d2 = to_d(x_2, sigma_2, denoised_2)
        
        # Stage 3
        x_3 = x + d2 * dt * 0.75
        sigma_3 = sigma + dt * 0.75
        denoised_3 = model(x_3, sigma_3 * s_in, **extra_args)
        d3 = to_d(x_3, sigma_3, denoised_3)
        
        # Final step with Bogacki-Shampine weights
        x = x + dt * (2/9 * d1 + 1/3 * d2 + 4/9 * d3)
    
    return x

# ----------------- registration -----------------
ksampling.sample_ralston = sample_ralston
ksampling.sample_bogacki = sample_bogacki

# Register samplers with ComfyUI
sampler_mappings = [
    ("ralston", sample_ralston),
    ("bogacki", sample_bogacki),
]

try:
    if hasattr(KSampler, "SAMPLERS"):
        if isinstance(KSampler.SAMPLERS, list):
            if KSampler.SAMPLERS and isinstance(KSampler.SAMPLERS[0], (list, tuple)):
                # ComfyUI expects list of (name, function) tuples
                for name, sampler_func in sampler_mappings:
                    if (name, name) not in KSampler.SAMPLERS:
                        KSampler.SAMPLERS.append((name, name))
            else:
                # Older format - list of names
                for name, sampler_func in sampler_mappings:
                    if name not in KSampler.SAMPLERS:
                        KSampler.SAMPLERS.append(name)
except Exception as e:
    print(f"Warning: sampler registration failed: {e}")

NODE_CLASS_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS"] + [name for name, _ in sampler_mappings]
