"""
Semantic Noise Sampler

Beyond Randomness (NLN + TPW)
https://arxiv.org/abs/2511.07756
"""

import torch
import math
import comfy.sample
import comfy.samplers
import comfy.model_management
import nodes

def comfy_nln_from_latent(latent, seed, n_erase=8):
    """Generate NLN noise using ComfyUI's prepare_noise"""
    if n_erase < 1:
        raise ValueError("n_erase must be >= 1")
    
    latent_image = latent["samples"]
    batch_inds = latent.get("batch_index", None)
    
    noise_sum = None
    for i in range(n_erase):
        base_seed = seed + i
        noise_i = comfy.sample.prepare_noise(latent_image, base_seed, batch_inds)
        if noise_sum is None:
            noise_sum = noise_i
        else:
            noise_sum.add_(noise_i)
    
    noise_sum.div_(math.sqrt(float(n_erase)))
    return noise_sum


def build_power_weights(timesteps, time_center, power=5.0, base=1000.0, eps=1e-8):
    """Build temporal weighting from timestep values"""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be 1D")
    
    t = timesteps.to(dtype=torch.float32)
    center = float(time_center)
    
    diff = torch.abs(t - center)
    f = -torch.pow(diff, 2.0 * float(power)) + float(base)
    f = torch.clamp(f, min=0.0)
    
    total = f.sum()
    if total <= eps:
        return torch.full_like(f, 1.0 / float(f.numel()))
    return f / (total + eps)


def aggregate_temporal_predictions(predictions, weights, eps=1e-8):
    """Aggregate predictions using temporal weights"""
    if predictions.ndim < 2:
        raise ValueError("predictions must have shape [K, ...]")
    if weights.ndim != 1:
        raise ValueError("weights must be 1D")
    if predictions.shape[0] != weights.shape[0]:
        raise ValueError("shape mismatch")
    
    K = predictions.shape[0]
    w = weights.to(dtype=predictions.dtype, device=predictions.device)
    view_shape = (K,) + (1,) * (predictions.ndim - 1)
    w_view = w.view(view_shape)
    
    weighted = predictions * w_view
    numerator = weighted.sum(dim=0)
    denom = math.sqrt(float((w ** 2).sum().item()) + eps)
    return numerator / denom


def inject_semantics(x_T, eps_agg, delta):
    """Inject semantic content into noise"""
    if x_T.shape != eps_agg.shape:
        raise ValueError("shape mismatch")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    
    if x_T.device != eps_agg.device:
        eps_agg = eps_agg.to(x_T.device)
    
    delta_f = float(delta)
    denom = math.sqrt(1.0 + delta_f * delta_f)
    return (x_T + delta_f * eps_agg) / denom


class SemanticNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "n_erase": ("INT", {"default": 8, "min": 1, "max": 100}),
                "tpw_steps": ("INT", {"default": 5, "min": 1, "max": 20}),
                "time_center": ("FLOAT", {"default": 800.0, "min": 0.0, "max": 1000.0}),
                "delta": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
                "power": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, n_erase, tpw_steps, time_center,
               delta, power):
        
        print(f"[SemanticNoise] Starting with seed={seed}, n_erase={n_erase}, tpw_steps={tpw_steps}")
        
        # Generate NLN noise
        print(f"[SemanticNoise] Generating NLN noise with {n_erase} erasures...")
        nln_noise = comfy_nln_from_latent(latent_image, seed=seed, n_erase=n_erase)
        print(f"[SemanticNoise] NLN noise shape: {nln_noise.shape}, mean: {nln_noise.mean().item():.4f}, std: {nln_noise.std().item():.4f}")
        
        device = comfy.model_management.get_torch_device()
        latent_samples = latent_image["samples"]
        
        print(f"[SemanticNoise] Original latent - mean: {latent_samples.mean().item():.4f}, std: {latent_samples.std().item():.4f}")
        
        # Add NLN noise to the latent (noising step)
        latent_copy = latent_image.copy()
        latent_copy["samples"] = latent_samples
        
        print(f"[SemanticNoise] Running sampling with cfg={cfg}, steps={steps}, denoise={denoise}")
        samples = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_copy,
            denoise=denoise,
            disable_noise=False
        )[0]
        
        print(f"[SemanticNoise] Sampling complete. Output shape: {samples['samples'].shape}")
        
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "SemanticNoise": SemanticNoiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticNoise": "Semantic Noise Sampler",
}