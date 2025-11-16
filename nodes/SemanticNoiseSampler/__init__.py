"""
Semantic Noise Sampler (NLN + TPW)

Implements:
- NLN (Noise Latent Normalization) using ComfyUI's prepare_noise
- TPW (Temporal Prediction Weighting) over denoised predictions
- Semantic injection into initial noise with variance control
- Auto-detection and adaptation for epsilon/v-pred/x0 prediction types
"""

import math
import torch

import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview


def comfy_nln_from_latent(latent, seed, n_erase=8):
    """
    Generate NLN noise using ComfyUI's prepare_noise:
    x_NLN = (1 / sqrt(n_erase)) * sum_i prepare_noise(seed + i)
    """
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


def detect_prediction_type(model):
    """
    Detect model prediction type: 'epsilon', 'v_prediction', or 'x0'.
    Returns tuple: (prediction_type: str, needs_adjustment: bool)
    """
    try:
        # Check model.model.model_type first (most reliable)
        if hasattr(model, 'model') and hasattr(model.model, 'model_type'):
            model_type_str = str(model.model.model_type)
            if 'V_PREDICTION' in model_type_str:
                return ('v_prediction', True)
            elif 'EPS' in model_type_str:
                return ('epsilon', False)
        
        # Check calculate_denoised method name
        model_sampling = model.get_model_object("model_sampling")
        if hasattr(model_sampling, 'calculate_denoised'):
            method_name = model_sampling.calculate_denoised.__qualname__
            if 'V_PREDICTION' in method_name:
                return ('v_prediction', True)
            elif 'X0' in method_name or 'CONST' in method_name:
                return ('x0', True)
        
        # Default to epsilon
        return ('epsilon', False)
        
    except Exception as e:
        print(f"[SemanticNoise] Warning: Could not detect prediction type: {e}")
        return ('epsilon', False)

def get_delta_scale(prediction_type):
    """
    Get delta scaling factor based on prediction type.
    
    - epsilon: 1.0 (no scaling)
    - v_prediction: 0.15 (very conservative, v-pred saturates easily)
    - x0: 0.6 (moderate scaling)
    """
    scale_map = {
        'epsilon': 1.0,
        'v_prediction': 0.15,
        'x0': 0.6,
    }
    return scale_map.get(prediction_type, 1.0)


def build_power_weights(timesteps, time_center, power=5.0, base=1000.0, eps=1e-8):
    """
    Build temporal weights w(t) ~ max(-|t - t_c|^(2p) + base, 0), then normalize.
    timesteps: 1D tensor of "time" values (e.g. mapped from step index to [0, 1000])
    """
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be 1D")

    t = timesteps.to(dtype=torch.float32)
    center = float(time_center)

    diff = torch.abs(t - center)
    f = -torch.pow(diff, 2.0 * float(power)) + float(base)
    f = torch.clamp(f, min=0.0)

    total = f.sum()
    if total <= eps:
        # fallback to uniform
        return torch.full_like(f, 1.0 / float(f.numel()))
    return f / (total + eps)


def aggregate_temporal_predictions(predictions, weights, eps=1e-8):
    """
    Aggregate predictions using temporal weights.

    predictions: [K, ...]
    weights:     [K]
    returns:     [...]
    """
    if predictions.ndim < 2:
        raise ValueError("predictions must have shape [K, ...]")
    if weights.ndim != 1:
        raise ValueError("weights must be 1D")
    if predictions.shape[0] != weights.shape[0]:
        raise ValueError("shape mismatch between predictions and weights")

    K = predictions.shape[0]
    w = weights.to(dtype=predictions.dtype, device=predictions.device)
    view_shape = (K,) + (1,) * (predictions.ndim - 1)
    w_view = w.view(view_shape)

    weighted = predictions * w_view
    numerator = weighted.sum(dim=0)
    denom = math.sqrt(float((w ** 2).sum().item()) + eps)
    return numerator / denom


def normalize_per_sample(x, eps=1e-8):
    """
    Normalize each sample in the batch to mean 0, std 1 over all non-batch dims.
    Works for [B, C, H, W] or [B, N] etc.
    """
    if x.ndim < 2:
        return x

    dims = tuple(range(1, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return (x - mean) / (std + eps)


def inject_semantics(x_T, eps_agg, delta):
    """
    Inject semantic content into noise with variance control:

    x_sem = (x_T + delta * eps_agg) / sqrt(1 + delta^2)

    Assumes x_T and eps_agg are per-sample normalized (mean 0, std 1).
    This keeps the overall scale close to unit variance while rotating
    the noise direction toward the semantic component.
    """
    if x_T.shape != eps_agg.shape:
        raise ValueError("inject_semantics: shape mismatch")
    if delta < 0.0:
        raise ValueError("inject_semantics: delta must be non-negative")

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

                # NLN
                "n_erase": ("INT", {"default": 8, "min": 1, "max": 100}),

                # TPW / semantics
                "tpw_steps": ("INT", {"default": 5, "min": 1, "max": 20}),
                "time_center": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0}),
                "power": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0}),
                "delta": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/semantic"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        n_erase,
        tpw_steps,
        time_center,
        power,
        delta,
    ):
        print(f"[SemanticNoise] seed={seed}, steps={steps}, cfg={cfg}, denoise={denoise}")
        print(
            f"[SemanticNoise] n_erase={n_erase}, tpw_steps={tpw_steps}, "
            f"time_center={time_center}, power={power}, delta={delta}"
        )

        device = comfy.model_management.get_torch_device()

        # Auto-detect prediction type
        pred_type, needs_adjustment = detect_prediction_type(model)
        
        delta_scale = get_delta_scale(pred_type)
        delta_adjusted = delta * delta_scale
        
        print(f"[SemanticNoise] Detected prediction type: {pred_type}")
        print(f"[SemanticNoise] Delta scale: {delta_scale:.2f} (effective delta: {delta_adjusted:.2f})")
        
        # Work on a copy of the latent dict
        latent = latent_image.copy()
        latent_samples = latent["samples"]

        # Ensure latent channels are compatible with the model
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
        latent["samples"] = latent_samples

        print(
            f"[SemanticNoise] Latent shape: {latent_samples.shape}, "
            f"mean={latent_samples.mean().item():.4f}, std={latent_samples.std().item():.4f}"
        )

        # 1) NLN: build initial noise
        print(f"[SemanticNoise] Generating NLN noise with n_erase={n_erase}...")
        nln_noise = comfy_nln_from_latent(latent, seed=seed, n_erase=n_erase)
        nln_noise = nln_noise.to(device)

        print(
            f"[SemanticNoise] NLN noise shape: {nln_noise.shape}, "
            f"mean={nln_noise.mean().item():.4f}, std={nln_noise.std().item():.4f}"
        )

        # 2) Build KSampler so we can control the loop
        ksampler = comfy.samplers.KSampler(
            model,
            steps,
            device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )

        sigmas = ksampler.sigmas
        num_sigmas = int(sigmas.shape[0])
        total_steps = max(num_sigmas - 1, 1)  # number of denoising steps
        print(f"[SemanticNoise] total_steps (sampler steps) = {total_steps}")

        tpw_steps_eff = max(1, min(int(tpw_steps), total_steps))
        if tpw_steps_eff != tpw_steps:
            print(
                f"[SemanticNoise] Clamping tpw_steps from {tpw_steps} "
                f"to {tpw_steps_eff} based on total_steps={total_steps}"
            )

        noise_mask = latent.get("noise_mask", None)

        # 3) First pass: run sampling to collect predictions at selected timesteps
        semantic_pred_dict = {}

        if tpw_steps_eff > 0 and total_steps > 0:
            if total_steps > 1:
                step_values = torch.linspace(0, total_steps - 1, steps=tpw_steps_eff)
            else:
                step_values = torch.zeros(tpw_steps_eff)

            step_indices = [int(x.item()) for x in step_values]
            idx_set = set(step_indices)

            print(f"[SemanticNoise] TPW will collect at step indices: {sorted(idx_set)}")

            def semantic_callback(i, denoised, x_t, total_steps_cb):
                i_int = int(i)
                # Collect only the first time we see each index
                if i_int in idx_set and i_int not in semantic_pred_dict:
                    semantic_pred_dict[i_int] = denoised.detach().cpu()

            print("[SemanticNoise] Running first pass (TPW collection)...")
            # First pass
            _ = ksampler.sample(
                nln_noise,
                positive,
                negative,
                cfg,
                latent_image=latent_samples,
                callback=semantic_callback,
                disable_pbar=True,
                seed=seed,
            )
            
        else:
            print("[SemanticNoise] Skipping TPW collection (no valid steps).")

        # 4) Aggregate predictions and inject semantics
        if len(semantic_pred_dict) == 0:
            print("[SemanticNoise] No TPW predictions collected, using pure NLN noise (normalized).")
            semantic_noise = normalize_per_sample(nln_noise)
        else:
            sorted_indices = sorted(semantic_pred_dict.keys())
            preds = torch.stack(
                [semantic_pred_dict[i] for i in sorted_indices],
                dim=0,
            )  # [K, B, C, H, W] or [K, ...] on CPU

            indices_tensor = torch.tensor(sorted_indices, dtype=torch.float32)
            if total_steps > 1:
                scale = 1000.0 / float(total_steps - 1)
            else:
                scale = 0.0
            timesteps = indices_tensor * scale

            print(
                f"[SemanticNoise] Aggregating {len(sorted_indices)} TPW predictions; "
                f"t range: [{timesteps.min().item():.2f}, {timesteps.max().item():.2f}]"
            )

            weights = build_power_weights(
                timesteps,
                time_center=time_center,
                power=power,
            )

            eps_agg = aggregate_temporal_predictions(preds, weights)  # on CPU
            eps_agg = eps_agg.to(device)

            # Normalize both NLN noise and aggregated semantics per sample
            nln_norm = normalize_per_sample(nln_noise)
            eps_norm = normalize_per_sample(eps_agg)
            
            semantic_noise = inject_semantics(nln_norm, eps_norm, delta_adjusted)
            print(
                f"[SemanticNoise] Semantic-injected noise (delta={delta_adjusted:.2f}): "
                f"mean={semantic_noise.mean().item():.4f}, std={semantic_noise.std().item():.4f}"
            )

        # 5) Second pass: actual sampling with preview callback
        preview_callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        print("[SemanticNoise] Running second pass from semantic-injected noise (with preview)...")
        # Second pass
        samples_tensor = ksampler.sample(
            semantic_noise,
            positive,
            negative,
            cfg,
            latent_image=latent_samples,
            callback=preview_callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = latent.copy()
        out["samples"] = samples_tensor
        print(f"[SemanticNoise] Done. Output shape: {out['samples'].shape}")
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SemanticNoise": SemanticNoiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticNoise": "Semantic Noise Sampler",
}