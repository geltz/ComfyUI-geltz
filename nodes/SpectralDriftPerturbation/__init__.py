import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft as tfft

from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch

from .guidance_utils import parse_unet_blocks  # default d2.2-9,d3 blocks

SDP_OPTION = "sdp"
_BASE_SEED = 1337  # baked seed

@torch.no_grad()
def _spectral_drift_noise(
    B: int,
    T: int,
    D: int,
    *,
    drift: float = 0.05,
    coherence: float = 0.6,
    device=None,
    dtype=None,
    generator: torch.Generator | None = None,
):
    import math
    import torch
    import torch.nn.functional as F
    from torch import fft as tfft

    dev = device
    cdt = torch.float32
    gen = generator
    pi = math.pi

    n_freq = T // 2 + 1

    # narrow low-pass band
    fc_frac = 0.06 + 0.04 * max(0.0, min(1.0, drift * 4.0))
    K = max(2, int(round(fc_frac * n_freq)))

    lp_mask = torch.zeros(n_freq, device=dev, dtype=cdt)
    lp_mask[:K] = torch.hann_window(K, periodic=True, device=dev, dtype=cdt)

    # small mid-band shelf to preserve texture (very gentle)
    mid_lo = max(K + 1, int(0.18 * n_freq))
    mid_hi = max(mid_lo + 2, int(0.38 * n_freq))
    mid_len = max(2, min(n_freq - mid_lo, mid_hi - mid_lo))
    mid_mask = torch.zeros(n_freq, device=dev, dtype=cdt)
    if mid_len > 1:
        mid_mask[mid_lo:mid_lo + mid_len] = torch.hann_window(mid_len, periodic=True, device=dev, dtype=cdt)

    # base spectrum shared across channels (time structure)
    A_base = torch.randn((B, 1, n_freq), device=dev, dtype=cdt, generator=gen) * lp_mask
    phi_base = 2.0 * pi * torch.rand((B, 1, n_freq), device=dev, dtype=cdt, generator=gen)
    spec_base = torch.complex(A_base * torch.cos(phi_base), A_base * torch.sin(phi_base))
    s = tfft.irfft(spec_base, n=T, dim=-1)  # (B,1,T)

    # light smoothing
    k = 5
    pad = k // 2
    kernel = torch.ones((1, 1, k), device=dev, dtype=cdt) / float(k)
    sv = s.view(B, 1, T)
    sv = F.pad(sv, (pad, pad), mode="reflect")
    s = F.conv1d(sv, kernel).view(B, 1, T)

    # normalize over time
    s = s - s.mean(dim=-1, keepdim=True)
    s = s / s.std(dim=-1, keepdim=True).clamp_min(1e-6)          # (B,1,T)

    # zero-mean channel pattern so it survives LayerNorm
    p = torch.randn((B, D, 1), device=dev, dtype=cdt, generator=gen)
    p = p - p.mean(dim=1, keepdim=True)
    p = p / p.std(dim=1, keepdim=True).clamp_min(1e-6)           # (B,D,1)

    delta = p * s                                                # (B,D,T)

    # tiny per-channel low-freq jitter
    eps = 0.03 * max(0.0, 1.0 - float(coherence))
    if eps > 0.0:
        Aj = torch.randn((B, D, n_freq), device=dev, dtype=cdt, generator=gen) * lp_mask
        phij = 2.0 * pi * torch.rand((B, D, n_freq), device=dev, dtype=cdt, generator=gen)
        spec_j = torch.complex(Aj * torch.cos(phij), Aj * torch.sin(phij))
        sj = tfft.irfft(spec_j, n=T, dim=-1)
        sj = sj / sj.std(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta + eps * sj

    # very gentle mid-band component to keep structure
    mid_gain = 0.12
    if mid_mask.any():
        Am = torch.randn((B, 1, n_freq), device=dev, dtype=cdt, generator=gen) * mid_mask
        phim = 2.0 * pi * torch.rand((B, 1, n_freq), device=dev, dtype=cdt, generator=gen)
        spec_m = torch.complex(Am * torch.cos(phim), Am * torch.sin(phim))
        s_m = tfft.irfft(spec_m, n=T, dim=-1)                    # (B,1,T)
        s_m = s_m / s_m.std(dim=-1, keepdim=True).clamp_min(1e-6)

        q = torch.randn((B, D, 1), device=dev, dtype=cdt, generator=gen)
        q = q - q.mean(dim=1, keepdim=True)
        q = q / q.std(dim=1, keepdim=True).clamp_min(1e-6)
        delta = delta + mid_gain * q * s_m

    # stabilize per-channel energy
    delta = delta / delta.std(dim=-1, keepdim=True).clamp_min(1e-6)

    return delta.to(dtype=dtype)


class SDPTransformerWrapper(nn.Module):
    def __init__(self, transformer_block: BasicTransformerBlock, seed_offset: int = 0) -> None:
        super().__init__()
        self.wrapped_block = transformer_block
        self.seed_offset = int(seed_offset) & 0x7FFFFFFF

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, transformer_options: dict[str, Any] = {}):
        if transformer_options.get(SDP_OPTION, False):
            B, T, D = x.shape
            dev, dt = x.device, x.dtype
            params = transformer_options.get("sdp_params", {})
            drift = float(params.get("drift", 0.05))
            coherence = float(params.get("coherence", 0.6))

            # deterministic per-block generator
            gen = torch.Generator(device=dev)
            gen.manual_seed(_BASE_SEED + self.seed_offset)

            delta = _spectral_drift_noise(
                B, T, D, drift=drift, coherence=coherence, device=dev, dtype=dt, generator=gen
            )  # (B,D,T)
            x = x + delta.transpose(1, 2)  # (B,T,D)

        return self.wrapped_block(x, context=context, transformer_options=transformer_options)


class SpectralDriftPerturbation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "min_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": "d2.2-9,d3"}),
                "drift": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "coherence": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float,
        min_scale: float = 1.0,
        unet_block_list: str = "d2.2-9,d3",
        drift: float = 0.05,
        coherence: float = 0.6,
    ):
        if min_scale > scale:
            raise ValueError("min_scale must be <= scale")
        if unet_block_list and "-d" in unet_block_list and "," not in unet_block_list:
            unet_block_list = unet_block_list.replace("-d", ",d")

        m = model.clone()
        inner_model: BaseModel = m.model

        # default blocks via guidance utils
        _, block_names = parse_unet_blocks(model, unet_block_list, None) if unet_block_list else (None, None)

        for name, module in inner_model.diffusion_model.named_modules():
            if isinstance(module, BasicTransformerBlock) and "wrapped_block" not in name:
                if block_names is None or name in block_names:
                    seed_offset = abs(hash(name)) & 0x7FFFFFFF
                    m.add_object_patch(f"diffusion_model.{name}", SDPTransformerWrapper(module, seed_offset=seed_offset))

        state = {"sigma_start": None}

        def _effective_scale(sig_val: float, start: float) -> float:
            if start is None or start <= 1e-6:
                return scale
            t = max(0.0, min(1.0, sig_val / start))
            w = 0.5 * (1.0 + math.cos(math.pi * t))
            return float(scale + (min_scale - scale) * w)

        def post_cfg_function(args):
            model_: BaseModel = args["model"]
            cond_pred = args["cond_denoised"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            cond = args["cond"]
            model_options = args["model_options"].copy()

            if scale == 0 and min_scale == 0:
                return cfg_result

            try:
                sig_val = float(sigma.flatten()[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
            except Exception:
                sig_val = 1.0
            if state["sigma_start"] is None:
                state["sigma_start"] = sig_val

            topts = model_options.get("transformer_options", {}).copy()
            topts[SDP_OPTION] = True
            topts["sdp_params"] = {"drift": float(drift), "coherence": float(coherence)}
            model_options["transformer_options"] = topts

            with torch.no_grad():
                (pag_cond_pred,) = calc_cond_batch(model_, [cond], x, sigma, model_options)

            eff = _effective_scale(sig_val, state["sigma_start"])
            return cfg_result + (cond_pred - pag_cond_pred) * eff

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {"SpectralDriftPerturbation": SpectralDriftPerturbation}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectralDriftPerturbation": "Spectral Drift Perturbation"}
