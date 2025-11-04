import math
from typing import Any

import torch
from torch import nn
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch

from .guidance_utils import parse_unet_blocks

# transformer_options keys
TDP_FLAG = "tdp_enable"          # enable token perturbation on this pass
TDP_EXEMPT_MASK = "tdp_exempt"   # optional: (B, T) bool, True = do NOT shuffle this token


class TDPTransformerWrapper(nn.Module):
    """
    Wrap a BasicTransformerBlock. When TDP_FLAG is set, partially shuffle only the
    non-exempt tokens in x. This avoids hardcoding PAD/EOS/CLS.
    """
    
    def __init__(self, transformer_block: BasicTransformerBlock, alpha: float = 0.35) -> None:
        super().__init__()
        self.wrapped_block = transformer_block
        self.alpha = alpha

    def _partial_shuffle(self, x: torch.Tensor, exempt_mask: torch.Tensor | None) -> torch.Tensor:
        if x is None or x.ndim != 3 or self.alpha <= 0.0:
            return x
        B, T, C = x.shape

        if exempt_mask is None:
            perm = torch.randperm(T, device=x.device)
            x_shuf = x[:, perm]
            return x * (1.0 - self.alpha) + x_shuf * self.alpha

        exempt_mask = exempt_mask.to(x.device).bool()
        x_out = x.clone()
        for b in range(B):
            keep = exempt_mask[b]
            movable_idx = (~keep).nonzero(as_tuple=False).flatten()
            if movable_idx.numel() < 2:
                continue
            shuffled_movable = movable_idx[torch.randperm(movable_idx.numel(), device=x.device)]
            x_out[b, movable_idx] = x[b, shuffled_movable]
        return x * (1.0 - self.alpha) + x_out * self.alpha

    def forward(self, x, context=None, transformer_options=None):
        transformer_options = transformer_options or {}
        if transformer_options.get("tdp_enable", False):
            exempt_mask = transformer_options.get("tdp_exempt", None)

            # only perturb context-like sequences, not spatial x
            if context is not None and context.ndim == 3:
                # heuristic: short sequences are probably text/cond
                if context.size(1) <= 256:
                    context = self._partial_shuffle(context, exempt_mask)
                # else: leave big sequences alone
        return self.wrapped_block(x, context=context, transformer_options=transformer_options)

class TokenDeltaPerturbation:
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
    ):
        if min_scale > scale:
            raise ValueError("min_scale should be <= scale")

        # normalize "d2.2-9-d3" into "d2.2-9,d3"
        if unet_block_list and "-d" in unet_block_list and "," not in unet_block_list:
            unet_block_list = unet_block_list.replace("-d", ",d")

        m = model.clone()
        inner_model: BaseModel = m.model

        # select UNet transformer blocks to wrap
        _, block_names = (
            parse_unet_blocks(model, unet_block_list, None) if unet_block_list else (None, None)
        )

        # wrap only selected transformer blocks
        for name, module in inner_model.diffusion_model.named_modules():
            if (
                isinstance(module, BasicTransformerBlock)
                and "wrapped_block" not in name
                and (block_names is None or name in block_names)
            ):
                m.add_object_patch(
                    f"diffusion_model.{name}",
                    TDPTransformerWrapper(module, alpha=0.35),
                )

        state: dict[str, Any] = {"sigma_start": None, "step": 0}

        def _effective_scale(sig_val: float, start: float | None) -> float:
            # cosine schedule from scale -> min_scale as sigma decreases
            if start is None or start <= 1e-6:
                return scale
            t = max(0.0, min(1.0, sig_val / start))
            w = 0.5 * (1.0 + math.cos(math.pi * t))
            return float(min_scale + (scale - min_scale) * w)

        def post_cfg_function(args):
            model: BaseModel = args["model"]
            cond_pred = args["cond_denoised"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            cond = args["cond"]
            model_options = args["model_options"].copy()

            if scale == 0 and min_scale == 0:
                return cfg_result

            # current sigma
            try:
                sig_val = (
                    float(sigma.flatten()[0].item())
                    if isinstance(sigma, torch.Tensor)
                    else float(sigma)
                )
            except Exception:
                sig_val = 1.0

            if state["sigma_start"] is None:
                state["sigma_start"] = sig_val

            state["step"] += 1

            # enable token perturbation for the second pass
            topts = model_options.get("transformer_options", {}).copy()
            topts[TDP_FLAG] = True

            # if caller supplied an exempt mask on the original pass, forward it
            orig_topts = args.get("model_options", {}).get("transformer_options", {})
            if TDP_EXEMPT_MASK in orig_topts:
                topts[TDP_EXEMPT_MASK] = orig_topts[TDP_EXEMPT_MASK]

            model_options["transformer_options"] = topts

            # run perturbed conditional pass
            with torch.no_grad():
                (pag_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)

            # delta between normal conditional and perturbed conditional
            delta = cond_pred - pag_cond_pred

            # 1) remove global bias
            delta = delta - delta.mean(dim=(1, 2, 3), keepdim=True)

            # 2) magnitude-aware normalize
            delta_mag = delta.abs().mean(dim=(1, 2, 3), keepdim=True)
            cond_mag = cond_pred.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            norm_factor = (cond_mag / (delta_mag + 1e-6)).clamp(0.0, 1.5)
            delta = delta * norm_factor

            # 3) clamp
            delta = delta.clamp(-1.0, 1.0)

            eff = _effective_scale(sig_val, state["sigma_start"])

            # optional debug
            if state["step"] % 5 == 1:
                print(
                    f"[TDP] step={state['step']} sigma={sig_val:.4f} eff={eff:.3f} "
                    f"|delta|={delta_mag.mean().item():.6f}"
                )

            return cfg_result + delta * eff

        # ensure we run even at CFG=1
        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {"TokenDeltaPerturbation": TokenDeltaPerturbation}
NODE_DISPLAY_NAME_MAPPINGS = {"TokenDeltaPerturbation": "Token Delta Perturbation"}
