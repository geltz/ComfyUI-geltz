import math
from typing import Any

import torch
from torch import nn
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch

from .guidance_utils import parse_unet_blocks

TDP_OPTION = "tdp"  # transformer_options flag


class TDPTransformerWrapper(nn.Module):
    """
    Wraps a BasicTransformerBlock and (optionally) shuffles QUERY tokens (x),
    same basis as TPG. Enabled only during the perturbed pass.
    """
    def __init__(self, transformer_block: BasicTransformerBlock) -> None:
        super().__init__()
        self.wrapped_block = transformer_block

    @staticmethod
    def _shuffle_tokens(x: torch.Tensor) -> torch.Tensor:
        if x is None or x.ndim < 3:
            return x
        perm = torch.randperm(x.shape[1], device=x.device)
        return x[:, perm]

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        transformer_options: dict[str, Any] = {},
    ):
        if transformer_options.get(TDP_OPTION, False):
            x = self._shuffle_tokens(x)
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

    def patch(self, model: ModelPatcher, scale: float, min_scale: float = 1.0, unet_block_list: str = "d2.2-9,d3"):
        if unet_block_list and "-d" in unet_block_list and "," not in unet_block_list:
            unet_block_list = unet_block_list.replace("-d", ",d")

        m = model.clone()
        inner_model: BaseModel = m.model

        _, block_names = parse_unet_blocks(model, unet_block_list, None) if unet_block_list else (None, None)

        # Patch only selected transformer blocks
        for name, module in inner_model.diffusion_model.named_modules():
            if (
                isinstance(module, BasicTransformerBlock)
                and "wrapped_block" not in name
                and (block_names is None or name in block_names)
            ):
                m.add_object_patch(f"diffusion_model.{name}", TDPTransformerWrapper(module))

        # Track the starting sigma so we can decay from scale -> min_scale
        state = {"sigma_start": None}

        def _effective_scale(sig_val: float, start: float) -> float:
            # Cosine decay: w goes 1 -> 0 across the schedule; floor at min_scale
            if start is None or start <= 1e-6:
                return scale
            t = max(0.0, min(1.0, sig_val / start))
            w = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))  # 1 at start, 0 near end
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

            # Get current sigma and lock the starting sigma on first use
            try:
                sig_val = float(sigma.flatten()[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
            except Exception:
                sig_val = 1.0
            if state["sigma_start"] is None:
                state["sigma_start"] = sig_val

            # Enable TDP only for the perturbed pass
            topts = model_options.get("transformer_options", {}).copy()
            topts[TDP_OPTION] = True
            model_options["transformer_options"] = topts

            with torch.no_grad():
                (pag_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)

            eff = _effective_scale(sig_val, state["sigma_start"])
            return cfg_result + (cond_pred - pag_cond_pred) * eff

        # Ensure hook runs even with CFG=1 fast path
        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {"TokenDeltaPerturbation": TokenDeltaPerturbation}
NODE_DISPLAY_NAME_MAPPINGS = {"TokenDeltaPerturbation": "Token Delta Perturbation"}
