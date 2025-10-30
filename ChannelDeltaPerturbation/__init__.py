import math
from typing import Any

import torch
from torch import nn
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch

from .guidance_utils import parse_unet_blocks

CDP_OPTION = "cdp"  # transformer_options flag
CDP_DROPOUT = "cdp_dropout"


class CDPTransformerWrapper(nn.Module):
    """
    Wraps BasicTransformerBlock and drops random channels in self-attention
    during perturbed pass.
    """
    def __init__(self, transformer_block: BasicTransformerBlock) -> None:
        super().__init__()
        self.wrapped_block = transformer_block

    @staticmethod
    def _channel_dropout(v: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        if v is None or v.ndim < 2 or dropout_rate <= 0:
            return v
        keep_prob = max(0.0, min(1.0, 1.0 - dropout_rate))
        ch_dim = v.shape[-1]
        # keep mask in v’s dtype to avoid fp16/fp32 mismatch
        mask = (torch.rand(ch_dim, device=v.device) < keep_prob).to(v.dtype)
        return v * mask

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        transformer_options: dict[str, Any] = {},
    ):
        # Apply channel dropout to input if CDP is enabled
        if transformer_options.get(CDP_OPTION, False):
            dropout_rate = transformer_options.get(CDP_DROPOUT, 0.4)
            x = self._channel_dropout(x, dropout_rate)
        return self.wrapped_block(x, context=context, transformer_options=transformer_options)


class ChannelDeltaPerturbation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "min_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "dropout_rate": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.9, "step": 0.05}),
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
        dropout_rate: float = 0.4,
        unet_block_list: str = "d2.2-9,d3"
    ):
        # Fix common input format issue
        if unet_block_list and "-d" in unet_block_list and "," not in unet_block_list:
            unet_block_list = unet_block_list.replace("-d", ",d")

        m = model.clone()
        inner_model: BaseModel = m.model

        # Parse which blocks to patch
        _, block_names = parse_unet_blocks(model, unet_block_list, None) if unet_block_list else (None, None)

        # Wrap selected transformer blocks
        wrapped_count = 0
        for name, module in inner_model.diffusion_model.named_modules():
            if (
                isinstance(module, BasicTransformerBlock)
                and "wrapped_block" not in name
                and (block_names is None or name in block_names)
            ):
                m.add_object_patch(f"diffusion_model.{name}", CDPTransformerWrapper(module))
                wrapped_count += 1

        if wrapped_count == 0:
            print(f"[CDP] Warning: No blocks were wrapped. Check unet_block_list: {unet_block_list}")

        # State for sigma scheduling
        state = {"sigma_start": None, "step": 0}

        def _effective_scale(sig_val: float, start: float) -> float:
            # Cosine decay from scale -> min_scale
            if start is None or start <= 1e-6:
                return scale
            t = max(0.0, min(1.0, sig_val / start))
            w = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))
            return float(min_scale + (scale - min_scale) * w)

        def post_cfg_function(args):
            model: BaseModel = args["model"]
            cond_pred = args["cond_denoised"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            cond = args["cond"]
            model_options = args["model_options"].copy()

            # Skip if both scales are zero
            if scale == 0 and min_scale == 0:
                return cfg_result

            # Extract sigma value safely
            try:
                sig_val = float(sigma.flatten()[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
            except Exception:
                sig_val = 1.0

            # Lock starting sigma on first step
            if state["sigma_start"] is None:
                state["sigma_start"] = sig_val
                print(f"[CDP] Initialized at sigma={sig_val:.4f}, wrapped {wrapped_count} blocks")

            state["step"] += 1

            # Enable CDP flag and set dropout rate for perturbed pass
            topts = model_options.get("transformer_options", {}).copy()
            topts[CDP_OPTION] = True
            topts[CDP_DROPOUT] = max(0.0, min(0.9, dropout_rate))
            model_options["transformer_options"] = topts

            # Perturbed forward pass
            with torch.no_grad():
                (perturbed_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)

            # Compute effective scale and apply delta
            eff_scale = _effective_scale(sig_val, state["sigma_start"])
            delta = (cond_pred - perturbed_pred) * eff_scale

            # Debug logging every 5 steps
            if state["step"] % 5 == 1:
                delta_mag = delta.abs().mean().item()
                print(f"[CDP] step={state['step']} sigma={sig_val:.4f} scale={eff_scale:.3f} |delta|={delta_mag:.6f}")

            return cfg_result + delta

        # Ensure hook runs even with CFG=1
        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {"ChannelDeltaPerturbation": ChannelDeltaPerturbation}
NODE_DISPLAY_NAME_MAPPINGS = {"ChannelDeltaPerturbation": "Channel Delta Perturbation"}