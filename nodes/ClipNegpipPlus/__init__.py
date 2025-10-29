# Enhanced NegPip with magnitude compensation, spectral regularization, and position encoding
from functools import partial
from typing import Any

import torch

from comfy import model_management
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_base import Flux, HunyuanVideo
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.sd1_clip import SDClipModel, gen_empty_tokens

from ..compat.advanced_encode import patch_adv_encode
from ..dit.flux_negpip import flux_forward_orig_negpip
from ..dit.hunyuan_video_negpip import (
    hunyuan_video_clip_encode_token_weights_negpip,
    hunyuan_video_forward_orig_negpip,
)

NEGPIP_PLUS_OPTION = "negpip_plus"
SUPPORTED_ENCODERS = [
    "clip_g",
    "clip_l",
    "t5xxl",
    "llama",
]


def has_negpip_plus(model_options: dict):
    return model_options.get(NEGPIP_PLUS_OPTION, False)


def negpip_attn(q, k, v, extra_options):
    new_k = k[:, 0::2]
    new_v = v[:, 1::2]
    return q, new_k, new_v


def apply_magnitude_compensation(zk, zv, z_empty, weight, alpha=1.7):
    """Compensate for attention softmax bias with negative weights."""
    if weight < 0:
        weight_abs = abs(weight)
        sign = -1
        weight_k = weight_abs
        weight_v = weight_abs * alpha
    else:
        sign = 1
        weight_k = weight
        weight_v = weight
    
    zk_comp = (zk - z_empty) * weight_k + z_empty
    zv_comp = sign * ((zv - z_empty) * weight_v + z_empty)
    
    return zk_comp, zv_comp


def apply_position_encoding(z, delta=0.02):
    """Add alternating positional bias to separate k/v positions."""
    seq_len = z.shape[1]
    pos_bias = torch.zeros(seq_len, device=z.device, dtype=z.dtype)
    pos_bias[::2] = delta
    pos_bias[1::2] = -delta
    
    return z + pos_bias.unsqueeze(0).unsqueeze(-1)


def apply_spectral_regularization(z, sigma=8.0):
    """Smooth high-frequency artifacts from sequence doubling."""
    z_fft = torch.fft.rfft(z, dim=1)
    freq = torch.arange(z_fft.shape[1], device=z.device, dtype=torch.float32)
    freq_filter = torch.exp(-freq**2 / (2 * sigma**2))
    z_fft = z_fft * freq_filter.unsqueeze(0).unsqueeze(-1)
    
    return torch.fft.irfft(z_fft, n=z.shape[1], dim=1)


def encode_token_weights_negpip_plus(
    self: SDClipModel, 
    token_weight_pairs,
    alpha=1.7,
    delta=0.02,
    sigma=8.0
):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        if hasattr(self, "gen_empty_tokens"):
            to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))  # type: ignore
        else:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

    o = self.encode(to_encode)
    out, pooled = o[:2]

    if pooled is not None:
        first_pooled = pooled[0:1].to(model_management.intermediate_device())
    else:
        first_pooled = pooled

    output = []
    for k in range(0, sections):
        zk = out[k : k + 1].clone()
        zv = out[k : k + 1].clone()
        if has_weights:
            z_empty = out[-1]
            for i in range(len(zk)):
                for j in range(len(zk[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        zk[i][j], zv[i][j] = apply_magnitude_compensation(
                            zk[i][j], zv[i][j], z_empty[j], weight, alpha
                        )

        # Interleave k and v
        z = torch.zeros_like(zk).repeat(1, 2, 1)
        for i in range(zk.shape[1]):
            z[:, 2 * i, :] += zk[:, i, :]
            z[:, 2 * i + 1, :] += zv[:, i, :]
        
        # Apply enhancements
        z = apply_position_encoding(z, delta)
        z = apply_spectral_regularization(z, sigma)
        
        output.append(z)

    if len(output) == 0:
        r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
    else:
        r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

    if len(o) > 2:
        extra = {}
        for k in o[2]:
            v = o[2][k]
            if k == "attention_mask":
                v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
            extra[k] = v

        r = r + (extra,)
    return r


class CLIPNegPipPlus(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "clip": (IO.CLIP, {}),
                "alpha": ("FLOAT", {"default": 1.7, "min": 1.0, "max": 3.0, "step": 0.1}),
                "delta": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005}),
                "sigma": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = (IO.MODEL, IO.CLIP)
    FUNCTION = "patch"

    CATEGORY = "conditioning"

    def patch(self, model: ModelPatcher, clip: CLIP, alpha: float, delta: float, sigma: float):
        m = model.clone()
        c = clip.clone()
        model_options: dict[str, Any] = m.model_options
        clip_options: dict[str, Any] = c.patcher.model_options

        encoders = [e for e in SUPPORTED_ENCODERS if hasattr(c.patcher.model, e)]
        if len(encoders) == 0:
            return (m, c)

        patch_adv_encode()

        if not has_negpip_plus(model_options):
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(
                        encode_token_weights_negpip_plus,
                        getattr(c.patcher.model, encoder),
                        alpha=alpha,
                        delta=delta,
                        sigma=sigma
                    ),
                )

            m.set_model_attn2_patch(negpip_attn)
            model_options[NEGPIP_PLUS_OPTION] = True
            clip_options[NEGPIP_PLUS_OPTION] = True
            self.patch_dit(m, c)

        return (m, c)

    @staticmethod
    def patch_dit(m: ModelPatcher, c: CLIP):
        diffusion_model = type(m.model)

        if issubclass(diffusion_model, Flux):
            m.add_object_patch(
                "diffusion_model.forward_orig", partial(flux_forward_orig_negpip, m.model.diffusion_model)
            )
        if issubclass(diffusion_model, HunyuanVideo):
            c.patcher.add_object_patch(
                "encode_token_weights",
                partial(hunyuan_video_clip_encode_token_weights_negpip, c.patcher.model),
            )
            m.add_object_patch(
                "diffusion_model.forward_orig",
                partial(hunyuan_video_forward_orig_negpip, m.model.diffusion_model),
            )


NODE_CLASS_MAPPINGS = {
    "CLIPNegPipPlus": CLIPNegPipPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPNegPipPlus": "CLIP NegPip+",
}

