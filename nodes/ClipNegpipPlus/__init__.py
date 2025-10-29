# Modified NegPip algorithm with orthogonal decomposition and smooth weight handling
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from comfy import model_management
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_base import Flux, HunyuanVideo
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.sd1_clip import SDClipModel, gen_empty_tokens

NEGPIP_PLUS_OPTION = "ppm_negpip_plus"
SUPPORTED_ENCODERS = [
    "clip_g",
    "clip_l",
    "t5xxl",
    "llama",
]


def has_negpip_plus(model_options: dict):
    return model_options.get(NEGPIP_PLUS_OPTION, False)


def negpip_plus_attn(q, k, v, extra_options):
    """Improved attention extraction with optional soft mixing."""
    k_extracted = k[:, 0::2]
    v_extracted = v[:, 1::2]
    
    # Optional soft mixing for gradient stability (default: pure negpip)
    mix_ratio = extra_options.get("negpip_mix_ratio", 0.0)
    if mix_ratio > 0:
        k_extracted = (1 - mix_ratio) * k_extracted + mix_ratio * k[:, 0::2]
        v_extracted = (1 - mix_ratio) * v_extracted + mix_ratio * v[:, 0::2]
    
    return q, k_extracted, v_extracted


def encode_token_weights_negpip_plus(self: SDClipModel, token_weight_pairs):
    """
    NegPip+ encoding with:
    - Smooth tanh-based weight interpolation
    - Orthogonal rotation-based k/v decomposition
    - Centered weight application
    """
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
            to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
        else:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

    o = self.encode(to_encode)
    out, pooled = o[:2]

    if pooled is not None:
        first_pooled = pooled[0:1].to(model_management.intermediate_device())
    else:
        first_pooled = pooled

    output = []
    
    for k_idx in range(0, sections):
        z_base = out[k_idx : k_idx + 1].clone()
        
        if has_weights:
            z_empty = out[-1]
            
            # Process key embeddings
            zk = z_base.clone()
            zv = z_base.clone()
            
            for i in range(len(z_base)):
                for j in range(len(z_base[i])):
                    weight = token_weight_pairs[k_idx][j][1]
                    if weight != 1.0:
                        # Extract sign for value vector
                        abs_weight = abs(weight)
                        w_eff = abs(weight) if weight > 0 else 1.0 / (1.0 + abs(weight))
                        # Use baseline vector as target
                        zk[i][j] = (zk[i][j] - z_empty[j]) * w_eff + z_empty[j]
                        zv[i][j] = (zv[i][j] - z_empty[j]) * w_eff + z_empty[j]
            
            z_weighted = zk  # Use zk as base for rotation
        else:
            z_weighted = z_base
        
        # Orthogonal decomposition using rotation
        dim = z_weighted.shape[-1]
        device = z_weighted.device
        
        # Position-dependent rotation angles (smaller for stability)
        theta = torch.linspace(0, torch.pi / 8, z_weighted.shape[1], device=device)
        cos_theta = torch.cos(theta).unsqueeze(-1).unsqueeze(0)
        sin_theta = torch.sin(theta).unsqueeze(-1).unsqueeze(0)
        
        # Create orthogonal k and v projections
        mid_dim = dim // 2
        z_k_rot = z_weighted.clone()
        z_v_rot = zv if has_weights else z_weighted.clone()
        
        # Rotation-based orthogonal decomposition (applied to both k and v separately)
        z_k_rot[..., :mid_dim] = z_weighted[..., :mid_dim] * cos_theta - z_weighted[..., mid_dim:] * sin_theta
        z_k_rot[..., mid_dim:] = z_weighted[..., :mid_dim] * sin_theta + z_weighted[..., mid_dim:] * cos_theta
        
        z_v_rot[..., :mid_dim] = z_v_rot[..., :mid_dim] * cos_theta + z_v_rot[..., mid_dim:] * sin_theta
        z_v_rot[..., mid_dim:] = -z_v_rot[..., :mid_dim] * sin_theta + z_v_rot[..., mid_dim:] * cos_theta
        
        z_k = z_k_rot
        z_v = z_v_rot
        
        # Interleave k and v
        z_interleaved = torch.zeros(z_k.shape[0], z_k.shape[1] * 2, z_k.shape[2], 
                                     device=z_k.device, dtype=z_k.dtype)
        z_interleaved[:, 0::2, :] = z_k
        z_interleaved[:, 1::2, :] = z_v
        
        output.append(z_interleaved)

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
            }
        }

    RETURN_TYPES = (IO.MODEL, IO.CLIP)
    FUNCTION = "patch"
    CATEGORY = "conditioning"

    def patch(self, model: ModelPatcher, clip: CLIP):
        m = model.clone()
        c = clip.clone()
        model_options: dict[str, Any] = m.model_options
        clip_options: dict[str, Any] = c.patcher.model_options

        encoders = [e for e in SUPPORTED_ENCODERS if hasattr(c.patcher.model, e)]
        if len(encoders) == 0:
            return (m, c)

        if not has_negpip_plus(model_options):
            # Patch CLIP encoders
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(encode_token_weights_negpip_plus, getattr(c.patcher.model, encoder)),
                )

            # Patch attention mechanism
            m.set_model_attn2_patch(negpip_plus_attn)
            model_options[NEGPIP_PLUS_OPTION] = True
            clip_options[NEGPIP_PLUS_OPTION] = True

        return (m, c)


NODE_CLASS_MAPPINGS = {
    "CLIPNegPipPlus": CLIPNegPipPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPNegPipPlus": "CLIP NegPip+",

}
