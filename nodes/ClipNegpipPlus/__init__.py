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


def encode_token_weights_negpip_plus(self: SDClipModel, token_weight_pairs):
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
    
    for k in range(0, sections):
        zk = out[k : k + 1].clone()
        zv = out[k : k + 1].clone()
        
        if has_weights:
            z_empty = out[-1]
            for i in range(len(zk)):
                for j in range(len(zk[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        if weight < 0:
                            weight = 1.0 / (1.0 + abs(weight))
                            zv[i][j] = (zv[i][j] - z_empty[j]) * weight + z_empty[j]
                        else:
                            zk[i][j] = (zk[i][j] - z_empty[j]) * weight + z_empty[j]
                            zv[i][j] = (zv[i][j] - z_empty[j]) * weight + z_empty[j]
        
        # Weak decorrelation: add perpendicular component without destroying structure
        alpha = 0.1
        zk_mean = zk.mean(dim=1, keepdim=True)
        zv_mean = zv.mean(dim=1, keepdim=True)
        zk_centered = zk - zk_mean
        zv_centered = zv - zv_mean
        
        correlation = torch.bmm(zv_centered.transpose(1, 2), zk_centered) / (zk.shape[1] + 1e-8)
        zv_correction = torch.bmm(zk_centered, correlation) * alpha
        zv = zv - zv_correction

        z = torch.zeros_like(zk).repeat(1, 2, 1)
        for i in range(zk.shape[1]):
            z[:, 2 * i, :] += zk[:, i, :]
            z[:, 2 * i + 1, :] += zv[:, i, :]
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


def negpip_plus_attn(q, k, v, extra_options):
    new_k = k[:, 0::2]
    new_v = v[:, 1::2]
    return q, new_k, new_v


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






