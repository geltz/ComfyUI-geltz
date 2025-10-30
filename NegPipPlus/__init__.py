# Improved NegPiP encoding
# Reflect negatives over neutral e (zv = 2*e - amp_k) for symmetric repulsion (avg k+v=e), fix z_empty indexing, limit mods to actual tokens.

# Original implementation by laksjdjf and hako-mikan licensed under AGPL-3.0
# Modified from ppm
# https://github.com/laksjdjf/cd-tuner_negpip-ComfyUI/blob/938b838546cf774dc8841000996552cef52cccf3/negpip.py#L43-L84
# https://github.com/hako-mikan/sd-webui-negpip
# https://github.com/pamparamm/ComfyUI-ppm

from functools import partial
from typing import Any

import torch

from comfy import model_management
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_base import Flux, HunyuanVideo
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.sd1_clip import SDClipModel, gen_empty_tokens

from .advanced_encode import patch_adv_encode
from .flux_negpip import flux_forward_orig_negpip
from .hunyuan_video_negpip import (
    hunyuan_video_clip_encode_token_weights_negpip,
    hunyuan_video_forward_orig_negpip,
)

NEGPIP_OPTION = "negpipplus"
SUPPORTED_ENCODERS = [
    "clip_g",
    "clip_l",
    "t5xxl",
    "llama",
]


def has_negpip(model_options: dict):
    return model_options.get(NEGPIP_OPTION, False)


def negpip_attn(q, k, v, extra_options):
    new_k = k[:, 0::2]
    new_v = v[:, 1::2]
    return q, new_k, new_v


def encode_token_weights_negpip(self: SDClipModel, token_weight_pairs):
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
        section_len = len(token_weight_pairs[k])
        z_empty = out[-1]
        for i in range(zk.shape[0]):
            for j in range(min(section_len, zk.shape[1])):
                weight = token_weight_pairs[k][j][1]
                if weight == 1.0:
                    continue
                e_j = z_empty[i][j]
                orig = zk[i][j]  # orig same for zk and zv
                alpha = abs(weight)
                amp_k = (orig - e_j) * alpha + e_j
                zk[i][j] = amp_k
                if weight > 0:
                    zv[i][j] = amp_k
                else:
                    zv[i][j] = 2 * e_j - amp_k

        z = torch.zeros_like(zk).repeat(1, 2, 1)
        for pos in range(zk.shape[1]):
            z[:, 2 * pos, :] += zk[:, pos, :]
            z[:, 2 * pos + 1, :] += zv[:, pos, :]
        output.append(z)

    if len(output) == 0:
        r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
    else:
        r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

    if len(o) > 2:
        extra = {}
        for key in o[2]:
            v = o[2][key]
            if key == "attention_mask":
                v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
            extra[key] = v

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

        patch_adv_encode()

        if not has_negpip(model_options):
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(encode_token_weights_negpip, getattr(c.patcher.model, encoder)),
                )

            m.set_model_attn2_patch(negpip_attn)
            model_options[NEGPIP_OPTION] = True
            clip_options[NEGPIP_OPTION] = True
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
