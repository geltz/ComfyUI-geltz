import math
import torch
import comfy.model_patcher

# (tok_len, device_str, dtype_str, seed, salt) -> 1D perm tensor
_PERM_CACHE = {}


def _extract_timestep_seed(extra_options):
    if extra_options is None:
        return None
    for key in ("timestep", "timesteps", "sigma", "sigmas"):
        if key in extra_options and extra_options[key] is not None:
            val = extra_options[key]
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    val = val.item()
                else:
                    val = val.flatten()[0].item()
            try:
                return int(float(val)) % (2**31)
            except Exception:
                return None
    return None


def _det_rand_from_seed(seed: int, salt: int) -> float:
    # simple deterministic mixer in [0,1)
    x = (seed ^ (salt * 0x9E3779B9)) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    return (x & 0xFFFFFFFF) / 0xFFFFFFFF


def _get_perm_indices(tok_len: int, device, dtype, seed: int, salt: int):
    key = (tok_len, str(device), str(dtype), seed, salt)
    perm = _PERM_CACHE.get(key, None)
    if perm is not None:
        return perm

    g = torch.Generator(device=device)
    g.manual_seed(seed ^ (salt * 0x9E3779B9))
    perm = torch.randperm(tok_len, generator=g, device=device)
    _PERM_CACHE[key] = perm
    return perm


def _apply_perm_indexed(x: torch.Tensor, perm: torch.Tensor, strength: float) -> torch.Tensor:
    # x: (..., T, D), perm: (T,)
    if strength <= 0.0:
        return x
    if x.dim() == 3:        # (bh, t, d)
        x_perm = x[:, perm, :]
    else:                   # (b, h, t, d)
        x_perm = x[..., perm, :]
    if strength >= 1.0:
        return x_perm
    return x + strength * (x_perm - x)


class TokenShuffler:
    """
    SDXL layout:
      - patch middle
      - patch input 2 (d2) except 2.0 and 2.1
      - patch input 3 (d3)
    with index-based, timestep-tied shuffle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shuffle_prob": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "shuffle_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(self, model, shuffle_prob=0.5, shuffle_strength=1.0):
        m = model.clone()

        def token_shuffle_attention(q, k, v, extra_options=None, mask=None, **kwargs):
            # extract timestep ONCE
            seed_val = _extract_timestep_seed(extra_options)
            if seed_val is None:
                return _vanilla_attention(q, k, v, mask)
        
            # Disable shuffle at very late steps (high timestep values)
            raw_ts = None
            if extra_options is not None:
                for key in ("timestep", "timesteps"):
                    if key in extra_options and extra_options[key] is not None:
                        fv = extra_options[key]
                        if isinstance(fv, torch.Tensor):
                            fv = fv.flatten()[0].item()
                        raw_ts = float(fv)
                        break
            
            if raw_ts is not None and raw_ts > 950:
                return _vanilla_attention(q, k, v, mask)
        
            # deterministic gate tied to timestep
            gate = _det_rand_from_seed(seed_val, 0)
            if gate >= shuffle_prob or shuffle_strength <= 0.0:
                return _vanilla_attention(q, k, v, mask)

            # final standard attention
            return _vanilla_attention(q, k, v, mask)

        def _vanilla_attention(q, k, v, mask):
            if q.dim() == 3:
                # (bh, t, d)
                bh, tq, d = q.shape
                tk = k.shape[1]
                scale = 1.0 / math.sqrt(d)
                scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (bh, tq, tk)
                if mask is not None:
                    scores = scores + mask
                attn = torch.softmax(scores, dim=-1)
                out = torch.bmm(attn, v)
                return out
            elif q.dim() == 4:
                # (b, h, t, d)
                b, h, tq, d = q.shape
                scale = 1.0 / math.sqrt(d)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (b, h, tq, tk)
                if mask is not None:
                    scores = scores + mask
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                return out
            else:
                return q

        mo = m.model_options

        # 1) middle block
        try:
            mo = comfy.model_patcher.set_model_options_patch_replace(
                mo,
                token_shuffle_attention,
                "attn2",
                "middle",
                0,
            )
        except Exception:
            pass
        # middle often has several transformer indices
        for tidx in (0, 1, 2, 3):
            try:
                mo = comfy.model_patcher.set_model_options_patch_replace(
                    mo,
                    token_shuffle_attention,
                    "attn2",
                    "middle",
                    0,
                    transformer_index=tidx,
                )
            except Exception:
                pass

        # 2) input block 2 (d2) — exclude 2.0 and 2.1
        # we try a few transformer indices; 0/1 excluded
        for tidx in (2, 3, 4, 5):
            try:
                mo = comfy.model_patcher.set_model_options_patch_replace(
                    mo,
                    token_shuffle_attention,
                    "attn2",
                    "input",
                    2,
                    transformer_index=tidx,
                )
            except Exception:
                pass

        # 3) input block 3 (d3) — patch all common transformer indices
        for tidx in (0, 1, 2, 3, 4):
            try:
                mo = comfy.model_patcher.set_model_options_patch_replace(
                    mo,
                    token_shuffle_attention,
                    "attn2",
                    "input",
                    3,
                    transformer_index=tidx,
                )
            except Exception:
                pass

        m.model_options = mo
        return (m,)


NODE_CLASS_MAPPINGS = {
    "TokenShuffler": TokenShuffler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TokenShuffler": "Token Shuffler",
}

