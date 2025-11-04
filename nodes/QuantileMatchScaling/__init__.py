import math
from typing import Any, Sequence

import torch
import torch.fft as fft


def _adaptive_freq_cutoffs(h: int, w: int, _unused=None) -> tuple[float, float]:
    """
    Return normalized low/high cutoffs in [0, 0.5].
    Keeps it simple and resolution-agnostic.
    """
    # tuned for SD-ish resolutions; adjust if you want finer control
    low_cut = 0.12
    high_cut = 0.35
    return low_cut, high_cut


def _freq_masks(h: int, w: int, low_cut: float, high_cut: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build 3 broadcastable masks in FFT frequency space (unshifted).
    Uses torch.fft.fftfreq so zero-frequency is at index 0.
    Shapes: (1, 1, h, w)
    """
    fy = torch.fft.fftfreq(h, d=1.0).to(device).view(h, 1)   # (h, 1)
    fx = torch.fft.fftfreq(w, d=1.0).to(device).view(1, w)   # (1, w)
    # radius in normalized frequency space
    fr = torch.sqrt(fx ** 2 + fy ** 2)  # (h, w)

    low_mask = (fr <= low_cut).unsqueeze(0).unsqueeze(0)     # (1, 1, h, w)
    mid_mask = ((fr > low_cut) & (fr <= high_cut)).unsqueeze(0).unsqueeze(0)
    high_mask = (fr > high_cut).unsqueeze(0).unsqueeze(0)
    return low_mask, mid_mask, high_mask


def _adaptive_quantiles(x: torch.Tensor, num_quantiles: int = 7) -> torch.Tensor:
    """
    Just produce fixed quantile positions in (0,1).
    """
    return torch.linspace(0.1, 0.9, steps=num_quantiles, device=x.device)


def _band_quantiles_safe_from_fft(
    xf: torch.Tensor,
    masks: Sequence[torch.Tensor],
    clamp_q: float,
    qs: torch.Tensor,
) -> list[torch.Tensor]:
    """
    For each band mask, collect magnitudes and compute quantiles.
    Returns list of tensors, each shape (len(qs),)
    """
    mag = xf.abs()  # (B, C, H, W)
    out = []
    for m in masks:
        # broadcast mask
        vals = mag[m.expand_as(mag)]
        if vals.numel() < 8:
            out.append(torch.full((qs.numel(),), 0.0, device=mag.device))
            continue
        qvals = torch.quantile(vals, qs)
        qvals = torch.clamp(qvals, 0.0, clamp_q)
        out.append(qvals)
    return out


def _fit_robust_map_from_qs(
    src_q: torch.Tensor,
    tgt_q: torch.Tensor,
    qs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit y = a * x + b in least squares on quantiles.
    src_q, tgt_q: (Q,)
    """
    mask = torch.isfinite(src_q) & torch.isfinite(tgt_q)
    if mask.sum() < 2:
        a = torch.tensor(1.0, device=src_q.device)
        b = torch.tensor(0.0, device=src_q.device)
        return a, b

    x = src_q[mask]
    y = tgt_q[mask]
    xm = x.mean()
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum().clamp(min=1e-6)
    a = num / den
    b = ym - a * xm
    return a, b


def _dynamic_rescale(cfg: float, base_rescale: float) -> float:
    """
    CFG-aware multiplier to keep high CFG under control.
    """
    if cfg <= 1.0:
        return base_rescale * 0.25
    if cfg <= 3.0:
        return base_rescale * 0.5
    if cfg <= 7.0:
        return base_rescale
    return base_rescale * 1.1


def _apply_qms(
    g: torch.Tensor,
    a_low: torch.Tensor,
    b_low: torch.Tensor,
    a_mid: torch.Tensor,
    b_mid: torch.Tensor,
    a_high: torch.Tensor,
    b_high: torch.Tensor,
    masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Apply per-band linear map to FFT magnitude, preserve phase.
    """
    xf = fft.fft2(g.float(), norm="ortho")
    mag = xf.abs()
    phase = torch.angle(xf)

    low_mask, mid_mask, high_mask = masks

    mag_new = mag.clone()

    # we want scalar broadcast
    mag_new[low_mask.expand_as(mag_new)] = (
        mag[low_mask.expand_as(mag)] * a_low + b_low
    ).clamp_min(0.0)
    mag_new[mid_mask.expand_as(mag_new)] = (
        mag[mid_mask.expand_as(mag)] * a_mid + b_mid
    ).clamp_min(0.0)
    mag_new[high_mask.expand_as(mag_new)] = (
        mag[high_mask.expand_as(mag)] * a_high + b_high
    ).clamp_min(0.0)

    # rebuild complex
    xf_new = torch.polar(mag_new, phase)
    g_new = fft.ifft2(xf_new, norm="ortho").real
    return g_new.to(g.dtype)


def _winsor(x: torch.Tensor, p: float = 99.9) -> torch.Tensor:
    """
    Global winsorization over spatial dims per (B,C).
    """
    b, c, h, w = x.shape
    x_flat = x.view(b, c, -1)
    hi_q = torch.quantile(x_flat, p / 100.0, dim=2, keepdim=True)
    lo_q = torch.quantile(x_flat, (1.0 - p / 100.0), dim=2, keepdim=True)
    x_clamped = torch.clamp(x_flat, min=lo_q, max=hi_q)
    return x_clamped.view_as(x)


def _safe_rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x ** 2).mean(dim=(1, 2, 3), keepdim=True) + 1e-12)


class QuantileMatchScaling:
    """
    Quantile Match Scaling as a traditional ComfyUI node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_qms"
    CATEGORY = "_for_testing"

    def apply_qms(self, model, rescale: float):
        state: dict[str, Any] = {
            "ema_a_low": None,
            "ema_b_low": None,
            "ema_a_mid": None,
            "ema_b_mid": None,
            "ema_a_high": None,
            "ema_b_high": None,
            "r_low": None,
            "r_mid": None,
            "r_high": None,
            "iter": 0,
            "total": None,
            "mask_key": None,
            "masks": None,
        }

        def custom_pre_cfg(args):
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                return conds_out

            cond, uncond = conds_out[0], conds_out[1]
            if cond is None or uncond is None:
                return conds_out

            w_cfg = float(args.get("cfg", 1.0))

            # track total sampling steps once
            total_steps = args.get("total", None)
            if state["total"] is None and total_steps is not None:
                state["total"] = max(1, int(total_steps))
            if state["total"] is None:
                state["total"] = 999999  # fallback
            p = state["iter"] / max(1, state["total"])

            # sigma-aware branch
            sigmas = args.get("sigmas", None)
            sigma_rel = 1.0
            if isinstance(sigmas, (list, tuple)) and state["iter"] < len(sigmas):
                cur_sigma = float(sigmas[state["iter"]])
                start_sigma = float(sigmas[0]) if len(sigmas) > 0 else None
                if start_sigma is not None and start_sigma > 0:
                    sigma_rel = float(cur_sigma / (start_sigma + 1e-12))
                    sigma_rel = max(0.0, min(1.0, sigma_rel))

            g = cond - uncond

            # masks
            h, w_ = cond.shape[-2:]
            low_cut, high_cut = _adaptive_freq_cutoffs(h, w_, None)
            key = (h, w_, low_cut, high_cut, cond.device)
            if state["mask_key"] != key:
                state["masks"] = _freq_masks(h, w_, low_cut, high_cut, cond.device)
                state["mask_key"] = key
            masks = state["masks"]

            # temp guided to measure overdrive
            xcfg_temp = uncond + w_cfg * g

            qs = _adaptive_quantiles(cond, num_quantiles=7)

            xf_cond = fft.fft2(cond.float(), norm="ortho")
            xf_xcfg = fft.fft2(xcfg_temp.float(), norm="ortho")
            cond_q = _band_quantiles_safe_from_fft(xf_cond, masks, 99.9, qs)
            xcfg_q = _band_quantiles_safe_from_fft(xf_xcfg, masks, 99.9, qs)

            a_low0, b_low0 = _fit_robust_map_from_qs(xcfg_q[0], cond_q[0], qs)
            a_mid0, b_mid0 = _fit_robust_map_from_qs(xcfg_q[1], cond_q[1], qs)
            a_high0, b_high0 = _fit_robust_map_from_qs(xcfg_q[2], cond_q[2], qs)

            one = torch.tensor(1.0, device=cond.device, dtype=cond.dtype)
            zero = torch.tensor(0.0, device=cond.device, dtype=cond.dtype)

            rescale_eff = _dynamic_rescale(w_cfg, base_rescale=rescale)

            fw0 = torch.tensor([1.15, 0.95, 1.00], device=cond.device)
            fw1 = torch.tensor([1.05, 1.00, 1.00], device=cond.device)
            freq_weights = fw0.lerp(fw1, p)

            a_low = one + freq_weights[0] * rescale_eff * (a_low0.to(cond.device, cond.dtype) - one)
            b_low = zero + freq_weights[0] * rescale_eff * b_low0.to(cond.device, cond.dtype)
            a_mid = one + freq_weights[1] * rescale_eff * (a_mid0.to(cond.device, cond.dtype) - one)
            b_mid = zero + freq_weights[1] * rescale_eff * b_mid0.to(cond.device, cond.dtype)
            a_high = one + freq_weights[2] * rescale_eff * (a_high0.to(cond.device, cond.dtype) - one)
            b_high = zero + freq_weights[2] * rescale_eff * b_high0.to(cond.device, cond.dtype)

            rho = 0.85
            base_r = 1.20
            if state["ema_a_low"] is None:
                state["ema_a_low"] = a_low.detach()
                state["ema_b_low"] = b_low.detach()
                state["ema_a_mid"] = a_mid.detach()
                state["ema_b_mid"] = b_mid.detach()
                state["ema_a_high"] = a_high.detach()
                state["ema_b_high"] = b_high.detach()
                state["r_low"] = base_r * 0.85
                state["r_mid"] = base_r
                state["r_high"] = base_r * 1.10
            else:
                r_low = state["r_low"]
                r_mid = state["r_mid"]
                r_high = state["r_high"]

                state["ema_a_low"] = rho * state["ema_a_low"] + (1.0 - rho) * a_low
                state["ema_b_low"] = rho * state["ema_b_low"] + (1.0 - rho) * b_low
                state["ema_a_mid"] = rho * state["ema_a_mid"] + (1.0 - rho) * a_mid
                state["ema_b_mid"] = rho * state["ema_b_mid"] + (1.0 - rho) * b_mid
                state["ema_a_high"] = rho * state["ema_a_high"] + (1.0 - rho) * a_high
                state["ema_b_high"] = rho * state["ema_b_high"] + (1.0 - rho) * b_high

                a_low = torch.clamp(a_low, state["ema_a_low"] / r_low, state["ema_a_low"] * r_low)
                b_low = torch.clamp(b_low, state["ema_b_low"] / r_low, state["ema_b_low"] * r_low)
                a_mid = torch.clamp(a_mid, state["ema_a_mid"] / r_mid, state["ema_a_mid"] * r_mid)
                b_mid = torch.clamp(b_mid, state["ema_b_mid"] / r_mid, state["ema_b_mid"] * r_mid)
                a_high = torch.clamp(a_high, state["ema_a_high"] / r_high, state["ema_a_high"] * r_high)
                b_high = torch.clamp(b_high, state["ema_b_high"] / r_high, state["ema_b_high"] * r_high)

            a_low = a_low.clamp(0.9, 1.4)
            a_mid = a_mid.clamp(0.75, 1.35)
            a_high = a_high.clamp(0.70, 1.35)

            if p < 0.6:
                a_low = a_low.clamp_min(1.0)
            else:
                a_low = a_low.clamp_min(0.9)

            bias_scale_step = 1.0 - 0.5 * p
            bias_scale_sigma = sigma_rel
            bias_scale = bias_scale_step * max(0.25, bias_scale_sigma)

            b_low = (b_low * bias_scale).clamp(-0.12, 0.12)
            b_mid = (b_mid * bias_scale).clamp(-0.10, 0.10)
            b_high = (b_high * bias_scale).clamp(-0.06, 0.06)

            g_scaled = _apply_qms(g, a_low, b_low, a_mid, b_mid, a_high, b_high, masks)
            g_scaled = _winsor(g_scaled, p=99.9)

            # remove tiny DC drift
            g_scaled = g_scaled - g_scaled.mean(dim=(2, 3), keepdim=True) * 0.05

            base_rms = _safe_rms(g)
            scaled_rms = _safe_rms(g_scaled)
            late = p
            cfg_term = max(0.0, (w_cfg - 4.0) / 8.0)
            cap = 0.9 - 0.2 * late * cfg_term
            cap = max(0.7, cap)
            scale_val = torch.clamp(cap * base_rms / (scaled_rms + 1e-12), max=1.25)

            cond_new = torch.nan_to_num(
                uncond + g_scaled * scale_val,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(cond.dtype)

            state["iter"] += 1
            return [cond_new, uncond] + conds_out[2:]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(custom_pre_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "QuantileMatchScaling": QuantileMatchScaling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantileMatchScaling": "Quantile Match Scaling",
}
