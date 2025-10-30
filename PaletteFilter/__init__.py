# Build a 3D LUT from a reference image using Sliced OT, then apply it as a color filter.

import torch
import numpy as np

def _to_np(t): return t.detach().cpu().numpy()
def _to_torch(a, device): return torch.from_numpy(a).to(device)

def _srgb_to_linear(x): return np.clip(x, 0, 1) ** 2.2
def _linear_to_srgb(x): return np.clip(x, 0, 1) ** (1.0 / 2.2)

def _interp_quantile_map(all_proj, s_sorted, r_sorted):
    n = min(len(s_sorted), len(r_sorted))
    if n < 2: return all_proj
    if len(s_sorted) != n:
        s_sorted = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(s_sorted)), s_sorted)
    if len(r_sorted) != n:
        r_sorted = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(r_sorted)), r_sorted)
    us = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5/n
    u = np.interp(all_proj, s_sorted, us, left=0.0, right=1.0)
    return np.interp(u, us, r_sorted)

def _sliced_ot_map(src_pts, ref_pts, projections=32, iters=1, max_samples=65536, seed=0, linearize=True):
    # src_pts/ref_pts: (N,3) [0,1]
    rng = np.random.default_rng(seed)
    src = src_pts.copy()
    ref = ref_pts.copy()
    if linearize:
        src = _srgb_to_linear(src); ref = _srgb_to_linear(ref)
    for _ in range(iters):
        for _k in range(projections):
            v = rng.normal(size=3).astype(np.float32); v /= (np.linalg.norm(v) + 1e-8)
            ps_all = (src @ v)
            s_proj = ps_all if src.shape[0] <= max_samples else (src[rng.choice(src.shape[0], max_samples, replace=False)] @ v)
            r_proj = (ref @ v) if ref.shape[0] <= max_samples else (ref[rng.choice(ref.shape[0], max_samples, replace=False)] @ v)
            s_sorted = np.sort(s_proj); r_sorted = np.sort(r_proj)
            mapped = _interp_quantile_map(ps_all, s_sorted, r_sorted).astype(np.float32)
            delta = (mapped - ps_all).astype(np.float32)
            src += (delta[:, None] * v[None, :])
            src = np.clip(src, 0.0, 1.0)
    if linearize:
        src = _linear_to_srgb(src)
    return np.clip(src, 0.0, 1.0).astype(np.float32)

def _build_identity_grid(n):
    g = np.linspace(0.0, 1.0, n, dtype=np.float32)
    R, G, B = np.meshgrid(g, g, g, indexing='ij')
    pts = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    return pts  # (n^3,3)

def _build_lut_from_ref(ref_rgb_flat, n, projections, iters, max_samples, seed, linearize):
    # Map identity grid through SOT toward reference distribution => LUT[n,n,n,3]
    grid = _build_identity_grid(n)                          # input sample positions
    mapped = _sliced_ot_map(grid, ref_rgb_flat, projections, iters, max_samples, seed, linearize)
    return mapped.reshape(n, n, n, 3).astype(np.float32)    # LUT

def _apply_lut_torch(img, lut):
    # img: [B,H,W,3] in [0,1], lut: [n,n,n,3]
    device = img.device
    n = lut.shape[0]
    lut_t = torch.from_numpy(lut).to(device)  # [n,n,n,3]

    x = img.clamp(0,1)
    s = (n - 1)
    xr = x[..., 0] * s
    xg = x[..., 1] * s
    xb = x[..., 2] * s

    i0r = xr.floor().long().clamp_(0, n-1); fr = (xr - i0r.float())
    i0g = xg.floor().long().clamp_(0, n-1); fg = (xg - i0g.float())
    i0b = xb.floor().long().clamp_(0, n-1); fb = (xb - i0b.float())
    i1r = (i0r + 1).clamp_(0, n-1)
    i1g = (i0g + 1).clamp_(0, n-1)
    i1b = (i0b + 1).clamp_(0, n-1)

    def S(ir, ig, ib): return lut_t[ir, ig, ib]  # [B,H,W,3]

    c000 = S(i0r, i0g, i0b); c100 = S(i1r, i0g, i0b); c010 = S(i0r, i1g, i0b); c110 = S(i1r, i1g, i0b)
    c001 = S(i0r, i0g, i1b); c101 = S(i1r, i0g, i1b); c011 = S(i0r, i1g, i1b); c111 = S(i1r, i1g, i1b)

    w000 = (1-fr)*(1-fg)*(1-fb); w100 = fr*(1-fg)*(1-fb); w010 = (1-fr)*fg*(1-fb); w110 = fr*fg*(1-fb)
    w001 = (1-fr)*(1-fg)*fb;     w101 = fr*(1-fg)*fb;     w011 = (1-fr)*fg*fb;     w111 = fr*fg*fb

    out = (
        c000 * w000.unsqueeze(-1) + c100 * w100.unsqueeze(-1) +
        c010 * w010.unsqueeze(-1) + c110 * w110.unsqueeze(-1) +
        c001 * w001.unsqueeze(-1) + c101 * w101.unsqueeze(-1) +
        c011 * w011.unsqueeze(-1) + c111 * w111.unsqueeze(-1)
    )
    return out.clamp(0,1)

class PaletteFilterLUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ref_image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lut_size": ("INT", {"default": 33, "min": 9, "max": 65, "step": 2}),
                "projections": ("INT", {"default": 32, "min": 1, "max": 2048, "step": 1}),
                "iters": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "max_samples": ("INT", {"default": 65536, "min": 2048, "max": 1048576, "step": 1024}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
                "linearize": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "color"

    def apply(self, image, ref_image, amount,
              lut_size=33, projections=32, iters=1, max_samples=65536, seed=0, linearize=True):
        device = image.device
        x = image.float().clamp(0,1)
        y = ref_image.float().clamp(0,1)

        Bx, H, W, C = x.shape
        By = y.shape[0]
        assert C == 3, "Expect RGB"
        assert By in (1, Bx), "ref_image batch must be 1 or match"

        out = torch.empty_like(x)
        for b in range(Bx):
            ref_b = 0 if By == 1 else b
            ref_flat = _to_np(y[ref_b].reshape(-1,3))
            lut = _build_lut_from_ref(
                ref_flat, lut_size,
                projections, iters, max_samples, seed, linearize
            )
            xb = x[b:b+1]  # keep dims for broadcasting
            yb = _apply_lut_torch(xb, lut)
            if amount < 1.0:
                yb = (1.0 - amount) * xb + amount * yb
            out[b:b+1] = yb

        return (out.to(device),)


NODE_CLASS_MAPPINGS = {
    "PaletteFilterLUT": PaletteFilterLUT,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaletteFilterLUT": "Palette Filter",
}
