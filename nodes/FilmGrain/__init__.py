import torch
import torch.nn.functional as F

class FilmGrainFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "image/filters"

    def apply_grain(
        self,
        image: torch.Tensor,
        intensity: float = 0.15,
        grain_size: int = 3,
        shadow: float = 0.6,
        midtone: float = 0.4,
        seed: int | None = None,
        frame: int = 0,
    ) -> torch.Tensor:
        """
        Inputs
          image: float tensor in [0,1], shape BxCxHxW or BxHxWxC, C=3
        Returns
          tensor with same shape and dtype as input
        """
        if image.dim() != 4:
            raise ValueError("image must be 4D")

        # Permute to NCHW if needed
        nhwc = False
        if image.shape[1] not in (1, 3):
            nhwc = True
            image = image.permute(0, 3, 1, 2)

        if image.shape[1] != 3:
            raise ValueError("expect 3 channels (RGB)")

        # Ensure float32 for math, remember original dtype
        orig_dtype = image.dtype
        x = image.to(torch.float32)

        # sRGB <-> linear helpers
        def srgb_to_linear(t):
            a = 0.055
            return torch.where(t <= 0.04045, t / 12.92, ((t + a) / (1 + a)).pow(2.4))

        def linear_to_srgb(t):
            a = 0.055
            return torch.where(t <= 0.0031308, t * 12.92, (1 + a) * t.clamp(min=0) ** (1 / 2.4) - a)

        # Work in linear light
        lin = srgb_to_linear(x.clamp(0, 1))

        # Luminance in linear light
        w_r, w_g, w_b = 0.2126, 0.7152, 0.0722
        y = lin[:, 0:1] * w_r + lin[:, 1:2] * w_g + lin[:, 2:3] * w_b  # N x1 xH xW

        # Tone weighting: more grain in shadows, some in mids
        w_mid = (1.0 - (y - 0.5).abs() * 2.0).clamp(0, 1)
        tone_w = (shadow * (1.0 - y) + midtone * w_mid).clamp(0, 1)    # N x1 xH xW

        # Deterministic RNG
        gen = None
        if seed is not None:
            gen = torch.Generator(device=lin.device)
            gen.manual_seed(int(seed + 1315423911 * frame))

        N, _, H, W = lin.shape

        # Monochrome grain in luma domain, then broadcast to RGB
        g = torch.randn((N, 1, H, W), device=lin.device, dtype=lin.dtype, generator=gen)

        # Band limit with box blur of size grain_size
        k = max(1, int(grain_size))
        if k % 2 == 0:
            k += 1
        if k > 1:
            kernel = torch.ones((1, 1, k, k), device=lin.device, dtype=lin.dtype) / (k * k)
            g = F.conv2d(g, kernel, padding=k // 2)

        # Normalize grain to unit std per image to keep intensity stable
        g_std = g.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-8).view(N, 1, 1, 1)
        g = g / g_std

        # Poisson-like response: scale a bit with sqrt(luma) to mimic photon noise
        poisson_w = (y + 1e-4).sqrt()
        strength = intensity * (0.5 * tone_w + 0.5 * poisson_w)  # N x1 xH xW

        # Apply same grain to all channels to avoid color speckle
        g_rgb = g.repeat(1, 3, 1, 1)
        lin_noisy = (lin + g_rgb * strength).clamp(0, 1)

        out = linear_to_srgb(lin_noisy).clamp(0, 1).to(orig_dtype)

        # Restore original layout
        if nhwc:
            out = out.permute(0, 2, 3, 1)
        return out


NODE_CLASS_MAPPINGS = {"FilmGrainFilter": FilmGrainFilter}

NODE_DISPLAY_NAME_MAPPINGS = {"FilmGrainFilter": "Film Grain"}
