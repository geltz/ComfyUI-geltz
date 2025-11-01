import torch
import torch.nn.functional as F

class FilmGrainFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.01}),
                "grain_size": ("INT", {"default": 3, "min": 1, "max": 9, "step": 2}),
                "shadow_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "midtone_weight": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "image/filters"

    def apply_grain(self, image, intensity, grain_size, shadow_weight, midtone_weight, seed):
        # Validate input
        if image.dim() != 4:
            raise ValueError("Image must be 4D tensor")

        # ComfyUI uses NHWC format
        B, H, W, C = image.shape
        if C != 3:
            raise ValueError("Image must have 3 channels (RGB)")

        # Convert to float32 for processing
        img = image.float()
        
        # sRGB to linear
        linear = torch.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055).pow(2.4)
        )

        # Calculate luminance in linear space
        luma = (
            linear[..., 0:1] * 0.2126 +
            linear[..., 1:2] * 0.7152 +
            linear[..., 2:3] * 0.0722
        )

        # Tone weighting: more grain in shadows and midtones
        mid_weight = (1.0 - (luma - 0.5).abs() * 2.0).clamp(0, 1)
        tone_weight = (shadow_weight * (1.0 - luma) + midtone_weight * mid_weight).clamp(0, 1)

        # Generate grain
        gen = torch.Generator(device=image.device).manual_seed(seed)
        grain = torch.randn(B, H, W, 1, device=image.device, generator=gen)

        # Blur grain if size > 1
        if grain_size > 1:
            # Convert to NCHW for conv2d
            grain_nchw = grain.permute(0, 3, 1, 2)
            kernel = torch.ones(1, 1, grain_size, grain_size, device=image.device) / (grain_size ** 2)
            grain_nchw = F.conv2d(grain_nchw, kernel, padding=grain_size // 2)
            grain = grain_nchw.permute(0, 2, 3, 1)

        # Normalize grain to unit std
        grain_std = grain.view(B, -1).std(dim=1, keepdim=True).clamp_min(1e-8)
        grain = grain / grain_std.view(B, 1, 1, 1)

        # Poisson-like scaling with luma
        poisson_scale = (luma + 1e-4).sqrt()
        strength = intensity * (0.5 * tone_weight + 0.5 * poisson_scale)

        # Apply grain (same to all channels)
        linear_out = (linear + grain * strength).clamp(0, 1)

        # Linear to sRGB
        output = torch.where(
            linear_out <= 0.0031308,
            linear_out * 12.92,
            1.055 * linear_out.pow(1 / 2.4) - 0.055
        ).clamp(0, 1)

        return (output,)  # Must return tuple


NODE_CLASS_MAPPINGS = {"FilmGrainFilter": FilmGrainFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"FilmGrainFilter": "Film Grain"}