import torch
import torch.nn.functional as F

class UNetHeatmap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmap",)
    FUNCTION = "generate_heatmap"
    CATEGORY = "latent"

    def apply_thermal_colormap(self, mag):
        # mag: (B, 1, H, W) normalized [0, 1]
        # Define points for interpolation: v from dark purple to yellow/orange
        points_v = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=mag.device)
        # Corresponding RGB colors: dark purple -> purple -> magenta -> yellow -> orange
        colors = torch.tensor([
            [0.05, 0.00, 0.20],  # dark purple
            [0.40, 0.00, 0.60],  # purple
            [0.80, 0.20, 0.40],  # magenta
            [1.00, 0.80, 0.00],  # yellow
            [1.00, 0.50, 0.00]   # orange
        ], device=mag.device)  # (5, 3)
        
        b, _, h, w = mag.shape
        mag_flat = mag.view(-1)  # (B*H*W,)
        n = mag_flat.numel()
        rgb = torch.zeros(n, 3, device=mag.device, dtype=torch.float32)
        
        # Find indices for each segment
        idx = torch.searchsorted(points_v, mag_flat, right=True) - 1  # [0,4]
        idx = torch.clamp(idx, 0, len(points_v) - 2)
        
        # Interpolate within segments
        low = points_v[idx]
        high = points_v[idx + 1]
        t = (mag_flat - low) / (high - low + 1e-8)
        t = torch.clamp(t, 0, 1)
        
        c_low = colors[idx]  # (n, 3)
        c_high = colors[idx + 1]  # (n, 3)
        rgb = c_low + t.unsqueeze(1) * (c_high - c_low)
        
        # Handle exact matches to points (interpolation covers)
        return rgb.view(b, h, w, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)

    def sharpen_image(self, image, strength=1.5):
        # image: (B, 3, H, W)
        # Simple unsharp mask: original + strength * (original - gaussian_blur)
        kernel_size = 3
        padding = kernel_size // 2
        # Gaussian kernel approx
        kernel = torch.tensor([[[[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]]]], device=image.device, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)  # (3, 1, 3, 3)
        # Blur
        blurred = F.conv2d(image, kernel, padding=padding, groups=3)
        # Sharpen
        sharpened = image + strength * (image - blurred)
        return torch.clamp(sharpened, 0, 1)

    def generate_heatmap(self, latent, upscale_factor=16, sharpen_strength=1.5):
        samples = latent["samples"]
        if samples.shape[0] > 1:
            samples = samples[0:1]
        # L2 norm across channels as attention proxy
        magnitude = torch.norm(samples, dim=1, keepdim=True)  # (B, 1, H, W)
        # Normalize to [0, 1]
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        if mag_max > mag_min:
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude = torch.zeros_like(magnitude)
        # Apply thermal colormap
        heatmap = self.apply_thermal_colormap(magnitude)
        # Upscale with bicubic for quality
        upscaled = F.interpolate(
            heatmap,
            scale_factor=upscale_factor,
            mode="bicubic",
            align_corners=False
        )
        # Clamp to [0, 1] for safety
        upscaled = torch.clamp(upscaled, 0, 1)
        # Apply sharpening to reduce blur from upscaling
        sharpened = self.sharpen_image(upscaled, sharpen_strength)
        # To BHWC
        image = sharpened.permute(0, 2, 3, 1)
        return (image,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "UNetHeatmap": UNetHeatmap
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UNetHeatmap": "UNet Heatmap"
}