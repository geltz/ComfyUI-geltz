import torch
import torch.nn.functional as F

class SimpleInpaint:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "mask_expand": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "Custom/Inpainting"

    def process(self, image, mask, vae, mask_expand, mask_blur):
        # 1. Handle Mask Dimensions
        # ComfyUI masks are [B, H, W], we need [B, 1, H, W] for processing
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        # 2. Resize mask to match image if they differ (common error prevention)
        B, H, W, C = image.shape
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        # 3. Expand Mask (Dilate) to ensure we cover edges
        if mask_expand > 0:
            kernel_size = 1 + 2 * mask_expand
            pad = mask_expand
            # Using MaxPool as a morphological dilation
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)

        # 4. Blur Mask (Gaussian) for smooth blending
        if mask_blur > 0:
            # Create simple gaussian kernel
            kernel_size = 2 * mask_blur + 1
            sigma = mask_blur / 2.0
            
            x_coord = torch.arange(kernel_size)
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
            
            mean = (kernel_size - 1)/2.
            variance = sigma**2.
            
            gaussian_kernel = (1./(2.*3.14159*variance)) * \
                              torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance) )
            
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.to(mask.device)
            
            # Pad to keep dimensions same
            pad = mask_blur
            mask = F.conv2d(mask, gaussian_kernel, padding=pad)

        # Clamp mask between 0 and 1 after blur
        mask = torch.clamp(mask, 0.0, 1.0)

        # 5. VAE Encode
        # We must move channels to [B, C, H, W] for VAE
        pixels = image.movedim(-1, 1)
        
        # Encode the pixels
        t = vae.encode(pixels[:,:,:,:])

        # 6. Apply the mask to the latent
        # Resize mask to latent dimensions (Image / 8)
        mask_latent = F.interpolate(mask, size=(t.shape[2], t.shape[3]), mode="bilinear", align_corners=False)
        
        # Remove the channel dimension for the latent noise mask requirement: [B, H, W]
        mask_latent = mask_latent[:, 0, :, :]

        # Create the return dictionary
        return ({"samples": t, "noise_mask": mask_latent}, )

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "SimpleInpaint": SimpleInpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleInpaint": "Simple Inpaint"
}