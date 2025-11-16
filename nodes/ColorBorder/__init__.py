import torch
import torch.nn.functional as F

class ColorBorderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_width": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_border"
    CATEGORY = "image/filters"

    def hex_to_rgb(self, hex_color):
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB (0-1 range)
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def add_border(self, image, border_width, color):
        # image shape: [batch, height, width, channels]
        batch, height, width, channels = image.shape
        
        # Parse color
        try:
            r, g, b = self.hex_to_rgb(color)
        except:
            r, g, b = 1.0, 1.0, 1.0  # Default to white on error
        
        # Permute to [batch, channels, height, width]
        image_perm = image.permute(0, 3, 1, 2)
        
        # Pad each channel with its corresponding color value
        bordered = torch.stack([
            F.pad(image_perm[:, 0], (border_width, border_width, border_width, border_width), value=r),
            F.pad(image_perm[:, 1], (border_width, border_width, border_width, border_width), value=g),
            F.pad(image_perm[:, 2], (border_width, border_width, border_width, border_width), value=b)
        ], dim=1)
        
        # Back to [batch, height, width, channels]
        bordered = bordered.permute(0, 2, 3, 1)
        
        return (bordered,)

NODE_CLASS_MAPPINGS = {
    "ColorBorderNode": ColorBorderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorBorderNode": "Color Border"
}