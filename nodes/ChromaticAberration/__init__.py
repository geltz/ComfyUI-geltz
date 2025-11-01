import torch
import torch.nn.functional as F

class ChromaticAberrationFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_aberration"
    CATEGORY = "image/filters"

    def apply_aberration(self, image, offset):
        img = image.permute(0, 3, 1, 2)
        b, c, h, w = img.shape
        
        # Shift red and blue channels
        px = int(offset)
        r = F.pad(img[:, 0:1], (px, 0, 0, 0))[:, :, :h, :w]
        g = img[:, 1:2]
        bl = F.pad(img[:, 2:3], (0, px, 0, 0))[:, :, :h, :w]
        
        result = torch.cat([r, g, bl], 1).permute(0, 2, 3, 1)
        return (result,)


NODE_CLASS_MAPPINGS = {"ChromaticAberrationFilter": ChromaticAberrationFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"ChromaticAberrationFilter": "Chromatic Aberration"}