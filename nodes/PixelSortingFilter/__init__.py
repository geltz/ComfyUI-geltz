import torch
import torch.nn.functional as F

class PixelSortingFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "direction": (["horizontal", "vertical"],),
                "span_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sort"
    CATEGORY = "image/filters"

    def apply_sort(self, image, threshold, direction, span_randomness):
        # ComfyUI passes (Batch, Height, Width, Channel)
        # We operate on (B, C, H, W) for some ops, but gather needs indices aligned
        img = image.permute(0, 3, 1, 2) # B, C, H, W
        
        if direction == "vertical":
            img = img.permute(0, 1, 3, 2) # Swap H/W to reuse horizontal logic
            
        b, c, h, w = img.shape
        
        # 1. Calculate Luminance for sorting key
        lum = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        
        # 2. Create Mask: Pixels > threshold are sortable
        mask = (lum > threshold).float()
        
        # 3. Create Sorting Keys
        # We want valid pixels to sort by luminance.
        # Invalid pixels (masked out) should stay in their original index.
        # We achieve this by making their sort-key their original index * large_constant
        
        x_indices = torch.arange(w, device=img.device).expand(b, h, w).float()
        
        # Add noise to the sort key to break up perfect banding (Span Randomness)
        noise = torch.randn_like(lum) * span_randomness * 0.5
        sort_score = lum + noise
        
        # Where mask is 0, force key to be very ordered (original index)
        # Where mask is 1, key is luminance
        # We create a "high value" offset for static pixels to push them to specific zones or keep structure?
        # Actually, standard strategy: 
        # Sort everything, but "locked" pixels rely on a fixed grid value + huge offset
        # This implementation is a "Span Sort" approximation:
        
        # Simpler Glitch Logic:
        # Sort the whole row, but weighted by mask? No.
        # Let's use the mask to select values, sort them, and put them back.
        # This is hard in pure tensor batch without loops.
        
        # Alternative Tensor Logic: Rank Order
        # Key = Luminance
        # If we just sort rows, it looks like a smear.
        # We will add the x_indices to the key heavily where we want to KEEP structure.
        
        structure_weight = (1.0 - mask) * 1000.0 # Huge weight to preserve order in dark areas
        final_key = sort_score + structure_weight * x_indices
        
        # 4. Get Sort Indices
        sorted_indices = torch.argsort(final_key, dim=-1)
        
        # 5. Gather pixels using the new indices
        # Expand indices for 3 channels
        idx_expanded = sorted_indices.unsqueeze(1).expand(-1, c, -1, -1)
        sorted_img = torch.gather(img, 3, idx_expanded)
        
        if direction == "vertical":
            sorted_img = sorted_img.permute(0, 1, 3, 2)
            
        # Convert back to BHWC
        result = sorted_img.permute(0, 2, 3, 1)
        return (result,)

NODE_CLASS_MAPPINGS = {"PixelSortingFilter": PixelSortingFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelSortingFilter": "Pixel Sorting (Glitch)"}