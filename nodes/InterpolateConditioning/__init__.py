import torch

class InterpolateConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("CONDITIONING", ),
                "interpolate": ("CONDITIONING", ),
                # Reduced step to 0.01 for finer control over the blend
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "interp_conditioning"
    CATEGORY = "conditioning/advanced"

    def interp_conditioning(self, original, interpolate, strength):
        # 1. Early exit if strength is 0 (return original unchanged)
        if strength == 0:
            return (original,)

        mixed_conditioning = []

        for i, t_orig in enumerate(original):
            orig_tensor = t_orig[0]
            orig_dict = t_orig[1].copy() # Keep metadata from original

            # Cycle through 'interpolate' inputs if there are fewer chunks than 'original'
            t_inj = interpolate[i % len(interpolate)]
            inj_tensor = t_inj[0]

            if isinstance(orig_tensor, torch.Tensor) and isinstance(inj_tensor, torch.Tensor):
                device = orig_tensor.device
                
                # --- SHAPE MATCHING ---
                # Ensures tensors are the same size before math operations
                if orig_tensor.shape != inj_tensor.shape:
                    target_len = orig_tensor.shape[1]
                    current_len = inj_tensor.shape[1]
                    
                    if current_len > target_len:
                        # Crop if new conditioning is longer
                        inj_tensor = inj_tensor[:, :target_len, ...]
                    elif current_len < target_len:
                        # Repeat if new conditioning is shorter
                        repeat_count = (target_len // current_len) + 1
                        inj_tensor = torch.cat([inj_tensor] * repeat_count, dim=1)[:, :target_len, ...]
                
                if inj_tensor.device != device:
                    inj_tensor = inj_tensor.to(device)

                # --- LINEAR INTERPOLATION (LERP) ---
                # Formula: (Original * (1 - Strength)) + (New * Strength)
                mixed_tensor = (orig_tensor * (1.0 - strength)) + (inj_tensor * strength)
                
                mixed_conditioning.append([mixed_tensor, orig_dict])
            else:
                # Fallback if data is not a tensor (e.g. area sizes)
                mixed_conditioning.append([orig_tensor, orig_dict])

        return (mixed_conditioning,)

NODE_CLASS_MAPPINGS = {
    "InterpolateConditioning": InterpolateConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InterpolateConditioning": "Interpolate Conditioning"
}