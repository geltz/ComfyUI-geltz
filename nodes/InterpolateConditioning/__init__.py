import torch
import math

class InterpolateConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("CONDITIONING", ),
                "interpolate": ("CONDITIONING", ),
                # Frequency adjusted to 0.0 - 1.0 range
                "frequency": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "interp_conditioning"
    CATEGORY = "conditioning/advanced"

    def set_cond_values(self, conditioning, start, end):
        """Helper to set start/end percent on a conditioning chunk."""
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['start_percent'] = start
            n[1]['end_percent'] = end
            c.append(n)
        return c

    def interp_conditioning(self, original, interpolate, frequency):
        out_conditioning = []
        
        # Resolution: Slices the timeline every 2% for smooth transitions
        step_size = 0.02 
        current_percent = 0.0

        while current_percent < 1.0:
            next_percent = min(current_percent + step_size, 1.0)
            
            # Calculate Sine Wave Strength
            # Angle covers 'frequency' full cycles over the generation
            angle = current_percent * frequency * 2 * math.pi
            
            # Map sine (-1 to 1) to strength (0 to 1)
            strength = (math.sin(angle) + 1.0) / 2.0
            
            mixed_slice = []
            for i, t_orig in enumerate(original):
                orig_tensor = t_orig[0]
                orig_dict = t_orig[1].copy()

                # Get corresponding interpolate tensor
                t_inj = interpolate[i % len(interpolate)]
                inj_tensor = t_inj[0]

                if isinstance(orig_tensor, torch.Tensor) and isinstance(inj_tensor, torch.Tensor):
                    device = orig_tensor.device
                    
                    # Match shapes
                    if orig_tensor.shape != inj_tensor.shape:
                        target_len = orig_tensor.shape[1]
                        current_len = inj_tensor.shape[1]
                        if current_len > target_len:
                            inj_tensor = inj_tensor[:, :target_len, ...]
                        elif current_len < target_len:
                            rep = (target_len // current_len) + 1
                            inj_tensor = torch.cat([inj_tensor] * rep, dim=1)[:, :target_len, ...]
                    
                    if inj_tensor.device != device:
                        inj_tensor = inj_tensor.to(device)

                    # Mix tensors based on sine strength
                    mixed_tensor = (orig_tensor * (1.0 - strength)) + (inj_tensor * strength)
                    mixed_slice.append([mixed_tensor, orig_dict])
                else:
                    mixed_slice.append([orig_tensor, orig_dict])

            # Register the slice in the timeline
            out_conditioning.extend(self.set_cond_values(mixed_slice, current_percent, next_percent))
            
            current_percent = next_percent

        return (out_conditioning,)

NODE_CLASS_MAPPINGS = {
    "InterpolateConditioning": InterpolateConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InterpolateConditioning": "Interpolate Conditioning"
}