import torch
import math

class InterpolateConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("CONDITIONING", ),
                "interpolate": ("CONDITIONING", ),
                # Frequency: Speed of the wave. 0.5 = One arch (0 -> 1 -> 0).
                "frequency": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                # Strength: Caps the maximum influence of the interpolate prompt.
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "interp_conditioning"
    CATEGORY = "conditioning/advanced"

    def set_cond_values(self, conditioning, start, end):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['start_percent'] = start
            n[1]['end_percent'] = end
            c.append(n)
        return c

    def interp_conditioning(self, original, interpolate, frequency, strength):
        out_conditioning = []
        step_size = 0.02 
        current_percent = 0.0

        while current_percent < 1.0:
            next_percent = min(current_percent + step_size, 1.0)
            
            # --- FIXED MATH ---
            # 1. Shift phase by -pi/2 so it starts at 0.0 (Original) instead of 0.5
            # 2. Multiplied by 'strength' to allow user cap
            angle = (current_percent * frequency * 2 * math.pi) - (math.pi / 2)
            sine_val = (math.sin(angle) + 1.0) / 2.0
            
            # Apply user strength cap
            current_strength = sine_val * strength
            
            mixed_slice = []
            for i, t_orig in enumerate(original):
                orig_tensor = t_orig[0]
                orig_dict = t_orig[1].copy()

                t_inj = interpolate[i % len(interpolate)]
                inj_tensor = t_inj[0]

                if isinstance(orig_tensor, torch.Tensor) and isinstance(inj_tensor, torch.Tensor):
                    device = orig_tensor.device
                    
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

                    mixed_tensor = (orig_tensor * (1.0 - current_strength)) + (inj_tensor * current_strength)
                    mixed_slice.append([mixed_tensor, orig_dict])
                else:
                    mixed_slice.append([orig_tensor, orig_dict])

            out_conditioning.extend(self.set_cond_values(mixed_slice, current_percent, next_percent))
            
            current_percent = next_percent

        return (out_conditioning,)

NODE_CLASS_MAPPINGS = {
    "InterpolateConditioning": InterpolateConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InterpolateConditioning": "Interpolate Conditioning"
}