import torch
import numpy as np

class MultiplySigmas:

    CONFIG = {
        "start": 0.2,       # Step percentage to start effect
        "end": 0.8,         # Step percentage to end effect
        "bias": 0.5,        # Peak position (0.5 = middle)
        "exponent": 1.0,    # Curve sharpness (1.0 = linear-ish/sine)
        "smooth": True,     # Use cosine smoothing (True = bell curve, False = triangle)
        # "cond": Multiplies sigma of positive prompt (Cleanest detail enhancement)
        # "uncond": Multiplies sigma of negative prompt
        # "both": Multiplies both (Stronger, similar to original script default)
        "mode": "cond"      
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "amount": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "positive = sharpen/detail, negative = smooth/blur"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_sigma_mult"
    CATEGORY = "custom/model"

    def apply_sigma_mult(self, model, amount):
        
        cfg = self.CONFIG
        
        # Pre-calculate the schedule curve logic
        def get_schedule_value(step_idx, total_steps):
            pct = step_idx / (total_steps - 1) if total_steps > 1 else 0.0
            
            # Outside range
            if pct < cfg["start"] or pct > cfg["end"]:
                return 0.0

            # Calculate relative position within the active window
            start_val = cfg["start"]
            end_val = cfg["end"]
            mid_val = start_val + cfg["bias"] * (end_val - start_val)

            val = 0.0
            if pct <= mid_val:
                # Rising edge
                norm = (pct - start_val) / (mid_val - start_val) if mid_val > start_val else 1.0
                val = norm
            else:
                # Falling edge
                norm = (end_val - pct) / (end_val - mid_val) if end_val > mid_val else 1.0
                val = norm

            # Apply smoothing and exponent
            if cfg["smooth"]:
                val = 0.5 * (1 - np.cos(val * np.pi))
            
            return val ** cfg["exponent"]

        def sigma_wrapper(model_function, params):
            x = params['input']
            sigma = params['timestep']
            transformer_options = params.get("transformer_options", {})
            
            # Determine current step from sigma schedule
            if "sigmas" in transformer_options:
                full_sigmas = transformer_options["sigmas"]
                total_steps = len(full_sigmas) - 1
                
                # Find closest step index
                current_sigma = sigma[0].item()
                diff = torch.abs(full_sigmas - current_sigma)
                step_idx = torch.argmin(diff).item()
                step_idx = min(step_idx, total_steps - 1)

                # Get curve multiplier for this step
                curve_strength = get_schedule_value(step_idx, total_steps)
                
                # If we are in the active zone, modify sigma
                if curve_strength > 0:
                    adjustment = curve_strength * amount
                    new_sigma = sigma.clone()
                    
                    # Apply to Cond (0) or Uncond (1) based on batch map
                    cond_or_uncond = transformer_options.get("cond_or_uncond", None)
                    
                    if cfg["mode"] == "both":
                        new_sigma *= (1.0 - adjustment)
                    elif cond_or_uncond is not None:
                        for i, type_idx in enumerate(cond_or_uncond):
                            if i < len(new_sigma):
                                if type_idx == 0 and cfg["mode"] == "cond":
                                    new_sigma[i] *= (1.0 - adjustment)
                                elif type_idx == 1 and cfg["mode"] == "uncond":
                                    new_sigma[i] *= (1.0 + adjustment)
                    else:
                        # Fallback if batch map missing
                        if cfg["mode"] == "cond": new_sigma *= (1.0 - adjustment)
                        elif cfg["mode"] == "uncond": new_sigma *= (1.0 + adjustment)

                    params['timestep'] = new_sigma

            return model_function(params["input"], params["timestep"], **params["c"])

        m = model.clone()
        m.set_model_unet_function_wrapper(sigma_wrapper)
        return (m,)

NODE_CLASS_MAPPINGS = {"MultiplySigmas": MultiplySigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiplySigmas": "Multiply Sigmas"}