from torch import nn
import copy

class ReflectionPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",)
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply"
    CATEGORY = "utils"

    def apply(self, vae):
        if vae.first_stage_model is None:
            return (vae,)

        vae_out = copy.copy(vae)
        vae_out.first_stage_model = copy.deepcopy(vae.first_stage_model)
        vae_out.first_stage_model.to(device=vae.device, dtype=vae.vae_dtype)

        # Apply reflection padding to all Conv2d layers with padding
        for module in vae_out.first_stage_model.modules():
            if isinstance(module, nn.Conv2d):
                pad = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                if any(p > 0 for p in pad):
                    module.padding_mode = "reflect"
        
        return (vae_out,)

NODE_CLASS_MAPPINGS = {
    "ReflectionPadding": ReflectionPadding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReflectionPadding": "Reflection Padding"
}