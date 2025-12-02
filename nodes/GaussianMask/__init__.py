# Modified from https://civitai.com/models/2184867
# This uses Gaussian noise instead.

import torch
import node_helpers

class GaussianMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "randomize_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "Percentage of embedding values to modify."}),
                "strength": ("FLOAT", {"default": 20, "min": 0.0, "max": 1000.0, "step": 0.1, "tooltip": "Scale of the random noise."}),
                "noise_insert": (["noise on beginning steps", "noise on ending steps", "noise on all steps"],),
                "steps_switchover_percent": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "Percentage of steps before switching conditioning."}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF, "step": 1})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "conditioning"

    def randomize_conditioning(self, conditioning, strength, seed, randomize_percent, noise_insert, steps_switchover_percent):
        if randomize_percent == 0 or strength == 0:
            return (conditioning,)

        randomize_percent = randomize_percent / 100.0
        switch_val = steps_switchover_percent / 100.0

        # Use a local generator so we don't change the global seed for other nodes
        generator = torch.Generator().manual_seed(seed)

        noisy_embedding = []
        
        for t in conditioning:
            current_tensor = t[0]
            current_dict = t[1].copy()

            if isinstance(current_tensor, torch.Tensor):
                device = current_tensor.device

                # Using randn (Gaussian/Normal) instead of rand (Uniform)
                # We do not need the (* 2 - 1) math because randn is already centered at 0
                noise = torch.randn(current_tensor.shape, generator=generator, device=device) * strength
                
                # Create mask
                mask = torch.bernoulli(torch.full(current_tensor.shape, randomize_percent, device=device), generator=generator).bool()
                
                modified_noise = noise * mask
                noisy_tensor = current_tensor + modified_noise
                
                noisy_embedding.append([noisy_tensor, current_dict])
            else:
                noisy_embedding.append([current_tensor, current_dict])

        if noise_insert == "noise on beginning steps":
            new_conditioning = node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": 0.0, "end_percent": switch_val})
            new_conditioning += node_helpers.conditioning_set_values(conditioning, {"start_percent": switch_val, "end_percent": 1.0})
        elif noise_insert == "noise on ending steps":
            new_conditioning = node_helpers.conditioning_set_values(conditioning, {"start_percent": 0.0, "end_percent": switch_val})
            new_conditioning += node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": switch_val, "end_percent": 1.0})
        else: 
            return (noisy_embedding,)

        return (new_conditioning,)

NODE_CLASS_MAPPINGS = {
    "GaussianMask": GaussianMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GaussianMask": "Gaussian Mask"
}
