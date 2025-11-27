import torch

class FlowMaskingEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "drop_probability": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "Advanced/Flow"
    
    def patch_model(self, model, drop_probability, seed, start_percent=0.0, end_percent=0.6):
        m = model.clone()

        target_keys = ["context", "txt", "y", "c_crossattn", "pooled_output"]

        def nudge_wrapper(model_function, params):
            
            # Timing check
            current_sigma = params["timestep"][0].item()
            
            model_sampling = m.get_model_object("model_sampling")
            percent = 0.5 
            
            if model_sampling:
                try:
                    percent = params.get("transformer_options", {}).get("percent_to_sigma_interval", None)
                    if percent is None:
                        max_sig = model_sampling.sigma(torch.tensor([0.0], device=params["input"].device))
                        percent = 1.0 - (current_sigma / max_sig.item())
                except:
                    percent = 0.5 

            # If outside range, run model normally
            if percent < start_percent or percent > end_percent:
                return model_function(params["input"], params["timestep"], **params["c"], **params.get("transformer_options", {}))

            # Apply masking
            input_x = params["input"]
            timestep = params["timestep"]
            c = params["c"]
            tf_options = params.get("transformer_options", {})
            
            step_seed = seed + int(current_sigma * 1000000)
            rng = torch.Generator().manual_seed(step_seed)

            c_perturbed = c.copy()

            for key in target_keys:
                if key in c_perturbed:
                    target_tensor = c_perturbed[key]
                    
                    if isinstance(target_tensor, torch.Tensor) and target_tensor.ndim >= 3:
                        
                        if drop_probability >= 1.0:
                            nudged_tensor = torch.zeros_like(target_tensor)
                        elif drop_probability <= 0.0:
                            nudged_tensor = target_tensor
                        else:
                            keep_prob = 1.0 - drop_probability
                            
                            # Generate mask on CPU to avoid device errors
                            rand_mask = torch.rand(target_tensor.shape, generator=rng, device="cpu")
                            mask = (rand_mask < keep_prob).to(target_tensor.dtype).to(target_tensor.device)
                            
                            nudged_tensor = target_tensor * mask * (1.0 / keep_prob)

                        c_perturbed[key] = nudged_tensor

            # Call model
            # Remove transformer_options from c_perturbed if it exists to avoid duplication error
            if "transformer_options" in c_perturbed:
                c_perturbed.pop("transformer_options")

            return model_function(input_x, timestep, **c_perturbed, transformer_options=tf_options)

        m.set_model_unet_function_wrapper(nudge_wrapper)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "FlowMaskingEmbeds": FlowMaskingEmbeds
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMaskingEmbeds": "Flow Masking (Embeds)"

}
