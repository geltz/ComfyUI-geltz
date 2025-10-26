import os
import torch
from torch import Tensor
import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.utils import ProgressBar

try:
    import sd_mecha
    from sd_mecha import merge_method, Parameter, Return
    SD_MECHA_AVAILABLE = True
except ImportError:
    SD_MECHA_AVAILABLE = False
    print("sd_mecha not found. please install: pip install sd-mecha")

CLAMP_QUANTILE = 0.99


def extract_lora_weights(diff: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Extract LoRA weights using SVD decomposition."""
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    # Use gesvd driver for better stability with CUDA
    try:
        U, S, Vh = torch.linalg.svd(diff.float(), driver='gesvd')
    except:
        U, S, Vh = torch.linalg.svd(diff.float())
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    
    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    
    return (U, Vh)

# sd_mecha merge_method decorator not used for extraction

def extract_lora_from_state_dict(sd_diff, rank, use_full_diff=False, include_bias=True, pbar=None):
    """Extract LoRA from a state dict difference."""
    output_sd = {}
    
    for key in sd_diff.keys():
        if pbar:
            pbar.update(1)
            
        if key.endswith(".weight"):
            weight_diff = sd_diff[key]
            base_key = key[:-7]  # Remove ".weight"
            
            if use_full_diff or weight_diff.ndim < 2:
                if weight_diff.ndim < 2 and include_bias:
                    output_sd[f"{base_key}.diff"] = weight_diff.contiguous().half().cpu()
                elif use_full_diff:
                    output_sd[f"{base_key}.diff"] = weight_diff.contiguous().half().cpu()
                continue
            
            try:
                lora_up, lora_down = extract_lora_weights(weight_diff, rank)
                output_sd[f"{base_key}.lora_up.weight"] = lora_up.contiguous().half().cpu()
                output_sd[f"{base_key}.lora_down.weight"] = lora_down.contiguous().half().cpu()
            except Exception as e:
                print(f"Could not extract LoRA for {key}: {e}")
                
        elif include_bias and key.endswith(".bias"):
            base_key = key[:-5]  # Remove ".bias"
            output_sd[f"{base_key}.diff_b"] = sd_diff[key].contiguous().half().cpu()
    
    return output_sd


class LoRAExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_diff": ("MODEL", {}),
                "clip_diff": ("CLIP", {}),
                "filename_prefix": ("STRING", {"default": "loras/extracted_lora"}),
                "rank": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "use_full_diff": ("BOOLEAN", {"default": False}),
                "include_bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "extract_lora"
    CATEGORY = "advanced/model_merging"

    def extract_lora(self, model_diff, clip_diff, filename_prefix, rank, use_full_diff, include_bias):
        print(f"\nLoRA Extract: rank={rank}, full_diff={use_full_diff}, bias={include_bias}")
        
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        
        output_sd = {}
        
        # Extract from model
        mm.load_models_gpu([model_diff], force_patch_weights=True)
        model_sd = model_diff.model_state_dict(filter_prefix="diffusion_model.")
        print(f"   Extracting {len(model_sd)} MODEL tensors...")
        pbar = ProgressBar(len(model_sd))
        
        # Add prefix to output keys
        model_lora = extract_lora_from_state_dict(model_sd, rank, use_full_diff, include_bias, pbar)
        for key, value in model_lora.items():
            output_sd[f"diffusion_model.{key}"] = value
        
        # Extract from CLIP
        try:
            clip_sd = clip_diff.cond_stage_model.state_dict()
            print(f"   Extracting {len(clip_sd)} CLIP tensors...")
            
            clip_lora = extract_lora_from_state_dict(clip_sd, rank, use_full_diff, include_bias)
            for key, value in clip_lora.items():
                output_sd[f"text_encoders.{key}"] = value
                    
        except Exception as e:
            print(f"CLIP extraction skipped: {e}")
        
        # Save
        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        
        print(f"LoRA saved: {output_checkpoint}\n")
        return ()


NODE_CLASS_MAPPINGS = {"LoRAExtract": LoRAExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"LoRAExtract": "LoRA Extract"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]