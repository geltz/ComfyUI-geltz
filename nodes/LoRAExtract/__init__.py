import os
import torch
from torch import Tensor
import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.utils import ProgressBar

CLAMP_QUANTILE = 0.99

def extract_lora_weights(diff: Tensor, rank: int, *, svd_mode: str = "lowrank",
                         oversample: int = 8, device: str = "cpu", quantile: float = 0.99) -> tuple[Tensor, Tensor]:
    """Extract LoRA weights using (approx) SVD with minimal memory.

    svd_mode: "lowrank" (memory-friendly, default) or "exact"
    oversample: extra components for lowrank SVD quality
    device: where to run SVD ("cpu" to save VRAM)
    quantile: clamp threshold (0.90–0.999)
    """
    with torch.inference_mode():
        conv2d = (diff.ndim == 4)
        kernel_size = None if not conv2d else diff.size()[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)

        out_dim, in_dim = diff.size()[:2]
        r = min(rank, in_dim, out_dim)

        x = diff.detach()
        if conv2d:
            if conv2d_3x3:
                x = x.flatten(start_dim=1)
            else:
                x = x.squeeze()

        x = x.to(device=device, dtype=torch.float32, copy=True)
        
        # Check for non-finite values and replace with zeros
        if not torch.isfinite(x).all():
            print(f"   Warning: Non-finite values detected, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            if svd_mode == "lowrank" and r < min(out_dim, in_dim) and hasattr(torch.linalg, "svd_lowrank"):
                q = min(r + oversample, min(out_dim, in_dim))
                U, S, V = torch.linalg.svd_lowrank(x, q=q, niter=2)
                Vh = V.T
                U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
            else:
                kwargs = {"driver": "gesvd"} if device == "cuda" else {}
                U, S, Vh = torch.linalg.svd(x, **kwargs)
                U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
        except Exception:
            U, S, Vh = torch.linalg.svd(x)
            U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        U = U * S.unsqueeze(0)

        hi_u = torch.quantile(U.abs().reshape(-1), quantile)
        hi_v = torch.quantile(Vh.abs().reshape(-1), quantile)
        hi_val = float(max(hi_u, hi_v))
        low_val = -hi_val
        U = U.clamp_(low_val, hi_val)
        Vh = Vh.clamp_(low_val, hi_val)

        if device == "cuda":
            torch.cuda.empty_cache()

        if conv2d:
            U = U.reshape(out_dim, r, 1, 1).to(dtype=torch.float16, device="cpu", copy=False)
            Vh = Vh.reshape(r, in_dim, kernel_size[0], kernel_size[1]).to(dtype=torch.float16, device="cpu", copy=False)
        else:
            U = U.reshape(out_dim, r).to(dtype=torch.float16, device="cpu", copy=False)
            Vh = Vh.reshape(r, in_dim).to(dtype=torch.float16, device="cpu", copy=False)

        return (U.contiguous(), Vh.contiguous())


def extract_lora_from_state_dict(sd_diff, rank, use_full_diff=False, include_bias=True, pbar=None,
                                 *, svd_mode="lowrank", oversample=8, svd_device="cpu", quantile=0.99):
    """Extract LoRA from a state dict difference with low VRAM usage."""
    output_sd = {}
    with torch.inference_mode():
        for key, tensor in sd_diff.items():
            if pbar:
                pbar.update(1)

            if key.endswith(".weight"):
                weight_diff = tensor
                base_key = key[:-7]  # strip ".weight"

                if use_full_diff or weight_diff.ndim < 2:
                    if weight_diff.ndim < 2 and include_bias:
                        output_sd[f"{base_key}.diff"] = weight_diff.contiguous().half().cpu()
                    elif use_full_diff:
                        output_sd[f"{base_key}.diff"] = weight_diff.contiguous().half().cpu()
                else:
                    try:
                        lora_up, lora_down = extract_lora_weights(
                            weight_diff, rank,
                            svd_mode=svd_mode, oversample=oversample,
                            device=svd_device, quantile=quantile
                        )
                        output_sd[f"{base_key}.lora_up.weight"] = lora_up
                        output_sd[f"{base_key}.lora_down.weight"] = lora_down
                    except Exception as e:
                        print(f"Could not extract LoRA for {key}: {e}")
            elif include_bias and key.endswith(".bias"):
                base_key = key[:-5]
                output_sd[f"{base_key}.diff_b"] = tensor.contiguous().half().cpu()

            # Free per-iter scratch ASAP
            del tensor
            if svd_device == "cuda":
                torch.cuda.empty_cache()

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
            },
            "optional": {
                "svd_mode": (["lowrank", "exact"], {"default": "lowrank"}),
                "oversample": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "svd_device": (["cpu", "cuda"], {"default": "cpu"}),
                "free_vram_early": ("BOOLEAN", {"default": True}),
                "quantile": ("FLOAT", {"default": 0.99, "min": 0.90, "max": 0.999, "step": 0.001}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "extract_lora"
    CATEGORY = "advanced/model_merging"

    def extract_lora(self, model_diff, clip_diff, filename_prefix, rank, use_full_diff, include_bias,
                     svd_mode="lowrank", oversample=8, svd_device="cpu", free_vram_early=True, quantile=0.99):
        print(f"\nLoRA Extract: rank={rank}, full_diff={use_full_diff}, bias={include_bias}, "
              f"svd_mode={svd_mode}, svd_device={svd_device}, quantile={quantile}")

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )

        output_sd = {}

        # --- Extract from model (load → capture sd → immediately free VRAM) ---
        with torch.inference_mode():
            mm.load_models_gpu([model_diff], force_patch_weights=True)
            model_sd = model_diff.model_state_dict(filter_prefix="diffusion_model.")

        if free_vram_early:
            try:
                mm.unload_all_models()
            except Exception:
                pass
            try:
                mm.soft_empty_cache()
            except Exception:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        print(f"   Extracting {len(model_sd)} MODEL tensors...")
        pbar = ProgressBar(len(model_sd))
        model_lora = extract_lora_from_state_dict(
            model_sd, rank, use_full_diff, include_bias, pbar,
            svd_mode=svd_mode, oversample=oversample, svd_device=svd_device, quantile=quantile
        )
        for key, value in model_lora.items():
            output_sd[f"diffusion_model.{key}"] = value

        # --- Extract from CLIP (CPU-safe by default) ---
        try:
            clip_sd = clip_diff.cond_stage_model.state_dict()
            print(f"   Extracting {len(clip_sd)} CLIP tensors...")
            clip_lora = extract_lora_from_state_dict(
                clip_sd, rank, use_full_diff, include_bias, None,
                svd_mode=svd_mode, oversample=oversample, svd_device=svd_device, quantile=quantile
            )
            for key, value in clip_lora.items():
                output_sd[f"text_encoders.{key}"] = value
        except Exception as e:
            print(f"CLIP extraction skipped: {e}")

        # --- Save ---
        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        print(f"LoRA saved: {output_checkpoint}\n")
        return ()


NODE_CLASS_MAPPINGS = {"LoRAExtract": LoRAExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"LoRAExtract": "LoRA Extract"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]