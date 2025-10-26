import re
import comfy.sd
import comfy.lora
import comfy.utils
import folder_paths


def _get_lora_path(name: str) -> str:
    return folder_paths.get_full_path("loras", name)


def _base_key_from_sd_key(sd_key: str) -> str:
    # Strip common LoRA suffixes to get the logical key
    if sd_key.endswith(".alpha"):
        return sd_key[:-len(".alpha")]
    for suf in (".lora_up.weight", ".lora_down.weight", ".lora_A.weight", ".lora_B.weight"):
        if sd_key.endswith(suf):
            return sd_key[:-len(suf)]
    return sd_key


def _block_multiplier_for_model_key(model_key: str, in_w: float, mid_w: float, out_w: float) -> float:
    m = re.search(r"diffusion_model\.(input_blocks|output_blocks|middle_block)", model_key)
    if not m:
        return 1.0
    which = m.group(1)
    if which == "input_blocks":
        return float(in_w)
    if which == "middle_block":
        return float(mid_w)
    return float(out_w)  # output_blocks


class SDXL_LoRA_BlockSlider_Loader:
    """
    Inputs:
      - model, clip: SDXL base model + clip
      - lora_name: file from models/loras
      - strength_model / strength_clip: global LoRA strengths
      - in_weight / mid_weight / out_weight: per-path multipliers
    Returns: (MODEL, CLIP) with LoRA applied using path sliders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "in_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mid_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "out_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load"
    CATEGORY = "loaders/LoRA"

    def load(self, model, clip, lora_name, strength_model, strength_clip, in_weight, mid_weight, out_weight):
        # 1) Load LoRA and standardize keys
        lora_path = _get_lora_path(lora_name)
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        if hasattr(comfy.lora, "standardize_lora_key_format"):
            sd = comfy.lora.standardize_lora_key_format(sd)

        # 2) Map logical LoRA keys to actual UNet module keys
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})

        # 3) Scale UNet LoRA tensors by path slider
        for k in list(sd.keys()):
            base = _base_key_from_sd_key(k)
            model_key = key_map.get(base)
            if model_key is None:
                continue  # not a UNet tensor (likely CLIP) -> leave as-is
            mult = _block_multiplier_for_model_key(model_key, in_weight, mid_weight, out_weight)
            if mult == 1.0:
                continue
            try:
                sd[k] = sd[k] * mult
            except Exception:
                pass  # non-tensor leaf

        # 4) Patch into the models
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, sd, strength_model, strength_clip)
        return (model_lora, clip_lora)


NODE_CLASS_MAPPINGS = {
    "SDXL_LoRA_BlockSlider_Loader": SDXL_LoRA_BlockSlider_Loader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL_LoRA_BlockSlider_Loader": "Load LoRA (SDXL Blocks)",
}
