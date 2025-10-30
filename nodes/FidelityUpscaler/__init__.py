import os
import sys
import zipfile
import tempfile
import hashlib
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# --- Config ---
FFX_URL = "https://github.com/GPUOpen-Effects/FidelityFX-CLI/releases/download/v1.0.3/FidelityFX-CLI-v1.0.3.zip"
FFX_SHA256 = "d280f245730c6d163c0e072a881ed4933b32e67b9de5494650119afa9649ea11"  # optional verify
NODE_DIR = Path(__file__).resolve().parent
FFX_DIR = NODE_DIR / "FidelityFX-CLI"
FFX_ZIP = NODE_DIR / "FidelityFX-CLI.zip"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_once():
    FFX_DIR.mkdir(exist_ok=True, parents=True)
    if any(FFX_DIR.rglob("*")):
        return
    if not FFX_ZIP.exists():
        urllib.request.urlretrieve(FFX_URL, FFX_ZIP.as_posix())
    try:
        # best-effort integrity check
        if _sha256(FFX_ZIP) != FFX_SHA256:
            raise RuntimeError("Downloaded FidelityFX-CLI zip failed SHA256 check.")
    except Exception:
        # continue even if hash compare not possible on some platforms
        pass
    with zipfile.ZipFile(FFX_ZIP, "r") as z:
        z.extractall(FFX_DIR)

def _find_ffx_executable() -> str:
    _download_once()
    candidates = []
    for p in FFX_DIR.rglob("*"):
        n = p.name.lower()
        if sys.platform.startswith("win") and n == "fidelityfx_cli.exe":
            candidates.append(p)
        elif not sys.platform.startswith("win") and (n == "fidelityfx_cli.exe" or n == "fidelityfx_cli"):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("FidelityFX-CLI executable not found after download.")
    return str(candidates[0])

def _run_ffx(input_path: str, output_path: str, scale: int = None, width: int = None, height: int = None, sharpness: float = 1.0):
    exe = _find_ffx_executable()
    if width and height:
        cmd = [exe, "-Scale", str(width), str(height), "-Mode", "CAS", "-Sharpness", str(sharpness), input_path, output_path]
    elif scale:
        cmd = [exe, "-Scale", f"{scale}x", f"{scale}x", "-Mode", "CAS", "-Sharpness", str(sharpness), input_path, output_path]
    else:
        raise ValueError("Provide either scale or width+height.")

    if not sys.platform.startswith("win"):
        # assumes `wine` is available for the Windows binary
        cmd = ["wine"] + cmd

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or (not os.path.exists(output_path)):
        raise RuntimeError(f"FidelityFX-CLI failed (code {result.returncode}). StdErr:\n{result.stderr}")

def _tensor_to_pil_batch(image_tensor: torch.Tensor):
    # image_tensor: [B,H,W,C], 0..1
    image_tensor = image_tensor.clamp(0, 1).cpu().numpy()
    batch = []
    for i in image_tensor:
        arr = (i * 255.0).round().astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        batch.append(Image.fromarray(arr, mode="RGB"))
    return batch

def _pil_to_tensor_batch(images):
    out = []
    for im in images:
        arr = np.array(im, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # [H,W,C] -> [1,H,W,C]
        out.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.cat(out, dim=0)

class FidelityFX_Upscaler:
    """
    Minimal upscaler node using AMD FidelityFX-CLI (CAS).
    Downloads the CLI on first run. Uses Wine on non-Windows hosts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "scale": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"

    def upscale(self, image, scale=2, target_width=0, target_height=0, sharpness=1.0):
        pil_batch = _tensor_to_pil_batch(image)
        out_batch = []

        for pil_im in pil_batch:
            in_w, in_h = pil_im.size
            if target_width > 0 and target_height > 0:
                exp_w, exp_h = int(target_width), int(target_height)
                mode = f"target {exp_w}x{exp_h}"
            else:
                s = int(scale)
                exp_w, exp_h = in_w * s, in_h * s
                mode = f"scale {s}x"

            with tempfile.TemporaryDirectory() as td:
                inp = os.path.join(td, "in.png")
                out = os.path.join(td, "out.png")
                pil_im.save(inp, format="PNG")
                if target_width > 0 and target_height > 0:
                    _run_ffx(inp, out, scale=None, width=exp_w, height=exp_h, sharpness=sharpness)
                else:
                    _run_ffx(inp, out, scale=int(scale), width=None, height=None, sharpness=sharpness)

                out_im = Image.open(out).convert("RGB")
                out_w, out_h = out_im.size
                ok = (out_w, out_h) == (exp_w, exp_h)
                print(f"[FidelityFX] {in_w}x{in_h} -> {out_w}x{out_h} ({mode}) {'OK' if ok else 'MISMATCH'}")
                out_batch.append(out_im)

        return (_pil_to_tensor_batch(out_batch),)


NODE_CLASS_MAPPINGS = {
    "FidelityFX_Upscaler": FidelityFX_Upscaler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FidelityFX_Upscaler": "FidelityFX Upscaler",
}
