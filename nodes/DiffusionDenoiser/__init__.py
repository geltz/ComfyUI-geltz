"""
Diffusion noise removal using OpenCV bilateral filter.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import folder_paths

class DiffusionDenoiser:
    """Apply bilateral filter to remove AI diffusion noise"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filter_diameter": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "sigma_color": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "sigma_space": ("FLOAT", {
                    "default": 64.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/filters"
    
    def denoise(self, image, filter_diameter, sigma_color, sigma_space):
        # Convert from ComfyUI format (B,H,W,C) float32 [0,1] to OpenCV uint8 [0,255]
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(
                img,
                d=filter_diameter,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
            
            # Convert back to ComfyUI format
            filtered = filtered.astype(np.float32) / 255.0
            results.append(filtered)
        
        output = np.stack(results, axis=0)
        return (torch.from_numpy(output),)


class DiffusionDenoiserBatch:
    """Apply bilateral filter to folder of images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "filter_diameter": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "sigma_color": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "sigma_space": ("FLOAT", {
                    "default": 64.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "recursive": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise_batch"
    CATEGORY = "image/filters"
    OUTPUT_IS_LIST = (True,)
    
    def imread_unicode(self, path):
        """Read image with Unicode support"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
    
    def denoise_batch(self, folder_path, filter_diameter, sigma_color, sigma_space, recursive):
        if not folder_path or not Path(folder_path).exists():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        input_path = Path(folder_path)
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []
        
        for pattern in patterns:
            if recursive:
                image_files.extend(input_path.rglob(pattern))
            else:
                image_files.extend(input_path.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")
        
        results = []
        for img_path in sorted(image_files):
            img = self.imread_unicode(img_path)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(
                img,
                d=filter_diameter,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
            
            # Convert to ComfyUI format
            filtered = filtered.astype(np.float32) / 255.0
            results.append(torch.from_numpy(filtered))
        
        return (results,)


NODE_CLASS_MAPPINGS = {
    "DiffusionDenoiser": DiffusionDenoiser,
    "DiffusionDenoiserBatch": DiffusionDenoiserBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionDenoiser": "Diffusion Denoiser",
    "DiffusionDenoiserBatch": "Diffusion Denoiser (Batch)",
}