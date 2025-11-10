import os
import torch
import numpy as np
import folder_paths

class ApplyLUT:
    @classmethod
    def INPUT_TYPES(cls):
        lut_dir = os.path.join(folder_paths.models_dir, "lut")
        os.makedirs(lut_dir, exist_ok=True)
        lut_files = [f for f in os.listdir(lut_dir) if f.endswith('.cube')]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (lut_files,),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lut"
    CATEGORY = "image/color"
    
    def parse_cube_lut(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        size = 33
        lut_data = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
            elif line and not line.startswith('#') and not line.startswith('TITLE'):
                try:
                    rgb = [float(x) for x in line.split()]
                    if len(rgb) == 3:
                        lut_data.append(rgb)
                except:
                    continue
        
        lut = np.array(lut_data).reshape(size, size, size, 3)
        return lut
    
    def apply_lut(self, image, lut_file):
        lut_path = os.path.join(folder_paths.models_dir, "lut", lut_file)
        lut = self.parse_cube_lut(lut_path)
        size = lut.shape[0]
        
        img = image.cpu().numpy()
        result = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            # Scale to LUT coordinates
            coords = img[i] * (size - 1)
            coords = np.clip(coords, 0, size - 1)
            
            # Trilinear interpolation
            r0, g0, b0 = coords.astype(int)[..., 0], coords.astype(int)[..., 1], coords.astype(int)[..., 2]
            r1, g1, b1 = np.minimum(r0 + 1, size - 1), np.minimum(g0 + 1, size - 1), np.minimum(b0 + 1, size - 1)
            
            dr = coords[..., 0] - r0
            dg = coords[..., 1] - g0
            db = coords[..., 2] - b0
            
            # Sample LUT corners
            c000 = lut[b0, g0, r0]
            c001 = lut[b0, g0, r1]
            c010 = lut[b0, g1, r0]
            c011 = lut[b0, g1, r1]
            c100 = lut[b1, g0, r0]
            c101 = lut[b1, g0, r1]
            c110 = lut[b1, g1, r0]
            c111 = lut[b1, g1, r1]
            
            # Interpolate
            c00 = c000 * (1 - dr[..., None]) + c001 * dr[..., None]
            c01 = c010 * (1 - dr[..., None]) + c011 * dr[..., None]
            c10 = c100 * (1 - dr[..., None]) + c101 * dr[..., None]
            c11 = c110 * (1 - dr[..., None]) + c111 * dr[..., None]
            
            c0 = c00 * (1 - dg[..., None]) + c01 * dg[..., None]
            c1 = c10 * (1 - dg[..., None]) + c11 * dg[..., None]
            
            result[i] = c0 * (1 - db[..., None]) + c1 * db[..., None]
        
        return (torch.from_numpy(result).to(image.device),)

NODE_CLASS_MAPPINGS = {
    "ApplyLUT": ApplyLUT
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyLUT": "Apply LUT"
}