import torch
import torch.nn.functional as F

class VoronoiTessellationFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_cells": ("INT", {"default": 100, "min": 10, "max": 2000, "step": 10, "display": "slider"}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_voronoi"
    CATEGORY = "image/filters"

    def apply_voronoi(self, image, num_cells, random_seed, smoothness):
        # image: B, H, W, C
        b, h, w, c = image.shape
        device = image.device
        
        torch.manual_seed(random_seed)
        
        # 1. Generate random seed coordinates in normalized range [-1, 1]
        # Shape: (B, num_cells, 2) (x, y)
        # We create specific indices for sampling
        seed_x = torch.randint(0, w, (b, num_cells), device=device)
        seed_y = torch.randint(0, h, (b, num_cells), device=device)
        
        # Gather colors at seed locations
        # Create batch indices
        batch_idx = torch.arange(b, device=device).unsqueeze(1).expand(-1, num_cells)
        
        # Sample colors: (B, N, C)
        seed_colors = image[batch_idx, seed_y, seed_x, :] 
        
        # 2. Create coordinate grid for the image
        # Shape: (1, H, W, 2)
        y_grid, x_grid = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0).float() # 1, H, W, 2
        
        # Flatten grid for distance calc: (1, H*W, 2)
        flat_grid = grid.reshape(1, h*w, 2)
        
        # Seed coords for distance: (B, N, 2)
        seed_coords = torch.stack((seed_x, seed_y), dim=-1).float()
        
        # 3. Calculate Distance from every pixel to every seed
        # Uses huge memory if N and Image are big. 
        # dist matrix: (B, H*W, N)
        # Optim: We rely on PyTorch cdist efficient implementation
        dist = torch.cdist(flat_grid.expand(b, -1, -1), seed_coords)
        
        # 4. Soft Voronoi (Smoothness) vs Hard Voronoi
        if smoothness > 0.01:
            # Softmin (Negative distance softmax) to blend colors
            # Scale smoothness: small value = sharper, large = blurrier
            # We need to invert dist so closer = higher weight
            # Normalizing factor is tricky, heuristic used here
            temperature = smoothness * 100.0
            weights = F.softmax(-dist / temperature, dim=-1) # (B, HW, N)
            
            # Weighted sum of colors
            # (B, HW, N) matmul (B, N, C) -> (B, HW, C)
            out_flat = torch.bmm(weights, seed_colors)
        else:
            # Hard assignment (Standard Voronoi)
            # Find index of closest seed
            nearest_idx = torch.argmin(dist, dim=-1) # (B, H*W)
            
            # Gather colors
            # We need to map nearest_idx to seed_colors
            # seed_colors is (B, N, C)
            # output needs to be (B, H*W, C)
            # gather expects index same dims as src except on dim
            idx_expanded = nearest_idx.unsqueeze(-1).expand(-1, -1, c) # B, HW, C
            out_flat = torch.gather(seed_colors, 1, idx_expanded)

        # Reshape back to image
        out = out_flat.reshape(b, h, w, c)
        
        return (out,)

NODE_CLASS_MAPPINGS = {"VoronoiTessellationFilter": VoronoiTessellationFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"VoronoiTessellationFilter": "Voronoi Tessellation"}