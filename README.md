## Image Processing

* **Apply LUT**
  Simple node that applies a LUT file (looks in `models/lut`) with interpolation.  

* **Chromatic Aberration**    
  Shifts the red, green, and blue channels separately to create controlled edge fringing. Good for adding a subtle lens-like imperfection without blurring the whole frame.    

* **Color Palette Extractor**    
  Finds a specified number of dominant colors in an image and exports both a small reference image and a CSV palette, so other steps in the pipeline can stay on the same color scheme.    

* **FidelityFX Upscaler**    
  AMD CAS–based upscaling with scale and sharpness controls. Lets you render at a lower resolution and recover detail on output.

* **Kuwahara Filter**    
  Fast, edge-aware smoothing that reduces noise but keeps important structure. Useful as a preprocessing pass for more stylized looks.

* **L₀ Smoothing**    
  Performs global smoothing while preserving strong edges, so you can flatten textures without losing object boundaries.

* **Local Laplacian**    
  Laplacian-pyramid–style tone and detail editing that avoids halo artifacts, suited to fine contrast or local enhancement work.

* **Palette Filter**    
  Builds a 3D LUT from a reference image using sliced optimal transport. This lets you steer an image toward a target grade while still keeping control over the result.

* **Temperature Adjust**    
  LAB-based white balance with saturation compensation in the range −1.0 to +1.0. Warms or cools the image while trying to maintain overall colorfulness.

* **UNet Heatmap**    
  Turns denoised UNet latents into a thermal-style map to show where the model is focusing during generation.

# Metadata & Utilities

* **Image Metadata Extractor**    
  Reads PNG/TIFF metadata and outputs a normalized prompt/settings string so runs can be reproduced.

* **Kohya LoRA Config**    
  Parses a LoRA header to JSON that works with [Kohya’s sd-scripts](https://github.com/kohya-ss/sd-scripts), making it easier to inspect or reuse training settings.

* **Load Image With Metadata**    
  Loads image, mask, and extracted text together so the asset and its context stay aligned.

* **Token Visualizer**    
  Shows token influence as a 2D wave so you can quickly see which parts of the text prompt are driving the output.

## Model & LoRA    

* **Load LoRA (SDXL Blocks)**    
  Loads SDXL-style LoRAs with per-block weights, which lets you adjust style or content while keeping the base structure stable.

* **LoRA Extract**    
  Diffs two models into a LoRA with adjustable rank (via [sd-mecha](https://github.com/ljleb/sd-mecha)).

## Sampling & Guidance

* **Perturbed Attention Delta**    
  A [PAG](https://arxiv.org/abs/2403.17377)-style attention method with sigma scheduling that adds controlled variation while staying near the main diffusion path.

* **Quantile Match Scaling**    
  Matches CFG frequency to the clean condition to avoid overdriving guidance when prompts are strong.

* **SADA Acceleration**    
  Skips diffusion steps based on trajectory stability, following the idea in [Stability-guided Adaptive Diffusion Acceleration](https://arxiv.org/abs/2507.17135). Aims to reduce steps while preserving the intended look.  

## Schedulers

* **Cosine**    
  Cosine-eased schedule for steady denoising.

* **Nonlinear**    
  Front-loads denoising early, can be unstable.

## Latent & Prompt

* **Danbooru Tags Transformer**    
  Implementation of [DART](https://github.com/p1atdev/danbooru-tags-transformer) as a single node.        

* **Structured Latent**    
  Creates seeded latents using several methods. Recommended: `smooth_gradient`.

* **Token Sculptor**    
  Adjusts CLIP embeddings toward soft top‑k neighbors, inspired by [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI), to fine‑tune how text concepts appear in the final image.










