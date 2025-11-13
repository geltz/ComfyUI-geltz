## Image Processing
- **Apply LUT** – Applies color lookup tables (.cube files)  
- **Chromatic Aberration** – Shifts RGB channels for lens-like effects  
- **Color Palette Extractor** – Finds dominant colors and exports a palette  
- **FidelityFX Upscaler** – AMD-based upscaling with sharpness controls  
- **Kuwahara Filter** – Edge-aware smoothing for stylized looks  
- **L₀ Smoothing** – Smooths textures while keeping edges sharp  
- **Local Laplacian** – Enhances contrast and details without halos  
- **Palette Filter** – Transfers color grading from a reference image  
- **Temperature Adjust** – Adjusts white balance while preserving saturation  
- **UNet Heatmap** – Shows where the AI model focuses during generation  

## Metadata & Utilities
- **Image Metadata Extractor** – Reads image metadata for reproducibility  
- **Kohya LoRA Config** – Converts LoRA headers to JSON for training with [Kohya](https://github.com/kohya-ss/sd-scripts)
- **Load Image With Metadata** – Loads images with their context and masks  
- **Token Visualizer** – Shows which parts of the prompt influence the output  

## Model & LoRA
- **Load LoRA (SDXL Blocks)** – Applies SDXL LoRAs with per-block control  
- **LoRA Extract** – Creates a LoRA by comparing the difference between two models using [sd-mecha](https://github.com/ljleb/sd-mecha)

## Sampling & Guidance
- **Perturbed Attention Delta** – Adds a second forward pass during generation via injected noise and reverses the path, based on [PAG](https://arxiv.org/abs/2403.17377)
- **Quantile Match Scaling** – Prevents over-guidance with strong prompts  
- **SADA Acceleration** – Skips stable steps to speed up generation based on [this paper](https://arxiv.org/abs/2507.17135) 
- **Token Shuffler** – Shuffles tokens during generation for better denoising path    

## Schedulers
- **Cosine** – Steady denoising schedule  
- **Nonlinear** – Fast early denoising (can be unstable)  

## Latent & Prompt
- **Danbooru Tags Transformer** – Transforms tags for anime-style generation  
- **Structured Latent** – Creates seeded starting images  
- **Token Sculptor** – Fine-tunes how text concepts appear in the image via top-k neighbors




