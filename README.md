## Image Processing
- **Apply LUT** – Applies color lookup tables (.cube files)  
- **Chromatic Aberration** – Shifts RGB channels for lens-like effects
- **Color Border** – Apply a border with adjustable width and color to an image    
- **Color Palette Extractor** – Finds dominant colors and exports a palette
- **Diffusion Denoiser** – Removes noise from generated images with a bilateral filter (accepts batch folder input)    
- **FidelityFX Upscaler** – Sharp upscaler, useful as a second-upscale pass
- **Kuwahara Filter** – Edge-aware smoothing to reduce noise and stylize     
- **L₀ Smoothing** – Smooths textures while keeping edges sharp  
- **Local Laplacian** – Enhances contrast and details without halos  
- **Palette Filter** – Transfers color grading from a reference image with an optimal transport      
- **Temperature Adjust** – Adjusts white balance while preserving saturation  
- **UNet Heatmap** – Creates a heatmap image of a denoised UNet latent   

## Metadata & Utilities  
- **Kohya LoRA Config** – Converts LoRA headers to JSON for training with [Kohya](https://github.com/kohya-ss/sd-scripts)
- **Load Image With Metadata** – Loads images with their context and masks  
- **Token Visualizer** – Renders a spiked wave with viridis-like coloring to see how tokens are weighted  

## Model & LoRA
- **Load LoRA (SDXL Blocks)** – Applies SDXL LoRAs with per-block control  
- **LoRA Extract** – Creates a LoRA by comparing the difference between two models using [sd-mecha](https://github.com/ljleb/sd-mecha)

## Sampling & Guidance
- **Perturbed Attention Delta** – Adds a second forward pass during generation via injected noise, based on [PAG](https://arxiv.org/abs/2403.17377)
- **Quantile Match Scaling** – Prevents over-guidance with FFT quantile filters
- **SADA Acceleration** – Skips stable steps to speed up generation based on [this paper](https://arxiv.org/abs/2507.17135)
- **Semantic Noise Sampler** – Builds NLN-based latent noise to improve sampling from latents. Not sure if it's a strict improvement yet, needs more testing. Based on [this paper](https://arxiv.org/abs/2511.07756)   
- **Token Shuffler** – Shuffles tokens during generation for better denoising path    

## Schedulers
- **Cosine** – Steady denoising in cosine-eased timestep space
- **Sigmoid** – Configurable steepness and center point for controlled denoising curves    

## Latent & Prompt
- **Danbooru Tags Transformer** – Single-node implementation of [DART](https://github.com/p1atdev/danbooru-tags-transformer), which generates danbooru tags
- **Prompt Shuffler** – Shuffles order of comma-separated tokens using a random seed    
- **Structured Latent** – Creates seeded empty latents using various noise methods    
- **Token Sculptor** – Fine-tunes tokens via top-k neighbors, based on [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)



















