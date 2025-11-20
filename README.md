## Image Processing
- **Apply LUT** – Apply color lookup tables (.cube files)  
- **Chromatic Aberration** – Shift RGB channels for lens-like effects
- **Color Border** – Apply a border with adjustable width and color to an image    
- **Color Palette Extractor** – Find dominant colors and exports a palette
- **Diffusion Denoiser** – Remove noise from generated images with a bilateral filter (accepts batch folder input)    
- **FidelityFX Upscaler** – Sharp upscaler, useful as a second-upscale pass
- **Kuwahara Filter** – Edge-aware smoothing to reduce noise and stylize     
- **L₀ Smoothing** – Smooth textures while keeping edges sharp  
- **Local Laplacian** – Enhance contrast and details without halos  
- **Palette Filter** – Transfer color grading from a reference image with an optimal transport
- **Pixel Sorting Filter** – Sort pixels of an image horizontally or vertically    
- **Temperature Adjust** – Adjust white balance while preserving saturation  
- **UNet Heatmap** – Create a heatmap image of a denoised UNet latent
- **Voronoi Tessellation Filter** – Creates voronoi segments from an image with adjustable size       

## Metadata & Utilities  
- **Kohya LoRA Config** – Convert LoRA headers to JSON for training with [Kohya](https://github.com/kohya-ss/sd-scripts)
- **Load Image With Metadata** – Load images with their context and masks  
- **Token Visualizer** – Render a spiked wave with viridis-like coloring to see how tokens are weighted  

## Model & LoRA
- **Load LoRA (SDXL Blocks)** – Apply SDXL LoRAs with per-block control  
- **LoRA Extract** – Create a LoRA by comparing the difference between two models using [sd-mecha](https://github.com/ljleb/sd-mecha)

## Sampling & Guidance
- **Perturbed Attention Delta** – Add a second forward pass during generation via injected noise, based on [PAG](https://arxiv.org/abs/2403.17377)
- **Quantile Match Scaling** – Prevent over-guidance with FFT quantile filters
- **SADA Acceleration** – Skip stable steps to speed up generation based on [this paper](https://arxiv.org/abs/2507.17135)
- **Semantic Noise Sampler** – Find better sampling noise by analyzing model semantics. Requires a full pass before denoising, so twice the steps. Based on [this paper](https://arxiv.org/abs/2511.07756)   
- **Token Shuffler** – Shuffle tokens during generation for better denoising path    

## Schedulers
- **Cosine** – Steady denoising in cosine-eased timestep space
- **Sigmoid** – Configurable steepness and center point for controlled denoising curves    

## Latent & Prompt
- **Danbooru Tags Transformer** – Single-node implementation of [DART](https://github.com/p1atdev/danbooru-tags-transformer), which generates danbooru tags
- **Dynamic Random Tokens** – Not a node! Global patch that handles nested syntax like {color {red|blue}|texture} for randomized token selection in a prompt		
- **Prompt Shuffler** – Shuffle order of comma-separated tokens using a random seed
- **Token Sculptor** – Fine-tune tokens via top-k neighbors, based on [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)
- **Reflection Padding** – Add reflection padding to conv2d layers on the VAE decoding. Use with models that have an [EQ-VAE](https://arxiv.org/abs/2502.09509)
- **Simple Inpaint** – Auto-resize masks and latent dimensions for inpainting		  
- **Structured Latent** – Create seeded empty latents using various noise methods      



























