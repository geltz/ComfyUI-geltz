## Image Processing

| Name | Description |
| :--- | :--- |
| **Apply LUT** | Apply color lookup tables (.cube files) to images. |
| **Chromatic Aberration** | Shift RGB channels to create lens distortion effects. |
| **Color Border** | Add a border to an image with adjustable width and color. |
| **Color Palette Extractor** | Analyze dominant colors and export a palette. |
| **Diffusion Denoiser** | Remove noise via bilateral filter; supports batch folder input. |
| **FidelityFX Upscaler** | Sharp upscaling method, ideal for second-pass upscaling. |
| **Kuwahara Filter** | Edge-aware smoothing for noise reduction and stylization. |
| **L₀ Smoothing** | Smooth textures while maintaining sharp edges. |
| **Local Laplacian** | Enhance contrast and details without introducing halos. |
| **Palette Filter** | Transfer color grading from a reference image using optimal transport. |
| **Pixel Sorting Filter** | Sort image pixels horizontally or vertically. |
| **Temperature Adjust** | Modify white balance while preserving saturation. |
| **UNet Heatmap** | Generate a heatmap visualization of a denoised UNet latent. |
| **Voronoi Tessellation** | Create Voronoi segments from an image with adjustable sizing. |

## Latent & Prompt

| Name | Description |
| :--- | :--- |
| **Danbooru Tags Transformer** | Implementation of [DART](https://huggingface.co/p1atdev/dart-v2-moe-sft) for generating Danbooru tags. |
| **Dynamic Random Tokens** | Global patch that enables nested random tokens e.g. {{red\|green\|blue}hair}. |
| **Prompt Shuffler** | Randomize the order of comma-separated tokens using a seed. |
| **Reflection Padding** | Add reflection padding to Conv2D layers (requires EQ-VAE models). |
| **Simple Inpaint** | Auto-resize masks and latent dimensions for inpainting tasks. |
| **Structured Latent** | Create seeded empty latents using various noise methods. |
| **Token Sculptor** | Fine-tune tokens via top-k neighbors (based on [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)). |

## Sampling & Guidance

| Name | Description |
| :--- | :--- |
| **Perturbed Attention Delta** | Alters attention values during forward pass. (based on [PAG](https://arxiv.org/abs/2403.17377)). |
| **Quantile Match Scaling** | FFT quantile filters to prevent over-guidance. |
| **SADA Acceleration** | Skip stable steps to speed up generation. [Paper](https://arxiv.org/abs/2507.17135). |
| **Semantic Noise Sampler** | Analyze model semantics to find optimal sampling noise (requires extra pass). |
| **Token Shuffler** | Timestep-aware cross-attention shuffling to improve UNet diffusion. |

## Model

| Name | Description |
| :--- | :--- |
| **Load LoRA (SDXL Blocks)** | Apply SDXL LoRAs with specific per-block control. |
| **LoRA Extract** | Create a LoRA by calculating the difference between two models. Uses [sd-mecha](https://github.com/ljleb/sd-mecha). |

## Metadata

| Name | Description |
| :--- | :--- |
| **Kohya LoRA Config** | Convert LoRA headers to JSON for [Kohya](https://github.com/kohya-ss/sd-scripts) training. |
| **Load Image (Metadata)** | Load images including their generation context and masks. |
| **Token Visualizer** | Render spiked waves to visualize token weighting. |

## Schedulers

| Name | Description |
| :--- | :--- |
| **Cosine** | Steady denoising using cosine-eased timestep space. |
| **Sigmoid** | Configurable steepness and center point for controlled denoising curves. |








