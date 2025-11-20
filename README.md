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

## Latent & Prompt Utilities

| Name | Description |
| :--- | :--- |
| **Danbooru Tags Transformer** | Implementation of DART for generating Danbooru tags. |
| **Dynamic Random Tokens** | Global patch that enables nested random tokens e.g. {{red\|green\|blue}hair}. |
| **Prompt Shuffler** | Randomize the order of comma-separated tokens using a seed. |
| **Reflection Padding** | Add reflection padding to Conv2D layers (requires EQ-VAE models). |
| **Simple Inpaint** | Auto-resize masks and latent dimensions for inpainting tasks. |
| **Structured Latent** | Create seeded empty latents using various noise methods. |
| **Token Sculptor** | Fine-tune tokens via top-k neighbors (based on Vector Sculptor). |

## Sampling & Guidance

| Name | Description |
| :--- | :--- |
| **Perturbed Attention Delta** | Inject noise during forward pass (based on PAG). |
| **Quantile Match Scaling** | FFT quantile filters to prevent over-guidance. |
| **SADA Acceleration** | Skip stable steps to speed up generation. |
| **Semantic Noise Sampler** | Analyze model semantics to find optimal sampling noise (requires extra pass). |
| **Token Shuffler** | Shuffle tokens during generation to alter denoising paths. |

## Model & LoRA

| Name | Description |
| :--- | :--- |
| **Load LoRA (SDXL Blocks)** | Apply SDXL LoRAs with specific per-block control. |
| **LoRA Extract** | Create a LoRA by calculating the difference between two models. |

## Metadata & Utilities

| Name | Description |
| :--- | :--- |
| **Kohya LoRA Config** | Convert LoRA headers to JSON for Kohya training. |
| **Load Image (Metadata)** | Load images including their generation context and masks. |
| **Token Visualizer** | Render spiked waves to visualize token weighting. |

## Schedulers

| Name | Description |
| :--- | :--- |
| **Cosine** | Steady denoising using cosine-eased timestep space. |
| **Sigmoid** | Configurable steepness and center point for controlled denoising curves. |



