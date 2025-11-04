## Image Processing

* **Chromatic Aberration**: Shift RGB channels for edge fringing.
* **Color Palette Extractor**: Find N dominant colors; exports palette image + CSV.
* **FidelityFX Upscaler**: AMD CAS upscaler (auto CLI download, Wine on non-Windows), target scale, sharpness control.
* **Film Grain**: Luma-weighted, band-limited grain with tone weighting and deterministic seed.
* **Kuwahara Filter**: Fast edge-preserving smoothing.
* **L₀ Smoothing**: Global flattening while keeping edges.
* **Local Laplacian**: Halo-free detail/tone edits via Laplacian pyramid.
* **Palette Filter**: 3D LUT from reference image using sliced OT; adjustable grade.
* **Temperature Adjust**: LAB white balance with saturation compensation (-1.0…+1.0).
* **UNet Heatmap**: Convert denoised UNet latents to thermal map (normalize, upscale, sharpen).

## Metadata & Utilities

* **Image Metadata Extractor**: Read PNG/TIFF, output normalized prompt/settings string.
* **Kohya LoRA Config**: Parse LoRA header to JSON; compatible with kohya sd-scripts.
* **Load Image With Metadata**: Load image, mask, and extracted text.
* **Token Visualizer**: Show token influence as 2D wave.

## Model & LoRA

* **Load LoRA (SDXL Blocks)**: Set per-block weights to keep structure but tune style.
* **LoRA Extract**: Diff two models into a LoRA with adjustable rank (via sd-mecha).

## Sampling & Guidance

* **NegPip+**: Symmetric negative repulsion with fixed indexing, limited to real tokens.
* **Perturbed Attention Delta**: PAG variant with sigma scheduling.
* **Quantile Match Scaling**: Match CFG freq to clean cond to avoid overdrive.
* **SADA Acceleration**: Skip diffusion steps using trajectory stability.
* **Spatial Split Attention**: Balance prompts over left/right regions with progressive merge.
* **Token Delta Perturbation**: Second UNet pass added to CFG for stronger guidance.

## Samplers

* **Ralston**: 3rd-order with optimal error.
* **Bogacki**: 3rd-order Bogacki-Shampine.

## Schedulers

* **Cosine**: Heavier early denoising.
* **Nonlinear**: Cosine-eased stable steps.

## Latent & Prompt

* **Structured Latent**: Seeded latents with perlin/gaussian/fractal.
* **Token Sculptor**: Nudge CLIP embeddings toward soft top-k neighbors.
