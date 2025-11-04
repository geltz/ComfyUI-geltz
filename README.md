## Image Processing

* **Chromatic Aberration**: Shift RGB channels for edge fringing.
* **Color Palette Extractor**: Find N dominant colors; exports palette image + CSV.
* **FidelityFX Upscaler**: AMD CAS upscaler, target scale, sharpness control.
* **Film Grain**: Luma-weighted, band-limited grain with tone weighting and deterministic seed.
* **Kuwahara Filter**: Fast edge-preserving smoothing.
* **L₀ Smoothing**: Global flattening while keeping edges.
* **Local Laplacian**: Halo-free detail/tone edits via Laplacian pyramid.
* **Palette Filter**: 3D LUT from reference image using sliced OT; adjustable grade.
* **Temperature Adjust**: LAB white balance with saturation compensation (-1.0…+1.0).
* **UNet Heatmap**: Convert denoised UNet latents to thermal map.

## Metadata & Utilities

* **Image Metadata Extractor**: Read PNG/TIFF, output normalized prompt/settings string.
* **Kohya LoRA Config**: Parse LoRA header to JSON; compatible with [Kohya’s sd-scripts](https://github.com/kohya-ss/sd-scripts).
* **Load Image With Metadata**: Load image, mask, and extracted text.
* **Token Visualizer**: Show token influence as 2D wave.

## Model & LoRA

* **Load LoRA (SDXL Blocks)**: Set per-block weights to keep structure but tune style.
* **LoRA Extract**: Diff two models into a LoRA with adjustable rank (via [sd-mecha](https://github.com/ljleb/sd-mecha)).

## Sampling & Guidance

* **NegPip+**: Symmetric negative repulsion with fixed indexing, limited to real tokens. Based on [ComfyUI-ppm](https://github.com/pamparamm/ComfyUI-ppm).
* **Perturbed Attention Delta**: [PAG](https://arxiv.org/abs/2403.17377) variant with sigma scheduling.
* **Quantile Match Scaling**: Match CFG freq to clean cond to avoid overdrive.
* **SADA Acceleration**: Skip diffusion steps using trajectory stability. Based on [Stability-guided Adaptive Diffusion Acceleration](https://arxiv.org/abs/2507.17135).
* **Spatial Split Attention**: Balance prompts over left/right regions with progressive merge.
* **Token Delta Perturbation**: Second UNet pass added to CFG for stronger guidance. Based on [Token Perturbation Guidance](https://arxiv.org/abs/2506.10036) and utilities from [ComfyUI-ppm](https://github.com/pamparamm/ComfyUI-ppm).

## Samplers

* **Ralston**: 3rd-order with optimal error.
* **Bogacki**: 3rd-order Bogacki-Shampine.

## Schedulers

* **Cosine**: Cosine-eased stable steps.
* **Nonlinear**: Heavier early denoising.

## Latent & Prompt

* **Structured Latent**: Seeded latents with perlin/gaussian/fractal.
* **Token Sculptor**: Nudge CLIP embeddings toward soft top-k neighbors. Inspired by [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI).


