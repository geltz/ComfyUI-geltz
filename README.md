## Image Processing

**Chromatic Aberration**	  
Shifts red and blue channels by a pixel offset to create edge fringing.		  

**Color Palette Extractor**  
Extracts N dominant colors via MiniBatchKMeans and outputs palette image plus CSV of hex codes.	

**FidelityFX Upscaler**  
Lightweight AMD FidelityFX CAS upscaler that auto-downloads the CLI (uses Wine on non-Windows) and supports target scale with adjustable sharpness.	

**Film Grain**	  
Luma-weighted, band-limited, linear-light grain with shadow/midtone weighting, Poisson-like scaling, deterministic seeding, and RGB-consistent noise.	  

**Kuwahara Filter**  
Fast edge-preserving filter selecting mean color from the minimum-variance quadrant.

**L₀ Smoothing Filter**  
Global edge-aware smoothing that flattens regions while preserving sharp boundaries.

**Local Laplacian Filter**  
Halo-free detail/tone manipulation via Laplacian pyramids with separable Gaussian blurs.

**Palette Filter**  
Builds a 3D LUT via sliced optimal transport from a reference image and applies it as a controllable color grade.

**Temperature Adjust**  
LAB-space white-balance adjustment with HSV saturation compensation, range -1.0…+1.0.

**UNet Heatmap**  
Turns denoised UNet latents into a thermal heatmap via L2 magnitude normalization, bicubic upscaling, and unsharp masking.

## Metadata & Utilities

**Image Metadata Extractor**  
Reads PNG/TIFF info and outputs normalized prompt/settings summary as a single string.

**Kohya Lora Config**  
Parses a LoRA's header and extracts human-readable metadata as JSON.  
*Output compatible with [Kohya's sd-scripts](https://github.com/kohya-ss/sd-scripts)*

**Load Image With Metadata**  
Loads image with embedded prompts/settings extraction, returns image, mask, and metadata text.

**Token Visualizer**  
Visualizes token influence via 2D wave path with normalized spikes.

## Model & LoRA Tools

**Load LoRA (SDXL Blocks)**  
Allows setting block weights for an SDXL LoRA. Useful to avoid structural changes but keep the style i.e. only weighting input blocks.

**LoRA Extract**  
Extracts the difference between two models as a LoRA with adjustable rank.  
*Uses the [sd-mecha](https://github.com/ljleb/sd-mecha) API*  

## Sampling & Guidance

**NegPip+**  
Modification of NegPip that reflects negatives over neutral embeddings for symmetric repulsion, fixes z_empty indexing, and limits effect to actual tokens.  
*Based on the implementation from [ppm](https://github.com/pamparamm/ComfyUI-ppm)*

**Perturbed Attention Delta**  
Small edit of [PAG](https://arxiv.org/abs/2403.17377) with smart sigma-based scheduling.

**Quantile Match Scaling**  
Stabilizes guidance by matching frequency-band quantiles to conditional distribution.

**SADA Model Acceleration**  
Skips redundant diffusion steps using trajectory stability analysis for faster sampling.  
*Based on [Stability-guided Adaptive Diffusion Acceleration](https://arxiv.org/abs/2507.17135)*

**Spatial Split Attention**  
Self-attention and cross-attention algorithm that equally weights left and right conditioning prompts, combining two regions with progressive convergence controlled by noise level.

**Spectral Drift Perturbation**		  
UNet patch that injects spectral-drift noise into transformer blocks and blends CFG with [PAG](https://arxiv.org/abs/2403.17377) via a cosine schedule; parameters control scale, drift, and coherence.		  

**Token Delta Perturbation**  
Shuffles attention tokens using a scaled delta, with a cosine-decayed perturbation scale.  
*Based on [Token Perturbation Guidance](https://arxiv.org/abs/2506.10036) with utilities from [ppm](https://github.com/pamparamm/ComfyUI-ppm)*

**Velocity Scaling**  
Reduces over-brightening in v-prediction models via epsilon scaling adaptation.  
*Based on [Elucidating the Exposure Bias in Diffusion Models](https://arxiv.org/abs/2308.15321)*

## Samplers

Loaded into KSampler's `sampler` selector.	  

**Ralston**	  
Third-order Ralston method with optimal error coefficients.		   

**Bogacki**	  
Third-order Bogacki-Shampine method.		  

## Schedulers

Loaded into KSampler's `scheduler` selector.	  

**Rewind**	  
Log-linear with sinusoidal warp (amp=0.15); non-uniform spacing for ODE steps with optional stall kicks.	  

**River**  
Designed for rectified flow. Step uniformly in alpha, then map to sigma via sigma = sqrt(1/α^2 - 1).     

**Power**	  
Power-law interpolation (rho=7.0); concentrates steps where noise changes matter most for improved quality.		

**SNR Uniform**  
Designed for v-prediction. Steps uniformly in log SNR and maps back to σ to produce a stable k-diffusion sigma schedule.  

**Momentum**  
Prediction-agnostic. Velocity-based spacing: accelerates through stable regions, decelerates at critical transitions.  

## Latent & Prompt Tools

**Structured Latent**    
Generate seeded empty latents with various initialization methods (perlin, gaussian, fractal, etc.)

**Token Sculptor**  
Strengthens prompt adherence by nudging CLIP embeddings toward soft top-k neighbor blends.  
*Inspired by [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)*  











