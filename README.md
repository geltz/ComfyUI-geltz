## comfyui-geltz

**Color Palette Extractor**	

Extracts N dominant colors via MiniBatchKMeans and outputs palette image plus CSV of hex codes.	

**Image Metadata Extractor**	

Reads PNG/TIFF info and outputs normalized prompt/settings summary as a single string.	

**Kuwahara Filter**	

Fast edge-preserving filter selecting mean color from the minimum-variance quadrant.	

**L₀ Gradient Minimization**	

Global edge-aware smoothing that flattens regions while preserving sharp boundaries.	

**Load Image With Metadata**	

Loads image with embedded prompts/settings extraction, returns image, mask, and metadata text.	

**Local Laplacian Filter**	

Halo-free detail/tone manipulation via Laplacian pyramids with separable Gaussian blurs.	

**Kohya Lora Config**	

Parses a LoRA's header and extracts human-readable metadata as JSON.  
*Output compatible with [Kohya's sd-scripts](https://github.com/kohya-ss/sd-scripts)*	

**ORBIT Merge**	

Direction-aware model merger decomposing deltas into parallel/orthogonal components with independent scaling.  
*Uses the [sd-mecha](https://github.com/ljleb/sd-mecha) API*	

**Perturbed Attention Delta**  

Small edit of [PAG](https://arxiv.org/abs/2403.17377) with smart sigma-based scheduling.	

**Quantile Match Scaling**	

Stabilizes generation by matching frequency-band quantiles to conditional distribution.	

**SADA Model Acceleration**	

Skips redundant diffusion steps using trajectory stability analysis for faster sampling.  
*Based on [Stability-guided Adaptive Diffusion Acceleration](https://arxiv.org/abs/2507.17135)*	

**Structured Latent**  

Generate seeded empty latents with various initialization methods (perlin, gaussian, fractal, etc.)  

**Temperature Adjust**	

LAB-space white-balance adjustment with HSV saturation compensation, range -1.0…+1.0.	

**Token Delta Perturbation**	

Shuffles attention tokens using a scaled delta, with a cosine-decayed perturbation scale.  
*Based on [Token Perturbation Guidance](https://arxiv.org/abs/2506.10036)* with utilities from [ppm](https://github.com/pamparamm/ComfyUI-ppm)	

**Token Visualizer**	

Visualizes token influence via 2D wave path with normalized spikes.	

**Token Sculptor**	

Strengthens prompt adherence by nudging CLIP embeddings toward soft top-k neighbor blends.  
*Inspired by [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)*	

**Velocity Scaling**  

Reduces over-brightening in v-prediction models via epsilon scaling adaptation.  
*Based on [Elucidating the Exposure Bias in Diffusion Models](https://arxiv.org/abs/2308.15321)*	








