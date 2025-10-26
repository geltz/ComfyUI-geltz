import math, torch
import comfy.model_patcher
import comfy.samplers

class PerturbedAttentionDelta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "min_scale_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(self, model, scale, min_scale_ratio=0.5):
        # where to hook
        unet_block = "middle"
        unet_block_id = 0

        # always start from a clone
        m = model.clone()

        def perturbed_attention(q, k, v, extra_options, mask=None):
            """
            Always return a tensor with q's sequence length, regardless of layout:
            works for (B,T,C) and (B,H,T,D).
            """
            # self-attn case (tq == tk): just pass v through.
            if q.shape == v.shape:
                return v

            # find the "token" axis = the axis where q and v differ and both are >1
            assert q.ndim == v.ndim, f"q.ndim={q.ndim} v.ndim={v.ndim}"
            seq_axes = [i for i, (aq, av) in enumerate(zip(q.shape, v.shape)) if aq != av and aq > 1 and av > 1]
            token_axis = seq_axes[0] if seq_axes else (-2)  # fallback to -2 (T) if heuristic fails

            # Pool v along its token axis to a single token, then broadcast to q's length
            pooled = v.mean(dim=token_axis, keepdim=True)   # shape matches v except token_axis==1
            out = pooled.expand(q.shape)                    # now matches q exactly
            return out

        # --- state for scheduling/telemetry ---
        state = {"sigma_start": None, "step_count": 0}

        def _sched_factor(sig_val, start):
            if start is None or start <= 1e-6:
                return 1.0, 1.0
            t = max(0.0, min(1.0, sig_val / start))
            # cosine schedule
            f = min_scale_ratio + (1.0 - min_scale_ratio) * (0.5 * (1.0 + math.cos(math.pi * (1.0 - t))))
            return f, t

        def _apply_patches(model_options):
            # Hit both self- and cross-attn; try common transformer indices.
            for name in ("attn1", "attn2"):
                model_options = comfy.model_patcher.set_model_options_patch_replace(
                    model_options, perturbed_attention, name, unet_block, unet_block_id
                )
                for tidx in (0, 1, 2, 3):
                    model_options = comfy.model_patcher.set_model_options_patch_replace(
                        model_options, perturbed_attention, name, unet_block, unet_block_id, transformer_index=tidx
                    )
            return model_options

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            cond = args["cond"]
            model_options = args["model_options"].copy()

            if scale == 0:
                return cfg_result

            state["step_count"] += 1

            try:
                sig_val = float(sigma.flatten()[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
            except Exception:
                sig_val = 1.0

            if state["sigma_start"] is None:
                state["sigma_start"] = sig_val
                print(f"[PAD] start sigma: {sig_val:.4f}")

            # ensure our hook actually maps to existing attention keys
            model_options = _apply_patches(model_options)

            # perturbed pass (no grad)
            with torch.no_grad():
                (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            sched, norm = _sched_factor(sig_val, state["sigma_start"])
            effective = scale * sched

            if state["step_count"] % 5 == 1:
                delta_mag = (cond_pred - pag).abs().mean().item()
                print(f"[PAD] step={state['step_count']} sigma={sig_val:.4f} "
                      f"sched={sched:.3f} scale={effective:.3f} |delta|={delta_mag:.6f}")

            return cfg_result + (cond_pred - pag) * effective

        # don't let CFG=1 fast-path skip our hook
        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)

        # return a 1-tuple (MODEL) at the METHOD level
        return (m,)

NODE_CLASS_MAPPINGS = {"PerturbedAttentionDelta": PerturbedAttentionDelta}
NODE_DISPLAY_NAME_MAPPINGS = {"PerturbedAttentionDelta": "Perturbed Attention Delta"}