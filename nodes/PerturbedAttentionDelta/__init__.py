import math, torch
import comfy.model_patcher
import comfy.samplers


class PerturbedAttentionDelta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "min_scale_ratio": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(self, model, scale, min_scale_ratio=0.5):
        # target a common UNet block; original code did this too
        unet_block = "middle"
        unet_block_id = 0

        m = model.clone()

        # ---------------------------------------------------------------------
        # 1. attention perturbation
        #    keep shape logic, but only PARTIALLY corrupt v instead of replacing
        # ---------------------------------------------------------------------
        def perturbed_attention(q, k, v, extra_options, mask=None):
            """
            Return something with q's sequence length while keeping v mostly intact.
            Works for (B,T,C) and (B,H,T,D).
            """
            # self-attn: same shape, don't touch
            if q.shape == v.shape:
                return v

            assert q.ndim == v.ndim, f"q.ndim={q.ndim} v.ndim={v.ndim}"

            # find token axis where q and v both have >1 and differ
            seq_axes = [
                i
                for i, (aq, av) in enumerate(zip(q.shape, v.shape))
                if aq != av and aq > 1 and av > 1
            ]
            token_axis = seq_axes[0] if seq_axes else (-2)

            # partial corruption: blend v with its pooled version
            # alpha controls how "wrong" the perturbed pass is
            alpha = 0.35
            pooled = v.mean(dim=token_axis, keepdim=True)  # (...,1,...)
            v_pert = v * (1.0 - alpha) + pooled * alpha

            # if after this we already match q, done
            if v_pert.shape == q.shape:
                return v_pert

            # otherwise, collapse to 1 token on v's token axis, then broadcast to q
            v_reduced = v_pert.mean(dim=token_axis, keepdim=True)
            v_q = v_reduced.expand(q.shape)
            return v_q

        # ---------------------------------------------------------------------
        # 2. state for schedule/debug
        # ---------------------------------------------------------------------
        state = {
            "sigma_start": None,
            "step_count": 0,
        }

        # cosine schedule from 1 -> min_scale_ratio as sigma drops
        def _sched_factor(sig_val, start):
            if start is None or start <= 1e-6:
                return 1.0, 1.0
            t = max(0.0, min(1.0, sig_val / start))
            f = min_scale_ratio + (1.0 - min_scale_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))
            )
            return f, t

        # patch all likely attention keys on demand
        def _apply_patches(model_options):
            # mirror original "hit self- and cross-attn" behavior
            for name in ("attn1", "attn2"):
                model_options = comfy.model_patcher.set_model_options_patch_replace(
                    model_options,
                    perturbed_attention,
                    name,
                    unet_block,
                    unet_block_id,
                )
                for tidx in (0, 1, 2, 3):
                    model_options = comfy.model_patcher.set_model_options_patch_replace(
                        model_options,
                        perturbed_attention,
                        name,
                        unet_block,
                        unet_block_id,
                        transformer_index=tidx,
                    )
            return model_options

        # ---------------------------------------------------------------------
        # 3. main hook: run a perturbed pass and inject a SMALL, shaped delta
        # ---------------------------------------------------------------------
        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]  # normal conditional
            cfg_result = args["denoised"]      # CFG-combined result
            sigma = args["sigma"]
            x = args["input"]
            cond = args["cond"]
            model_options = args["model_options"].copy()

            if scale == 0:
                return cfg_result

            state["step_count"] += 1

            # numeric sigma
            try:
                sig_val = (
                    float(sigma.flatten()[0].item())
                    if isinstance(sigma, torch.Tensor)
                    else float(sigma)
                )
            except Exception:
                sig_val = 1.0

            if state["sigma_start"] is None:
                state["sigma_start"] = sig_val
                print(f"[PAD] start sigma: {sig_val:.4f}")

            # make sure our attention is actually used
            model_options = _apply_patches(model_options)

            # one perturbed forward
            with torch.no_grad():
                (pag,) = comfy.samplers.calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )

            # schedule for this sigma
            sched, _ = _sched_factor(sig_val, state["sigma_start"])
            effective = scale * sched

            # -----------------------------------------------------------------
            # delta shaping to avoid over-bright images
            # -----------------------------------------------------------------
            delta = cond_pred - pag  # raw difference

            # remove global DC bias per-sample
            delta = delta - delta.mean(dim=(1, 2, 3), keepdim=True)

            # magnitude awareness: don't let delta energy exceed cond energy too much
            delta_mag = delta.abs().mean(dim=(1, 2, 3), keepdim=True)
            cond_mag = cond_pred.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            norm_factor = (cond_mag / (delta_mag + 1e-6)).clamp(0.0, 1.5)
            delta = delta * norm_factor

            # small safety clamp; tune if needed
            delta = delta.clamp(-1.0, 1.0)

            if state["step_count"] % 5 == 1:
                print(
                    f"[PAD] step={state['step_count']} "
                    f"sigma={sig_val:.4f} sched={sched:.3f} eff={effective:.3f} "
                    f"|delta|={delta_mag.mean().item():.6f}"
                )

            return cfg_result + delta * effective

        # make sampler call our hook
        m.set_model_sampler_post_cfg_function(
            post_cfg_function, disable_cfg1_optimization=True
        )

        return (m,)


NODE_CLASS_MAPPINGS = {"PerturbedAttentionDelta": PerturbedAttentionDelta}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttentionDelta": "Perturbed Attention Delta"
}


