import math
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_patcher import ModelPatcher

COND, UNCOND = 0, 1

def _lcm_list(xs):
    v = xs[0]
    for y in xs[1:]:
        v = math.lcm(v, y)
    return v

def _split_kv_cond(cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k, v = cond[:, 0::2], cond[:, 1::2]
    return (k, v) if k.shape == v.shape else (cond, cond)

class SpatialSplitAttention(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "left_cond": (IO.CONDITIONING, {}),
                "right_cond": (IO.CONDITIONING, {}),
                "center": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "separation": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.5, "step": 0.01}),
                "convergence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "token_influence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (IO.MODEL, IO.CONDITIONING)
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: ModelPatcher, left_cond, right_cond, center: float, feather: float, sharpness: float, separation: float, convergence: float, token_influence: float):
        m = model.clone()
        dtype = m.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        left = left_cond[0][0].to(device=device, dtype=dtype)
        right = right_cond[0][0].to(device=device, dtype=dtype)
        left_k, left_v = _split_kv_cond(left)
        right_k, right_v = _split_kv_cond(right)

        def get_progressive_params(sigma):
            t = (sigma / 14.0).clamp(0.0, 1.0).item() if torch.is_tensor(sigma) else 0.0
            # Start separation earlier and more aggressively
            separation_start = convergence * 0.7  # Start separation before full convergence
            if t > separation_start:
                progress = (t - separation_start) / (1.0 - separation_start)
                offset = separation * progress
                curr_feather = 0.01 + (feather - 0.01) * (1.0 - progress)  # Tighter feathering
            else:
                offset = 0.0
                curr_feather = feather
            return center - offset, center + offset, curr_feather, t

        def make_mask(h, w, c, f, dev, dt):
            cx = int(round(w * c))
            cx = max(1, min(w - 1, cx))
            fw = int(round(w * f))
            fw = min(max(fw, 0), w // 2)

            # always create in the same dtype as model (dt)
            x = torch.arange(w, device=dev, dtype=dt).view(1, 1, w)
            one = torch.tensor(1.0, device=dev, dtype=dt)
            zero = torch.tensor(0.0, device=dev, dtype=dt)

            if fw > 0:
                left_core = (x <= (cx - fw)).to(dt)
                right_core = (x >= (cx + fw)).to(dt)
                u = (x - (cx - fw)).clamp(zero, torch.tensor(2 * fw, device=dev, dtype=dt)) / max(1, 2 * fw)
                ramp = one - torch.sigmoid(torch.tensor(sharpness * 3.0, device=dev, dtype=dt) * (u - 0.5))
                left_band = (one - left_core) * (one - right_core) * ramp
                left_line = left_core + left_band
            else:
                left_line = (x < cx).to(dt)

            right_line = one - left_line
            left_mask = left_line.view(1, w).expand(h, w)
            right_mask = right_line.view(1, w).expand(h, w)
            mask = torch.stack([left_mask, right_mask], dim=0)
            s = mask.sum(dim=0, keepdim=True).clamp_min(torch.tensor(1e-6, device=dev, dtype=dt))
            return mask / s

        def mask_tokens(mask_2hw, size_hw, bs, n_tok):
            m = F.interpolate(mask_2hw.unsqueeze(1), size=size_hw, mode="nearest").squeeze(1)
            return m.view(2, n_tok, 1).repeat_interleave(bs, dim=0)
        
        def attn1_patch(q, k, v, extra):
            cond_types = extra.get("cond_or_uncond", [])
            if not cond_types:
                return q, k, v

            num_chunks = len(cond_types)
            bs = q.shape[0] // num_chunks
            h, w = extra["activations_shape"][-2:]
            n_tokens = q.shape[1]

            sigma = extra.get("sigmas", [0.0])[0] if "sigmas" in extra else 0.0
            left_c, right_c, curr_f, t_val = get_progressive_params(sigma)

            if t_val <= convergence:
                return q, k, v

            # build spatial masks in same dtype/device as q/k/v
            mask_l = make_mask(h, w, left_c, curr_f, q.device, q.dtype)[0:1]
            mask_r = make_mask(h, w, right_c, curr_f, q.device, q.dtype)[1:2]
            mask_2 = torch.cat([mask_l, mask_r], dim=0)

            # tokens mask: [2, n_tokens, 1]
            mask_tok = mask_tokens(mask_2, (h, w), 1, n_tokens)
            left_mask_values = mask_tok[0:1].expand(bs * num_chunks, -1, -1)
            right_mask_values = mask_tok[1:2].expand(bs * num_chunks, -1, -1)

            # use same-dtype scalars to avoid silent upcasting
            one = torch.tensor(1.0, device=k.device, dtype=k.dtype)
            strength = torch.tensor(token_influence, device=k.device, dtype=k.dtype)  # FIXED: Add this line
            spatial_strength = torch.tensor(2.0, device=k.device, dtype=k.dtype)

            # Apply spatial-aware suppression based on actual mask strength
            k_left = k * (left_mask_values + (1.0 - left_mask_values) * torch.exp(-strength * spatial_strength))
            v_left = v * (left_mask_values + (1.0 - left_mask_values) * torch.exp(-strength * spatial_strength))

            k_right = k * (right_mask_values + (1.0 - right_mask_values) * torch.exp(-strength * spatial_strength))
            v_right = v * (right_mask_values + (1.0 - right_mask_values) * torch.exp(-strength * spatial_strength))

            # duplicate q, route k/v
            q_double = torch.cat([q, q], dim=0)
            k_double = torch.cat([k_left, k_right], dim=0)
            v_double = torch.cat([v_left, v_right], dim=0)

            extra["spatial_split_active"] = True
            return q_double, k_double, v_double


        def attn1_output_patch(out, extra):
            if not extra.get("spatial_split_active", False):
                return out
            
            h, w = extra["activations_shape"][-2:]
            n_tokens = out.shape[1]
            
            # out is already doubled: [2*original_bs, n_tokens, dim]
            half = out.shape[0] // 2
            left_out = out[:half]
            right_out = out[half:]
            
            sigma = extra.get("sigmas", [0.0])[0] if "sigmas" in extra else 0.0
            left_c, right_c, curr_f, _ = get_progressive_params(sigma)
            
            # spatial merge masks - use half for bs since we already split
            mask_l = make_mask(h, w, left_c, curr_f, out.device, out.dtype)[0:1]
            mask_r = make_mask(h, w, right_c, curr_f, out.device, out.dtype)[1:2]
            mask_2 = torch.cat([mask_l, mask_r], dim=0)
            mask_tok = mask_tokens(mask_2, (h, w), half, n_tokens)
            
            # hard spatial decision
            left_mask = (mask_tok[:half, :, 0:1] > 0.5).to(dtype=out.dtype)
            right_mask = 1.0 - left_mask
            
            merged = left_out * left_mask + right_out * right_mask
            
            extra["spatial_split_active"] = False
            return merged
        
        def attn2_patch(q, k, v, extra):
            cond_types = list(extra.get("cond_or_uncond", []))
            if not cond_types:
                return q, k, v

            num_chunks = len(cond_types)
            bs = q.shape[0] // num_chunks
            h, w = extra["activations_shape"][-2:]

            sigma = extra.get("sigmas", [0.0])[0] if "sigmas" in extra else 0.0
            left_c, right_c, curr_f, t_val = get_progressive_params(sigma)

            Lk = _lcm_list([left_k.shape[1], right_k.shape[1]])
            Lv = _lcm_list([left_v.shape[1], right_v.shape[1]])

            Qs, Ks, Vs, new_types = [], [], [], []
            q_chunks = q.chunk(num_chunks, 0)
            k_chunks = k.chunk(num_chunks, 0)
            v_chunks = v.chunk(num_chunks, 0)

            for i, t in enumerate(cond_types):
                q_i = q_chunks[i]
                if t == UNCOND:
                    Qs.append(q_i)
                    Ks.append(k_chunks[i])
                    Vs.append(v_chunks[i])
                    new_types.append(UNCOND)
                    continue

                left_k_rep  = left_k.repeat(bs, Lk // left_k.shape[1], 1)
                right_k_rep = right_k.repeat(bs, Lk // right_k.shape[1], 1)
                left_v_rep  = left_v.repeat(bs, Lv // left_v.shape[1], 1)
                right_v_rep = right_v.repeat(bs, Lv // right_v.shape[1], 1)

                # More aggressive separation in cross-attention
                if t_val > convergence:
                    n_tokens = q_i.shape[1]
                    mask_l = make_mask(h, w, left_c,  curr_f, q_i.device, q_i.dtype)[0:1]
                    mask_r = make_mask(h, w, right_c, curr_f, q_i.device, q_i.dtype)[1:2]
                    mask_2 = torch.cat([mask_l, mask_r], dim=0)
                    mask_lr = mask_tokens(mask_2, (h, w), bs, n_tokens).view(2, bs, n_tokens, 1)
                    
                    # Use much stronger spatial separation
                    one = torch.tensor(1.0, device=q_i.device, dtype=q_i.dtype)
                    strength = torch.tensor(token_influence, device=q_i.device, dtype=q_i.dtype)  # FIXED: Add this line
                    separation_strength = torch.tensor(3.0, device=q_i.device, dtype=q_i.dtype)
                    
                    # Apply spatial separation directly without preservation buffers
                    left_q = q_i * (mask_lr[0] + (1.0 - mask_lr[0]) * torch.exp(-strength * separation_strength))
                    right_q = q_i * (mask_lr[1] + (1.0 - mask_lr[1]) * torch.exp(-strength * separation_strength))
                    
                else:
                    left_q = q_i
                    right_q = q_i

                Qs += [left_q, right_q]
                Ks += [left_k_rep, right_k_rep]
                Vs += [left_v_rep, right_v_rep]
                new_types += [COND, COND]

            extra["cond_or_uncond"] = new_types
            return torch.cat(Qs, 0), torch.cat(Ks, 0), torch.cat(Vs, 0)


        def attn2_output_patch(out, extra):
            types = extra.get("cond_or_uncond")
            if not types:
                return out

            h, w = extra["activations_shape"][-2:]
            bs = out.shape[0] // len(types)
            n_tokens = out.shape[1]

            sigma = extra.get("sigmas", [0.0])[0] if "sigmas" in extra else 0.0
            left_c, right_c, curr_f, t_val = get_progressive_params(sigma)

            mask_l = make_mask(h, w, left_c,  curr_f, out.device, out.dtype)[0:1]
            mask_r = make_mask(h, w, right_c, curr_f, out.device, out.dtype)[1:2]
            mask_2 = torch.cat([mask_l, mask_r], dim=0)
            # shape: [2, bs, n_tokens, 1]
            mask_lr = mask_tokens(mask_2, (h, w), bs, n_tokens).view(2, bs, n_tokens, 1)

            # Use sharper decision boundaries
            if t_val > convergence:
                # Much sharper decision boundary
                decision_sharpness = torch.tensor(500.0, device=out.device, dtype=out.dtype)
                mask_left = torch.sigmoid(decision_sharpness * (mask_lr[0] - 0.5))
            else:
                # Binary decision for early steps
                mask_left = (mask_lr[0] > 0.6).to(dtype=out.dtype)  # Higher threshold
                
            mask_right = 1.0 - mask_left

            outputs, idx = [], 0
            for t in types:
                if t == UNCOND:
                    outputs.append(out[idx:idx + bs])
                    idx += bs
                else:
                    left_o  = out[idx:idx + bs]
                    right_o = out[idx + bs:idx + 2 * bs]
                    idx += 2 * bs
                    # Clean merge without bias adjustments - FIXED: removed the problematic bias code
                    blended = left_o * mask_left + right_o * mask_right
                    outputs.append(blended)

            return torch.cat(outputs, 0)

        m.set_model_attn1_patch(attn1_patch)
        m.set_model_attn1_output_patch(attn1_output_patch)
        m.set_model_attn2_patch(attn2_patch)
        m.set_model_attn2_output_patch(attn2_output_patch)

        return (m, left_cond)

NODE_CLASS_MAPPINGS = {"SpatialSplitAttention": SpatialSplitAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"SpatialSplitAttention": "Spatial Split Attention"}