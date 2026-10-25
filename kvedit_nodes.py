# -*- coding: UTF-8 -*-
"""ComfyUI nodes for KV-Edit (native FLUX DiT stack).

The plugin deliberately owns **no** CLIP, no VAE and no sampler of its own:

    Load Diffusion Model -> MODEL ─────────────────────────────────┐
    Load CLIP -> CLIPTextEncode(source prompt) -> CONDITIONING ────┤
    Load CLIP -> CLIPTextEncode(target prompt) -> CONDITIONING ────┤
    Load Image -> VAEEncode -> LATENT (the source image) ──────────┤
    Load Image -> MaskEditor / solid mask      -> MASK ────────────┤
                                                                   ▼
                                                        KV_Edit_Sample -> LATENT
                                                                   ▼
                                                             VAEDecode -> IMAGE

Because CLIP / VAE / DiT live in different nodes, comfy can unload each of them when
the next one needs the VRAM - the old wrapper materialised a second copy of the flux
weights (accelerate load_checkpoint_and_dispatch on top of the comfy model) which is
what blew up a 12 GB card and made the plugin depend on accelerate + huggingface_hub.

All LATENT inputs/outputs follow the stock convention: **raw (unscaled) latents**, the
same space VAEEncode / KSampler / VAEDecode use; the scaling to the DiT's space happens
inside the node via ``model.process_latent_in/out``.
"""

import os

import torch

import comfy.model_management
import comfy.utils

from . import kvedit_core as core
from . import kvedit_sampling as sampling


def _activation_budget(patcher, latent_shape):
    """VRAM to keep free for one forward (comfy's own estimate is tuned for B=1).

    KV-Edit runs the DiT twice per step and additionally holds one step of K/V, so we
    scale comfy's formula and keep a 2 GB floor.
    """
    base_model = patcher.model
    batch = max(1, int(latent_shape[0]))
    area = batch * int(latent_shape[-2]) * int(latent_shape[-1])
    dtype_size = comfy.model_management.dtype_size(base_model.get_dtype_inference())
    factor = getattr(base_model, "memory_usage_factor", 3.1) or 3.1
    est = int(area * dtype_size * 0.01 * factor * (1024 * 1024))
    floor = int(float(os.environ.get("KVEDIT_VRAM_BUDGET", "2")) * (1024 ** 3))
    return max(est * 2, floor)


def _prepare_model(patcher, latent_shape):
    need = _activation_budget(patcher, latent_shape)
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    comfy.model_management.load_models_gpu([patcher], memory_required=need,
                                           minimum_memory_required=need)
    try:
        patcher.pre_run()
    except Exception:
        pass
    return patcher.model, patcher.load_device


def _make_apply(base_model, transformer_options, model_conds):
    def apply_fn(x, t):
        t_tensor = torch.full((x.shape[0],), float(t), device=x.device,
                              dtype=torch.float32)
        return base_model.apply_model(x, t_tensor,
                                      transformer_options=transformer_options,
                                      **model_conds).float()
    return apply_fn


def _block_count(base_model):
    dm = base_model.diffusion_model
    return len(dm.double_blocks) + len(dm.single_blocks)


def _set_guidance(conditioning, guidance):
    """CONDITIONING with the flux guidance value forced (FluxGuidance node optional)."""
    if conditioning is None or len(conditioning) == 0:
        raise ValueError("empty conditioning")
    out = []
    for c in conditioning:
        emb, extra = c[0], c[1]
        d = dict(extra)
        d["guidance"] = float(guidance)
        out.append([emb, d])
    return out


class KV_Edit_Sample:
    """Training-free image editing with precise background preservation (KV-Edit)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A native flux MODEL (UNETLoader / "
                                               "CheckpointLoaderSimple / + LoraLoader). "
                                               "flux1-dev-fp8 works."}),
                "source_conditioning": ("CONDITIONING", {"tooltip": "CLIPTextEncode of the "
                                                                    "prompt that describes the INPUT image."}),
                "target_conditioning": ("CONDITIONING", {"tooltip": "CLIPTextEncode of the "
                                                                    "prompt that describes the DESIRED image."}),
                "latent_image": ("LATENT", {"tooltip": "Raw latent of the source image "
                                                       "(VAEEncode)."}),
                "mask": ("MASK", {"tooltip": "Region to EDIT (white = edited). The "
                                             "complement is preserved pixel-exact."}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200, "step": 1}),
                "skip_step": ("INT", {"default": 1, "min": 0, "max": 30, "step": 1,
                                      "tooltip": "Upstream default 1: drop the noisiest "
                                                 "step of the schedule."}),
                "attn_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                                         "tooltip": "Additive bias on the background keys "
                                                    "during the target pass. 0 disables it. "
                                                    "1.0 = upstream default."}),
                "attn_mask": ("BOOLEAN", {"default": False,
                                          "tooltip": "Upstream `attn_mask`: hard-block "
                                                     "mask<->background attention during the "
                                                     "source pass. Off upstream-default; "
                                                     "it forces a slow attention backend."}),
                "use_inf": ("BOOLEAN", {"default": True,
                                        "tooltip": "Inversion-free path (upstream default). "
                                                   "OFF = a real RF inversion pass first."}),
                "source_guidance": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0,
                                              "step": 0.1}),
                "target_guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0,
                                              "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "display": "number",
                                 "tooltip": "Only used by re_init on the two-pass path; "
                                            "upstream fixes the source-trajectory noise to "
                                            "seed 0 on the inversion-free path."}),
            },
            "optional": {
                "re_init": ("BOOLEAN", {"default": False,
                                        "tooltip": "Two-pass path only: restart the edited "
                                                   "region from fresh noise instead of the "
                                                   "inverted latent."}),
                "kv_offload": ("BOOLEAN", {"default": True,
                                          "tooltip": "Keep the per-step K/V cache on the CPU "
                                                     "(~0.7 GB at 512, ~2.9 GB at 1024). "
                                                     "On by default: on a 12 GB card the GPU "
                                                     "copy does not fit next to the model."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "KV_Edit"
    DESCRIPTION = (
        "KV-Edit (https://github.com/Xilluill/KV-Edit) on comfy's native FLUX DiT.\n"
        "The source pass caches the K/V of the background for every block; the target pass "
        "restores them, so the unmasked area survives the edit exactly and only the masked "
        "region follows the target prompt.\n"
        "Wire: VAEEncode(source image) -> latent_image; MASK = the region to edit -> mask; "
        "then VAEDecode the output. \n"
        "Guidance: set it here (source/target separately) OR with a FluxGuidance node - this "
        "node's values win. With an 8-step turbo LoRA use steps=8 and the flux sigma table "
        "is still driven by the resolution-dependent shift (comfy's default 1.15 is not)."
    )

    def run(self, model, source_conditioning, target_conditioning, latent_image, mask,
            steps, skip_step, attn_scale, attn_mask, use_inf, source_guidance,
            target_guidance, seed, re_init=False, kv_offload=True):

        raw = latent_image["samples"]
        if raw.shape[0] != 1:
            raise ValueError("KV-Edit works on one image at a time (batch must be 1)")
        device = model.load_device
        raw = raw.to(device=device, dtype=torch.float32)
        scaled = model.model.process_latent_in(raw)

        h, w = int(scaled.shape[-2]), int(scaled.shape[-1])
        h_tok, w_tok = core.token_grid(scaled.shape)
        n_img = h_tok * w_tok

        edited, bg, tok = core.mask_to_tokens(mask, h, w)
        if edited.numel() == 0:
            raise ValueError("the mask selects no token - nothing to edit")
        if bg.numel() == 0:
            print("[KV_Edit] warning: the mask covers the whole image, nothing is preserved")

        sigmas = core.get_schedule(int(steps), n_img).to(device)

        state = core.KVState()
        state.edited_idx = edited.to(device)
        state.bg_idx = bg.to(device)
        state.tok_mask = tok.to(device)
        state.use_attn_mask = bool(attn_mask)
        state.attn_scale = float(attn_scale) if float(attn_scale) != 0.0 else None
        # 57 blocks x K,V x 24 heads x L x 128 x 2 B: ~0.7 GB/step at 512, ~2.9 GB at 1024
        state.kv_device = torch.device("cpu") if kv_offload else None
        state.enabled = True

        n_blocks = _block_count(model.model)
        kv_mem = core.kv_bytes(n_img, n_blocks)
        print(f"[KV_Edit] {h_tok}x{w_tok} tokens | {core.describe(steps, n_img)} | "
              f"edited tokens={edited.numel()} ({100.0 * edited.numel() / n_img:.1f}%) | "
              f"K/V cache {kv_mem / 2**30:.2f}GB/step x {n_blocks} blocks")

        transformer_options = core.build_transformer_options(model, state)
        base_model, device = _prepare_model(model, scaled.shape)

        noise_shape = torch.zeros_like(scaled)
        conds_src = core.rows_to_model_conds(
            base_model, core._cond_rows(_set_guidance(source_conditioning, source_guidance)),
            noise_shape, device)
        conds_tgt = core.rows_to_model_conds(
            base_model, core._cond_rows(_set_guidance(target_conditioning, target_guidance)),
            noise_shape, device)
        apply_src = _make_apply(base_model, transformer_options, conds_src)
        apply_tgt = _make_apply(base_model, transformer_options, conds_tgt)

        z0_tok = core.pack_tokens(scaled, core.PATCH)
        pbar = comfy.utils.ProgressBar(max(1, int(steps)))

        with comfy.model_management.cuda_device_context(device):
            with core.raw_velocity(base_model):
                if use_inf:
                    ts = sigmas[int(skip_step):]
                    z_fe = sampling.kv_edit_inversion_free(
                        apply_src, apply_tgt, z0_tok, h, w, ts, state,
                        progress=lambda i, n: pbar.update_absolute(i, n))
                else:
                    saved = {}
                    sampling.kv_invert(apply_src, z0_tok, h, w, sigmas, state, saved,
                                       offload=bool(kv_offload),
                                       progress=lambda i, n: pbar.update_absolute(i, n))
                    z_fe = sampling.kv_denoise(
                        apply_tgt, z0_tok, h, w, sigmas, skip_step, state, saved,
                        n_inv=len(sigmas) - 1, re_init=bool(re_init), seed=seed,
                        progress=lambda i, n: pbar.update_absolute(i, n))
                    saved.clear()

        out_tok = z0_tok.clone()
        out_tok[:, state.edited_idx, :] = z_fe
        out = model.model.process_latent_out(core.unpack_tokens(out_tok, h, w, core.PATCH))
        # composite onto the input latent: the untouched area is then *bit-identical*,
        # not merely "a raw->scaled->raw round trip of it" (that costs 1 fp32 ulp).
        pm = core.pixel_mask_from_tokens(tok, h_tok, w_tok).to(device=out.device,
                                                                dtype=out.dtype)
        out = raw * (1.0 - pm) + out * pm

        print(f"[KV_Edit] done | patch fired {state.n_fire}x")
        state.reset()
        state.enabled = False
        del state
        return ({"samples": out.to(device="cpu")},)


NODE_CLASS_MAPPINGS = {
    "KV_Edit_Sample": KV_Edit_Sample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KV_Edit_Sample": "KV_Edit_Sample (native DiT)",
}
