# -*- coding: UTF-8 -*-
"""KV-Edit (https://github.com/Xilluill/KV-Edit) re-implemented on comfy's *native*
FLUX DiT stack.

Upstream ships its own copy of the BFL flux model (``flux/model.py`` +
``flux/modules/layers.py``) and mods two things:

  1. ``DoubleStreamBlock_kv`` / ``SingleStreamBlock_kv``: during the *source* pass the
     image part of K and V is cached per block; during the *target* pass the cached
     K/V rows of the **background** are restored (while the edited region keeps the
     fresh K/V), which is what makes the background pixel-exact.
  2. two ODE drivers: an inversion-free one (``denoise_kv_inf``, upstream default) and
     a real RF-inversion one (``denoise_kv`` + ``Flux_kv_edit``).

This module re-implements both against comfy's own ``comfy.ldm.flux`` so the plugin
needs **no diffusers, no transformers, no accelerate, no huggingface_hub**.

Why a full token sequence works here (the key porting decision)
--------------------------------------------------------------
Upstream feeds the target pass **only the edited tokens** (``L_mask`` rows) and, inside
every block, splices their K/V into the cached source K/V.  comfy's ``Flux._forward``
always builds a rectangular token grid from a ``[B, C, H, W]`` latent, so we cannot feed
a token subset.  We therefore feed the **whole sequence** and, in the target phase,
overwrite the *background* rows of K/V with the cached ones.  That is exactly
equivalent:

  * attention is only ever read at the text rows and at the edited rows - the
    K/V tensors they see are ``[txt | cached_bg | fresh_mask]`` in both stacks;
  * the per-row projections / MLP / modulation are applied independently to each row,
    so the discarded background rows never leak into the edited rows;
  * RoPE: because upstream's target sequence is a subset, ``Flux_kv.forward`` has to
    slice the full-grid rope down to the mask positions (``info['pe_mask']``, the
    [txt | edited tokens] slice) and feed that instead of the plain ``pe``.  With the
    full grid the stock RoPE is already the right position for every row, so that
    whole slice disappears;
  * the token grid (``h/16 x w/16``, row major) and the RoPE id layout are identical
    in the two stacks - no index remapping.

The upstream quirks that are *reproduced on purpose* (do not "fix" them):
  * the source-trajectory noise is drawn from ``Generator(device).manual_seed(0)``
    **inside** the loop -> every entry of ``noise_list`` is the same tensor, and the
    node's ``seed`` has no effect in the inversion-free path;
  * in the two-pass path the K/V cache is stored under the key of ``t_prev`` during
    inversion but looked up under ``t_curr`` during denoising -> the source K/V of one
    step is the one produced *one step later* on the schedule;
  * the saved source latent of the previous step is blended back with weight
    ``(1 - mask)`` before every denoise step (a no-op for a hard mask).
"""

import math

import torch

import comfy.model_management

PATCH = 2
BASE_SEQ = 256
MAX_SEQ = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


# --------------------------------------------------------------------------------------
# schedule - verbatim upstream get_schedule()
# --------------------------------------------------------------------------------------

def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def get_schedule(num_steps, image_seq_len, shift=True):
    """Upstream ``flux.sampling.get_schedule``.

    Note that ``linspace(1, 0, n+1)`` + shift is bit-identical to diffusers'
    ``linspace(1, 1/n, n)`` + shift + terminal 0, i.e. this *is* the official FLUX
    sigma table - with a **resolution dependent** mu, which comfy's
    ``ModelSamplingFlux`` (hardcoded shift=1.15) is not.  So we drive it ourselves.
    """
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=BASE_SHIFT, y2=MAX_SHIFT)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps


def describe(steps, tokens):
    mu = get_lin_function(y1=BASE_SHIFT, y2=MAX_SHIFT)(tokens)
    return f"steps={steps} tokens={tokens} mu={mu:.4f} shift={math.exp(mu):.3f}"


# --------------------------------------------------------------------------------------
# token <-> latent
# --------------------------------------------------------------------------------------

def pack_tokens(x, patch_size=PATCH):
    """[B, C, H, W] -> [B, L, C*p*p], the inverse of comfy's rearrange in Flux._forward."""
    b, c, h, w = x.shape
    p = patch_size
    return (x.view(b, c, h // p, p, w // p, p)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(b, (h // p) * (w // p), c * p * p))


def unpack_tokens(t, h, w, patch_size=PATCH):
    """[B, L, C*p*p] -> [B, C, H, W]."""
    b, l, d = t.shape
    p = patch_size
    c = d // (p * p)
    return (t.view(b, h // p, w // p, c, p, p)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(b, c, h, w))


def token_grid(latent_shape, patch_size=PATCH):
    h = int(latent_shape[-2])
    w = int(latent_shape[-1])
    return ((h + patch_size // 2) // patch_size, (w + patch_size // 2) // patch_size)


# --------------------------------------------------------------------------------------
# mask -> token indices (upstream semantics: bilinear -> >0 -> 16ch -> patchify)
# --------------------------------------------------------------------------------------

def mask_to_tokens(mask, h_lat, w_lat, patch_size=PATCH):
    """comfy MASK [B,H,W] in [0,1] -> (edited_idx, background_idx, token_mask[1,L,C*p*p]).

    Upstream resizes the mask with **bilinear** to the latent resolution and keeps every
    non-zero pixel (``mask[mask > 0] = 1``), then repeats it to 16 channels and
    patchifies.  A token counts as edited when at least one of its 4 sub-pixels is > 0.
    The per-channel token mask is kept because upstream scales the velocity update with
    it channel-wise.
    """
    m = mask
    if m is None:
        raise ValueError("mask is required")
    if m.ndim == 4:                      # [B, 1, H, W] is accepted too
        m = m[:, 0]
    if m.ndim == 2:
        m = m.unsqueeze(0)
    m = m[0].detach().float().cpu()
    # crop to what the latent covers (upstream crops the image to a multiple of 16)
    m = m[: h_lat * 8, : w_lat * 8]
    m = (m > 0).float().unsqueeze(0).unsqueeze(0)
    m = torch.nn.functional.interpolate(m, size=(h_lat, w_lat), mode="bilinear",
                                        align_corners=False)
    m = (m > 0).float()
    m = m.repeat(1, 16, 1, 1)
    tok = pack_tokens(m, patch_size)                 # [1, L, 64]
    keep = tok.sum(-1)[0] > 0.5                      # [L]
    edited = torch.nonzero(keep, as_tuple=False).reshape(-1)
    bg = torch.nonzero(~keep, as_tuple=False).reshape(-1)
    return edited, bg, tok


def pixel_mask_from_tokens(tok, h_tok, w_tok, patch_size=PATCH):
    """[h,w] float mask, 1 on the pixels a *edited* token owns.

    Used to composite the result back onto the input latent, so the untouched area
    stays **bit-identical** instead of surviving a raw->scaled->raw round trip
    (that round trip costs 1 fp32 ulp, ~9.5e-07 at the magnitude flux latents live
    at, and it would accumulate over chained edits).
    """
    keep = (tok[0].sum(-1) > 0.5).view(h_tok, w_tok).float()
    return torch.kron(keep, torch.ones((patch_size, patch_size), device=keep.device,
                                       dtype=keep.dtype))


# --------------------------------------------------------------------------------------
# the attention patch
# --------------------------------------------------------------------------------------

class KVState:
    """Mutable per-run state read by the ``attn1_patch``.

    ``phase`` 0 = source pass (cache the image K/V), 1 = target pass (restore the
    background rows of K/V from the cache).
    """

    def __init__(self):
        self.enabled = False
        self.phase = 0
        self.store = {}            # (block_type, block_index) -> (k_img, v_img)
        self.vault = {}            # timestep key -> store   (two-pass mode only)
        self.edited_idx = None     # token indices of the edited region
        self.bg_idx = None         # token indices that keep the source K/V
        self.use_attn_mask = False
        self.attn_scale = None     # None or a float
        self.kv_device = None      # where the cached K/V live
        self.n_fire = 0
        self._cache = {}

    def reset(self):
        self.store = {}
        self.vault = {}
        self.n_fire = 0
        self._cache = {}

    def attn_mask_for(self, txt_len, n_img, device):
        key = ("mask", txt_len, n_img)
        hit = self._cache.get(key)
        if hit is None:
            s = txt_len + n_img
            am = torch.zeros(s, s, dtype=torch.bool, device=device)
            am[:txt_len, :] = True
            t = torch.arange(txt_len, device=device)
            if self.edited_idx is not None and self.edited_idx.numel():
                si = txt_len + self.edited_idx
                am[si.unsqueeze(1), si.unsqueeze(0)] = True
                am[si.unsqueeze(1), t.unsqueeze(0)] = True
            if self.bg_idx is not None and self.bg_idx.numel():
                sb = txt_len + self.bg_idx
                am[sb.unsqueeze(1), sb.unsqueeze(0)] = True
                am[sb.unsqueeze(1), t.unsqueeze(0)] = True
            hit = am.unsqueeze(0)
            self._cache[key] = hit
        return hit

    def attn_scale_for(self, txt_len, n_img, dtype, device):
        key = ("scale", txt_len, n_img, dtype)
        hit = self._cache.get(key)
        if hit is None:
            s = txt_len + n_img
            a = torch.zeros(1, 1, s, dtype=dtype, device=device)
            if self.bg_idx is not None and self.bg_idx.numel():
                a[0, 0, txt_len + self.bg_idx] = float(self.attn_scale)
            hit = a
            self._cache[key] = hit
        return hit


def make_attn_patch(state: KVState):
    """comfy ``transformer_options["patches"]["attn1_patch"]`` callable.

    Called by every flux block (19 double + 38 single) with q/k/v of shape
    ``[B, heads, seq, head_dim]``; the sequence is ``[text | image]`` in both stacks.
    """

    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        if not state.enabled:
            return {}
        eo = extra_options or {}
        img_slice = eo.get("img_slice", None)
        if img_slice is None:
            return {}
        txt_len = int(img_slice[0])
        n_img = int(k.shape[2]) - txt_len
        blk = (eo.get("block_type", "?"), int(eo.get("block_index", 0)))
        out = {}

        if state.phase == 0:
            # ---- source pass: remember the image rows of K/V ----------------------
            k_img = k[:, :, txt_len:, :]
            v_img = v[:, :, txt_len:, :]
            if state.kv_device is not None and k.device != state.kv_device:
                k_img = k_img.to(state.kv_device)
                v_img = v_img.to(state.kv_device)
            else:
                # clone so the (much bigger) text part of the cat() can be freed
                k_img = k_img.clone()
                v_img = v_img.clone()
            state.store[blk] = (k_img, v_img)
            if state.use_attn_mask and state.bg_idx is not None and state.bg_idx.numel():
                out["attn_mask"] = state.attn_mask_for(txt_len, n_img, q.device)
        else:
            # ---- target pass: put the cached background rows back -----------------
            entry = state.store.get(blk)
            if entry is not None and state.bg_idx is not None and state.bg_idx.numel():
                src_k, src_v = entry
                dev = k.device
                # select the background rows *where the cache lives*: with a CPU cache
                # only the rows we actually need cross the bus (a 1024x1024 image is
                # 2.9 GB of K/V per step - too much for a 12 GB card).
                bi = state.bg_idx.to(src_k.device)
                sel_k = src_k.index_select(2, bi).to(dev)
                sel_v = src_v.index_select(2, bi).to(dev)
                idx = txt_len + state.bg_idx.to(dev)
                k = k.index_copy(2, idx, sel_k)
                v = v.index_copy(2, idx, sel_v)
                state.n_fire += 1
                out["k"] = k
                out["v"] = v
            if state.attn_scale:
                out["attn_mask"] = state.attn_scale_for(txt_len, n_img, q.dtype, q.device)
        return out

    return patch


def build_transformer_options(patcher, state: KVState):
    """The patcher's own options + our patch (so other hooks keep working)."""
    base = {}
    model_options = getattr(patcher, "model_options", None) or {}
    own = model_options.get("transformer_options", None) or {}
    if own:
        base = dict(own)
    patches = dict(base.get("patches", {}) or {})
    patches["attn1_patch"] = list(patches.get("attn1_patch", []) or []) + [make_attn_patch(state)]
    base["patches"] = patches
    return base


# --------------------------------------------------------------------------------------
# raw velocity proxy (comfy's calculate_denoised returns x0 = x - sigma*v)
# --------------------------------------------------------------------------------------

class RawVelocitySampling(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def calculate_input(self, sigma, noise):
        return self.base.calculate_input(sigma, noise)

    def timestep(self, sigma):
        return self.base.timestep(sigma)

    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output.float()

    def percent_to_sigma(self, percent):
        return self.base.percent_to_sigma(percent)

    def sigma(self, timestep):
        return self.base.sigma(timestep)

    @property
    def sigma_min(self):
        return self.base.sigma_min

    @property
    def sigma_max(self):
        return self.base.sigma_max


class raw_velocity:
    """Context manager: swap the model's model_sampling for the raw-velocity proxy."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.orig = None

    def __enter__(self):
        self.orig = self.base_model.model_sampling
        self.base_model.model_sampling = RawVelocitySampling(self.orig)
        return self.base_model

    def __exit__(self, *args):
        self.base_model.model_sampling = self.orig
        self.orig = None
        return False


# --------------------------------------------------------------------------------------
# conditioning helpers
# --------------------------------------------------------------------------------------

def _cond_rows(conditioning):
    """comfy CONDITIONING ([[emb, dict], ...]) -> a cond row dict for extra_conds."""
    if conditioning is None or len(conditioning) == 0:
        raise ValueError("empty conditioning input")
    emb, extra = conditioning[0][0], conditioning[0][1]
    row = dict(extra)
    row["cross_attn"] = emb
    return row


def rows_to_model_conds(base_model, row, noise, device):
    d = dict(row)
    d.setdefault("noise", noise)
    d.setdefault("device", device)
    return comfy.samplers.cond_cat([base_model.extra_conds(**d)])


def kv_bytes(n_img, n_blocks, heads=24, head_dim=128, itemsize=2):
    return n_blocks * 2 * heads * n_img * head_dim * itemsize
