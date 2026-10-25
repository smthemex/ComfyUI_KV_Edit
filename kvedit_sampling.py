# -*- coding: UTF-8 -*-
"""The two KV-Edit ODE drivers, replayed on comfy's native FLUX stack.

Both take the *scaled* clean latent in token space (``[1, L, 64]``), exactly like the
upstream ``flux.sampling.denoise_kv*`` helpers, and return the edited tokens.
"""

import torch

from .kvedit_core import PATCH, pack_tokens, unpack_tokens


# --------------------------------------------------------------------------------------
# 1) inversion-free (upstream ``denoise_kv_inf``, the upstream default)
# --------------------------------------------------------------------------------------

def kv_edit_inversion_free(apply_src, apply_tgt, z0_tok, h, w, sigmas, state,
                           progress=None):
    """Upstream ``Flux_kv_edit_inf.forward`` + ``denoise_kv_inf``.

    One loop, two model calls per step:
        z_src = (1 - t) * z0 + t * noise          (analytic source trajectory)
        z_tar = z_src[edited] - z0[edited] + z_fe (the edited region's own trajectory)
        v_fe  = v_tar - v_src[edited]             (velocity of the edit)
        z_fe += (t_prev - t_curr) * v_fe * mask
    """
    device = z0_tok.device
    edited = state.edited_idx
    tok_mask = state.tok_mask[:, edited, :]
    n = len(sigmas)
    steps = max(n - 1, 0)

    # Upstream creates a fresh ``Generator(device).manual_seed(0)`` *inside* the loop, so
    # every entry of noise_list is the same tensor.  Reproduced verbatim (one draw).
    gen = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(z0_tok.size(), dtype=z0_tok.dtype, layout=z0_tok.layout,
                        device=device, generator=gen)

    z_fe = z0_tok[:, edited, :].clone()

    for i in range(steps):
        t_curr = float(sigmas[i])
        t_prev = float(sigmas[i + 1])

        z_src = (1.0 - t_curr) * z0_tok + t_curr * noise
        z_tar = z_src[:, edited, :] - z0_tok[:, edited, :] + z_fe
        # full token grid: the edited rows carry the target trajectory, the background
        # rows are irrelevant (their K/V is replaced by the cache inside every block).
        full_tar = z_src.clone()
        full_tar[:, edited, :] = z_tar

        state.store = {}
        state.phase = 0
        v_src = apply_src(unpack_tokens(z_src, h, w, PATCH), t_curr)

        state.phase = 1
        v_tar = apply_tgt(unpack_tokens(full_tar, h, w, PATCH), t_curr)

        v_fe = (pack_tokens(v_tar, PATCH)[:, edited, :]
                - pack_tokens(v_src, PATCH)[:, edited, :])
        z_fe = z_fe + (t_prev - t_curr) * v_fe * tok_mask

        state.store = {}
        if progress is not None:
            progress(i + 1, steps)

    return z_fe


# --------------------------------------------------------------------------------------
# 2) real RF inversion + denoise (upstream ``denoise_kv``)
# --------------------------------------------------------------------------------------

def kv_invert(apply_src, z0_tok, h, w, sigmas, state, saved, offload=False,
              progress=None):
    """Upstream ``Flux_kv_edit.inverse``.

    Walks the schedule from t=0 to t=1 adding the velocity (the naive RF inversion),
    and stores, for every step, the K/V of every block *and* the pre-update latent.
    Upstream keys the store with ``t_prev``; we keep integer keys and reuse them.
    """
    ts = sigmas.flip(0)          # ascending: [0.0, s27, ..., s1, 1.0]
    z = z0_tok
    n = len(ts) - 1
    for i in range(n):
        t_curr = float(ts[i])
        t_prev = float(ts[i + 1])
        state.store = {}
        saved[i] = {"z": z.detach().clone(), "kv": state.store}
        state.phase = 0
        v = apply_src(unpack_tokens(z, h, w, PATCH), t_curr)
        if offload:
            for blk in list(state.store.keys()):
                k_img, v_img = state.store[blk]
                state.store[blk] = (k_img.to("cpu", non_blocking=False),
                                    v_img.to("cpu", non_blocking=False))
            saved[i]["z"] = saved[i]["z"].to("cpu", non_blocking=False)
        z = z + (t_prev - t_curr) * pack_tokens(v, PATCH)
        if progress is not None:
            progress(i + 1, n)
    return z


def kv_denoise(apply_tgt, z0_tok, h, w, sigmas_full, skip, state, saved, n_inv,
               re_init=False, seed=0, progress=None):
    """Upstream ``Flux_kv_edit.denoise``.

    Upstream keys the inversion store with ``t_prev`` and looks it up with ``t_curr``.
    Resolving that: denoise iteration j reads the entry produced by inversion iteration
    ``n_inv - 1 - skip - j``, i.e. the source K/V always come from the state **one step
    later** on the schedule.  That off-by-one is reproduced on purpose.
    """
    edited = state.edited_idx
    tok_mask = state.tok_mask[:, edited, :]
    skip = int(skip)
    ts = sigmas_full[skip:]
    n = len(ts) - 1

    if re_init:
        # upstream: noise = randn_like(zt); zt = z0*(1-t) + noise*t ; t = ts[0]
        gen = torch.Generator(device=z0_tok.device).manual_seed(int(seed) & 0xFFFFFFFF)
        noise = torch.randn(z0_tok.size(), dtype=z0_tok.dtype, layout=z0_tok.layout,
                            device=z0_tok.device, generator=gen)
        full = (z0_tok * (1.0 - float(ts[0])) + noise * float(ts[0])).contiguous()
    else:
        entry = saved.get(n_inv - 1 - skip)
        if entry is None:
            raise RuntimeError("empty inversion: nothing was stored")
        full = entry["z"].to(z0_tok.device).clone()

    for j in range(n):
        t_curr = float(ts[j])
        t_prev = float(ts[j + 1])
        entry = saved.get(n_inv - 1 - skip - j)
        if entry is None:
            raise RuntimeError(f"missing inversion cache for step {j}")
        source = entry["z"].to(z0_tok.device)
        # upstream blends the previous source latent back with weight (1 - mask);
        # a no-op for a hard mask, kept for bit-fidelity with soft masks.
        full[:, edited, :] = (source[:, edited, :] * (1.0 - tok_mask)
                              + full[:, edited, :] * tok_mask)

        state.store = entry["kv"]
        state.phase = 1
        v = apply_tgt(unpack_tokens(full, h, w, PATCH), t_curr)

        full = full + (t_prev - t_curr) * pack_tokens(v, PATCH)
        if progress is not None:
            progress(j + 1, n)

    out = z0_tok.clone()
    out[:, edited, :] = (z0_tok[:, edited, :] * (1.0 - tok_mask)
                         + full[:, edited, :] * tok_mask)
    return out[:, edited, :]
