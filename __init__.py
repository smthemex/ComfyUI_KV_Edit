# -*- coding: UTF-8 -*-
"""ComfyUI_KV_Edit - native-diT rewrite.

The old wrapper (``kv_edit_node.py`` + ``flux/`` + ``gradio_*.py``) is still on disk for
reference but is **not imported any more**: it built a private copy of the FLUX model
(next to the one comfy already had), loaded it through ``accelerate`` and reached into
``huggingface_hub``, so every diffusers / transformers / accelerate bump could break it
and it doubled the VRAM.

The native path needs nothing beyond torch: the diffusion model, the CLIP, the VAE and
the sampler all stay in their own nodes.
"""

from .kvedit_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
