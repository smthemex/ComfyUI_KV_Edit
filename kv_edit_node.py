# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from PIL import Image
import torch
import numpy as np
from .flux.sampling import prepare
from einops import rearrange
from .flux.util import load_ae_cf
from .node_utils import cleanup,pil2narry
from .gradio_kv_edit import FluxEditor_kv_Wrapper
from .flux.kv_edit import Flux_kv_edit_inf,Flux_kv_edit
from .gradio_kv_edit_inf import FluxEditor_kv_Wrapper_inf,SamplingOptions
from comfy import model_management
import comfy.model_management
import folder_paths
MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


class KV_Edit_Load:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet": (["none"]+folder_paths.get_filename_list("diffusion_models"),),
                "offload":("BOOLEAN",{"default":True}),
                "use_inf":("BOOLEAN",{"default":True}),
                },
            "optional": { "model":("MODEL",),},
            }

    RETURN_TYPES = ("MODEL_KVEDIT", )
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "KV_Edit"

    def main(self, unet,offload,use_inf,**kwargs):
        cf_model=kwargs.get("model")
        if cf_model is None:
            if unet=="none":
                raise Exception("Please select a model")
            else:
                cf_model=folder_paths.get_full_path("diffusion_models", unet)
        device_val = "cpu" if offload else device
        if not use_inf:
            model=Flux_kv_edit(device_val,cf_model,'flux-dev')
            if not isinstance(cf_model, str):
                del cf_model
                cleanup()
            pipeline=FluxEditor_kv_Wrapper(offload,device)
        else:
            model=Flux_kv_edit_inf(device_val,cf_model,'flux-dev')
            if not isinstance(cf_model, str):
                del cf_model
                cleanup()
            pipeline=FluxEditor_kv_Wrapper_inf(offload,device)
        
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        cleanup()
        return ({"pipeline":pipeline,"model":model,"use_inf":use_inf},)

    
class KV_Edit_PreData:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip":("CLIP",),
                "vae":("VAE",),
                "image": ("IMAGE", ), 
                "mask":("MASK",),# B H W 
                "source_prompt": ("STRING", {"default": "in a cluttered wooden cabin,a workbench holds a green neon sign that reads 'I love nana'.", "multiline": True,"tooltip": "The source_prompt to be encoded."}),
                "target_prompt": ("STRING", {"default": "in a cluttered wooden cabin,a workbench holds a green neon sign that reads 'I love here'.", "multiline": True,"tooltip": "The target_prompt to be encoded."}),
                 },
            
            }

    RETURN_TYPES = ("CONDITION_KV", )
    RETURN_NAMES = ("condition",)
    FUNCTION = "main"
    CATEGORY = "KV_Edit"


    @torch.inference_mode()
    def encode(self,init_image,ae, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device).to(torch.bfloat16)
        ae.encoder.to(torch_device)

        init_image = ae.encode(init_image).to(torch.bfloat16)
        return init_image
    
    def main(self,clip, vae,image,mask,source_prompt,target_prompt,):
        ae=load_ae_cf(vae).to(device=device,dtype=torch.bfloat16)

        # encode image
        np_image=image.squeeze().mul(255).clamp(0, 255).byte().numpy()
        shape = np_image.shape 
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        np_image = np_image[:height, :width, :]
        latent_image = self.encode(np_image,ae, device).to(device)
        #print(latent_image.shape) #torch.Size([1, 16, 64, 64])

        width,height = latent_image.shape[3]*8,latent_image.shape[2]*8

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        # mask crop
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = mask[:, 0:height, 0:width]
        mask = out.unsqueeze(0).to(device)
        mask[mask > 0] = 1   
        mask=mask.to(torch.bfloat16)
        #print(mask.shape) #torch.Size([1, 1, 512, 512])
      
        with torch.no_grad():
            if clip is None:
                raise Exception("clip is None")
            inp = prepare(clip,device,latent_image, prompt=source_prompt)
            inp_target = prepare(clip,device, latent_image, prompt=target_prompt)
        
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        cleanup()
       
        return ({"inp":inp,"inp_target":inp_target,"mask":mask,"source_prompt":source_prompt,"target_prompt":target_prompt,"size":(width,height),"ae":ae},)

class KV_Edit_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_KVEDIT",),
                "condition": ("CONDITION_KV",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "step": 1, "display": "number"}),
                "inversion_steps": ("INT", {"default": 28, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "inversion_guidance": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "denoise_steps": ("INT", {"default": 28, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "denoise_guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "skip_step":("INT", {"default": 1, "min": 0, "max":30 ,"step": 1, "display": "number"}),
                "attn_scale":("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "re_init":("BOOLEAN",{"default":False}),
                "attn_mask":("BOOLEAN",{"default":False}),},
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "KV_Edit"

    def main(self,model, condition,seed,inversion_steps,inversion_guidance,denoise_steps,denoise_guidance,skip_step,attn_scale,re_init,attn_mask):

        use_inf=model.get("use_inf")
        pipeline=model.get("pipeline")
        model_int=model.get("model")

        inp=condition.get("inp")
        inp_target=condition.get("inp_target")
        mask=condition.get("mask")
        ae=condition.get("ae")
        
        opts = SamplingOptions(
            source_prompt=condition.get("source_prompt"),
            target_prompt=condition.get("target_prompt"),
            width=condition.get("size")[0],
            height=condition.get("size")[1],
            inversion_num_steps=inversion_steps,
            denoise_num_steps=denoise_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
            attn_scale=attn_scale
        )

        # inverse
        if use_inf:
            print("start edit")
            try:
                output_latent=pipeline.edit(model_int,opts,inp,inp_target,mask)
            except model_management.OOM_EXCEPTION:
                print("get OOM,try again ")
                output_latent=pipeline.edit(model_int,opts,inp,inp_target,mask)
        else:
            try:
                print("start inverse")
                if attn_mask:
                    mask_inverse = mask
                else:
                    mask_inverse=None
                opts.skip_step=0
                pipeline.inverse(model_int,opts,mask_inverse,inp)
                print("inverse done,start edit")
                #  edit
                opts.skip_step=skip_step
                output_latent=pipeline.edit(opts,inp_target,mask)
            except model_management.OOM_EXCEPTION:
                if attn_mask:
                    mask_inverse = mask
                else:
                    mask_inverse=None
                opts.skip_step=0
                pipeline.inverse(model_int,opts,mask_inverse,inp)
                print("inverse done,start edit")
                #  edit
                opts.skip_step=skip_step
                output_latent=pipeline.edit(opts,inp_target,mask)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x = ae.decode(output_latent.to(dtype=torch.bfloat16))
        
        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
       
        # out = {}
        # out["samples"] = output_latent
        cleanup()
        return (pil2narry(img),)


NODE_CLASS_MAPPINGS = {
    "KV_Edit_Load": KV_Edit_Load,
    "KV_Edit_PreData": KV_Edit_PreData,
    "KV_Edit_Sampler":KV_Edit_Sampler,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KV_Edit_Load": "KV_Edit_Load",
    "KV_Edit_PreData": "KV_Edit_PreData",
    "KV_Edit_Sampler":"KV_Edit_Sampler",

}
