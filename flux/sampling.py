import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux,Flux_kv
from .modules.conditioner import HFEmbedder
from tqdm import tqdm
from tqdm.contrib import tzip

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    
def prepare(clip,device,img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # if isinstance(prompt, str):
    #     prompt = [prompt]
    
    # txt = t5(prompt)
    clip.tokenizer.min_length=512
    tokens = clip.tokenize(prompt)
    tokens["t5xxl"] = clip.tokenize(prompt)["t5xxl"]
    txt = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True).pop("cond")


    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    #vec = clip(prompt)
    tokens["l"] = clip.tokenize(prompt)["l"]
    vec = clip.encode_from_tokens(tokens,  return_dict=True).pop("pooled_output")

    #print(vec.shape, img.shape, txt.shape, img_ids.shape, txt_ids.shape) # torch.Size([1, 768]) torch.Size([1, 1024, 64]) torch.Size([1, 512, 4096]) torch.Size([1, 1024, 3]) torch.Size([1, 512, 3])
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img.to(device,torch.bfloat16),
        "img_ids": img_ids.to(device,torch.bfloat16),
        "txt": txt.to(device,torch.bfloat16),
        "txt_ids": txt_ids.to(device,torch.bfloat16),
        "vec": vec.to(device,torch.bfloat16),
    }

def prepare_(t5: HFEmbedder, clip: HFEmbedder,device, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
    #print(img.shape,vec.shape, txt.shape, img_ids.shape) #torch.Size([1, 1024, 64]) torch.Size([1, 768]) torch.Size([1, 512, 4096]) torch.Size([1, 1024, 3])
    return {
        "img": img.to(device,torch.bfloat16),
        "img_ids": img_ids.to(device,torch.bfloat16),
        "txt": txt.to(device,torch.bfloat16),
        "txt_ids": txt_ids.to(device,torch.bfloat16),
        "vec": vec.to(device,torch.bfloat16),
    }

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def denoise_kv(
    model: Flux_kv,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):

    if inverse:
        timesteps = timesteps[::-1]
        
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    for i, (t_curr, t_prev) in enumerate(tzip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        
        if inverse:
            img_name = str(info['t']) + '_' + 'img'
            info['feature'][img_name] = img.cpu()
        else:
            img_name = str(info['t']) + '_' + 'img'
            source_img = info['feature'][img_name].to(img.device)
            img = source_img[:, info['mask_indices'],...] * (1 - info['mask'][:, info['mask_indices'],...]) + img * info['mask'][:, info['mask_indices'],...]
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        img = img + (t_prev - t_curr) * pred
    return img, info

def denoise_kv_inf(
    model: Flux_kv,
    # model input
    img: Tensor,
    img_ids: Tensor,
    source_txt: Tensor,
    source_txt_ids: Tensor,
    source_vec: Tensor,
    target_txt: Tensor,
    target_txt_ids: Tensor,
    target_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    target_guidance: float = 4.0,
    source_guidance: float = 4.0,
    info: dict = {},
):
        
    target_guidance_vec = torch.full((img.shape[0],), target_guidance, device=img.device, dtype=img.dtype)
    source_guidance_vec = torch.full((img.shape[0],), source_guidance, device=img.device, dtype=img.dtype)
    
    mask_indices = info['mask_indices']
    init_img = img.clone() 
    z_fe = img[:, mask_indices,...]
    
    noise_list = []
    for i in range(len(timesteps)):
        noise = torch.randn(init_img.size(), dtype=init_img.dtype, 
                        layout=init_img.layout, device=init_img.device,
                        generator=torch.Generator(device=init_img.device).manual_seed(0)) 
        noise_list.append(noise)

    for i, (t_curr, t_prev) in enumerate(tzip(timesteps[:-1], timesteps[1:])): 
        
        info['t'] = t_curr
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        z_src = (1 - t_curr) * init_img + t_curr * noise_list[i]
        z_tar = z_src[:, mask_indices,...] - init_img[:, mask_indices,...] + z_fe
        
        info['inverse'] = True
        info['feature'] = {}
        v_src = model(
            img=z_src,
            img_ids=img_ids,
            txt=source_txt,
            txt_ids=source_txt_ids,
            y=source_vec,
            timesteps=t_vec,
            guidance=source_guidance_vec,
            info=info
        )
        
        info['inverse'] = False
        v_tar = model(
            img=z_tar,
            img_ids=img_ids,
            txt=target_txt,
            txt_ids=target_txt_ids,
            y=target_vec,
            timesteps=t_vec,
            guidance=target_guidance_vec,
            info=info
        )
    
        v_fe = v_tar - v_src[:, mask_indices,...]
        z_fe = z_fe + (t_prev - t_curr) * v_fe * info['mask'][:, mask_indices,...]
    return z_fe, info
