import os
from dataclasses import dataclass
import gc
from shutil import copy
import torch
import torch.nn as nn 
from einops import rearrange
from huggingface_hub import hf_hub_download
#from imwatermark import WatermarkEncoder
from safetensors.torch import load_file as load_sft
from .modules.layers import MLPEmbedder,DoubleStreamBlock_kv,DoubleStreamBlock,SingleStreamBlock,SingleStreamBlock_kv
from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder
import comfy.model_management
import json
import types  # 新增导入
import torch.nn.functional as F  # 确保F.interpolate可用
@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None

configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


# 修改加载函数
def load_flux_model(ckpt_path, device, flux_cls=Flux):
   
    with torch.device("meta" if ckpt_path is not None else device):
        model = flux_cls(configs["flux-dev"].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        if "fp8" in ckpt_path:
            from optimum.quanto import requantize
            import folder_paths
            json_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI_KV_Edit/flux/config.json") #config is for pass block
            with open(json_path,'r') as f:
                quantization_map = json.load(f)
            print(f"Start fp8 requantization process...")
            requantize(model, sd, quantization_map, device=device)
            print("Model is requantized!")
        else:
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        del sd
        torch.cuda.empty_cache()
        #print_load_warning(missing, unexpected)

    return model


# def load_flux_model_cf(cf_model,device,  flux_cls=Flux):
#     original = cf_model.model.diffusion_model
    
#     # 保持原参数引用（不复制）
#     params = original.params
#     hidden_size = original.hidden_size
#     num_heads = original.num_heads
    
#     # 动态创建新模块（保持量化参数）
#     new_blocks = nn.ModuleList([
#         DoubleStreamBlock_kv(
#             hidden_size,
#             num_heads,
#             mlp_ratio=params.mlp_ratio,
#             qkv_bias=params.qkv_bias
#         ) for _ in range(params.depth)
#     ])
    
#     # 参数嫁接（仅替换block参数）
#     for new_block, old_block in zip(new_blocks, original.double_blocks):
#         new_block.load_state_dict(old_block.state_dict(), assign=True)
    
#     # 原子替换（保持原模型内存布局）
#     original.__class__ = flux_cls
#     original.double_blocks = new_blocks
#     original.single_blocks = nn.ModuleList([
#         SingleStreamBlock_kv(hidden_size, num_heads, params.mlp_ratio)
#         for _ in range(params.depth_single_blocks)
#     ])
    
#     return original

def load_flux_model_(cf_model, device, flux_cls=Flux):
    original_sd = cf_model.model.diffusion_model.state_dict()
    del cf_model
    gc.collect()
    new_model = flux_cls(configs["flux-dev"].params).to(torch.bfloat16)
    new_model.load_state_dict(original_sd, strict=False)
    del original_sd
    gc.collect()
    torch.cuda.empty_cache()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    #del cf_model.model.diffusion_model
    torch.cuda.empty_cache()
    return new_model

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True, flux_cls=Flux) -> Flux:
    # Loading Flux
    print("Init model")
    
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device("meta" if ckpt_path is not None else device):
        model = flux_cls(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        del sd
        torch.cuda.empty_cache()
        #print_load_warning(missing, unexpected)
    return model


def load_t5_(repo,device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    # return HFEmbedder("black-forest-labs/FLUX.1-dev", max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)
    return HFEmbedder(repo, max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)


def load_clip_(repo,device: str | torch.device = "cuda") -> HFEmbedder:
    # return HFEmbedder("black-forest-labs/FLUX.1-dev", max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)
    return HFEmbedder('F:/test/ComfyUI/models/diffusers/black-forest-labs/FLUX.1-dev', max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)

def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    # return HFEmbedder("black-forest-labs/FLUX.1-dev", max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    # return HFEmbedder("black-forest-labs/FLUX.1-dev", max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)


def load_ae(ckpt_path: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    # ckpt_path = configs[name].ae_path
    # if (
    #     ckpt_path is None
    #     and configs[name].repo_id is not None
    #     and configs[name].repo_ae is not None
    #     and hf_download
    # ):
    #     ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    #with torch.device("meta" if ckpt_path is not None else device):
    ae = AutoEncoder(configs["flux-dev"].ae_params).to(torch.bfloat16)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device="cpu")
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        #print_load_warning(missing, unexpected)
        del sd
        torch.cuda.empty_cache()
    return ae

def load_ae_cf(ckpt, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    # ckpt_path = configs[name].ae_path
    # if (
    #     ckpt_path is None
    #     and configs[name].repo_id is not None
    #     and configs[name].repo_ae is not None
    #     and hf_download
    # ):
    #     ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    #with torch.device("meta" if ckpt_path is not None else device):
    ae = AutoEncoder(configs["flux-dev"].ae_params).to(torch.bfloat16)

    if ckpt is not None:
        sd = ckpt.get_sd()
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        #print_load_warning(missing, unexpected)
        del sd
        torch.cuda.empty_cache()
    return ae

def load_ae_(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        #print_load_warning(missing, unexpected)
        del sd
        torch.cuda.empty_cache()
    return ae

# class WatermarkEmbedder:
#     def __init__(self, watermark):
#         self.watermark = watermark
#         self.num_bits = len(WATERMARK_BITS)
#         self.encoder = WatermarkEncoder()
#         self.encoder.set_watermark("bits", self.watermark)

#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Adds a predefined watermark to the input image

#         Args:
#             image: ([N,] B, RGB, H, W) in range [-1, 1]

#         Returns:
#             same as input but watermarked
#         """
#         image = 0.5 * image + 0.5
#         squeeze = len(image.shape) == 4
#         if squeeze:
#             image = image[None, ...]
#         n = image.shape[0]
#         image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
#         # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
#         # watermarking libary expects input as cv2 BGR format
#         for k in range(image_np.shape[0]):
#             image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
#         image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
#             image.device
#         )
#         image = torch.clamp(image / 255, min=0.0, max=1.0)
#         if squeeze:
#             image = image[0]
#         image = 2 * image - 1
#         return image


# # A fixed 48-bit message that was chosen at random
# WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# # bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
# WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
# #embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
