from dataclasses import dataclass
from .util import load_flux_model_,load_flux_model
from einops import rearrange,repeat
import torch
import torch.nn.functional as F
from torch import Tensor

from .sampling import get_schedule, unpack,denoise_kv,denoise_kv_inf
from .util import load_flux_model_,load_flux_model
from .model import Flux_kv

# @dataclass
# class SamplingOptions:
#     source_prompt: str = ''
#     target_prompt: str = ''
#     # prompt: str
#     width: int = 1366
#     height: int = 768
#     inversion_num_steps: int = 0
#     denoise_num_steps: int = 0
#     skip_step: int = 0
#     inversion_guidance: float = 1.0
#     denoise_guidance: float = 1.0
#     seed: int = 42
#     re_init: bool = False
#     attn_mask: bool = False

class only_Flux(torch.nn.Module): 
    def __init__(self, device,model,name='flux-dev'):
        super().__init__()
        self.device = device
        self.name = name
        self.is_dev = self.name != "flux-schnell"
        #self.model = load_flow_model(self.name, device=self.device,flux_cls=Flux_kv)
        if isinstance(model, str):
            self.model = load_flux_model(model,self.device, flux_cls=Flux_kv)
        else:
            self.model = load_flux_model_(model,self.device, flux_cls=Flux_kv)

    def create_attention_mask(self,seq_len, mask_indices, text_len=512 , device='cuda'):
        # """
        # 创建自定义的注意力掩码。

        # Args:
        #     seq_len (int): 序列长度。
        #     mask_indices (List[int]): 图像令牌中掩码区域的索引。
        #     text_len (int): 文本令牌的长度，默认 512。改成comfy的时候用256
        #     device (str): 设备类型，如 'cuda' 或 'cpu'。

        # Returns:
        #     torch.Tensor: 形状为 (seq_len, seq_len) 的注意力掩码。
        # """
        # 初始化掩码为全 False

        
        #text_len=512 if self.is_dev else 256

        attention_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # 文本令牌索引
        text_indices = torch.arange(0, text_len, device=device)

        # 掩码区域令牌索引
        mask_token_indices = torch.tensor([idx + text_len for idx in mask_indices], device=device)

        # 背景区域令牌索引
        all_indices = torch.arange(text_len, seq_len, device=device)
        background_token_indices = torch.tensor([idx for idx in all_indices if idx not in mask_token_indices])

        # 设置文本查询可以关注所有键
        attention_mask[text_indices.unsqueeze(1).expand(-1, seq_len)] = True
        attention_mask[text_indices.unsqueeze(1), text_indices] = True# 关注文本
        attention_mask[text_indices.unsqueeze(1), background_token_indices] = True # 关注背景

        
        # attention_mask[mask_token_indices.unsqueeze(1), background_token_indices] = True  # 关注背景
        attention_mask[mask_token_indices.unsqueeze(1), text_indices] = True  # 关注文本
        attention_mask[mask_token_indices.unsqueeze(1), mask_token_indices] = True  # 关注掩码区域

        
        # attention_mask[background_token_indices.unsqueeze(1).expand(-1, seq_len), :] = False
        # attention_mask[background_token_indices.unsqueeze(1), mask_token_indices] = True  # 关注掩码
        attention_mask[background_token_indices.unsqueeze(1), text_indices] = True  # 关注文本
        attention_mask[background_token_indices.unsqueeze(1), background_token_indices] = True  # 关注背景区域

        return attention_mask.unsqueeze(0)
    def create_attention_scale(self,seq_len, mask_indices,text_len=512, device='cuda',scale = 0):

        attention_scale = torch.zeros(1, seq_len, dtype=torch.bfloat16, device=device) # 相加时广播


        text_indices = torch.arange(0, text_len, device=device)
        
        mask_token_indices = torch.tensor([idx + text_len for idx in mask_indices], device=device)

        all_indices = torch.arange(text_len, seq_len, device=device)
        background_token_indices = torch.tensor([idx for idx in all_indices if idx not in mask_token_indices])
        
        attention_scale[0, background_token_indices] = scale #
        
        return attention_scale.unsqueeze(0)
    
class Flux_kv_edit_inf(only_Flux):
    def __init__(self, device,model,name):
        super().__init__(device,model,name)
        
      
    @torch.inference_mode()
    def forward(self,inp,inp_target,mask,opts):
       
        info = {}
        info['feature'] = {}
        bs, L, d = inp["img"].shape

        _,tokens_L,_= inp["txt"].shape #comfy tokens_L=256

        h = opts.height // 8
        w = opts.width // 8
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        mask[mask > 0] = 1
        #print("mask",mask.shape) #mask torch.Size([1, 1, 64, 64])
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        info['mask'] = mask
        bool_mask = (mask.sum(dim=2) > 0.5)
        info['mask_indices'] = torch.nonzero(bool_mask)[:,1] 
        #单独分离inversion
        if opts.attn_mask and (~bool_mask).any():
            attention_mask = self.create_attention_mask(L+tokens_L, info['mask_indices'],tokens_L, device=self.device) #使用512令牌时须将256改成512
            print("attention_mask",attention_mask.shape)
        else:
            attention_mask = None   
        info['attention_mask'] = attention_mask

        if opts.attn_scale != 0 and (~bool_mask).any():
            attention_scale = self.create_attention_scale(L+tokens_L, info['mask_indices'], tokens_L,device=mask.device,scale = opts.attn_scale)#使用512令牌时须将256改成512
        else:
            attention_scale = None
        info['attention_scale'] = attention_scale

        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
    
        z0 = inp["img"]

        with torch.no_grad():
            info['inject'] = True
            z_fe, info = denoise_kv_inf(self.model, img=inp["img"], img_ids=inp['img_ids'], 
                                    source_txt=inp['txt'], source_txt_ids=inp['txt_ids'], source_vec=inp['vec'],
                                    target_txt=inp_target['txt'], target_txt_ids=inp_target['txt_ids'], target_vec=inp_target['vec'],
                                    timesteps=denoise_timesteps, source_guidance=opts.inversion_guidance, target_guidance=opts.denoise_guidance,
                                    info=info)
        mask_indices = info['mask_indices'] 
     
        z0[:, mask_indices,...] = z_fe

        z0 = unpack(z0.float(),  opts.height, opts.width)
        del info
        return z0

class Flux_kv_edit(only_Flux):
    def __init__( self, device,model_path,name):
        super().__init__(device,model_path,name)
    
    @torch.inference_mode()
    def forward(self,inp,inp_target,mask:Tensor,opts):
        z0,zt,info = self.inverse(inp,mask,opts)
        z0 = self.denoise(z0,zt,inp_target,mask,opts,info)
        return z0
    @torch.inference_mode()
    def inverse(self,inp,mask,opts):
        info = {}
        info['feature'] = {}
        bs, L, d = inp["img"].shape
        _,tokens_L,_= inp["txt"].shape #comfy tokens_L=256
        h = opts.height // 8
        w = opts.width // 8

        if opts.attn_mask:
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
            mask[mask > 0] = 1
            
            mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            bool_mask = (mask.sum(dim=2) > 0.5)
            mask_indices = torch.nonzero(bool_mask)[:,1] 
            
            #单独分离inversion
            assert not (~bool_mask).all(), "mask is all false"
            assert not (bool_mask).all(), "mask is all true"
            attention_mask = self.create_attention_mask(L+tokens_L, mask_indices,tokens_L, device=mask.device)#512->256
            info['attention_mask'] = attention_mask
    
        
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
        
        # 加噪过程
        z0 = inp["img"].clone()        
        info['inverse'] = True
        zt, info = denoise_kv(self.model, **inp, timesteps=denoise_timesteps, guidance=opts.inversion_guidance, inverse=True, info=info)
        return z0,zt,info
    
    @torch.inference_mode()
    def denoise(self,z0,zt,inp_target,mask:Tensor,opts,info):
        
        h = opts.height // 8
        w = opts.width // 8
        L = h * w // 4 
        _,tokens_L,_= inp_target["txt"].shape #comfy tokens_L=256
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        mask[mask > 0] = 1
        
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
      
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        info['mask'] = mask
        bool_mask = (mask.sum(dim=2) > 0.5)
        info['mask_indices'] = torch.nonzero(bool_mask)[:,1]
        
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
       
        mask_indices = info['mask_indices']
        if opts.re_init:
            noise = torch.randn_like(zt)
            t  = denoise_timesteps[0]
            zt_noise = z0 *(1 - t) + noise * t
            inp_target["img"] = zt_noise[:, mask_indices,...]
        else:
            inp_target["img"] = zt[:, mask_indices,...]

        if opts.attn_scale != 0 and (~bool_mask).any():
            attention_scale = self.create_attention_scale(L+tokens_L, mask_indices,tokens_L, device=mask.device,scale = opts.attn_scale)#512->256
        else:
            attention_scale = None
        info['attention_scale'] = attention_scale

        info['inverse'] = False
        x, _ = denoise_kv(self.model, **inp_target, timesteps=denoise_timesteps, guidance=opts.denoise_guidance, inverse=False, info=info)
       
        z0[:, mask_indices,...] = z0[:, mask_indices,...] * (1 - info['mask'][:, mask_indices,...]) + x * info['mask'][:, mask_indices,...]
        
        z0 = unpack(z0.float(),  opts.height, opts.width)
        del info
        return z0
