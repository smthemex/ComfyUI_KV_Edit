
import time
from dataclasses import dataclass
from comfy import model_management
import torch
from .flux.kv_edit import Flux_kv_edit

@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 1.0

class FluxEditor_kv_Wrapper:
    def __init__(self,model_path,offload,device):
        
        self.device = device
        self.offload = offload
        device_val = "cpu" if self.offload else self.device
        self.model=Flux_kv_edit(device_val,model_path,name='flux-dev')
        self.model.eval()
        self.info = {}
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
        #     self.ae.encoder.to(self.device)
        
    @torch.inference_mode()
    def inverse(self,opts,mask,inp):
        self.z0 = None
        self.zt = None

        if 'feature' in self.info:
            key_list = list(self.info['feature'].keys())
            for key in key_list:
                del self.info['feature'][key]
        self.info = {}

        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()

        t0 = time.perf_counter()

        self.model = self.model.to(self.device)
        self.z0,self.zt,self.info = self.model.inverse(inp,mask,opts)
        
        # if self.offload:
        #     self.model.cpu()
        #     torch.cuda.empty_cache()
            
        t1 = time.perf_counter()
        print(f"inversion Done in {t1 - t0:.1f}s.")
        return None

        
        
    @torch.inference_mode()
    def edit(self, 
             opts,inp_target,mask,
             ):
       
        torch.cuda.empty_cache()
            
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        
        torch.cuda.empty_cache()
        #self.model = self.model.to(self.device)
        x = self.model.denoise(self.z0,self.zt,inp_target,mask,opts,self.info)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        return x

    
    @torch.inference_mode()
    def encode(self,init_image, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
        
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image
