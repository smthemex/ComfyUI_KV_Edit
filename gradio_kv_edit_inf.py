
from dataclasses import dataclass

import torch
from .flux.kv_edit import Flux_kv_edit_inf

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


class FluxEditor_kv_Wrapper_inf:
    def __init__(self, model_path,offload,device):

        self.device = device
        self.offload = offload
        self.is_schnell = False
        self.model = Flux_kv_edit_inf(device="cpu" if self.offload else self.device,model_path=model_path,name='flux-dev')
        self.model.eval()

            
    @torch.inference_mode()
    def edit(self, 
            opts,inp,inp_target,mask
             ):
        
        torch.cuda.empty_cache()
            
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
      
        self.model = self.model.to(self.device) 
        x = self.model(inp, inp_target, mask, opts)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        print("End Edit")
        return x
