import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
import gradio as gr
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit import Flux_kv_edit

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

def resize_image(image_array, max_width=1360, max_height=768):
    # 将numpy数组转换为PIL图像
    if image_array.shape[-1] == 4:
        mode = 'RGBA'
    else:
        mode = 'RGB'
    
    pil_image = Image.fromarray(image_array, mode=mode)
    
    # 获取原始图像的宽度和高度
    original_width, original_height = pil_image.size
    
    # 计算缩放比例
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    
    # 选择较小的缩放比例以确保图像不超过最大宽度和高度
    scale_ratio = min(width_ratio, height_ratio)
    
    # 如果图像已经小于或等于最大分辨率，则不进行缩放
    if scale_ratio >= 1:
        return image_array
    
    # 计算新的宽度和高度
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    
    # 缩放图像
    resized_image = pil_image.resize((new_width, new_height))
    
    # 将PIL图像转换回numpy数组
    resized_array = np.array(resized_image)
    
    return resized_array

class FluxEditor_kv_demo:
    def __init__(self, args):
        self.args = args
        self.gpus = args.gpus
        if self.gpus:
            self.device = [torch.device("cuda:0"), torch.device("cuda:1")]
        else:
            self.device = [torch.device(args.device), torch.device(args.device)]

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'regress_result'

        self.t5 = load_t5(self.device[1], max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device[1])
        self.model = Flux_kv_edit(self.device[0], name=self.name)
        self.ae = load_ae(self.name, device=self.device[1])

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        self.info = {}
    @torch.inference_mode()
    def inverse(self, brush_canvas,
             source_prompt, target_prompt, 
             inversion_num_steps, denoise_num_steps, 
             skip_step, 
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask
             ):
        if hasattr(self, 'z0'):
            del self.z0
            del self.zt
        # self.info = {}
        # gc.collect()
        
        if 'feature' in self.info:
            key_list = list(self.info['feature'].keys())
            for key in key_list:
                del self.info['feature'][key]
        self.info = {}
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        # init_image = resize_image(init_image)
        shape = init_image.shape        
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=0, # no skip step in inverse leads chance to adjest skip_step in edit
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask
        )
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()
        
        if opts.attn_mask:
            # rgba_mask = resize_image(brush_canvas["layers"][0])[:height, :width, :]
            rgba_mask = brush_canvas["layers"][0][:height, :width, :]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(int)
        
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        else:
            mask = None
        
        self.init_image = self.encode(init_image, self.device[1]).to(self.device[0])

        t0 = time.perf_counter()

        with torch.no_grad():
            inp = prepare(self.t5, self.clip,self.init_image, prompt=opts.source_prompt)
            self.z0,self.zt,self.info = self.model.inverse(inp,mask,opts)
        t1 = time.perf_counter()
        print(f"inversion Done in {t1 - t0:.1f}s.")
        return None

        
        
    @torch.inference_mode()
    def edit(self, brush_canvas,
             source_prompt, target_prompt, 
             inversion_num_steps, denoise_num_steps, 
             skip_step, 
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask
             ):
        
        torch.cuda.empty_cache()
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        shape = init_image.shape        
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]
        # brush_canvas = brush_canvas["composite"][:,:,:3][:height, :width, :]
    
        # if np.all(brush_canvas[:,:,0] == brush_canvas[:,:,1]) and np.all(brush_canvas[:,:,1] == brush_canvas[:,:,2]):
        rgba_mask = brush_canvas["layers"][0][:height, :width, :]
        mask = rgba_mask[:,:,3]/255
        mask = mask.astype(int)
        
        
        rgba_mask[:,:,3] = rgba_mask[:,:,3]//2
        masked_image = Image.alpha_composite(Image.fromarray(rgba_init_image, 'RGBA'), Image.fromarray(rgba_mask, 'RGBA'))
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        
        seed = int(seed)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask
        )
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)

        t0 = time.perf_counter()

    
        with torch.no_grad():
            inp_target = prepare(self.t5, self.clip, self.init_image, prompt=opts.target_prompt)
    
        x = self.model.denoise(self.z0.clone(),self.zt,inp_target,mask,opts,self.info)
            
        with torch.autocast(device_type=self.device[1].type, dtype=torch.bfloat16):
            x = self.ae.decode(x.to(self.device[1]))
    
        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0
      
        
        fn = output_name.format(idx=idx)
    
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        exif_data[ExifTags.Base.ImageDescription] = target_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        masked_image.save(fn.replace(".jpg", "_mask.png"),  format='PNG')
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        
        print("End Edit")
        return img

    
    @torch.inference_mode()
    def encode(self,init_image, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
    
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image
    
def create_demo(model_name: str):
    editor = FluxEditor_kv_demo(args)
    is_schnell = model_name == "flux-schnell"
    
    title = r"""
        <h1 align="center">🎨 KV-Edit: Training-Free Image Editing for Precise Background Preservation</h1>
        """
    one = r"""
    We recommend that you try our code locally, you can try several different edits after inverting the image only once!
    """
        
    description = r"""
        <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/Xilluill/KV-Edit' target='_blank'><b>KV-Edit: Training-Free Image Editing for Precise Background Preservation</b></a>.<br>
    
        💫💫 <b>Here is editing steps:</b> <br>
        1️⃣ Upload your image that needs to be edited. <br>
        2️⃣ Fill in your source prompt and click the "Inverse" button to perform image inversion. <br>
        3️⃣ Use the brush tool to draw your mask area. <br>
        4️⃣ Fill in your target prompt, then adjust the hyperparameters. <br>
        5️⃣ Click the "Edit" button to generate your edited image! <br>
        
        🔔🔔 [<b>Important</b>] We suggest trying "re_init" and "attn_mask" only when the result is too similar to the original content (e.g. removing objects).<br>
        """
    article = r"""
    If our work is helpful, please help to ⭐ the <a href='https://github.com/Xilluill/KV-Edit' target='_blank'>Github Repo</a>. Thanks! 
    """
    
    with gr.Blocks() as demo:
        gr.HTML(title)
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value='' )
                inversion_num_steps = gr.Slider(1, 50, 28, step=1, label="Number of inversion steps")
                target_prompt = gr.Textbox(label="Target Prompt", value='' )
                denoise_num_steps = gr.Slider(1, 50, 28, step=1, label="Number of denoise steps")
                # init_image = gr.Image(label="Input Image", visible=True,scale=1)
                brush_canvas = gr.ImageEditor(label="Brush Canvas",
                                                sources=('upload'), 
                                                brush=gr.Brush(colors=["#ff0000"],color_mode='fixed'),
                                                interactive=True,
                                                transforms=[],
                                                container=True,
                                                format='png')
                
                inv_btn = gr.Button("inverse")
                edit_btn = gr.Button("edit")
                
            
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    skip_step = gr.Slider(0, 30, 4, step=1, label="Number of skip steps")
                    inversion_guidance = gr.Slider(1.0, 10.0, 1.5, step=0.1, label="inversion Guidance", interactive=not is_schnell)
                    denoise_guidance = gr.Slider(1.0, 10.0, 5.5, step=0.1, label="denoise Guidance", interactive=not is_schnell)
                    seed = gr.Textbox('0', label="Seed (-1 for random)", visible=True)
                    with gr.Row():
                        re_init = gr.Checkbox(label="re_init", value=False)
                        attn_mask = gr.Checkbox(label="attn_mask", value=False)
                
                output_image = gr.Image(label="Generated Image")
                gr.Markdown(article)
        inv_btn.click(
            fn=editor.inverse,
            inputs=[brush_canvas,
                    source_prompt, target_prompt, 
                    inversion_num_steps, denoise_num_steps, 
                    skip_step, 
                    inversion_guidance,
                    denoise_guidance,seed,
                    re_init, attn_mask
                    ],
            outputs=[output_image]
        )
        edit_btn.click(
            fn=editor.edit,
            inputs=[ brush_canvas,
                    source_prompt, target_prompt, 
                    inversion_num_steps, denoise_num_steps, 
                    skip_step, 
                    inversion_guidance,
                    denoise_guidance,seed,
                    re_init, attn_mask
                    ],
            outputs=[output_image]
        )
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--gpus", action="store_true", help="2 gpu to use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=41032)
    args = parser.parse_args()

    demo = create_demo(args.name)
    
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)