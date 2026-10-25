# ComfyUI_KV_Edit

[KV_Edit](https://github.com/Xilluill/KV-Edit): Training-Free Image Editing for Precise
Background Preservation, re-implemented **on ComfyUI's own FLUX DiT stack**.

# update
* 改成comfyUI的内核封装 /changed to ComfyUI's own FLUX DiT stack
  
# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_KV_Edit
```

# 2.Model
* 2.1 download  'ae.safetensor' and 'flux1-dev.safetensors' from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) 下载ae和flux单体模型模型,文件结构如下图
* 2.2 download 'clip_l.safetensors' and 't5xxl_fp8_e4m3fn.safetensors' from [here](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main)下载T5和clip l单体模型模型,文件结构如下图

```
--   ComfyUI/models/diffusion_models  # or unet
    ├── flux1-dev.safetensors #23.8G 全量
    ├── flux1-kj-dev-fp8.safetensors #11G 名字中有fp8自动开启量化，注意不支持comfy量化的版本
--   ComfyUI/models/clip
    ├── clip_l.safetensors
    ├── t5xxl_fp8_e4m3fn.safetensorsvae
--   ComfyUI/models/vae
    ├── ae.safetensor
```

# 3 Example
* new
![](https://github.com/smthemex/ComfyUI_KV_Edit/blob/main/example_workflows/example.png)

# 4 Citation

```
@article{zhu2025kv,
  title={KV-Edit: Training-Free Image Editing for Precise Background Preservation},
  author={Zhu, Tianrui and Zhang, Shiyi and Shao, Jiawei and Tang, Yansong},
  journal={arXiv preprint arXiv:2502.17363},
  year={2025}
}
```
