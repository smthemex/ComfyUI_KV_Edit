# ComfyUI_KV_Edit
 [KV_Edit](https://github.com/Xilluill/KV-Edit): Training-Free Image Editing for Precise Background Preservation，you can use it in comfyUI

# update
* use single dit like kj or other /修改模型加载占用及去掉重复量化，加入加速lora支持
  

# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_KV_Edit
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  'ae.safetensor' and 'flux1-dev.safetensors' from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) 下载ae和flux单体模型模型,文件结构如下图
* 3.1.2 download 'clip_l.safetensors' and 't5xxl_fp8_e4m3fn.safetensors' from [here](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main)下载T5和clip l单体模型模型,文件结构如下图

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

# Example
* new
![ ](https://github.com/smthemex/ComfyUI_KV_Edit/blob/main/example_workflows/example.png)
* use comfy T5 and clip and flux dev single checkpoints用comfy的text encoder和flux的fp16/fp8模型，推荐
![](https://github.com/smthemex/ComfyUI_KV_Edit/blob/main/resources/example1.png)
* use comfy or flux dev single checkpoints 仅使用单体模型，速度更慢，更占内存，主要是量化两次模型
![](https://github.com/smthemex/ComfyUI_KV_Edit/blob/main/resources/example0.png)



# Citation

If you find our work helpful, please **star 🌟** this[ repo](https://github.com/Xilluill/KV-Edit) and **cite 📑** our paper. Thanks for your support!
```
@article{zhu2025kv,
  title={KV-Edit: Training-Free Image Editing for Precise Background Preservation},
  author={Zhu, Tianrui and Zhang, Shiyi and Shao, Jiawei and Tang, Yansong},
  journal={arXiv preprint arXiv:2502.17363},
  year={2025}
}
```

# 👍🏻 Acknowledgements
Our code is modified based on [FLUX](https://github.com/black-forest-labs/flux) and [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit). Special thanks to [Wenke Huang](https://wenkehuang.github.io/) for his early inspiration and helpful guidance to this project!

# 📧 Contact
This repository is currently under active development and restructuring. The codebase is being optimized for better stability and reproducibility. While we strive to maintain code quality, you may encounter temporary issues during this transition period. For any questions or technical discussions, feel free to open an issue or contact us via email at xilluill070513@gmail.com.
