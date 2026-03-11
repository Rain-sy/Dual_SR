import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
from transformers import CLIPTextModel, T5EncoderModel

model_name = "black-forest-labs/FLUX.1-dev"
controlnet_name = "jasperai/Flux.1-dev-Controlnet-Upscaler"

print("正在下载 VAE...")
AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.bfloat16)

print("正在下载 Text Encoders...")
CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16)
T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

print("正在下载 Transformer (最慢的部分，请耐心等待)...")
FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.bfloat16)

print("正在下载 ControlNet...")
FluxControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.bfloat16)

print("✅ 所有模型已下载并缓存完成！现在可以运行 8 卡训练了。")