#!/usr/bin/env python
"""
Dual-Stream FLUX SR ControlNet Training with CLEAR Acceleration

特性：
1. CLEAR 线性注意力加速 (O(N²) → O(N))
2. PixelFeatureExtractor 带 GroupNorm（稳定输出）
3. Flow Matching 训练
4. 启动时自动检测关键数值范围

Usage:
    accelerate launch --num_processes=8 --gradient_accumulation_steps 8 \
        train_clear_control_v2.py \
        --hr_dir Data/Mix10K_HR \
        --lr_dir Data/Mix10K_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --batch_size 2 \
        --start_t 0.8 --epochs 60 --lr 1e-5 \
        --clear_ckpt ckpt/clear_local_16_down_4.safetensors
"""

import os
import gc
import math
import argparse
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch._inductor.config as inductor_config
inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True
inductor_config.verbose_progress = False  # 🚀 禁用详细输出

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

# 设置 Triton 环境变量（防止 shared memory 错误）
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["TRITON_NUM_STAGES"] = "2"


# ============================================================================
# Pixel Feature Extractor (带 GroupNorm，稳定输出)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，映射到 Latent 空间维度。
    使用 GroupNorm 确保输出稳定，Zero Conv 确保初始化时不破坏预训练模型。
    
    Input: RGB image [B, 3, H, W] (H, W = 512)
    Output: Latent-space features [B, 16, H/8, W/8] (64x64)
    """
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            # 256 → 128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # 128 → 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )
        
        # Zero Conv: 初始化权重为 0，训练初期不影响 ControlNet
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    """Super-Resolution Dataset with random cropping"""
    
    def __init__(self, hr_dir, lr_dir, resolution=512, num_crops=1, is_val=False):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.num_crops = num_crops
        self.scale = 4
        self.is_val = is_val
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"[Dataset] Found {len(self.hr_files)} images, {num_crops} crops each = {len(self)} samples")
    
    def __len__(self):
        return len(self.hr_files) * self.num_crops
    
    def _find_lr_file(self, hr_name):
        """查找对应的 LR 文件"""
        base = os.path.splitext(hr_name)[0]
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base + suffix + ext)
                if os.path.exists(candidate):
                    return candidate
        return os.path.join(self.lr_dir, hr_name)
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_crops
        hr_name = self.hr_files[img_idx]
        
        hr_img = Image.open(os.path.join(self.hr_dir, hr_name)).convert('RGB')
        lr_img = Image.open(self._find_lr_file(hr_name)).convert('RGB')
        
        # 随机裁剪
        hr_w, hr_h = hr_img.size
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale
        
        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                # 验证集使用中心裁剪 (Center Crop)
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
            else:
                # 训练集保持随机裁剪
                x = np.random.randint(0, hr_w - crop_size + 1)
                y = np.random.randint(0, hr_h - crop_size + 1)
        else:
            # 图像小于 crop_size，resize 到目标尺寸
            x, y = 0, 0
            hr_img = hr_img.resize((crop_size, crop_size), Image.BICUBIC)
            lr_img = lr_img.resize((lr_crop_size, lr_crop_size), Image.BICUBIC)

        hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
        lr_x, lr_y = x // self.scale, y // self.scale
        lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        
        # LR 上采样到 HR 尺寸
        lr_up = lr_crop.resize((crop_size, crop_size), Image.BICUBIC)
        
        # 转换为 tensor，范围 [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_crop)).permute(2, 0, 1).float() / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).permute(2, 0, 1).float() / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Main System
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """Dual Stream FLUX SR with CLEAR acceleration"""
    
    def __init__(self, model_name, device, pixel_weight=1.0,
                 window_size=16, down_factor=4, clear_ckpt=None):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.pixel_weight = pixel_weight
        self.window_size = window_size
        self.down_factor = down_factor
        self.clear_ckpt = clear_ckpt
        
        self._load_models()
    
    def _load_models(self):
        from diffusers import AutoencoderKL, FluxTransformer2DModel, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # Text Encoders (只用于生成 empty prompt embedding，然后释放)
        text_enc = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        text_enc_2 = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")
        
        with torch.no_grad():
            clip_ids = tokenizer("", padding="max_length", max_length=77,
                                 return_tensors="pt").input_ids.to(self.device)
            clip_out = text_enc(clip_ids, output_hidden_states=False)
            
            t5_ids = tokenizer_2("", padding="max_length", max_length=512,
                                 return_tensors="pt").input_ids.to(self.device)
            t5_out = text_enc_2(t5_ids)
            
            self.text_embeds = {
                'pooled': clip_out.pooler_output.to(dtype),
                'prompt': t5_out[0].to(dtype),
                'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }
        
        # 释放 text encoders
        del text_enc, text_enc_2, tokenizer, tokenizer_2
        torch.cuda.empty_cache()
        gc.collect()
        
        # Transformer (FLUX)
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        self.transformer.enable_gradient_checkpointing()
        
        # ControlNet
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=dtype
        ).to(self.device)
        self.controlnet.enable_gradient_checkpointing()
        
        # Pixel Feature Extractor
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        
        # 初始化 CLEAR
        self._init_clear()
    
    def _init_clear(self):
        """初始化 CLEAR 注意力"""
        import attention_processor
        from attention_processor import (
            LocalDownsampleFlexAttnProcessor, LocalFlexAttnProcessor,
            init_local_downsample_mask_flex, init_local_mask_flex
        )
        
        dtype = torch.bfloat16
        
        # 初始化 mask（固定 512x512 分辨率，对应 32x32 patches）
        patch_size = 32  # 512 / 16 = 32
        device_str = str(self.device)
        
        if self.down_factor > 1:
            init_local_downsample_mask_flex(
                height=patch_size, width=patch_size, text_length=512,
                window_size=self.window_size, down_factor=self.down_factor, device=device_str
            )
        else:
            init_local_mask_flex(
                height=patch_size, width=patch_size, text_length=512,
                window_size=self.window_size, device=device_str
            )
        
        # 设置全局变量
        attention_processor.HEIGHT = patch_size
        attention_processor.WIDTH = patch_size
        
        # 加载 CLEAR 权重
        if self.clear_ckpt and os.path.exists(self.clear_ckpt):
            from safetensors.torch import load_file
            clear_weights = load_file(self.clear_ckpt)
            
            # 只对 transformer_blocks（不是 single_transformer_blocks）应用 CLEAR
            for name, module in self.transformer.named_modules():
                if hasattr(module, 'set_processor') and 'transformer_blocks.' in name and 'single' not in name:
                    block_idx = int(name.split('transformer_blocks.')[1].split('.')[0])
                    prefix = f"transformer_blocks.{block_idx}.attn."
                    
                    block_weights = {
                        k.replace(prefix, ''): v.to(dtype)
                        for k, v in clear_weights.items()
                        if k.startswith(prefix)
                    }
                    
                    if block_weights:
                        if self.down_factor > 1:
                            processor = LocalDownsampleFlexAttnProcessor(down_factor=self.down_factor)
                        else:
                            processor = LocalFlexAttnProcessor()
                        
                        processor.load_state_dict(block_weights, strict=False)
                        processor = processor.to(self.device, dtype)
                        module.set_processor(processor)
            
    
    def _pack(self, x):
        """Pack latent for transformer: [B, C, H, W] -> [B, H*W/4, C*4]"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // 2) * (W // 2), C * 4)
    
    def _unpack(self, x, H, W):
        """Unpack transformer output: [B, H*W/4, C*4] -> [B, C, H, W]"""
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)
    
    def _img_ids(self, H, W, device, dtype):
        """Generate image position IDs"""
        h, w = H // 2, W // 2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h * w, 3)
    
    def encode(self, img):
        """Encode image to latent"""
        lat = self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample()
        # FLUX VAE shift_factor
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            lat = lat * self.vae.config.scaling_factor
        return lat
    
    def decode(self, lat):
        """Decode latent to image"""
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            lat = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            lat = lat / self.vae.config.scaling_factor
        return self.vae.decode(lat.to(self.vae.dtype)).sample
    
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """Forward pass: predict velocity"""
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        # Pixel features（PixelExtractor 输出 /8，与 VAE latent 一致）
        pixel_feat = self.pixel_extractor(lr_pixel)
        
        # 尺寸检查（通常不需要，但保留以防万一）
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Fuse conditions
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        # Pack for transformer
        noisy_packed = self._pack(noisy.to(dtype))
        fused_packed = self._pack(fused_cond)
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Text embeddings
        pooled = self.text_embeds['pooled'].expand(B, -1)
        prompt = self.text_embeds['prompt'].expand(B, -1, -1)
        text_ids = self.text_embeds['text_ids']
        
        # Timestep（FLUX 期望 [0, 1] 范围）
        t_input = t.to(dtype)
        
        # Guidance
        guidance_tensor = torch.full((B,), guidance, device=device, dtype=dtype)
        
        # ControlNet
        ctrl_out = self.controlnet(
            hidden_states=noisy_packed,
            controlnet_cond=fused_packed,
            timestep=t_input,
            guidance=guidance_tensor,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        ctrl_block, ctrl_single = ctrl_out
        
        # Transformer
        out = self.transformer(
            hidden_states=noisy_packed,
            timestep=t_input,
            guidance=guidance_tensor,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=text_ids,
            img_ids=img_ids,
            controlnet_block_samples=ctrl_block,
            controlnet_single_block_samples=ctrl_single,
            return_dict=False,
        )[0]
        
        return self._unpack(out, H, W)
    
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, start_t=0.8):
        """Flow Matching inference with official scheduler"""
        from diffusers import FlowMatchEulerDiscreteScheduler
        
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
        # 计算 mu（动态 shift）
        _, _, h_lat, w_lat = lr_lat.shape
        image_seq_len = (h_lat // 2) * (w_lat // 2)
        
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name, subfolder="scheduler"
        )
        
        def calculate_shift(seq_len, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.16):
            m = (max_shift - base_shift) / (max_seq - base_seq)
            b = base_shift - m * base_seq
            return seq_len * m + b
        
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        scheduler.set_timesteps(num_steps, device=device, mu=mu)
        
        # 从 LR 插值开始（不是纯噪声）
        noise = torch.randn_like(lr_lat)
        if start_t < 1.0:
            start_idx = int((1.0 - start_t) * num_steps)
            timesteps = scheduler.timesteps[start_idx:]
            
            # 获取真实的起始 t（scheduler 可能是 [0,1] 或 [0,1000]）
            first_t = timesteps[0].item()
            actual_start_t = first_t / scheduler.config.num_train_timesteps if first_t > 1.0 else first_t
            
            lat = actual_start_t * noise + (1 - actual_start_t) * lr_lat
        else:
            lat = noise * scheduler.init_noise_sigma
            timesteps = scheduler.timesteps
        
        # 去噪循环
        for t in timesteps:
            t_val = t.item()
            if t_val > 1.0:
                t_val = t_val / scheduler.config.num_train_timesteps
            
            t_batch = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t_batch, guidance)
            lat = scheduler.step(v, t, lat, return_dict=False)[0]
        
        return lat


# ============================================================================
# Training Functions
# ============================================================================

def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target"""
    pred = torch.clamp(pred, -1.0, 1.0)
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(4.0 / mse).item()  # 范围 [-1,1]，所以 max=2


def validate(system, accelerator, val_loader, device, num_samples=5, num_steps=20, start_t=0.8):
    """Validation"""
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.controlnet.eval()
    
    psnr_list = []
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        hr = batch['hr'].to(device).to(torch.bfloat16)
        lr = batch['lr'].to(device).to(torch.bfloat16)
        
        with torch.no_grad():
            hr_lat = unwrapped.encode(hr)
            lr_lat = unwrapped.encode(lr)
            sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps, start_t=start_t)
            sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
    
    unwrapped.pixel_extractor.train()
    unwrapped.controlnet.train()
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, args, path):
    """Save checkpoint"""
    unwrapped = accelerator.unwrap_model(system)
    torch.save({
        'epoch': epoch,
        'controlnet': unwrapped.controlnet.state_dict(),
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'pixel_weight': args.pixel_weight,
        'window_size': args.window_size,
        'down_factor': args.down_factor,
        'resolution': args.resolution,
        'start_t': args.start_t,
        'loss': loss,
        'psnr': psnr,
    }, path)


def debug_first_batch(system, batch, device, accelerator):
    """调试第一个 batch，检测关键数值范围"""
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("DEBUG: First Batch Check")
        print("="*60)
    
    unwrapped = accelerator.unwrap_model(system)
    hr = batch['hr'].to(device).to(torch.bfloat16)
    lr = batch['lr'].to(device).to(torch.bfloat16)
    
    with torch.no_grad():
        # 输入范围
        if accelerator.is_main_process:
            print(f"Input:")
            print(f"  hr shape: {hr.shape}, range: [{hr.min():.2f}, {hr.max():.2f}]")
            print(f"  lr shape: {lr.shape}, range: [{lr.min():.2f}, {lr.max():.2f}]")
        
        # VAE encode
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        
        if accelerator.is_main_process:
            print(f"VAE Latents:")
            print(f"  hr_lat shape: {hr_lat.shape}, range: [{hr_lat.min():.2f}, {hr_lat.max():.2f}]")
            print(f"  lr_lat shape: {lr_lat.shape}, range: [{lr_lat.min():.2f}, {lr_lat.max():.2f}]")
        
        # Pixel Extractor
        encoder_out = unwrapped.pixel_extractor.encoder(lr)
        pixel_feat = unwrapped.pixel_extractor(lr)
        
        if accelerator.is_main_process:
            print(f"PixelFeatureExtractor:")
            print(f"  encoder output shape: {encoder_out.shape}, range: [{encoder_out.min():.2f}, {encoder_out.max():.2f}]")
            print(f"  pixel_feat shape: {pixel_feat.shape}, range: [{pixel_feat.min():.2f}, {pixel_feat.max():.2f}]")
            print(f"  zero_conv.weight range: [{unwrapped.pixel_extractor.zero_conv.weight.min():.4f}, {unwrapped.pixel_extractor.zero_conv.weight.max():.4f}]")
        
        # Fused condition
        fused = lr_lat + unwrapped.pixel_weight * pixel_feat
        
        if accelerator.is_main_process:
            print(f"Fused Condition:")
            print(f"  pixel_weight: {unwrapped.pixel_weight}")
            print(f"  fused shape: {fused.shape}, range: [{fused.min():.2f}, {fused.max():.2f}]")
        
        # 模拟一个 forward pass
        t = torch.rand(hr.shape[0], device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(hr_lat)
        x_t = t.view(-1, 1, 1, 1) * noise + (1 - t.view(-1, 1, 1, 1)) * hr_lat
        
        v_pred = unwrapped.forward(x_t, lr_lat, lr, t)
        target_v = noise - hr_lat
        
        if accelerator.is_main_process:
            print(f"Forward Pass:")
            print(f"  t range: [{t.min():.3f}, {t.max():.3f}]")
            print(f"  x_t (noisy) range: [{x_t.min():.2f}, {x_t.max():.2f}]")
            print(f"  v_pred range: [{v_pred.min():.2f}, {v_pred.max():.2f}]")
            print(f"  target_v range: [{target_v.min():.2f}, {target_v.max():.2f}]")
        
        # 检查是否有异常值
        has_nan = torch.isnan(v_pred).any() or torch.isnan(target_v).any()
        has_inf = torch.isinf(v_pred).any() or torch.isinf(target_v).any()
        pixel_feat_ok = pixel_feat.abs().max() < 100  # 应该在正常范围内
        
        if accelerator.is_main_process:
            print(f"\nHealth Check:")
            print(f"  NaN detected: {has_nan}")
            print(f"  Inf detected: {has_inf}")
            print(f"  pixel_feat in normal range (<100): {pixel_feat_ok.item()}")
            
            if has_nan or has_inf or not pixel_feat_ok:
                print("  ⚠️  WARNING: Potential numerical issues detected!")
            else:
                print("  ✅ All checks passed!")
            print("="*60 + "\n")
    
    return not (has_nan or has_inf or not pixel_feat_ok)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_crops', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--start_t', type=float, default=0.8)
    parser.add_argument('--guidance', type=float, default=3.5)
    
    # CLEAR
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--down_factor', type=int, default=4)
    parser.add_argument('--clear_ckpt', type=str, default='ckpt/clear_local_16_down_4.safetensors')
    
    parser.add_argument('--output_dir', type=str, default='./checkpoints/clear_control')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 初始化 accelerator
    accelerator = Accelerator(mixed_precision='bf16')
    set_seed(args.seed)
    
    device = accelerator.device
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_clear_res{args.resolution}_crop{args.num_crops}"
    output_dir = os.path.join(args.output_dir, exp_name)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"FLUX SR Training with CLEAR Acceleration")
        print(f"{'='*60}")
        print(f"Output: {output_dir}")
        print(f"Resolution: {args.resolution}, Crops: {args.num_crops}")
        print(f"pixel_weight: {args.pixel_weight}, start_t: {args.start_t}")
        print(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}")
        print(f"{'='*60}\n")
    
    # 创建数据集
    train_dataset = SRDataset(args.hr_dir, args.lr_dir, args.resolution, args.num_crops)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution, num_crops=1)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # 创建模型
    system = DualStreamFLUXSR(
        args.model_name, device, args.pixel_weight,
        args.window_size, args.down_factor, args.clear_ckpt
    )
    
    # 设置可训练参数
    trainable_params = list(system.controlnet.parameters()) + list(system.pixel_extractor.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Prepare
    system, optimizer, train_loader = accelerator.prepare(system, optimizer, train_loader)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)
    
    # 调试第一个 batch
    first_batch = next(iter(train_loader))
    debug_ok = debug_first_batch(system, first_batch, device, accelerator)
    if not debug_ok:
        if accelerator.is_main_process:
            print("⚠️  Debug check failed! Please review the output above.")
    
    # 训练循环
    best_psnr = 0
    steps_per_epoch = len(train_loader)
    
    for epoch in range(args.epochs):
        system.train()
        epoch_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                        disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(progress):
            hr = batch['hr'].to(torch.bfloat16)
            lr = batch['lr'].to(torch.bfloat16)
            
            with torch.no_grad():
                hr_lat = accelerator.unwrap_model(system).encode(hr)
                lr_lat = accelerator.unwrap_model(system).encode(lr)
            
            # Flow matching
            t = torch.rand(hr.shape[0], device=device, dtype=torch.bfloat16)
            noise = torch.randn_like(hr_lat)
            x_t = t.view(-1, 1, 1, 1) * noise + (1 - t.view(-1, 1, 1, 1)) * hr_lat
            target_v = noise - hr_lat
            
            # Forward
            v_pred = system(x_t, lr_lat, lr, t, args.guidance)
            loss = F.mse_loss(v_pred, target_v)
            
            # Backward
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / steps_per_epoch
        
        # Validation
        val_psnr = 0
        if val_loader and (epoch + 1) % args.val_every == 0:
            val_psnr = validate(system, accelerator, val_loader, device,
                               num_samples=5, num_steps=20, start_t=args.start_t)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(system, accelerator, epoch + 1, avg_loss, val_psnr, args,
                               os.path.join(output_dir, f'epoch_{epoch+1}.pt'))
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(system, accelerator, epoch + 1, avg_loss, val_psnr, args,
                               os.path.join(output_dir, 'best_model.pt'))
                print(f"  → New best PSNR: {best_psnr:.2f} dB")
    
    if accelerator.is_main_process:
        print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
