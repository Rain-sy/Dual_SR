#!/usr/bin/env python
"""
FLUX SR Training with CLEAR + ControlNet (Direct Image Training)

Usage:
    accelerate launch --num_processes=8 train_clear_control.py \
        --hr_dir Data/Mix10K_HR \
        --lr_dir Data/Mix10K_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --resolution 512 \
        --batch_size 2 \
        --epochs 60 \
        --lr 1e-5 \
        --clear_ckpt ckpt/clear_local_16_down_4.safetensors
"""

import os
# 🚀 Suppress Triton autotuning logs
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
# 🚀 限制 Triton shared memory 使用，避免 OutOfResources
os.environ["TRITON_NUM_STAGES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch._inductor.config as inductor_config
inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True
inductor_config.verbose_progress = False  # 🚀 禁用详细输出

from PIL import Image
import numpy as np
import argparse
import time
import gc
from datetime import datetime
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import FluxTransformer2DModel, FluxControlNetModel, AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import logging

# Suppress verbose logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("triton").setLevel(logging.ERROR)


# ============================================================================
# FLUX Scheduler Helper
# ============================================================================

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """计算 FLUX scheduler 的动态 mu 参数"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    """SR Dataset - directly loads images
    
    Args:
        hr_dir: HR images directory
        lr_dir: LR images directory  
        resolution: Target resolution (default 512)
        num_crops: Number of random crops per image (default 5)
        full_image: If True, resize whole image instead of cropping (default False)
    """
    
    def __init__(self, hr_dir, lr_dir, resolution=512, num_crops=5, full_image=False):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.num_crops = num_crops if not full_image else 1  # 整图模式只有 1 个 sample
        self.full_image = full_image
        
        # Find all images
        self.hr_files = sorted([
            f for f in os.listdir(hr_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if full_image:
            print(f"[Dataset] Found {len(self.hr_files)} images (full image mode, resize to {resolution})")
        else:
            print(f"[Dataset] Found {len(self.hr_files)} images, {num_crops} crops each = {len(self.hr_files) * num_crops} samples")
    
    def __len__(self):
        return len(self.hr_files) * self.num_crops
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_crops
        crop_idx = idx % self.num_crops
        
        hr_name = self.hr_files[img_idx]
        base_name = os.path.splitext(hr_name)[0]
        
        # Load HR image
        hr_path = os.path.join(self.hr_dir, hr_name)
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Find LR image
        lr_path = None
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:  # 支持多种命名格式
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base_name + suffix + ext)
                if os.path.exists(candidate):
                    lr_path = candidate
                    break
            if lr_path:
                break
        
        if lr_path is None:
            lr_path = os.path.join(self.lr_dir, hr_name)
        
        lr_img = Image.open(lr_path).convert('RGB')
        
        if self.full_image:
            # 🚀 整图模式：直接 resize 到目标分辨率
            hr_resized = hr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
            lr_up = lr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
            
            # To tensor [-1, 1]
            hr_t = torch.from_numpy(np.array(hr_resized)).permute(2, 0, 1).float() / 127.5 - 1
            lr_t = torch.from_numpy(np.array(lr_up)).permute(2, 0, 1).float() / 127.5 - 1
        else:
            # Random crop from HR, corresponding crop from LR
            crop_hr = self.resolution
            crop_lr = self.resolution // 4
            
            W, H = hr_img.size
            max_x = max(0, W - crop_hr)
            max_y = max(0, H - crop_hr)
            
            # Use crop_idx for reproducibility within same image
            torch.manual_seed(idx)
            x = torch.randint(0, max_x + 1, (1,)).item() if max_x > 0 else 0
            y = torch.randint(0, max_y + 1, (1,)).item() if max_y > 0 else 0
            
            # Crop HR
            hr_crop = hr_img.crop((x, y, x + crop_hr, y + crop_hr))
            
            # Corresponding LR crop (4x downscale)
            lr_x, lr_y = x // 4, y // 4
            lr_crop = lr_img.crop((lr_x, lr_y, lr_x + crop_lr, lr_y + crop_lr))
            
            # Upsample LR to match HR resolution (bicubic)
            lr_up = lr_crop.resize((self.resolution, self.resolution), Image.BICUBIC)
            
            # To tensor [-1, 1]
            hr_t = torch.from_numpy(np.array(hr_crop)).permute(2, 0, 1).float() / 127.5 - 1
            lr_t = torch.from_numpy(np.array(lr_up)).permute(2, 0, 1).float() / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


class ValDataset(Dataset):
    """Validation Dataset - center crop only"""
    
    def __init__(self, hr_dir, lr_dir, resolution=512):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        
        self.hr_files = sorted([
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_name = self.hr_files[idx]
        base_name = os.path.splitext(hr_name)[0]
        
        hr_path = os.path.join(self.hr_dir, hr_name)
        hr_img = Image.open(hr_path).convert('RGB')
        
        lr_path = None
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:  # 支持多种命名格式
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base_name + suffix + ext)
                if os.path.exists(candidate):
                    lr_path = candidate
                    break
            if lr_path:
                break
        if lr_path is None:
            lr_path = os.path.join(self.lr_dir, hr_name)
        
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Center crop
        W, H = hr_img.size
        crop_hr = self.resolution
        crop_lr = self.resolution // 4
        
        x = (W - crop_hr) // 2
        y = (H - crop_hr) // 2
        
        hr_crop = hr_img.crop((x, y, x + crop_hr, y + crop_hr))
        
        lr_x, lr_y = x // 4, y // 4
        lr_crop = lr_img.crop((lr_x, lr_y, lr_x + crop_lr, lr_y + crop_lr))
        lr_up = lr_crop.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        hr_t = torch.from_numpy(np.array(hr_crop)).permute(2, 0, 1).float() / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).permute(2, 0, 1).float() / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """Lightweight CNN to extract pixel-level features from LR image"""
    
    def __init__(self, in_channels=3, latent_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /4
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # /8
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, 3, stride=2, padding=1),  # /16
        )
        # Zero conv for stable training
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        feat = self.net(x)
        return self.zero_conv(feat)


# ============================================================================
# Main System
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """Dual Stream FLUX SR with CLEAR acceleration"""
    
    def __init__(self, model_name, device, pretrained_controlnet,
                 train_controlnet=True, pixel_weight=1.0,
                 window_size=16, down_factor=4, clear_ckpt=None, resolution=512):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.pretrained_controlnet = pretrained_controlnet
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.window_size = window_size
        self.down_factor = down_factor
        self.clear_ckpt = clear_ckpt
        self.resolution = resolution
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self.text_embeds = None
    
    def setup(self, local_rank):
        """Setup models (called after DDP)"""
        dtype = torch.bfloat16
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        # 🌟 加上这一行！极大地节省高分辨率下的 VAE 显存
        self.vae.enable_tiling()
        # Pre-compute empty text embeddings
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
                                truncation=True, return_tensors="pt").input_ids.to(self.device)
            clip_out = text_enc(clip_ids, output_hidden_states=False)
            
            t5_ids = tokenizer_2("", padding="max_length", max_length=512,
                                truncation=True, return_tensors="pt").input_ids.to(self.device)
            t5_out = text_enc_2(t5_ids)
            
            self.text_embeds = {
                'pooled': clip_out.pooler_output.to(dtype),
                'prompt': t5_out[0].to(dtype),
                'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }
        
        del text_enc, text_enc_2
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load Transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Enable gradient checkpointing for memory efficiency
        try:
            self.transformer.enable_gradient_checkpointing()
        except:
            pass
        
        if self.clear_ckpt:
            self._enable_clear(local_rank, dtype)
        
        # Load ControlNet
        self.controlnet = FluxControlNetModel.from_pretrained(
            self.pretrained_controlnet, torch_dtype=dtype
        ).to(self.device)
        
        if self.train_controlnet:
            self.controlnet.requires_grad_(True)
            try:
                self.controlnet.enable_xformers_memory_efficient_attention()
            except:
                pass
            try:
                self.controlnet.enable_gradient_checkpointing()
            except:
                pass
        else:
            self.controlnet.requires_grad_(False)
        
        # Pixel Extractor
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(True)
        
        print(f"[Rank {local_rank}] GPU memory: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
    
    def _enable_clear(self, local_rank, dtype):
        from safetensors.torch import load_file
        from attention_processor import (
            LocalDownsampleFlexAttnProcessor,
            LocalFlexAttnProcessor,
            init_local_downsample_mask_flex,
            init_local_mask_flex
        )
        
        latent_size = self.resolution // 16
        text_length = 512
        device_str = str(self.device)
        
        if self.down_factor > 1:
            init_local_downsample_mask_flex(
                height=latent_size, width=latent_size, text_length=text_length,
                window_size=self.window_size, down_factor=self.down_factor, device=device_str
            )
        else:
            init_local_mask_flex(
                height=latent_size, width=latent_size, text_length=text_length,
                window_size=self.window_size, device=device_str
            )
        
        clear_weights = load_file(self.clear_ckpt)
        
        # CLEAR only for transformer_blocks (19 dual-stream blocks)
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
                        processor = LocalDownsampleFlexAttnProcessor(
                            down_factor=self.down_factor
                        )
                    else:
                        processor = LocalFlexAttnProcessor()
                    processor.load_state_dict(block_weights, strict=False)
                    processor = processor.to(self.device, dtype)
                    module.set_processor(processor)
    
    def get_trainable_params(self):
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return params
    
    @torch.no_grad()
    def encode(self, img):
        """Encode image to latent"""
        img = img.to(self.vae.dtype).to(self.device)
        lat = self.vae.encode(img).latent_dist.sample()
        
        # 🌟 FLUX VAE 需要 shift_factor
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor is not None:
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            lat = lat * self.vae.config.scaling_factor
        return lat
    
    @torch.no_grad()
    def decode(self, lat):
        """Decode latent to image"""
        lat = lat.to(self.device)
        
        # 🌟 解码时加回 shift_factor
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor is not None:
            lat = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            lat = lat / self.vae.config.scaling_factor
            
        sample = self.vae.decode(lat).sample
        return sample
    
    def _pack(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H//2)*(W//2), C*4)
    
    def _unpack(self, x, H, W):
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)
    
    def _img_ids(self, H, W, device, dtype):
        h, w = H//2, W//2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h*w, 3)
    
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """Forward pass"""
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel)
        
        # 🚀 修复尺寸不匹配：PixelExtractor 输出 /16，VAE latent 是 /8
        # 需要 2x 上采样 pixel_feat 以匹配 lr_lat
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Fuse conditions
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        # Pack for transformer
        noisy_packed = self._pack(noisy)
        fused_packed = self._pack(fused_cond)
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Text embeddings
        pooled = self.text_embeds['pooled'].expand(B, -1)
        prompt = self.text_embeds['prompt'].expand(B, -1, -1)
        text_ids = self.text_embeds['text_ids']  # 2D tensor, 不需要 batch 维度
        
        # Timestep - FLUX 期望 [0, 1] 范围
        t_input = t.to(dtype)
        
        # Guidance (for ControlNet conditioning strength)
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
        B = lr_lat.shape[0]
        dtype = torch.bfloat16
        device = lr_lat.device
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name, subfolder="scheduler"
        )
        
        # 🌟 动态计算当前分辨率需要的 mu 参数
        _, _, h_lat, w_lat = lr_lat.shape
        # FLUX 打包后的序列长度是 (H/2) * (W/2)
        image_seq_len = (h_lat // 2) * (w_lat // 2)
        
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        
        # 传入 mu
        scheduler.set_timesteps(num_steps, device=device, mu=mu)
        
        # Start from LR interpolation
        noise = torch.randn_like(lr_lat)
        if start_t < 1.0:
            lat = start_t * noise + (1 - start_t) * lr_lat
            start_idx = int((1.0 - start_t) * num_steps)
            timesteps = scheduler.timesteps[start_idx:]
        else:
            lat = noise * scheduler.init_noise_sigma
            timesteps = scheduler.timesteps
        
        for t in timesteps:
            # 🌟 安全归一化：确保传入 forward 的 t 永远在 [0, 1] 之间
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

def compute_flow_matching_loss(system, hr, lr, device, scheduler, debug=False):
    """Flow Matching Loss"""
    dtype = torch.bfloat16
    
    hr = hr.to(device).to(dtype)
    lr = lr.to(device).to(dtype)
    
    # Encode
    with torch.no_grad():
        hr_lat = system.encode(hr)
        lr_lat = system.encode(lr)
    
    B = hr_lat.shape[0]
    
    # Flow matching
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
    # 映射到 [0, 1] 之间的浮点数 t
    t = (timesteps.float() / scheduler.config.num_train_timesteps).to(dtype)
    noise = torch.randn_like(hr_lat)
    
    x_t = t.view(B, 1, 1, 1) * noise + (1 - t.view(B, 1, 1, 1)) * hr_lat
    target_v = noise - hr_lat
    
    pred_v = system(x_t, lr_lat, lr, t)
    
    # 🔍 调试信息
    if debug:
        print(f"  [DEBUG] hr_lat range: [{hr_lat.min():.2f}, {hr_lat.max():.2f}]")
        print(f"  [DEBUG] noise range: [{noise.min():.2f}, {noise.max():.2f}]")
        print(f"  [DEBUG] target_v range: [{target_v.min():.2f}, {target_v.max():.2f}]")
        print(f"  [DEBUG] pred_v range: [{pred_v.min():.2f}, {pred_v.max():.2f}]")
        print(f"  [DEBUG] t values: {t.tolist()}")
    
    loss = F.mse_loss(pred_v.float(), target_v.float())
    return loss


def calculate_psnr(pred, target):
    # 🌟 clamp 到 [-1, 1] 避免异常值影响 PSNR
    pred = torch.clamp(pred, -1.0, 1.0)
    
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(4.0 / mse).item()


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
        
        # 🔍 调试信息
        if i == 0:
            print(f"  [VAL DEBUG] hr range: [{hr.min():.2f}, {hr.max():.2f}]")
            print(f"  [VAL DEBUG] sr range: [{sr.min():.2f}, {sr.max():.2f}]")
            print(f"  [VAL DEBUG] hr_lat range: [{hr_lat.min():.2f}, {hr_lat.max():.2f}]")
            print(f"  [VAL DEBUG] sr_lat range: [{sr_lat.min():.2f}, {sr_lat.max():.2f}]")
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
    
    unwrapped.pixel_extractor.train()
    unwrapped.controlnet.train()
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, args, path):
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


def format_time(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_crops', type=int, default=5, help='Number of random crops per image')
    parser.add_argument('--full_image', action='store_true', default=False, 
                        help='Use full image resize instead of random crops')
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--down_factor', type=int, default=4)
    parser.add_argument('--clear_ckpt', type=str, required=True)
    
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--val_steps', type=int, default=20)
    parser.add_argument('--start_t', type=float, default=0.8)
    
    parser.add_argument('--output_dir', type=str, default='./checkpoints/clear_control')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.freeze_controlnet:
        args.train_controlnet = False
    
    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=4  # 🌟 比如 batch_size=1，累积4步相当于 batch=4
    )
    device = accelerator.device
    local_rank = accelerator.process_index
    is_main = accelerator.is_main_process
    
    torch.manual_seed(args.seed + local_rank)
    
    # Dataset
    train_dataset = SRDataset(
        args.hr_dir, args.lr_dir, args.resolution, 
        num_crops=args.num_crops, full_image=args.full_image
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = ValDataset(args.val_hr_dir, args.val_lr_dir, args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "ctrl" if args.train_controlnet else "pix"
    img_mode = "full" if args.full_image else f"crop{args.num_crops}"
    save_dir = os.path.join(args.output_dir, f"{timestamp}_{mode}_res{args.resolution}_{img_mode}")
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'train.log')
        
        with open(log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FLUX SR Training with CLEAR + ControlNet\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"HR Dir: {args.hr_dir}\n")
            f.write(f"LR Dir: {args.lr_dir}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"Image Mode: {'Full image resize' if args.full_image else f'{args.num_crops} crops per image'}\n")
            f.write(f"CLEAR: window={args.window_size}, down={args.down_factor}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            f.write(f"Start T (inference): {args.start_t}\n\n")
        
        print("=" * 70)
        print("🚀 FLUX SR with CLEAR + ControlNet")
        print("=" * 70)
        print(f"HR Dir: {args.hr_dir}")
        print(f"LR Dir: {args.lr_dir}")
        print(f"Resolution: {args.resolution}")
        print(f"Image Mode: {'Full image resize' if args.full_image else f'{args.num_crops} crops per image'}")
        print(f"CLEAR: window={args.window_size}, down={args.down_factor}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Start T (inference): {args.start_t}")
        print("=" * 70)
    
    # Model
    train_controlnet = args.train_controlnet
    system = DualStreamFLUXSR(
        args.model_name, device, args.pretrained_controlnet,
        train_controlnet=train_controlnet, pixel_weight=args.pixel_weight,
        window_size=args.window_size, down_factor=args.down_factor,
        clear_ckpt=args.clear_ckpt, resolution=args.resolution,
    )

    from diffusers import FlowMatchEulerDiscreteScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )
    system.setup(local_rank)
    
    trainable_params = system.get_trainable_params()
    total_params = sum(p.numel() for p in trainable_params)
    
    if train_controlnet:
        optimizer_params = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10},
        ]
    else:
        optimizer_params = [
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10},
        ]
    
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    
    # LR scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Accelerate
    system, optimizer, train_loader, scheduler = accelerator.prepare(
        system, optimizer, train_loader, scheduler
    )
    
    if is_main:
        print(f"Trainable params: {total_params/1e6:.2f}M")
        print(f"Steps/epoch: {len(train_loader)}")
        print(f"Total steps: {total_steps}")
        print("=" * 70)
    
    # Resume
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.controlnet.load_state_dict(ckpt['controlnet'])
        unwrapped.pixel_extractor.load_state_dict(ckpt['pixel_extractor'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        if is_main:
            print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        system.train()
        
        # 🚀 使用 tqdm 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for step, batch in enumerate(pbar):
            # 🔍 第一个 epoch 的第一个 batch 打印调试信息
            debug = (epoch == start_epoch and step == 0 and is_main)
            
            loss = compute_flow_matching_loss(
                accelerator.unwrap_model(system), batch['hr'], batch['lr'], device, noise_scheduler, debug=debug
            )
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_losses.append(loss.item())
            
            # 更新进度条
            if is_main:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        
        if is_main:
            val_psnr = 0.0
            if val_loader and (epoch + 1) % args.val_interval == 0:
                val_psnr = validate(system, accelerator, val_loader, device,
                                   num_samples=5, num_steps=args.val_steps, start_t=args.start_t)
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Val PSNR: {val_psnr:.2f} dB, Time: {format_time(epoch_time)}")
            else:
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Time: {format_time(epoch_time)}")
            
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, Time={format_time(epoch_time)}\n")
            
            torch.cuda.empty_cache()
            
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr, args,
                              os.path.join(save_dir, f'epoch{epoch+1}.pt'))
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr, args,
                              os.path.join(save_dir, 'best_model.pt'))
                print(f"  ✓ New best PSNR: {best_psnr:.2f} dB")
        
        accelerator.wait_for_everyone()
    
    if is_main:
        save_checkpoint(system, accelerator, args.epochs - 1, avg_loss, best_psnr, args,
                       os.path.join(save_dir, 'final_model.pt'))
        print("=" * 70)
        print(f"✅ Training complete! Best PSNR: {best_psnr:.2f} dB")
        print(f"   Checkpoints saved to: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()