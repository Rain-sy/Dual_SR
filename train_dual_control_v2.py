#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training - V2
======================================================================

V2 改进：
1. 训练 log 文件保存（training_log.txt）
2. num_crops 参数 - 每张图像多次随机裁剪
3. 验证集中心裁剪
4. LR 文件名自动匹配（适配 DIV2K 等数据集）
5. 更好的进度显示
6. Loss float32 计算

Usage:
    accelerate launch --num_processes=8 \
        --gradient_accumulation_steps=4 \
        train_dual_control_v2.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --batch_size 2 --epochs 120 --lr 1e-5  --flow_mode mean 

        --flow_mode mixed --mix_prob 0.5
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

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，映射到 Latent 空间维度
    使用 Zero Conv 确保初始化时不破坏预训练 ControlNet
    
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
        
        # Zero Conv: 初始化权重为 0
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Dataset (V2: 支持 num_crops 和 center crop)
# ============================================================================

class SRDataset(Dataset):
    """
    Super-Resolution Dataset
    
    V2 改进:
    - num_crops: 每张图像随机裁剪次数
    - is_val: 验证集使用中心裁剪
    - 自动匹配 LR 文件名
    """
    
    def __init__(self, hr_dir, lr_dir, resolution=512, num_crops=1, is_val=False):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.num_crops = num_crops
        self.scale = 4
        self.is_val = is_val
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.hr_files) * self.num_crops
    
    def _find_lr_file(self, hr_name):
        """自动查找对应的 LR 文件（适配 DIV2K 等数据集）"""
        base = os.path.splitext(hr_name)[0]
        
        # 尝试多种后缀
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base + suffix + ext)
                if os.path.exists(candidate):
                    return candidate
        
        # 回退：直接使用相同文件名
        return os.path.join(self.lr_dir, hr_name)
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_crops
        hr_name = self.hr_files[img_idx]
        
        hr_img = Image.open(os.path.join(self.hr_dir, hr_name)).convert('RGB')
        lr_img = Image.open(self._find_lr_file(hr_name)).convert('RGB')
        
        # 裁剪逻辑
        hr_w, hr_h = hr_img.size
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale
        
        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                # 验证集使用中心裁剪
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
            else:
                # 训练集使用随机裁剪
                x = np.random.randint(0, hr_w - crop_size + 1)
                y = np.random.randint(0, hr_h - crop_size + 1)
            
            hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
            
            # 对应 LR 区域
            lr_x, lr_y = x // self.scale, y // self.scale
            lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        else:
            # 图像太小，直接 resize
            hr_crop = hr_img.resize((crop_size, crop_size), Image.BICUBIC)
            lr_crop = lr_img.resize((lr_crop_size, lr_crop_size), Image.BICUBIC)
        
        # Bicubic upsample LR to match HR size
        lr_up = lr_crop.resize((crop_size, crop_size), Image.BICUBIC)
        
        # To tensor [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_crop)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """
    Dual-Stream FLUX SR System
    
    - pixel_weight: pixel 特征融合权重
    - 标准 flow matching（从纯噪声出发）
    """
    
    def __init__(self, model_name, device, pretrained_controlnet=None, 
                 train_controlnet=True, pixel_weight=1.0):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)
    
    def _load_models(self, pretrained_controlnet):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        import time
        
        dtype = torch.bfloat16
        
        # 错峰加载
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank > 0:
            time.sleep(local_rank * 5)
        
        # Load VAE
        print(f"[Rank {local_rank}] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        # Cache text embeddings
        print(f"[Rank {local_rank}] Caching text embeddings...")
        text_enc = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        text_enc_2 = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        tok_2 = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")
        
        with torch.no_grad():
            clip_out = text_enc(tok([""], padding="max_length", max_length=77,
                                    truncation=True, return_tensors="pt").input_ids.to(self.device))
            t5_out = text_enc_2(tok_2([""], padding="max_length", max_length=512,
                                      truncation=True, return_tensors="pt").input_ids.to(self.device))
            self._cached_embeds = {
                'pooled': clip_out.pooler_output.to(dtype),
                'prompt': t5_out[0].to(dtype),
                'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }
        
        del text_enc, text_enc_2
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load Transformer
        print(f"[Rank {local_rank}] Loading FLUX Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        controlnet_path = pretrained_controlnet or "jasperai/Flux.1-dev-Controlnet-Upscaler"
        print(f"[Rank {local_rank}] Loading ControlNet from {controlnet_path}...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype
        ).to(self.device)
        
        # Pixel Feature Extractor
        print(f"[Rank {local_rank}] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        
        # 启用 Flash Attention
        self._enable_flash_attention(local_rank)
        
        print(f"[Rank {local_rank}] ✓ All models loaded")
    
    def _enable_flash_attention(self, local_rank=0):
        """启用 Flash Attention 加速（xformers 或 PyTorch 2.0 SDPA）"""
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
            if local_rank == 0:
                print("[Flash] ✓ Enabled xformers memory efficient attention")
        except Exception as e:
            if local_rank == 0:
                print(f"[Flash] xformers not available ({e}), using PyTorch 2.0 SDPA")
    
    def encode(self, img):
        """Encode image to latent (FLUX VAE with shift_factor)"""
        lat = self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample()
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            lat = lat * self.vae.config.scaling_factor
        return lat
    
    def decode(self, lat):
        """Decode latent to image"""
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            lat = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            lat = lat / self.vae.config.scaling_factor
        return self.vae.decode(lat.to(self.vae.dtype)).sample
    
    def _pack(self, x):
        """Pack latent for transformer: [B, C, H, W] -> [B, H*W/4, C*4]"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // 2) * (W // 2), C * 4)
    
    def _unpack(self, x, H, W):
        """Unpack transformer output"""
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
    
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """Forward pass: predict velocity"""
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel)
        
        # 尺寸对齐
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
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        text_ids = self._cached_embeds['text_ids']
        
        # Timestep and guidance
        t_input = t.to(dtype)
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
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, 
                  start_mode='standard', start_t=1.0):
        """
        Euler 推理
        
        Args:
            start_mode:
                - 'standard': 从纯噪声开始
                - 'mean': 从 lr_lat 开始
                - 'mixed': 从 start_t*noise + (1-start_t)*lr_lat 开始
            start_t: mixed 模式下的混合比例（1.0=纯噪声，0.0=纯lr_lat）
        """
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        noise = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        # 🌟 根据起点模式计算 start_step
        if start_mode == 'standard':
            lat = noise
            start_step = 0  # 从 t=1.0 开始
        elif start_mode == 'mean':
            lat = lr_lat.clone()
            start_step = 0  # Mean Flow 的 t=1.0 就是 lr_lat
        elif start_mode == 'mixed':
            lat = start_t * noise + (1 - start_t) * lr_lat
            # 🌟 关键：跳过前 (1-start_t) 比例的步数
            start_step = round((1.0 - start_t) * num_steps)
        else:
            lat = noise
            start_step = 0
        
        # 🌟 从正确的时间步开始积分
        for i in range(start_step, num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t, guidance)
            lat = lat - dt * v
        
        return lat
    
    def get_trainable_params(self):
        """Get trainable parameters"""
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return params


# ============================================================================
# Training Functions
# ============================================================================

def compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel, flow_mode='standard', 
                                mix_prob=0.5, mean_noise_scale=0.1):
    """
    Flow Matching Loss with multiple modes
    
    Args:
        flow_mode: 
            - 'standard': noise → hr_lat (原始 flow matching)
            - 'mean': lr_lat → hr_lat (Mean Flow，学残差)
            - 'mixed': 混合两种模式
        mix_prob: mixed 模式下使用 noise 的概率
        mean_noise_scale: mean/mixed 模式下给 lr_lat 添加的微扰动强度
    
    轨迹：source (t=1) → hr_lat (t=0)
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(hr_lat)
    
    # 根据模式选择 source
    if flow_mode == 'standard':
        source = noise
    elif flow_mode == 'mean':
        # 🌟 给 lr_lat 加微扰动，防止模型"偷懒"
        source = lr_lat + mean_noise_scale * torch.randn_like(lr_lat)
    elif flow_mode == 'mixed':
        # 每个 batch 随机选择
        use_noise = torch.rand(B, device=device) < mix_prob
        use_noise = use_noise.view(B, 1, 1, 1).float()
        
        # 🌟 给 lr_lat 加微扰动，逼迫 Pixel Extractor 强力发力
        mean_flow_start = lr_lat + mean_noise_scale * torch.randn_like(lr_lat)
        
        source = use_noise * noise + (1 - use_noise) * mean_flow_start
    else:
        raise ValueError(f"Unknown flow_mode: {flow_mode}")
    
    # 插值：x_t = t * source + (1 - t) * hr_lat
    t_expand = t.view(B, 1, 1, 1)
    x_t = t_expand * source + (1 - t_expand) * hr_lat
    
    # 目标速度：v = source - hr_lat
    target_v = source - hr_lat
    
    # 预测
    unwrapped = system.module if hasattr(system, 'module') else system
    v_pred = unwrapped.forward(x_t, lr_lat, lr_pixel, t)
    
    # Loss (float32 计算)
    loss = F.mse_loss(v_pred.float(), target_v.float())
    
    return loss


def calculate_psnr(pred, target):
    """Calculate PSNR between tensors in [-1, 1]"""
    pred = (pred.clamp(-1, 1) + 1) / 2
    target = (target + 1) / 2
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def validate(system, accelerator, val_loader, device, num_samples=5, num_steps=20, 
             flow_mode='standard'):
    """Validation"""
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.controlnet.eval()
    
    psnr_list = []
    
    # 根据训练模式选择推理起点（必须与训练分布一致！）
    if flow_mode == 'standard':
        start_mode = 'standard'  # 从纯噪声开始
        start_t = 1.0
    elif flow_mode == 'mean':
        start_mode = 'mean'  # 从 lr_lat 开始（与训练一致）
        start_t = 1.0
    else:  # mixed
        start_mode = 'mixed'
        start_t = 0.5  # 混合模式用中间值
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device).to(torch.bfloat16)
        lr = batch['lr'].to(device).to(torch.bfloat16)
        
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps, 
                                     start_mode=start_mode, start_t=start_t)
        sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
    
    # 恢复训练模式
    unwrapped.pixel_extractor.train()
    unwrapped.controlnet.train()
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, flow_mode, path):
    """Save checkpoint"""
    unwrapped = accelerator.unwrap_model(system)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'pixel_weight': pixel_weight,
        'flow_mode': flow_mode,
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'controlnet': unwrapped.controlnet.state_dict(),
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Stream FLUX SR Training V2')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_crops', type=int, default=2,
                        help='Random crops per image per epoch')
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default=None)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--guidance', type=float, default=3.5)
    
    # Flow mode
    parser.add_argument('--flow_mode', type=str, default='standard',
                        choices=['standard', 'mean', 'mixed'],
                        help='Flow matching mode: standard (noise→hr), mean (lr→hr), mixed')
    parser.add_argument('--mix_prob', type=float, default=0.5,
                        help='Probability of using noise in mixed mode')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/dual_controlv2')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.freeze_controlnet:
        args.train_controlnet = False
    
    # Initialize accelerator
    # 注意：gradient_accumulation_steps 由 DeepSpeed 配置文件控制
    accelerator = Accelerator(
        mixed_precision='bf16',
    )
    
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_dual_v2_{args.flow_mode}_res{args.resolution}_crop{args.num_crops}"
    save_dir = os.path.join(args.save_dir, exp_name)
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        # 打印配置
        print("\n" + "=" * 70)
        print("FLUX SR Training with Dual-Stream ControlNet - V2")
        print("=" * 70)
        print(f"\nHR Dir: {args.hr_dir}")
        print(f"LR Dir: {args.lr_dir}")
        print(f"Resolution: {args.resolution}")
        print(f"Num Crops: {args.num_crops}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Flow Mode: {args.flow_mode}" + (f" (mix_prob={args.mix_prob})" if args.flow_mode == 'mixed' else ""))
        print(f"Learning Rate: {args.lr} (PixelExtractor: {args.lr * 10})")
        print(f"Warmup Epochs: {args.warmup_epochs}")
        print(f"Save Dir: {save_dir}")
        print("=" * 70 + "\n")
    
    # Create model
    train_controlnet = args.train_controlnet
    system = DualStreamFLUXSR(
        args.model_name, device, args.pretrained_controlnet,
        train_controlnet=train_controlnet, pixel_weight=args.pixel_weight
    )
    
    # Enable gradient checkpointing
    if hasattr(system.transformer, 'enable_gradient_checkpointing'):
        system.transformer.enable_gradient_checkpointing()
    if hasattr(system.controlnet, 'enable_gradient_checkpointing'):
        system.controlnet.enable_gradient_checkpointing()
    
    # Optimizer groups
    if train_controlnet:
        optimizer_grouped_parameters = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    
    trainable_params = system.get_trainable_params()
    
    if is_main:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Create datasets
    train_dataset = SRDataset(
        args.hr_dir, args.lr_dir, args.resolution,
        num_crops=args.num_crops, is_val=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(
            args.val_hr_dir, args.val_lr_dir, args.resolution,
            num_crops=1, is_val=True
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            print(f"[Data] Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    else:
        if is_main:
            print(f"[Data] Training: {len(train_dataset)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerator
    system, optimizer, train_loader, scheduler = accelerator.prepare(
        system, optimizer, train_loader, scheduler
    )
    
    # Resume
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        unwrapped = accelerator.unwrap_model(system)
        
        if 'pixel_extractor' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
            unwrapped.pixel_extractor.load_state_dict(state)
        
        if train_controlnet and 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            unwrapped.controlnet.load_state_dict(state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # 初始化 log 文件
    log_path = os.path.join(save_dir, 'training_log.txt')
    if is_main:
        with open(log_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FLUX SR Training with Dual-Stream ControlNet - V2\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"HR Dir: {args.hr_dir}\n")
            f.write(f"LR Dir: {args.lr_dir}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"Num Crops: {args.num_crops}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            f.write(f"Flow Mode: {args.flow_mode}" + (f" (mix_prob={args.mix_prob})" if args.flow_mode == 'mixed' else "") + "\n")
            f.write(f"Learning Rate: {args.lr} (PixelExtractor: {args.lr * 10})\n")
            f.write(f"Warmup Epochs: {args.warmup_epochs}\n\n")
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Set train mode
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        if train_controlnet:
            unwrapped.controlnet.train()
        
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            hr = batch['hr'].to(device).to(torch.bfloat16)
            lr = batch['lr'].to(device).to(torch.bfloat16)
            
            with torch.no_grad():
                hr_lat = unwrapped.encode(hr)
                lr_lat = unwrapped.encode(lr)
            
            # DeepSpeed 自动处理梯度累积和裁剪（在 dszero2.json 中配置）
            loss = compute_flow_matching_loss(
                system, hr_lat, lr_lat, lr,
                flow_mode=args.flow_mode, mix_prob=args.mix_prob
            )
            
            accelerator.backward(loss)
            optimizer.step()
            
            # 🌟 关键修复：只在梯度同步时才更新学习率
            # 否则累积步数为 N 时，LR 会以 N 倍速度衰减
            if accelerator.sync_gradients:
                scheduler.step()
            
            optimizer.zero_grad()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        val_psnr = 0.0
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_psnr = validate(system, accelerator, val_loader, device, 
                               num_samples=5, flow_mode=args.flow_mode)
        
        # Logging and saving (main process)
        if is_main:
            lr_current = scheduler.get_last_lr()[0]
            
            # 写入 log 文件
            log_line = f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, LR={lr_current:.2e}\n"
            with open(log_path, 'a') as f:
                f.write(log_line)
            
            # 打印
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, lr={lr_current:.2e}")
            
            # Save best
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    system, accelerator, epoch, avg_loss, val_psnr, 
                    args.pixel_weight, args.flow_mode,
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"  → New best PSNR: {best_psnr:.2f} dB")
            
            # Periodic save
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    system, accelerator, epoch, avg_loss, val_psnr, 
                    args.pixel_weight, args.flow_mode,
                    os.path.join(save_dir, f'epoch{epoch+1}.pt')
                )
            
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
    
    # Final save
    if is_main:
        save_checkpoint(
            system, accelerator, args.epochs - 1, avg_loss, best_psnr, 
            args.pixel_weight, args.flow_mode,
            os.path.join(save_dir, 'final_model.pt')
        )
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()