#!/usr/bin/env python
"""
Dual-Stream FLUX SR Training with CLEAR + Latent Caching

accelerate launch --num_processes=8 --gradient_accumulation_steps 2 train_dual_control_clear_cached.py \
    --latent_dir Data/Mix10K_latents \
    --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
    --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
    --batch_size 2 \
    --epochs 80 \
    --lr 1e-5 \
    --warmup_epochs 3 \
    --val_interval 5 \
    --save_interval 10 \
    --clear_ckpt ckpt/clear_local_16_down_4.safetensors
"""

import os
import sys

# 🚀 禁用 Triton AUTOTUNE 日志（必须在 import torch 之前）
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCH_COMPILE_DEBUG"] = "0"

import gc
import math
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch._inductor.config as inductor_config
inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True
inductor_config.verbose_progress = False  # 🚀 禁用详细输出

# 禁用 torch 相关日志
import logging
for name in ['torch', 'torch._dynamo', 'torch._inductor', 'torch._functorch', 'torch.fx', 'triton']:
    logging.getLogger(name).setLevel(logging.ERROR)

# 🚀 CUDA 优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from attention_processor import (
    LocalFlexAttnProcessor, 
    LocalDownsampleFlexAttnProcessor,
    init_local_mask_flex, 
    init_local_downsample_mask_flex
)


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels), nn.SiLU(),
        )
        
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        return self.zero_conv(self.encoder(x))


# ============================================================================
# Cached Latent Dataset
# ============================================================================

class CachedLatentDataset(Dataset):
    """从磁盘加载预缓存的 latent"""
    
    def __init__(self, latent_dir):
        self.latent_dir = Path(latent_dir)
        
        # 加载索引
        index_path = self.latent_dir / 'index.pt'
        if index_path.exists():
            index = torch.load(index_path, weights_only=True)
            self.files = index['files']
            self.resolution = index['resolution']
        else:
            # 直接扫描目录
            self.files = [f.stem for f in sorted(self.latent_dir.glob("*.pt")) 
                         if f.stem not in ['config', 'index']]
            config = torch.load(self.latent_dir / 'config.pt', weights_only=True)
            self.resolution = config['resolution']
        
        # 展开所有 crops
        self.samples = []
        for f in self.files:
            data = torch.load(self.latent_dir / f"{f}.pt", weights_only=True)
            num_crops = data.get('num_crops', 1)
            for c in range(num_crops):
                self.samples.append((f, c))
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, crop_idx = self.samples[idx]
        data = torch.load(self.latent_dir / f"{filename}.pt", weights_only=True)
        
        hr_lat = data['hr_latents'][crop_idx].squeeze(0).float()  # [16, H, W]
        lr_lat = data['lr_latents'][crop_idx].squeeze(0).float()  # [16, H, W]
        lr_pixel = data['lr_pixels'][crop_idx].squeeze(0).float()  # [3, H, W]
        
        return {
            'hr_lat': hr_lat,
            'lr_lat': lr_lat,
            'lr_pixel': lr_pixel,
        }


# ============================================================================
# Validation Dataset (still uses images)
# ============================================================================

class SRDataset(Dataset):
    """用于验证的原始图像数据集"""
    def __init__(self, hr_dir, lr_dir, resolution=512):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.hr_files) == len(self.lr_files)
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        
        lr_w, lr_h = lr_img.size
        crop_lr = self.resolution // 4
        
        if lr_w > crop_lr and lr_h > crop_lr:
            x = np.random.randint(0, lr_w - crop_lr)
            y = np.random.randint(0, lr_h - crop_lr)
            lr_img = lr_img.crop((x, y, x + crop_lr, y + crop_lr))
            hr_img = hr_img.crop((x * 4, y * 4, (x + crop_lr) * 4, (y + crop_lr) * 4))
        
        lr_up = lr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        hr_img = hr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System (No VAE in training loop)
# ============================================================================

class DualStreamFLUXSR_Cached(nn.Module):
    """使用缓存 latent 的训练系统，不需要 VAE"""
    
    def __init__(self, model_name, device, pretrained_controlnet=None,
                 train_controlnet=True, pixel_weight=0.1,
                 window_size=16, down_factor=4, clear_ckpt=None, resolution=512):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.window_size = window_size
        self.down_factor = down_factor
        self.clear_ckpt = clear_ckpt
        self.resolution = resolution
        
        self.vae = None  # 只在验证时加载
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)
    
    def _load_models(self, pretrained_controlnet):
        from diffusers import FluxControlNetModel
        from transformer_flux import FluxTransformer2DModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        import time as time_module
        
        dtype = torch.bfloat16
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if local_rank > 0:
            time_module.sleep(local_rank * 5)
        
        # 缓存 text embeddings
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
        
        # 加载 Transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # 🚀 Transformer 也开启 gradient checkpointing（节省激活值显存）
        try:
            self.transformer.enable_gradient_checkpointing()
        except:
            pass
        
        if self.clear_ckpt:
            self._enable_clear(local_rank, dtype)
        
        # 加载 ControlNet
        self.controlnet = FluxControlNetModel.from_pretrained(
            pretrained_controlnet, torch_dtype=dtype
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
        
        latent_size = self.resolution // 16
        text_length = 512
        
        if self.down_factor == 1:
            init_local_mask_flex(latent_size, latent_size, text_length, self.window_size, self.device)
        else:
            init_local_downsample_mask_flex(latent_size, latent_size, text_length, 
                                           self.window_size, self.down_factor, self.device)
        
        # 只替换双流模块
        original_processors = self.transformer.attn_processors
        attn_processors = {}
        
        for k, proc in original_processors.items():
            if k.startswith("transformer_blocks"):
                if self.down_factor > 1:
                    attn_processors[k] = LocalDownsampleFlexAttnProcessor(self.down_factor).to(self.device, dtype)
                else:
                    attn_processors[k] = LocalFlexAttnProcessor()
            else:
                attn_processors[k] = proc
        
        self.transformer.set_attn_processor(attn_processors)
        clear_state_dict = load_file(self.clear_ckpt)
        self.transformer.load_state_dict(clear_state_dict, strict=False)
        
        clear_count = sum(1 for k in attn_processors if k.startswith("transformer_blocks"))
    
    def load_vae_for_validation(self):
        """验证时才加载 VAE"""
        if self.vae is None:
            from diffusers import AutoencoderKL
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.vae = AutoencoderKL.from_pretrained(
                self.model_name, subfolder="vae", torch_dtype=torch.bfloat16
            ).to(self.device)
            self.vae.requires_grad_(False)
            self.vae.enable_tiling()
    
    def unload_vae(self):
        """验证后卸载 VAE 释放显存"""
        if self.vae is not None:
            del self.vae
            self.vae = None
            torch.cuda.empty_cache()
    
    def get_trainable_params(self):
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return [p for p in params if p.requires_grad]
    
    def _pack(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)
    
    def _unpack(self, x, H, W):
        B, _, D = x.shape
        C = D // 4
        return x.view(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    
    def _img_ids(self, H, W, device, dtype):
        h, w = H//2, W//2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h*w, 3)
    
    @torch.no_grad()
    def encode(self, img):
        """只在验证时使用"""
        return self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, lat):
        """只在验证时使用"""
        return self.vae.decode(lat / self.vae.config.scaling_factor).sample
    
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """前向传播（直接使用缓存的 latent）"""
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        noisy = noisy.to(dtype)
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        t = t.to(dtype)
        
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel).to(dtype)
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(fused_cond)
        
        # ControlNet
        ctrl_out = self.controlnet(
            hidden_states=packed_noisy,
            controlnet_cond=packed_cond,
            timestep=t,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=torch.full((B,), guidance, device=device, dtype=dtype),
            return_dict=False,
        )
        
        ctrl_block = [x.to(dtype) for x in ctrl_out[0]] if ctrl_out[0] else None
        ctrl_single = [x.to(dtype) for x in ctrl_out[1]] if ctrl_out[1] else None
        
        # Transformer
        pred = self.transformer(
            hidden_states=packed_noisy,
            timestep=t,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=torch.full((B,), guidance, device=device, dtype=dtype),
            controlnet_block_samples=ctrl_block,
            controlnet_single_block_samples=ctrl_single,
            return_dict=False,
        )[0]
        
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5):
        B = lr_lat.shape[0]
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        lat = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t, guidance)
            lat = lat - dt * v
        
        return lat


# ============================================================================
# Training Functions
# ============================================================================

def compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel):
    """Flow Matching Loss (直接使用缓存的 latent)"""
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    hr_lat = hr_lat.to(dtype)
    lr_lat = lr_lat.to(dtype)
    lr_pixel = lr_pixel.to(dtype)
    
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(hr_lat)
    
    x_t = t.view(B, 1, 1, 1) * noise + (1 - t.view(B, 1, 1, 1)) * hr_lat
    target_v = noise - hr_lat
    
    pred_v = system(x_t, lr_lat, lr_pixel, t)
    
    loss = F.mse_loss(pred_v.float(), target_v.float())
    return loss


def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(4.0 / mse).item()  # 4.0 = 2^2 for [-1, 1] range


def validate(system, accelerator, val_loader, device, num_samples=5, num_steps=20):
    """验证（需要临时加载 VAE）"""
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.controlnet.eval()
    
    # 加载 VAE
    unwrapped.load_vae_for_validation()
    
    psnr_list = []
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        hr = batch['hr'].to(device).to(torch.bfloat16)
        lr = batch['lr'].to(device).to(torch.bfloat16)
        
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps)
        sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
    
    # 卸载 VAE
    unwrapped.unload_vae()
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
    parser.add_argument('--latent_dir', type=str, required=True, help='Cached latent directory')
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    parser.add_argument('--pixel_weight', type=float, default=1)
    
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--down_factor', type=int, default=4)
    parser.add_argument('--clear_ckpt', type=str, required=True)
    
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=10)
    
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dual_clear_cached')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 从缓存目录读取 resolution
    config = torch.load(os.path.join(args.latent_dir, 'config.pt'), weights_only=True)
    args.resolution = config['resolution']
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    device = accelerator.device
    is_main = accelerator.is_main_process
    
    set_seed(args.seed)
    
    train_controlnet = args.train_controlnet and not args.freeze_controlnet
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "ctrl+pixel" if train_controlnet else "pixel_only"
    save_dir = os.path.join(args.output_dir, f"{timestamp}_{mode}_pw{args.pixel_weight}_cached")
    
    log_file = None
    training_start_time = time.time()
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'training_log.txt')
        
        with open(log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FLUX SR Training with CLEAR + Latent Caching\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Latent Dir: {args.latent_dir}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"CLEAR: window={args.window_size}, down={args.down_factor}\n")
            f.write(f"Batch Size: {args.batch_size}\n\n")
        
        print("=" * 70)
        print("🚀 FLUX SR with CLEAR + Latent Caching")
        print("=" * 70)
        print(f"Latent Dir: {args.latent_dir}")
        print(f"Resolution: {args.resolution}")
        print(f"CLEAR: window={args.window_size}, down={args.down_factor}")
        print(f"Batch Size: {args.batch_size}")
        print("=" * 70)
    
    system = DualStreamFLUXSR_Cached(
        args.model_name, device, args.pretrained_controlnet,
        train_controlnet=train_controlnet, pixel_weight=args.pixel_weight,
        window_size=args.window_size, down_factor=args.down_factor,
        clear_ckpt=args.clear_ckpt, resolution=args.resolution,
    )
    
    trainable_params = system.get_trainable_params()
    total_params = sum(p.numel() for p in trainable_params)
    
    if train_controlnet:
        optimizer_params = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    else:
        optimizer_params = [{"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}]
    
    # 使用缓存的 Dataset
    train_dataset = CachedLatentDataset(args.latent_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    
    if is_main:
        print(f"[Training] Trainable parameters: {total_params:,}")
        print(f"[Data] Training samples: {len(train_dataset)}")
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            print(f"[Data] Validation samples: {len(val_dataset)}")
    
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    system, optimizer, train_loader, scheduler = accelerator.prepare(system, optimizer, train_loader, scheduler)
    
    start_epoch, best_psnr = 0, 0.0
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()})
        if train_controlnet and 'controlnet' in ckpt:
            unwrapped.controlnet.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()})
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
    
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        if train_controlnet:
            unwrapped.controlnet.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            # 🚀 直接使用缓存的 latent，无需 VAE 编码
            hr_lat = batch['hr_lat'].to(device).to(torch.bfloat16)
            lr_lat = batch['lr_lat'].to(device).to(torch.bfloat16)
            lr_pixel = batch['lr_pixel'].to(device).to(torch.bfloat16)
            
            with accelerator.autocast():
                loss = compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel)
            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        
        if is_main:
            val_psnr = 0.0
            if val_loader and (epoch + 1) % args.val_interval == 0:
                val_psnr = validate(system, accelerator, val_loader, device, num_samples=5)
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
                print(f"[Epoch {epoch+1}] ✓ New best PSNR: {best_psnr:.2f} dB")
        
        accelerator.wait_for_everyone()
    
    if is_main:
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Output: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()