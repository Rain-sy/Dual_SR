#!/usr/bin/env python
"""
Dual-Stream FLUX SR ControlNet Training - Simplified Version

改动说明：
1. 去掉 MeanFlow - 回到标准 flow matching（从纯噪声出发）
2. 添加 pixel_weight 参数 - 控制 pixel 特征的融合强度
3. 保留 Zero Conv 初始化
4. 🌟 自动启用 Flash Attention 加速（PyTorch 2.0 SDPA 或 xformers）

关键参数：
    --pixel_weight 0.1    # 推荐从小值开始，保留 FLUX 的生成能力

加速效果（启用 Flash Attention 后）：
    - 训练速度提升 1.5-2x
    - 显存占用减少 20-40%

Usage: 
    accelerate launch --num_processes=8  train_dual_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --epochs 100 --lr 1e-5

依赖（可选，用于额外加速）：
    pip install xformers  # 如果 PyTorch < 2.0
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
# Pixel Feature Extractor (与原版一致)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，并映射到 Latent 空间维度。
    使用 Zero Conv 确保初始化时不破坏预训练 ControlNet。
    
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
# Dataset
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, resolution=512):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR/LR count mismatch: {len(self.hr_files)} vs {len(self.lr_files)}"
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        
        # Random crop
        lr_w, lr_h = lr_img.size
        crop_lr = self.resolution // 4
        
        if lr_w > crop_lr and lr_h > crop_lr:
            x = np.random.randint(0, lr_w - crop_lr)
            y = np.random.randint(0, lr_h - crop_lr)
            lr_img = lr_img.crop((x, y, x + crop_lr, y + crop_lr))
            hr_img = hr_img.crop((x * 4, y * 4, (x + crop_lr) * 4, (y + crop_lr) * 4))
        
        # Bicubic upscale LR to match HR size
        lr_up = lr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        hr_img = hr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        # To tensor [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System (Simplified)
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """
    Dual-Stream FLUX SR System - Simplified Version
    
    改动：
    1. 添加 pixel_weight 参数控制融合强度
    2. 去掉 MeanFlow，使用标准 flow matching
    """
    
    def __init__(self, model_name, device, pretrained_controlnet=None, 
                 train_controlnet=True, pixel_weight=0.1):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight  # 🌟 新增：pixel 特征权重
        
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
        
        # Load Transformer (frozen)
        print(f"[Rank {local_rank}] Loading FLUX Transformer (frozen)...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        print(f"[Rank {local_rank}] Loading ControlNet...")
        if pretrained_controlnet and pretrained_controlnet.lower() != 'none':
            self.controlnet = FluxControlNetModel.from_pretrained(
                pretrained_controlnet, torch_dtype=dtype
            ).to(self.device)
        else:
            raise ValueError("pretrained_controlnet is required")
        
        if self.train_controlnet:
            self.controlnet.requires_grad_(True)
            print(f"[Rank {local_rank}] ControlNet: trainable")
        else:
            self.controlnet.requires_grad_(False)
            print(f"[Rank {local_rank}] ControlNet: frozen")
        
        # Initialize Pixel Feature Extractor
        print(f"[Rank {local_rank}] Initializing Pixel Feature Extractor (weight={self.pixel_weight})...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(True)
        
        # 🌟 Enable Flash Attention / Memory Efficient Attention
        self._enable_flash_attention(local_rank)
        
        print(f"[Rank {local_rank}] Pixel Extractor params: {sum(p.numel() for p in self.pixel_extractor.parameters()):,}")
        print(f"[Rank {local_rank}] GPU memory: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
    
    def _enable_flash_attention(self, local_rank=0):
        """
        🌟 修复后的 Flash Attention 加速逻辑
        注意：千万不要使用普通的 AttnProcessor2_0 覆盖 FLUX 的注意力层！
        """
        flash_enabled = False
        
        # 方法 1：优先尝试启用 Xformers (极致省显存，尤其在长序列和多卡训练下表现优异)
        try:
            import xformers
            # Diffusers 的这个方法很聪明，它会自动为 FLUX 映射专用的 FluxXFormersAttnProcessor
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
            flash_enabled = True
            if local_rank == 0:
                print(f"[Flash] ✓ Enabled xformers memory efficient attention (v{xformers.__version__})")
        except ImportError:
            if local_rank == 0:
                print("[Flash] xformers not installed, trying native PyTorch SDPA...")
        except Exception as e:
            if local_rank == 0:
                print(f"[Flash] xformers activation failed: {e}")
        
        # 方法 2：如果没装 xformers，检查 PyTorch 是否原生支持 SDPA
        if not flash_enabled:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # 🌟 什么都不用 Set！Diffusers 默认就会使用带有 SDPA 的 FluxAttnProcessor2_0
                if local_rank == 0:
                    print("[Flash] ✓ Using Diffusers default PyTorch 2.0 SDPA (Flash Attention supported under the hood)")
            else:
                if local_rank == 0:
                    print("[Flash] ⚠ No acceleration available (PyTorch < 2.0 and xformers not installed). Training will be slow.")
    
    def get_trainable_params(self):
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return [p for p in params if p.requires_grad]
    
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
    
    @torch.no_grad()
    def encode(self, img):
        return self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, lat):
        return self.vae.decode(lat / self.vae.config.scaling_factor).sample
    
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """
        Forward pass with controllable pixel weight
        """
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        noisy = noisy.to(dtype)
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        t = t.to(dtype)
        
        # Get cached embeddings
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        # 🌟 Pixel features with controllable weight
        pixel_feat = self.pixel_extractor(lr_pixel).to(dtype)
        
        # 🌟 Weighted fusion: lr_lat + pixel_weight * pixel_feat
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        # Pack for transformer
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(fused_cond)
        
        # ControlNet forward
        ctrl_kwargs = {
            'hidden_states': packed_noisy.to(dtype),
            'controlnet_cond': packed_cond.to(dtype),
            'timestep': t.to(dtype),
            'encoder_hidden_states': prompt.to(dtype),
            'pooled_projections': pooled.to(dtype),
            'txt_ids': txt_ids.to(dtype),
            'img_ids': img_ids.to(dtype),
            'guidance': torch.full((B,), guidance, device=device, dtype=dtype),
            'return_dict': False,
        }
        
        ctrl_out = self.controlnet(**ctrl_kwargs)
        
        ctrl_block_samples = [x.to(dtype) for x in ctrl_out[0]] if ctrl_out[0] is not None else None
        ctrl_single_samples = [x.to(dtype) for x in ctrl_out[1]] if ctrl_out[1] is not None else None
        
        # Transformer forward
        trans_kwargs = {
            'hidden_states': packed_noisy.to(dtype),
            'timestep': t.to(dtype),
            'encoder_hidden_states': prompt.to(dtype),
            'pooled_projections': pooled.to(dtype),
            'txt_ids': txt_ids.to(dtype),
            'img_ids': img_ids.to(dtype),
            'guidance': torch.full((B,), guidance, device=device, dtype=dtype),
            'controlnet_block_samples': ctrl_block_samples,
            'controlnet_single_block_samples': ctrl_single_samples,
            'return_dict': False,
        }
        
        pred = self.transformer(**trans_kwargs)[0]
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5):
        """
        🌟 Standard Flow Matching 推理：从纯噪声出发（去掉 MeanFlow）
        """
        B = lr_lat.shape[0]
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # 🌟 Standard: 从纯噪声出发
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
    """
    🌟 Standard Flow Matching Loss（去掉 MeanFlow）
    
    轨迹：noise (t=1) → hr_lat (t=0)
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(hr_lat)
    
    # 🌟 Standard flow matching: 从纯噪声出发
    # z_t = (1-t) * hr_lat + t * noise
    t_expand = t.view(B, 1, 1, 1)
    z_t = (1 - t_expand) * hr_lat + t_expand * noise
    
    # 目标速度: v = noise - hr_lat (指向噪声方向)
    v_target = noise - hr_lat
    
    # Forward prediction
    v_pred = system(z_t, lr_lat, lr_pixel, t)
    
    return F.mse_loss(v_pred.float(), v_target.float())


def calculate_psnr(img1, img2):
    """Calculate PSNR between two tensors in [-1, 1]"""
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def validate(system, accelerator, val_loader, device, num_samples=5, num_steps=20):
    """Validate the model"""
    # 🌟 必须先脱壳，才能安全访问子模块和自定义方法
    unwrapped_sys = accelerator.unwrap_model(system)
    
    unwrapped_sys.pixel_extractor.eval()
    if hasattr(unwrapped_sys, 'controlnet'):
        unwrapped_sys.controlnet.eval()
    
    psnr_list = []
    dtype = torch.bfloat16
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device).to(dtype)
        lr = batch['lr'].to(device).to(dtype)
        
        # 🌟 使用脱壳后的系统调用方法
        hr_lat = unwrapped_sys.encode(hr)
        lr_lat = unwrapped_sys.encode(lr)
        sr_lat = unwrapped_sys.inference(lr_lat, lr, num_steps=num_steps)
        sr = unwrapped_sys.decode(sr_lat)
        
        psnr = calculate_psnr(sr.float(), hr.float())
        psnr_list.append(psnr)
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, path):
    unwrapped_sys = accelerator.unwrap_model(system)
    
    torch.save({
        'epoch': epoch,
        'controlnet': unwrapped_sys.controlnet.state_dict(),
        'pixel_extractor': unwrapped_sys.pixel_extractor.state_dict(),
        'pixel_weight': pixel_weight,  # 🌟 保存 pixel_weight
        'loss': loss,
        'psnr': psnr,
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, 
                        default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    
    # 🌟 Pixel weight control
    parser.add_argument('--pixel_weight', type=float, default=1,
                        help='Weight for pixel features (0.0-1.0). Lower = more texture, higher = more fidelity')
    
    # Training
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    
    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--low_memory', action='store_true', default=False)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dual_simple')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
    )
    
    device = accelerator.device
    is_main = accelerator.is_main_process
    
    set_seed(args.seed)
    
    # Determine training mode
    train_controlnet = args.train_controlnet and not args.freeze_controlnet and not args.low_memory
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "ctrl+pixel" if train_controlnet else "pixel_only"
    save_dir = os.path.join(args.output_dir, f"{timestamp}_{mode}_pw{args.pixel_weight}")
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 70)
        print("Dual-Stream FLUX SR - Simplified (Standard Flow Matching)")
        print("=" * 70)
        print(f"Model: {args.model_name}")
        print(f"ControlNet: {args.pretrained_controlnet}")
        print(f"Mode: {'ControlNet + Pixel Extractor' if train_controlnet else 'Pixel Extractor only'}")
        print(f"🌟 Pixel Weight: {args.pixel_weight}")
        print(f"Resolution: {args.resolution}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Output: {save_dir}")
        print(f"GPUs: {accelerator.num_processes}")
        print("=" * 70)
        print("\n⚠️  Using STANDARD Flow Matching (no MeanFlow)")
        print(f"    Pixel weight = {args.pixel_weight} (lower = more texture)\n")
    
    # Create model
    system = DualStreamFLUXSR(
        args.model_name, 
        device, 
        args.pretrained_controlnet,
        train_controlnet=train_controlnet,
        pixel_weight=args.pixel_weight,  # 🌟 传入 pixel_weight
    )
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(system.transformer, 'enable_gradient_checkpointing'):
            system.transformer.enable_gradient_checkpointing()
        if hasattr(system.controlnet, 'enable_gradient_checkpointing'):
            system.controlnet.enable_gradient_checkpointing()
        if is_main:
            print("[Memory] Gradient checkpointing enabled")
    
    # Get trainable parameters
    trainable_params = system.get_trainable_params()
    total_params = sum(p.numel() for p in trainable_params)

    if train_controlnet:
        optimizer_grouped_parameters = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0} # 🌟 放大了 10 倍！
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
        
    if is_main:
        print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Create datasets
    train_dataset = SRDataset(args.hr_dir, args.lr_dir, args.resolution)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    if is_main:
        print(f"[Data] Training samples: {len(train_dataset)}")
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            print(f"[Data] Validation samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    # Learning rate scheduler
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
    
    # Resume if specified
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load Pixel Extractor
        pixel_ext = system.pixel_extractor.module if hasattr(system.pixel_extractor, 'module') else system.pixel_extractor
        raw_state = ckpt['pixel_extractor']
        clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
        pixel_ext.load_state_dict(clean_state)
        
        # Load ControlNet if training it
        if train_controlnet and 'controlnet' in ckpt:
            ctrl = system.controlnet.module if hasattr(system.controlnet, 'module') else system.controlnet
            raw_state = ckpt['controlnet']
            clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
            ctrl.load_state_dict(clean_state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Training history
    history = {'train_loss': [], 'val_psnr': []}
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Set train mode
        if hasattr(system.pixel_extractor, 'module'):
            system.pixel_extractor.module.train()
        else:
            system.pixel_extractor.train()
        
        if train_controlnet:
            if hasattr(system.controlnet, 'module'):
                system.controlnet.module.train()
            else:
                system.controlnet.train()
        
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            hr = batch['hr'].to(device).to(torch.bfloat16)
            lr = batch['lr'].to(device).to(torch.bfloat16)
            
            unwrapped_sys = accelerator.unwrap_model(system)
            
            with torch.no_grad():
                hr_lat = unwrapped_sys.encode(hr)
                lr_lat = unwrapped_sys.encode(lr)
            
            with accelerator.accumulate(system):
                with accelerator.autocast():
                    loss = compute_flow_matching_loss(system, hr_lat, lr_lat, lr)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        # Validation and saving (main process only)
        if is_main:
            val_psnr = 0.0
            if val_loader and (epoch + 1) % args.val_interval == 0:
                val_psnr = validate(system, accelerator, val_loader, device, num_samples=5)
                history['val_psnr'].append(val_psnr)
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
                torch.cuda.empty_cache()
            else:
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    system, accelerator, epoch, avg_loss, val_psnr, args.pixel_weight,
                    os.path.join(save_dir, f'epoch{epoch+1}.pt')
                )
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    system, accelerator, epoch, avg_loss, val_psnr, args.pixel_weight,
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"[Epoch {epoch+1}] ✓ New best PSNR: {best_psnr:.2f} dB")
        
        accelerator.wait_for_everyone()
    
    # Final save
    if is_main:
        save_checkpoint(
            system, accelerator, args.epochs - 1, avg_loss, best_psnr, args.pixel_weight,
            os.path.join(save_dir, 'final_model.pt')
        )
        
        # Training summary
        summary_path = os.path.join(save_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Dual-Stream FLUX SR - Simplified (Standard Flow Matching)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"ControlNet: {args.pretrained_controlnet}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            f.write(f"Mode: {'ControlNet + Pixel Extractor' if train_controlnet else 'Pixel Extractor only'}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"GPUs: {accelerator.num_processes}\n")
            f.write(f"Best PSNR: {best_psnr:.2f} dB\n")
            f.write(f"Final loss: {avg_loss:.6f}\n")
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()