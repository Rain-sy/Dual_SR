#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLUX SR ControlNet Training - Fixed Version with DeepSpeed ZeRO-2

Key Fix: When using pretrained ControlNet (jasperai/Flux.1-dev-Controlnet-Upscaler),
         we must pass VAE latent DIRECTLY to ControlNet, NOT through a random encoder!
         The pretrained ControlNet expects raw VAE latents as input.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python train_sr_flux_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4

    # Multi-GPU with DeepSpeed ZeRO-2
    accelerate launch train_sr_flux_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --use_deepspeed
"""

import os
import gc
import random
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, resolution: int = 512, augment: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.augment = augment
        
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
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        w, h = hr_img.size
        if w < self.resolution or h < self.resolution:
            scale = max(self.resolution / w, self.resolution / h) * 1.1
            hr_img = hr_img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            lr_up = lr_up.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = hr_img.size
        
        x, y = random.randint(0, w-self.resolution), random.randint(0, h-self.resolution)
        hr_img = hr_img.crop((x, y, x+self.resolution, y+self.resolution))
        lr_up = lr_up.crop((x, y, x+self.resolution, y+self.resolution))
        
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
        
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2,0,1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2,0,1) / 127.5 - 1
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# FLUX SR System - Fixed for Pretrained ControlNet
# ============================================================================

class FLUXSRSystem:
    """
    FLUX SR training system.
    
    IMPORTANT: When using pretrained ControlNet, we pass VAE latent DIRECTLY
    to ControlNet without any intermediate encoder. The pretrained model
    expects raw VAE latents!
    """
    
    def __init__(self, model_name: str, ctrl_layers: int = 4, ctrl_single: int = 10, 
                 device='cuda', pretrained_controlnet: str = None):
        self.model_name = model_name
        self.ctrl_layers = ctrl_layers
        self.ctrl_single = ctrl_single
        self.device = device
        self.pretrained_controlnet = pretrained_controlnet
        self.use_pretrained = pretrained_controlnet is not None and pretrained_controlnet.lower() != 'none'
        
        self.transformer = None
        self.vae = None
        self.controlnet = None
        
        self._cached_embeds = None
    
    def load(self):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        print(f"[FLUX] Loading components to {self.device}...")
        print(f"[FLUX] Using pretrained ControlNet: {self.use_pretrained}")
        
        # Load VAE
        print("[FLUX] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        # Cache text embeddings
        print("[FLUX] Caching text embeddings...")
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
        
        # Load transformer
        print("[FLUX] Loading transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        self.transformer.enable_gradient_checkpointing()
        
        # Load or create ControlNet
        if self.use_pretrained:
            print(f"[FLUX] Loading pretrained ControlNet: {self.pretrained_controlnet}")
            self.controlnet = FluxControlNetModel.from_pretrained(
                self.pretrained_controlnet,
                torch_dtype=dtype
            ).to(self.device)
            print("[FLUX] ✓ Using pretrained ControlNet - will pass VAE latent directly!")
        else:
            print(f"[FLUX] Creating ControlNet from scratch ({self.ctrl_layers}+{self.ctrl_single} layers)...")
            num_blocks = len(self.transformer.transformer_blocks)
            num_single = len(self.transformer.single_transformer_blocks)
            
            self.controlnet = FluxControlNetModel.from_transformer(
                self.transformer,
                num_layers=min(self.ctrl_layers, num_blocks),
                num_single_layers=min(self.ctrl_single, num_single),
                attention_head_dim=self.transformer.config.attention_head_dim,
                num_attention_heads=self.transformer.config.num_attention_heads,
            ).to(self.device)
        
        if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
            self.controlnet.enable_gradient_checkpointing()
        
        # Stats
        t_params = sum(p.numel() for p in self.transformer.parameters())
        c_params = sum(p.numel() for p in self.controlnet.parameters())
        
        print(f"[FLUX] Transformer: {t_params/1e9:.2f}B (frozen)")
        print(f"[FLUX] ControlNet: {c_params/1e6:.0f}M (trainable)")
        print(f"[FLUX] GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    def get_trainable_params(self):
        """Only ControlNet is trainable - no extra encoder needed for pretrained!"""
        return [p for p in self.controlnet.parameters() if p.requires_grad]
    
    @torch.no_grad()
    def encode(self, img):
        return self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, lat):
        return self.vae.decode(lat / self.vae.config.scaling_factor).sample
    
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
    
    def forward(self, noisy, lr_lat, t, guidance=3.5):
        """
        Forward pass.
        
        CRITICAL FIX: Pass lr_lat DIRECTLY to ControlNet (no encoder)!
        The pretrained ControlNet expects raw VAE latents.
        """
        B, C, H, W = noisy.shape
        device, dtype = noisy.device, noisy.dtype
        
        # Embeddings
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Pack latents
        packed_noisy = self._pack(noisy)
        # ★ CRITICAL: Pass VAE latent directly - no encoder!
        packed_cond = self._pack(lr_lat.to(dtype))
        
        # Check if ControlNet uses guidance_embeds
        # Build kwargs for ControlNet
        controlnet_kwargs = {
            'hidden_states': packed_noisy,
            'controlnet_cond': packed_cond,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'return_dict': False,
        }
        
        # 🌟 FOOLPROOF FIX: 直接检查底层的网络层类名中是否包含 "Guidance"
        if "Guidance" in type(self.controlnet.time_text_embed).__name__:
            controlnet_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        # ControlNet forward
        ctrl_out = self.controlnet(**controlnet_kwargs)
        
        # Build kwargs for Transformer
        transformer_kwargs = {
            'hidden_states': packed_noisy,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'controlnet_block_samples': ctrl_out[0],
            'controlnet_single_block_samples': ctrl_out[1],
            'return_dict': False,
        }
        
        # 🌟 FOOLPROOF FIX: 对主干 Transformer 做同样的真实类名检查
        if "Guidance" in type(self.transformer.time_text_embed).__name__:
            transformer_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        pred = self.transformer(**transformer_kwargs)[0]
        
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, num_steps=4, guidance=3.5):
        """Generate SR from LR using Euler integration"""
        B = lr_lat.shape[0]
        lat = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=lr_lat.dtype)
            v = self.forward(lat, lr_lat, t, guidance=guidance)
            lat = lat - dt * v
        
        return lat


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(system, loader, optimizer, device, args, accelerator=None):
    """Train one epoch"""
    system.controlnet.train()
    losses = []
    
    pbar = tqdm(loader, desc="Training", disable=accelerator and not accelerator.is_main_process)
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        hr = batch['hr'].to(device)
        lr = batch['lr'].to(device)
        B = hr.shape[0]
        
        # Encode
        with torch.no_grad():
            hr_lat = system.encode(hr)
            lr_lat = system.encode(lr)
        
        # Flow matching
        t = torch.rand(B, device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(hr_lat)
        t_exp = t[:, None, None, None]
        noisy = (1 - t_exp) * hr_lat + t_exp * noise
        v_target = noise - hr_lat
        
        # Forward + backward
        if accelerator:
            with accelerator.accumulate(system.controlnet):
                with accelerator.autocast():
                    v_pred = system.forward(noisy, lr_lat, t)
                    loss = F.mse_loss(v_pred.float(), v_target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(system.get_trainable_params(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                v_pred = system.forward(noisy, lr_lat, t)
                loss = F.mse_loss(v_pred.float(), v_target.float())
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(system.get_trainable_params(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            loss = loss * args.gradient_accumulation_steps
            
            # Clean up to prevent memory fragmentation
            del noisy, v_pred, v_target
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg': f"{np.mean(losses[-50:]):.4f}"})
    
    return np.mean(losses)


@torch.no_grad()
def validate(system, loader, device, num_samples=10, num_steps=4):
    """Validate model"""
    system.controlnet.eval()
    psnrs = []
    
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        hr = batch['hr'].to(device)
        lr = batch['lr'].to(device)
        
        hr_lat = system.encode(hr)
        lr_lat = system.encode(lr)
        sr_lat = system.inference(lr_lat, num_steps=num_steps)
        sr = system.decode(sr_lat)
        
        hr_01 = (hr + 1) / 2
        sr_01 = ((sr + 1) / 2).clamp(0, 1)
        mse = F.mse_loss(sr_01, hr_01)
        psnr = 10 * torch.log10(1.0 / mse)
        psnrs.append(psnr.item())
    
    return np.mean(psnrs)


def save_checkpoint(system, optimizer, epoch, loss, psnr, path):
    """Save checkpoint"""
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'controlnet': system.controlnet.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def save_training_summary(save_dir, args, history, best_psnr):
    """Save training summary"""
    losses = history['train_loss']
    
    # Analyze convergence
    if len(losses) >= 10:
        early_loss = np.mean(losses[:10])
        late_loss = np.mean(losses[-10:])
        loss_reduction = (early_loss - late_loss) / early_loss * 100
        loss_std = np.std(losses[-10:])
    else:
        early_loss = late_loss = losses[-1] if losses else 0
        loss_reduction = 0
        loss_std = 0
    
    # Determine status
    if loss_reduction > 50 and loss_std < 0.05:
        status = "✅ CONVERGED"
    elif loss_reduction > 20:
        status = "⚠️ PARTIALLY CONVERGED"
    else:
        status = "❌ NOT CONVERGED"
    
    report = f"""
================================================================================
                    FLUX SR ControlNet Training Summary
================================================================================

📋 Configuration:
   Model: {args.model_name}
   ControlNet: {args.pretrained_controlnet}
   Resolution: {args.resolution}
   Epochs: {args.epochs}
   Batch: {args.batch_size} x {args.gradient_accumulation_steps}
   Learning Rate: {args.lr} -> {args.min_lr}

📈 Results:
   Initial Loss: {early_loss:.4f}
   Final Loss: {late_loss:.4f}
   Loss Reduction: {loss_reduction:.1f}%
   Best Val PSNR: {best_psnr:.2f} dB

🎯 Status: {status}

================================================================================
"""
    
    print(report)
    
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write(report)
        f.write("\n\nTraining History:\n")
        f.write("Epoch\tLoss\t\tLR\t\tVal_PSNR\n")
        f.write("-" * 60 + "\n")
        for i, ep in enumerate(history['epoch']):
            val_p = history['val_psnr'][i]
            val_str = f"{val_p:.2f}" if val_p else "-"
            f.write(f"{ep}\t{history['train_loss'][i]:.6f}\t{history['lr'][i]:.2e}\t{val_str}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev',
                        help='FLUX model: schnell (4 steps) or dev (20+ steps). Use dev with jasperai ControlNet!')
    parser.add_argument('--controlnet_layers', type=int, default=4)
    parser.add_argument('--controlnet_single_layers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_base', type=str, default='./checkpoints/flux_control')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_controlnet', type=str, 
                        default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    parser.add_argument('--num_inference_steps', type=int, default=None,
                        help='Inference steps for validation. Default: 4 for schnell, 20 for dev')
    parser.add_argument('--use_deepspeed', action='store_true', help='Use DeepSpeed ZeRO-2')
    parser.add_argument('--local_rank', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup accelerator if using DeepSpeed
    accelerator = None
    if args.use_deepspeed:
        try:
            from accelerate import Accelerator
            from accelerate.utils import set_seed
            import os
            
            # 🌟 修复 accelerate 在多进程传递时的 Base64 padding bug
            ds_env = os.environ.get("ACCELERATE_CONFIG_DS_FIELDS", "")
            if ds_env:
                os.environ["ACCELERATE_CONFIG_DS_FIELDS"] = ds_env + "=" * (4 - len(ds_env) % 4)
            accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                mixed_precision='bf16',
            )
            device = accelerator.device
            is_main = accelerator.is_main_process
            set_seed(args.seed)
        except ImportError:
            print("Warning: accelerate not installed, falling back to single GPU")
            accelerator = None
            device = 'cuda'
            is_main = True
    else:
        device = 'cuda'
        is_main = True
    
    # Generate save directory
    if args.save_dir is None and is_main:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.split('/')[-1].replace('FLUX.1-', '')
        pretrained_short = 'pretrained' if args.pretrained_controlnet.lower() != 'none' else 'scratch'
        args.save_dir = os.path.join(
            args.save_base, 
            f"{ts}_{model_short}_ctrl{args.controlnet_layers}+{args.controlnet_single_layers}_{pretrained_short}"
        )
    
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Auto-detect model type for display
        is_schnell = 'schnell' in args.model_name.lower()
        
        print("=" * 70)
        print("FLUX SR ControlNet Training - Fixed Version")
        print("=" * 70)
        print(f"Model: {args.model_name} ({'schnell - 4 steps' if is_schnell else 'dev - 20+ steps'})")
        print(f"ControlNet: {args.pretrained_controlnet}")
        print(f"Resolution: {args.resolution}")
        print(f"ControlNet Layers: {args.controlnet_layers}+{args.controlnet_single_layers}")
        print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps}")
        print(f"LR: {args.lr} -> {args.min_lr}")
        print(f"DeepSpeed: {args.use_deepspeed}")
        print(f"Save Dir: {args.save_dir}")
        print("=" * 70)
        print("\n⚠️  IMPORTANT: Using pretrained ControlNet - passing VAE latent directly!")
        print("    (No intermediate encoder - this is the fix for loss=0.4 issue)\n")
    
    # Dataset
    dataset = SRDataset(args.hr_dir, args.lr_dir, args.resolution, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_set = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution, augment=False)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    
    if is_main:
        print(f"Dataset: {len(dataset)} training images")
        if val_loader:
            print(f"Validation: {len(val_set)} images")
    
    # Load system
    pretrained = args.pretrained_controlnet if args.pretrained_controlnet.lower() != 'none' else None
    system = FLUXSRSystem(
        args.model_name,
        args.controlnet_layers,
        args.controlnet_single_layers,
        str(device),
        pretrained_controlnet=pretrained
    )
    system.load()
    
    # Optimizer
    optimizer = torch.optim.AdamW(system.get_trainable_params(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler - use CosineAnnealingLR per epoch (simpler and more stable)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=args.warmup_epochs
    )
    
    # Cosine decay scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    
    # Combine: warmup then cosine decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    
    # Prepare with accelerator
    if accelerator:
        system.controlnet, optimizer, loader = accelerator.prepare(
            system.controlnet, optimizer, loader
        )
        if val_loader:
            val_loader = accelerator.prepare(val_loader)
    
    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if accelerator:
            accelerator.unwrap_model(system.controlnet).load_state_dict(ckpt['controlnet'])
        else:
            system.controlnet.load_state_dict(ckpt['controlnet'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {'epoch': [], 'train_loss': [], 'val_psnr': [], 'lr': []}
    best_psnr = 0
    
    # Auto-detect inference steps based on model
    if args.num_inference_steps is None:
        if 'schnell' in args.model_name.lower():
            args.num_inference_steps = 4
        else:  # dev or other
            args.num_inference_steps = 20
    
    if is_main:
        print("\nStarting training...")
        print(f"Inference steps for validation: {args.num_inference_steps}")
        print(f"Initial GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(system, loader, optimizer, device, args, accelerator)
        
        # Step scheduler per epoch (not per step!)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['lr'].append(current_lr)
        
        # Validation
        val_psnr = None
        if val_loader and (epoch + 1) % args.save_every == 0:
            val_psnr = validate(system, val_loader, device, 
                               num_samples=10, num_steps=args.num_inference_steps)
        history['val_psnr'].append(val_psnr)
        
        if is_main:
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            log_str = f"Epoch {epoch+1}: loss={avg_loss:.6f}, lr={current_lr:.2e}, GPU={peak_mem:.2f}GB"
            if val_psnr:
                log_str += f", PSNR={val_psnr:.2f}dB"
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    if accelerator:
                        state = accelerator.unwrap_model(system.controlnet).state_dict()
                    else:
                        state = system.controlnet.state_dict()
                    save_checkpoint(system, optimizer, epoch, avg_loss, val_psnr,
                                    os.path.join(args.save_dir, 'best_model.pt'))
                    log_str += " ★ Best!"
            print(log_str)
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                if accelerator:
                    state = accelerator.unwrap_model(system.controlnet).state_dict()
                else:
                    state = system.controlnet.state_dict()
                save_checkpoint(system, optimizer, epoch, avg_loss, val_psnr or 0,
                                os.path.join(args.save_dir, f'epoch{epoch+1}.pt'))
    
    # Final save
    if is_main:
        if accelerator:
            state = accelerator.unwrap_model(system.controlnet).state_dict()
        else:
            state = system.controlnet.state_dict()
        save_checkpoint(system, optimizer, args.epochs-1, avg_loss, best_psnr,
                        os.path.join(args.save_dir, 'final.pt'))
        
        save_training_summary(args.save_dir, args, history, best_psnr)
        print(f"\n✅ Training complete! Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()