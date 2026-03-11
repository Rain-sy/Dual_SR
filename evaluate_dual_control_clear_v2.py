#!/usr/bin/env python
"""
Dual-Stream FLUX SR Evaluation with CLEAR Acceleration

与 train_clear_control_v2.py 配套使用。

Usage:
    python evaluate_dual_control_clear_v2.py \
        --checkpoint checkpoints/clear_control/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --force_512
"""

import os
import gc
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置 Triton 环境变量
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["TRITON_NUM_STAGES"] = "2"


# ============================================================================
# Pixel Feature Extractor (与训练代码完全一致)
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
        
        # Zero Conv
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (numpy arrays, range [0, 255])"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, channel_axis=2, data_range=255)
    except ImportError:
        return 0.0


# ============================================================================
# Evaluator
# ============================================================================

class CLEAREvaluator:
    """FLUX SR Evaluator with CLEAR acceleration"""
    
    def __init__(self, model_name, device, checkpoint_path=None, pixel_weight=1.0,
                 window_size=16, down_factor=4, clear_ckpt='ckpt/clear_local_16_down_4.safetensors'):
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.pixel_weight = pixel_weight
        self.window_size = window_size
        self.down_factor = down_factor
        self.clear_ckpt = clear_ckpt
        
        self._cached_embeds = None
        self._current_mask_size = None
        self._debug_printed = False
        
        self._load_models()
    
    def _load_models(self):
        from diffusers import AutoencoderKL, FluxTransformer2DModel, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        import attention_processor
        from attention_processor import (
            LocalDownsampleFlexAttnProcessor, LocalFlexAttnProcessor,
            init_local_downsample_mask_flex, init_local_mask_flex
        )
        
        dtype = torch.bfloat16
        print(f"[Eval] Loading models (CLEAR: w={self.window_size}, d={self.down_factor})...", end=" ")
        
        # VAE (tiled for large images)
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        )
        self.vae.enable_tiling()
        self.vae.to("cpu")  # CPU-GPU 调度，用时搬到 GPU
        
        # Text encoders (生成 empty prompt，然后释放)
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
                                    return_tensors="pt").input_ids.to(self.device))
            t5_out = text_enc_2(tok_2([""], padding="max_length", max_length=512,
                                      return_tensors="pt").input_ids.to(self.device))
            self._cached_embeds = {
                'pooled': clip_out.pooler_output.to(dtype),
                'prompt': t5_out[0].to(dtype),
                'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }
        
        del text_enc, text_enc_2, tok, tok_2
        torch.cuda.empty_cache()
        gc.collect()
        
        # Transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # ControlNet
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=dtype
        ).to(self.device)
        self.controlnet.requires_grad_(False)
        
        # Pixel Extractor
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(False)
        
        # 初始化 CLEAR mask（默认 512x512）
        patch_size = 32
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
        
        attention_processor.HEIGHT = patch_size
        attention_processor.WIDTH = patch_size
        self._current_mask_size = (patch_size, patch_size)
        
        # 加载 CLEAR 权重
        if os.path.exists(self.clear_ckpt):
            from safetensors.torch import load_file
            clear_weights = load_file(self.clear_ckpt)
            
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
        else:
            raise FileNotFoundError(f"CLEAR checkpoint not found: {self.clear_ckpt}")
        
        # 加载训练的 checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'pixel_weight' in ckpt:
                self.pixel_weight = ckpt['pixel_weight']
            
            if 'pixel_extractor' in ckpt:
                state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
                self.pixel_extractor.load_state_dict(state)
            
            if 'controlnet' in ckpt:
                state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
                self.controlnet.load_state_dict(state)
            
            print(f"Checkpoint(epoch={ckpt.get('epoch', '?')}, psnr={ckpt.get('psnr', 0):.2f})")
        else:
            print("No checkpoint loaded")
        
        self.controlnet.eval()
        self.pixel_extractor.eval()
        self.transformer.eval()
        
        print(f"[Eval] Ready. pixel_weight={self.pixel_weight}, GPU={torch.cuda.memory_allocated(self.device)/1e9:.1f}GB")
    
    def update_mask_for_size(self, height, width):
        """动态更新 CLEAR mask（用于不同分辨率）"""
        import attention_processor
        from attention_processor import init_local_downsample_mask_flex, init_local_mask_flex
        
        patch_h = height // 16
        patch_w = width // 16
        
        if self._current_mask_size == (patch_h, patch_w):
            return
        
        torch.cuda.empty_cache()
        gc.collect()
        
        device_str = str(self.device)
        if self.down_factor > 1:
            init_local_downsample_mask_flex(
                height=patch_h, width=patch_w, text_length=512,
                window_size=self.window_size, down_factor=self.down_factor, device=device_str
            )
        else:
            init_local_mask_flex(
                height=patch_h, width=patch_w, text_length=512,
                window_size=self.window_size, device=device_str
            )
        
        attention_processor.HEIGHT = patch_h
        attention_processor.WIDTH = patch_w
        self._current_mask_size = (patch_h, patch_w)
    
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
        self.vae.to(self.device)
        lat = self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample()
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            lat = lat * self.vae.config.scaling_factor
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        return lat
    
    @torch.no_grad()
    def decode(self, lat):
        self.vae.to(self.device)
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            lat = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            lat = lat / self.vae.config.scaling_factor
        img = self.vae.decode(lat.to(self.vae.dtype)).sample
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        return img
    
    @torch.no_grad()
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        noisy = noisy.to(dtype)
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        t = t.to(dtype)
        
        # 调试输出（只打印一次）
        if not self._debug_printed:
            print(f"\n[DEBUG] Forward pass:")
            print(f"  noisy shape: {noisy.shape}, range: [{noisy.min():.2f}, {noisy.max():.2f}]")
            print(f"  lr_lat shape: {lr_lat.shape}, range: [{lr_lat.min():.2f}, {lr_lat.max():.2f}]")
            print(f"  lr_pixel shape: {lr_pixel.shape}, range: [{lr_pixel.min():.2f}, {lr_pixel.max():.2f}]")
            print(f"  t: {t}")
        
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel).to(dtype)
        
        # 尺寸检查
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )
        
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        # 调试输出
        if not self._debug_printed:
            encoder_out = self.pixel_extractor.encoder(lr_pixel)
            print(f"  encoder output range: [{encoder_out.min():.2f}, {encoder_out.max():.2f}]")
            print(f"  pixel_feat range: [{pixel_feat.min():.2f}, {pixel_feat.max():.2f}]")
            print(f"  fused_cond range: [{fused_cond.min():.2f}, {fused_cond.max():.2f}]")
            print(f"  pixel_weight: {self.pixel_weight}")
            self._debug_printed = True
        
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(fused_cond)
        
        guidance_tensor = torch.full((B,), guidance, device=device, dtype=dtype)
        
        # ControlNet
        ctrl_out = self.controlnet(
            hidden_states=packed_noisy,
            controlnet_cond=packed_cond,
            timestep=t,
            guidance=guidance_tensor,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        
        # Transformer
        pred = self.transformer(
            hidden_states=packed_noisy,
            timestep=t,
            guidance=guidance_tensor,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=txt_ids,
            img_ids=img_ids,
            controlnet_block_samples=ctrl_out[0],
            controlnet_single_block_samples=ctrl_out[1],
            return_dict=False,
        )[0]
        
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, start_t=0.8):
        from diffusers import FlowMatchEulerDiscreteScheduler
        
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
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
        
        noise = torch.randn_like(lr_lat)
        if start_t < 1.0:
            start_idx = int((1.0 - start_t) * num_steps)
            timesteps = scheduler.timesteps[start_idx:]
            
            first_t = timesteps[0].item()
            actual_start_t = first_t / scheduler.config.num_train_timesteps if first_t > 1.0 else first_t
            
            lat = actual_start_t * noise + (1 - actual_start_t) * lr_lat
        else:
            lat = noise * scheduler.init_noise_sigma
            timesteps = scheduler.timesteps
        
        for t in timesteps:
            t_val = t.item()
            if t_val > 1.0:
                t_val = t_val / scheduler.config.num_train_timesteps
            
            t_batch = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t_batch, guidance)
            lat = scheduler.step(v, t, lat, return_dict=False)[0]
        
        return lat


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--start_t', type=float, default=0.8)
    
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--down_factor', type=int, default=4)
    parser.add_argument('--clear_ckpt', type=str, default='ckpt/clear_local_16_down_4.safetensors')
    
    parser.add_argument('--force_512', action='store_true',
                        help='Force 512x512 evaluation (recommended for training consistency)')
    parser.add_argument('--max_size', type=int, default=1536)
    
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 自动检测数据集名称
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower:
            args.dataset = 'Urban100'
        elif 'div2k' in hr_lower:
            args.dataset = 'DIV2K'
        elif 'set5' in hr_lower:
            args.dataset = 'Set5'
        elif 'set14' in hr_lower:
            args.dataset = 'Set14'
        elif 'bsd100' in hr_lower:
            args.dataset = 'BSD100'
        elif 'manga' in hr_lower:
            args.dataset = 'Manga109'
        else:
            args.dataset = 'Unknown'
    
    # 输出目录：outputs/数据集名称/CLEARControl/实验名称/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{timestamp}_clear_w{args.window_size}_d{args.down_factor}_{args.num_steps}step"
    output_dir = os.path.join(args.output_base, args.dataset, 'CLEARControl', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FLUX SR Evaluation with CLEAR")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}, start_t: {args.start_t}")
    if args.force_512:
        print(f"⚠️  Force 512x512 mode")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # 创建 evaluator
    evaluator = CLEAREvaluator(
        args.model_name, device, args.checkpoint,
        window_size=args.window_size, down_factor=args.down_factor, clear_ckpt=args.clear_ckpt
    )
    
    # LPIPS
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except ImportError:
        print("Warning: LPIPS not available")
    
    # 获取图片列表
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    
    for hf in tqdm(hr_files, desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        
        # 找 LR 文件
        lr_path = None
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(args.lr_dir, base_name + suffix + ext)
                if os.path.exists(candidate):
                    lr_path = candidate
                    break
            if lr_path:
                break
        if lr_path is None:
            lr_path = os.path.join(args.lr_dir, hf)
        
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # 处理尺寸
        if args.force_512:
            W, H = 512, 512
            hr_img = hr_img.resize((W, H), Image.BICUBIC)
            lr_img = lr_img.resize((W // 4, H // 4), Image.BICUBIC)
        else:
            orig_W, orig_H = hr_img.size
            max_dim = max(orig_W, orig_H)
            if max_dim > args.max_size:
                scale = args.max_size / max_dim
                orig_W = int(orig_W * scale)
                orig_H = int(orig_H * scale)
                hr_img = hr_img.resize((orig_W, orig_H), Image.BICUBIC)
                lr_img = lr_img.resize((orig_W // 4, orig_H // 4), Image.BICUBIC)
            
            W = orig_W - (orig_W % 64)
            H = orig_H - (orig_H % 64)
            hr_img = hr_img.crop((0, 0, W, H))
            lr_img = lr_img.crop((0, 0, W // 4, H // 4))
        
        hr_np = np.array(hr_img)
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        # 转 tensor
        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device).to(torch.bfloat16)
        
        # 更新 mask
        if not args.force_512:
            evaluator.update_mask_for_size(H, W)
        
        # 推理
        lr_lat = evaluator.encode(lr_t)
        sr_lat = evaluator.inference(lr_lat, lr_t, num_steps=args.num_steps, 
                                      guidance=args.guidance, start_t=args.start_t)
        sr_t = evaluator.decode(sr_lat)
        
        # 转 numpy
        sr_np = ((sr_t.float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        
        # 计算指标
        psnr_val = calculate_psnr(sr_np, hr_np)
        ssim_val = calculate_ssim(sr_np, hr_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        if lpips_fn:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            hr_t = hr_t.to(device)
            lr_t_lpips = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lr_t_lpips = lr_t_lpips.to(device)
            # 🌟 必须 clamp，扩散模型输出可能略微溢出 [-1, 1]，LPIPS 对此敏感
            lpips_val = lpips_fn(sr_t.float().clamp(-1, 1), hr_t).item()
            lpips_bic = lpips_fn(lr_t_lpips, hr_t).item()
            lpips_list.append(lpips_val)
            lpips_bic_list.append(lpips_bic)
        
        # 保存图片
        if args.save_images:
            Image.fromarray(sr_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        # 保存对比图
        if args.save_comparisons:
            comp = Image.new('RGB', (W * 4, H))
            comp.paste(lr_img.resize((W, H), Image.NEAREST), (0, 0))
            comp.paste(lr_bicubic, (W, 0))
            comp.paste(Image.fromarray(sr_np), (W * 2, 0))
            comp.paste(hr_img, (W * 3, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
        
        # 释放显存
        del lr_lat, sr_lat, lr_t, sr_t
        torch.cuda.empty_cache()
    
    # 汇总结果
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"Results ({len(hr_files)} images)")
    print(f"{'='*70}")
    print(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    if lpips_fn:
        print(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"CLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"Δ:            {avg_psnr - avg_psnr_bic:+.4f}, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
    else:
        print(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}")
        print(f"CLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
    print(f"{'='*70}")
    
    # 保存结果到文件
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}, start_t: {args.start_t}\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n\n")
        f.write(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}")
        if lpips_fn:
            f.write(f", LPIPS={avg_lpips_bic:.4f}")
        f.write(f"\nCLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
        if lpips_fn:
            f.write(f", LPIPS={avg_lpips:.4f}")
        f.write(f"\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Images saved to: {output_dir}")


if __name__ == '__main__':
    main()