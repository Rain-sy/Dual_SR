#!/usr/bin/env python
"""
Dual-Stream FLUX SR Evaluation - Simplified Version

与 train_dual_control_simple.py 完全匹配

Usage:
    python evaluate_dual_control_simple.py \
        --checkpoint checkpoints/dual_simple/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --tile_size 1024 --overlap 128
"""

import os
import gc
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Note: lpips not installed. Run: pip install lpips")


# ============================================================================
# Pixel Feature Extractor (与训练代码完全一致)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )
        
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
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


# ============================================================================
# Evaluator
# ============================================================================

class DualStreamEvaluator:
    def __init__(self, model_name, device, checkpoint_path=None, pixel_weight=0.1):
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.pixel_weight = pixel_weight  # 🌟 可从 checkpoint 加载
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        self.ckpt_info = {}
    
    def load(self):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        print(f"[Eval] Loading to {self.device}...")
        
        # Load VAE
        print("[Eval] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        # Cache embeddings
        print("[Eval] Caching text embeddings...")
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
        print("[Eval] Loading FLUX Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        print("[Eval] Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=dtype
        ).to(self.device)
        
        # Initialize Pixel Extractor
        print("[Eval] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        
        # Load checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"[Eval] Loading checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.ckpt_info = ckpt
            
            # 🌟 从 checkpoint 加载 pixel_weight
            if 'pixel_weight' in ckpt:
                self.pixel_weight = ckpt['pixel_weight']
                print(f"[Eval] Loaded pixel_weight: {self.pixel_weight}")
            
            # Load Pixel Extractor
            if 'pixel_extractor' in ckpt:
                raw_state = ckpt['pixel_extractor']
                clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
                self.pixel_extractor.load_state_dict(clean_state)
                print("[Eval] ✓ Loaded Pixel Extractor")
            
            # Load ControlNet
            if 'controlnet' in ckpt:
                raw_state = ckpt['controlnet']
                clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
                self.controlnet.load_state_dict(clean_state)
                print("[Eval] ✓ Loaded ControlNet")
            
            print(f"[Eval] Checkpoint info: epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', 0):.4f}, psnr={ckpt.get('psnr', 0):.2f}")
        else:
            print("[Eval] ⚠ No checkpoint provided, using pretrained weights only")
        
        self.controlnet.requires_grad_(False)
        self.pixel_extractor.requires_grad_(False)
        self.controlnet.eval()
        self.pixel_extractor.eval()
        
        print(f"[Eval] Pixel weight: {self.pixel_weight}")
        print(f"[Eval] Ready. GPU: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
    
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
    
    @torch.no_grad()
    def forward(self, noisy, packed_cond, t, guidance=3.5):
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        noisy = noisy.to(dtype)
        t = t.to(dtype)
        
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        packed_noisy = self._pack(noisy)
        
        # ControlNet
        ctrl_kwargs = {
            'hidden_states': packed_noisy,
            'controlnet_cond': packed_cond,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'guidance': torch.full((B,), guidance, device=device, dtype=dtype),
            'return_dict': False,
        }
        
        ctrl_out = self.controlnet(**ctrl_kwargs)
        
        ctrl_block_samples = [x.to(dtype) for x in ctrl_out[0]] if ctrl_out[0] is not None else None
        ctrl_single_samples = [x.to(dtype) for x in ctrl_out[1]] if ctrl_out[1] is not None else None
        
        # Transformer
        trans_kwargs = {
            'hidden_states': packed_noisy,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
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
        🌟 Standard Flow Matching 推理：从纯噪声出发
        """
        B = lr_lat.shape[0]
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # 🌟 预计算 fused condition
        pixel_feat = self.pixel_extractor(lr_pixel).to(dtype)
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        packed_cond = self._pack(fused_cond)
        
        # 🌟 Standard: 从纯噪声出发
        lat = torch.randn_like(lr_lat)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=dtype)
            
            # 模型预测使用 bfloat16
            v = self.forward(lat, packed_cond, t, guidance)
            
            # 🌟 核心改进：在进行 ODE 步进更新时，转换为 float32 保证数值绝对精准
            lat_f32 = lat.float()
            v_f32 = v.float()
            lat_f32 = lat_f32 - dt * v_f32
            
            # 更新完再转回 bfloat16 给下一步使用
            lat = lat_f32.to(dtype)
        
        return lat


# ============================================================================
# Tiled Inference
# ============================================================================

@torch.no_grad()
def run_sr_tiled(evaluator, lr_tensor, device, num_steps=20, guidance=3.5,
                 tile_size=1024, overlap=128, blend_mode='gaussian'):
    _, _, H, W = lr_tensor.shape
    
    # Small image: direct inference
    if H <= tile_size and W <= tile_size:
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            lr_padded = lr_tensor
        
        lr_lat = evaluator.encode(lr_padded)
        sr_lat = evaluator.inference(lr_lat, lr_padded, num_steps, guidance)
        sr_padded = evaluator.decode(sr_lat)
        return sr_padded[:, :, :H, :W]
    
    # Large image: tiled inference
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros((1, 1, H, W), device=device)
    stride = tile_size - overlap
    
    # Gaussian blending weights
    if blend_mode == 'gaussian':
        sigma = tile_size / 6
        y_coords = torch.arange(tile_size, device=device).float() - tile_size / 2
        x_coords = torch.arange(tile_size, device=device).float() - tile_size / 2
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)).unsqueeze(0).unsqueeze(0)
    
    # Calculate tile positions
    h_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if h_starts[-1] + tile_size < H:
        h_starts.append(max(0, H - tile_size))
    
    w_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if w_starts[-1] + tile_size < W:
        w_starts.append(max(0, W - tile_size))
    
    total_tiles = len(h_starts) * len(w_starts)
    pbar = tqdm(total=total_tiles, desc="Tiles", leave=False)
    
    for h_start in h_starts:
        for w_start in w_starts:
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            th, tw = tile.shape[2], tile.shape[3]
            
            if th < tile_size or tw < tile_size:
                tile_padded = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            else:
                tile_padded = tile
            
            lr_lat = evaluator.encode(tile_padded)
            sr_lat = evaluator.inference(lr_lat, tile_padded, num_steps, guidance)
            tile_out_padded = evaluator.decode(sr_lat)
            
            tile_out = tile_out_padded[:, :, :th, :tw]
            
            if blend_mode == 'gaussian':
                tile_weight = gaussian[:, :, :th, :tw]
            else:
                tile_weight = torch.ones(1, 1, th, tw, device=device)
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_out * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
            pbar.update(1)
    
    pbar.close()
    return output / (weight + 1e-8)


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
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--blend_mode', type=str, default='gaussian')
    
    # 🌟 可手动覆盖 pixel_weight
    parser.add_argument('--pixel_weight', type=float, default=None,
                        help='Override pixel weight (default: load from checkpoint)')
    
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset
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
    
    # Experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{ts}_dual_simple_{args.num_steps}step"
    
    # Output directory
    output_dir = os.path.join(args.output_base, args.dataset, 'DualSimple', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    # 初始 pixel_weight（可能被 checkpoint 覆盖）
    initial_pixel_weight = args.pixel_weight if args.pixel_weight is not None else 0.1
    
    print("=" * 70)
    print("Dual-Stream FLUX SR Evaluation (Simplified)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load model
    evaluator = DualStreamEvaluator(args.model_name, device, args.checkpoint, initial_pixel_weight)
    evaluator.load()
    
    # 🌟 如果命令行指定了 pixel_weight，覆盖 checkpoint 中的值
    if args.pixel_weight is not None:
        evaluator.pixel_weight = args.pixel_weight
        print(f"[Eval] Overriding pixel_weight to: {args.pixel_weight}")
    
    # Load LPIPS
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
    
    # Get files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(hr_files) == len(lr_files), f"HR/LR mismatch: {len(hr_files)} vs {len(lr_files)}"
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    # Metrics
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    filenames = []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files), desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        filenames.append(base_name)
        
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        
        hr_np = np.array(hr_img)
        H, W = hr_np.shape[0], hr_np.shape[1]
        
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device).to(torch.bfloat16)
        
        sr_t = run_sr_tiled(
            evaluator, lr_t, device,
            num_steps=args.num_steps,
            guidance=args.guidance,
            tile_size=args.tile_size,
            overlap=args.overlap,
            blend_mode=args.blend_mode
        )
        
        sr_np = ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Metrics - SR
        psnr_val = calculate_psnr(sr_np, hr_np)
        ssim_val = calculate_ssim(sr_np, hr_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        # Metrics - Bicubic
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        # LPIPS
        if lpips_fn:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            sr_t_lpips = torch.from_numpy(sr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lr_t_lpips = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            
            lpips_val = lpips_fn(sr_t_lpips.to(device), hr_t.to(device)).item()
            lpips_bic = lpips_fn(lr_t_lpips.to(device), hr_t.to(device)).item()
            lpips_list.append(lpips_val)
            lpips_bic_list.append(lpips_bic)
        
        # Save images
        if args.save_images:
            Image.fromarray(sr_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        if args.save_comparisons:
            comp = Image.new('RGB', (W * 4, H))
            lr_display = lr_img.resize((W, H), Image.NEAREST)
            comp.paste(lr_display, (0, 0))
            comp.paste(lr_bicubic, (W, 0))
            comp.paste(Image.fromarray(sr_np), (W * 2, 0))
            comp.paste(hr_img, (W * 3, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
        
        torch.cuda.empty_cache()
    
    # Results
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    if LPIPS_AVAILABLE:
        print(f"Bicubic:     PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"Dual-Simple: PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"Δ:           {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
    else:
        print(f"Bicubic:     PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
        print(f"Dual-Simple: PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
    print("=" * 70)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("Dual-Stream FLUX SR Evaluation (Simplified)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}\n")
        f.write(f"Tile: {args.tile_size}, Overlap: {args.overlap}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary:\n")
        f.write("=" * 60 + "\n")
        if LPIPS_AVAILABLE:
            f.write(f"Bicubic:     PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}\n")
            f.write(f"Dual-Simple: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}\n")
            f.write(f"Δ:           {avg_psnr - avg_psnr_bic:+.4f}, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}\n")
        else:
            f.write(f"Bicubic:     PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}\n")
            f.write(f"Dual-Simple: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Per-image results:\n")
        f.write("=" * 60 + "\n")
        for i, fname in enumerate(filenames):
            delta_psnr = psnr_list[i] - psnr_bic_list[i]
            if LPIPS_AVAILABLE:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (Δ{delta_psnr:+.2f}), LPIPS={lpips_list[i]:.4f}\n")
            else:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (Δ{delta_psnr:+.2f})\n")
    
    print(f"\n✅ Results saved: {output_dir}")


if __name__ == '__main__':
    main()
