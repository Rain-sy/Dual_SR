#!/usr/bin/env python
"""
FLUX SR ControlNet Evaluation Script (Tiled Inference + Multi-GPU)

Output structure:
    outputs/
    └── DIV2K/
        └── FluxControl/
            └── 20250220_pretrained_20step/
                ├── predictions/
                ├── comparisons/
                └── results.txt

Usage:
    # Single GPU
    python evaluate_flux_control.py \
        --use_pretrained \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4

    # Multi-GPU (each GPU processes different images)
    CUDA_VISIBLE_DEVICES=0 python evaluate_flux_control.py --use_pretrained --gpu_id 0 --num_gpus 4 ... &
    CUDA_VISIBLE_DEVICES=1 python evaluate_flux_control.py --use_pretrained --gpu_id 1 --num_gpus 4 ... &
    CUDA_VISIBLE_DEVICES=2 python evaluate_flux_control.py --use_pretrained --gpu_id 2 --num_gpus 4 ... &
    CUDA_VISIBLE_DEVICES=3 python evaluate_flux_control.py --use_pretrained --gpu_id 3 --num_gpus 4 ... &
    wait
    # Then merge results manually or use --merge_results
"""

import os
import gc
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Note: lpips not installed. Run: pip install lpips")


# ============================================================================
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two numpy arrays (H, W, C) in [0, 255]"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two numpy arrays (H, W, C) in [0, 255]"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


# ============================================================================
# FLUX SR Evaluator
# ============================================================================

class FLUXSREvaluator:
    """FLUX SR evaluator"""
    
    def __init__(self, model_name, device, checkpoint_path=None):
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        self.transformer = None
        self.vae = None
        self.controlnet = None
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
        
        # Cache text embeddings
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
        
        # Load transformer
        print("[Eval] Loading transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        print("[Eval] Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=dtype
        ).to(self.device)
        
        # Load finetuned weights if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"[Eval] Loading finetuned weights: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # 🌟 核心修复：自动剥离多卡训练带来的 'module.' 前缀
            raw_state_dict = ckpt['controlnet']
            clean_state_dict = {}
            for k, v in raw_state_dict.items():
                clean_key = k.replace('module.', '')
                clean_state_dict[clean_key] = v
            
            self.controlnet.load_state_dict(clean_state_dict)
            self.ckpt_info = ckpt
            print(f"[Eval] Loaded: epoch={ckpt.get('epoch', '?')}, psnr={ckpt.get('psnr', '?')}")
        else:
            print("[Eval] Using pretrained ControlNet (no finetuning)")
        
        self.controlnet.requires_grad_(False)
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
    def forward(self, noisy, lr_lat, t, guidance=3.5):
        B, C, H, W = noisy.shape
        device, dtype = noisy.device, noisy.dtype
        
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(lr_lat)
        
        # ControlNet
        ctrl_kwargs = {
            'hidden_states': packed_noisy,
            'controlnet_cond': packed_cond,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'return_dict': False,
        }
        if "Guidance" in type(self.controlnet.time_text_embed).__name__:
            ctrl_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        ctrl_out = self.controlnet(**ctrl_kwargs)
        
        # Transformer
        trans_kwargs = {
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
        if "Guidance" in type(self.transformer.time_text_embed).__name__:
            trans_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        pred = self.transformer(**trans_kwargs)[0]
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, num_steps=20, guidance=3.5):
        """Generate SR using Euler integration (flow matching)"""
        B = lr_lat.shape[0]
        lat = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=lr_lat.dtype)
            v = self.forward(lat, lr_lat, t, guidance)
            lat = lat - dt * v
        
        return lat
    
    @torch.no_grad()
    def encode(self, img):
        return self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, lat):
        return self.vae.decode(lat / self.vae.config.scaling_factor).sample


# ============================================================================
# Tiled Inference
# ============================================================================

@torch.no_grad()
def run_sr_tiled_flux(evaluator, lr_tensor, device, num_steps=20, guidance=3.5, 
                      tile_size=1024, overlap=128, blend_mode='gaussian'):
    """
    Tiled inference with Gaussian blending for arbitrary high resolutions.
    直接处理全尺寸原图，无需裁剪。
    """
    _, _, H, W = lr_tensor.shape
    
    # 如果图像足够小，直接进行单次推理
    if H <= tile_size and W <= tile_size:
        pad_h, pad_w = (16 - H % 16) % 16, (16 - W % 16) % 16
        lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        lr_lat = evaluator.encode(lr_padded)
        sr_lat = evaluator.inference(lr_lat, num_steps, guidance)
        sr_padded = evaluator.decode(sr_lat)
        return sr_padded[:, :, :H, :W]
    
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    stride = tile_size - overlap
    
    # 预计算高斯融合权重掩码
    if blend_mode == 'gaussian':
        sigma = tile_size / 6
        y_coords = torch.arange(tile_size, device=device).float() - tile_size / 2
        x_coords = torch.arange(tile_size, device=device).float() - tile_size / 2
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)).unsqueeze(0).unsqueeze(0)
    
    # 计算切割点
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
            h_end, w_end = min(h_start + tile_size, H), min(w_start + tile_size, W)
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            th, tw = tile.shape[2], tile.shape[3]
            
            # 边缘 Tile 补齐到 tile_size，使用 reflect 防止黑边伪影
            if th < tile_size or tw < tile_size:
                tile_padded = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            else:
                tile_padded = tile
            
            # FLUX 编码 -> 推理 -> 解码
            lr_lat = evaluator.encode(tile_padded)
            sr_lat = evaluator.inference(lr_lat, num_steps, guidance)
            tile_out_padded = evaluator.decode(sr_lat)
            
            # 切除多余的 Pad
            tile_out = tile_out_padded[:, :, :th, :tw]
            
            # 融合权重
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
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    
    # Tiled inference
    parser.add_argument('--tile_size', type=int, default=1024, help='Tile size (48GB: 1024, 24GB: 512)')
    parser.add_argument('--overlap', type=int, default=128, help='Overlap between tiles')
    parser.add_argument('--blend_mode', type=str, default='gaussian', choices=['gaussian', 'linear'])
    
    # Multi-GPU support
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for multi-GPU split')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs')
    
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if not args.use_pretrained and not args.checkpoint:
        print("Error: --checkpoint is required unless --use_pretrained is set")
        return
    
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
        elif 'bsd100' in hr_lower or 'b100' in hr_lower:
            args.dataset = 'BSD100'
        elif 'manga' in hr_lower:
            args.dataset = 'Manga109'
        else:
            args.dataset = 'Unknown'
    
    # Build experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = 'pretrained' if args.use_pretrained else 'finetuned'
        exp_name = f"{ts}_{mode}_{args.num_steps}step"
    
    # Output dir
    output_dir = os.path.join(args.output_base, args.dataset, 'FluxControl', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("=" * 70)
    print("FLUX SR ControlNet Evaluation (Tiled + High-Res)")
    print("=" * 70)
    print(f"Mode: {'Pretrained' if args.use_pretrained else 'Finetuned'}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Inference steps: {args.num_steps}, Guidance: {args.guidance}")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}, Blend: {args.blend_mode}")
    if args.num_gpus > 1:
        print(f"Multi-GPU: GPU {args.gpu_id + 1}/{args.num_gpus}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load model
    checkpoint = args.checkpoint if not args.use_pretrained else None
    evaluator = FLUXSREvaluator(args.model_name, device, checkpoint)
    evaluator.load()
    
    # Show checkpoint info
    if evaluator.ckpt_info:
        if 'epoch' in evaluator.ckpt_info:
            print(f"Trained for: {evaluator.ckpt_info['epoch'] + 1} epochs")
        if 'psnr' in evaluator.ckpt_info:
            print(f"Training best PSNR: {evaluator.ckpt_info['psnr']:.2f} dB")
        if 'loss' in evaluator.ckpt_info:
            print(f"Final loss: {evaluator.ckpt_info['loss']:.6f}")
        print("=" * 70)
    
    # Load LPIPS
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
        print("[Eval] LPIPS loaded")
    
    # Get files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(hr_files) == len(lr_files), f"HR/LR mismatch: {len(hr_files)} vs {len(lr_files)}"
    
    # Multi-GPU: split files
    if args.num_gpus > 1:
        total_files = len(hr_files)
        files_per_gpu = (total_files + args.num_gpus - 1) // args.num_gpus
        start_idx = args.gpu_id * files_per_gpu
        end_idx = min(start_idx + files_per_gpu, total_files)
        hr_files = hr_files[start_idx:end_idx]
        lr_files = lr_files[start_idx:end_idx]
        print(f"[GPU {args.gpu_id}] Processing images {start_idx}-{end_idx-1} ({len(hr_files)} images)")
    
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    # Metrics storage
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    filenames = []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files), desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        filenames.append(base_name)
        
        # Load images
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        
        hr_np = np.array(hr_img)
        H, W = hr_np.shape[0], hr_np.shape[1]
        
        # Bicubic upscale
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        # To tensor
        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device)
        
        # 🌟 Tiled inference (全尺寸原图)
        sr_t = run_sr_tiled_flux(
            evaluator, lr_t, device,
            num_steps=args.num_steps,
            guidance=args.guidance,
            tile_size=args.tile_size,
            overlap=args.overlap,
            blend_mode=args.blend_mode
        )
        
        sr_np = ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Calculate metrics (全尺寸)
        psnr_val = calculate_psnr(sr_np, hr_np)
        ssim_val = calculate_ssim(sr_np, hr_np)
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
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
            # Comparison: LR | Bicubic | SR | GT
            comp = Image.new('RGB', (W * 4, H))
            lr_display = lr_img.resize((W, H), Image.NEAREST)  # Nearest 凸显像素感
            comp.paste(lr_display, (0, 0))
            comp.paste(lr_bicubic, (W, 0))
            comp.paste(Image.fromarray(sr_np), (W * 2, 0))
            comp.paste(hr_img, (W * 3, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
    
    # Calculate averages
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    if LPIPS_AVAILABLE:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"Δ:        {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
    else:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
        print(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
        print(f"Δ:        {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}")
    print("=" * 70)
    
    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    if avg_psnr > avg_psnr_bic:
        print(f"✓ PSNR: Model is better by {avg_psnr - avg_psnr_bic:.2f} dB")
    else:
        print(f"⚠ PSNR: Model is lower by {avg_psnr_bic - avg_psnr:.2f} dB (生成式模型特点)")
    
    if LPIPS_AVAILABLE:
        if avg_lpips < avg_lpips_bic:
            print(f"✓ LPIPS: Model has better perceptual quality ({avg_lpips_bic - avg_lpips:.4f} improvement)")
        else:
            print(f"⚠ LPIPS: Model has worse perceptual quality")
    print("=" * 70)
    
    # Save results.txt
    suffix = f"_gpu{args.gpu_id}" if args.num_gpus > 1 else ""
    results_path = os.path.join(output_dir, f'results{suffix}.txt')
    
    with open(results_path, 'w') as f:
        f.write("FLUX SR ControlNet Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {'Pretrained' if args.use_pretrained else 'Finetuned'}\n")
        if args.checkpoint:
            f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n")
        f.write(f"Inference steps: {args.num_steps}\n")
        f.write(f"Guidance: {args.guidance}\n")
        f.write(f"Tile size: {args.tile_size}, Overlap: {args.overlap}\n")
        
        if evaluator.ckpt_info:
            if 'epoch' in evaluator.ckpt_info:
                f.write(f"Epochs trained: {evaluator.ckpt_info['epoch'] + 1}\n")
            if 'loss' in evaluator.ckpt_info:
                f.write(f"Final loss: {evaluator.ckpt_info['loss']:.6f}\n")
        
        f.write("\n")
        if LPIPS_AVAILABLE:
            f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}\n")
            f.write(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}\n")
            f.write(f"Δ:        {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}\n")
        else:
            f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}\n")
            f.write(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}\n")
            f.write(f"Δ:        {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Per-image results:\n")
        f.write("-" * 80 + "\n")
        if LPIPS_AVAILABLE:
            f.write(f"{'Filename':<25} {'SR PSNR':<10} {'Bic PSNR':<10} {'Δ PSNR':<10} {'SR LPIPS':<10} {'Bic LPIPS':<10}\n")
            for i, fname in enumerate(filenames):
                delta = psnr_list[i] - psnr_bic_list[i]
                f.write(f"{fname:<25} {psnr_list[i]:<10.2f} {psnr_bic_list[i]:<10.2f} {delta:+<10.2f} {lpips_list[i]:<10.4f} {lpips_bic_list[i]:<10.4f}\n")
        else:
            f.write(f"{'Filename':<25} {'SR PSNR':<10} {'Bic PSNR':<10} {'Δ':<10}\n")
            for i, fname in enumerate(filenames):
                delta = psnr_list[i] - psnr_bic_list[i]
                f.write(f"{fname:<25} {psnr_list[i]:<10.2f} {psnr_bic_list[i]:<10.2f} {delta:+.2f}\n")
    
    print(f"\n✅ Results saved: {output_dir}")
    print(f"   - results{suffix}.txt")
    if args.save_images:
        print(f"   - predictions/")
    if args.save_comparisons:
        print(f"   - comparisons/ (LR | Bicubic | SR | GT)")
    
    return avg_psnr


if __name__ == '__main__':
    main()