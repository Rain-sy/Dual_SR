#!/usr/bin/env python
"""
Latent Caching Script for FLUX SR Training

将 HR/LR 图片预先编码为 latent 并保存到磁盘，加速后续训练。

Usage:
    python cache_latents.py \
        --hr_dir Data/Mix15K_HR \
        --lr_dir Data/Mix15K_LR_bicubic_X4 \
        --output_dir Data/Mix15K_latents \
        --resolution 512

效果：
    - 训练时跳过 VAE 编码，每个 batch 省 ~0.5s
    - 无需加载 VAE 到 GPU，省 ~3GB 显存
    - Mix15K (15000张) 预处理约需 30-60 分钟
"""

import os
import gc
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def process_image(img_path, resolution):
    """加载并预处理图片"""
    img = Image.open(img_path).convert('RGB')
    
    # 获取 LR crop 大小
    lr_w, lr_h = img.size
    crop_lr = resolution // 4
    
    # 如果图片足够大，随机裁剪；否则 resize
    if lr_w >= crop_lr and lr_h >= crop_lr:
        # 对于缓存，我们使用中心裁剪而非随机裁剪
        # 这样每次运行结果一致
        x = (lr_w - crop_lr) // 2
        y = (lr_h - crop_lr) // 2
        img = img.crop((x, y, x + crop_lr, y + crop_lr))
    
    # Resize 到目标分辨率
    img = img.resize((resolution, resolution), Image.BICUBIC)
    
    # 转为 tensor [-1, 1]
    img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
    return img_t


def main():
    parser = argparse.ArgumentParser(description='Cache latents for FLUX SR training')
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for cached latents')
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--resolution', type=int, default=512, help='Target resolution')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = {
        'hr_dir': args.hr_dir,
        'lr_dir': args.lr_dir,
        'resolution': args.resolution,
        'model_name': args.model_name,
    }
    torch.save(config, output_dir / 'config.pt')
    
    print("=" * 70)
    print("🚀 Latent Caching for FLUX SR")
    print("=" * 70)
    print(f"HR Dir: {args.hr_dir}")
    print(f"LR Dir: {args.lr_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Resolution: {args.resolution}")
    print("=" * 70)
    
    # 加载 VAE
    print("\n[1/3] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_name, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(args.device)
    vae.requires_grad_(False)
    vae.eval()
    
    # 获取文件列表
    hr_files = sorted([f for f in os.listdir(args.hr_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(hr_files) == len(lr_files), f"HR/LR mismatch: {len(hr_files)} vs {len(lr_files)}"
    print(f"\n[2/3] Found {len(hr_files)} image pairs")
    
    # 编码并保存
    print(f"\n[3/3] Encoding and caching latents...")
    
    scaling_factor = vae.config.scaling_factor
    
    for i, (hr_f, lr_f) in enumerate(tqdm(zip(hr_files, lr_files), total=len(hr_files))):
        # 生成输出文件名
        base_name = os.path.splitext(hr_f)[0]
        output_path = output_dir / f"{base_name}.pt"
        
        # 如果已存在则跳过
        if output_path.exists():
            continue
        
        # 加载 HR 图片
        hr_path = os.path.join(args.hr_dir, hr_f)
        hr_img = Image.open(hr_path).convert('RGB')
        
        # 加载 LR 图片
        lr_path = os.path.join(args.lr_dir, lr_f)
        lr_img = Image.open(lr_path).convert('RGB')
        
        # 获取 crop 参数
        lr_w, lr_h = lr_img.size
        crop_lr = args.resolution // 4
        
        # 随机位置（但为了可复现性，用固定种子）
        # 训练时会有数据增强，这里用多个 crop 增加多样性
        crops_to_save = []
        
        if lr_w >= crop_lr and lr_h >= crop_lr:
            # 保存多个随机 crop（增加数据多样性）
            np.random.seed(i)  # 可复现
            
            # ✅ 根据图片大小动态决定 crop 数量
            # 大图切更多块，最多 8 个（平衡存储和多样性）
            max_crops = (lr_w // crop_lr) * (lr_h // crop_lr)
            num_crops = min(8, max(1, max_crops))  # 1-8 个 crop
            
            for c in range(num_crops):
                x = np.random.randint(0, lr_w - crop_lr + 1)
                y = np.random.randint(0, lr_h - crop_lr + 1)
                
                # Crop LR
                lr_crop = lr_img.crop((x, y, x + crop_lr, y + crop_lr))
                # Crop HR (4x)
                hr_crop = hr_img.crop((x * 4, y * 4, (x + crop_lr) * 4, (y + crop_lr) * 4))
                
                # Resize
                lr_up = lr_crop.resize((args.resolution, args.resolution), Image.BICUBIC)
                hr_resized = hr_crop.resize((args.resolution, args.resolution), Image.BICUBIC)
                
                # 转 tensor
                lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1.0
                hr_t = torch.from_numpy(np.array(hr_resized)).float().permute(2, 0, 1) / 127.5 - 1.0
                
                crops_to_save.append((hr_t, lr_t))
        else:
            # 图片太小，直接 resize
            lr_up = lr_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            hr_resized = hr_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            
            lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1.0
            hr_t = torch.from_numpy(np.array(hr_resized)).float().permute(2, 0, 1) / 127.5 - 1.0
            
            crops_to_save.append((hr_t, lr_t))
        
        # 编码所有 crops
        all_hr_latents = []
        all_lr_latents = []
        all_lr_pixels = []
        
        for hr_t, lr_t in crops_to_save:
            hr_t = hr_t.unsqueeze(0).to(args.device).to(torch.bfloat16)
            lr_t = lr_t.unsqueeze(0).to(args.device).to(torch.bfloat16)
            
            with torch.no_grad():
                hr_lat = vae.encode(hr_t).latent_dist.sample() * scaling_factor
                lr_lat = vae.encode(lr_t).latent_dist.sample() * scaling_factor
            
            # ✅ 修复：保持 bfloat16，避免 FP16 溢出风险
            all_hr_latents.append(hr_lat.cpu())  # 保持 bfloat16
            all_lr_latents.append(lr_lat.cpu())  # 保持 bfloat16
            all_lr_pixels.append(lr_t.cpu())     # 保持 bfloat16
        
        # 保存
        torch.save({
            'hr_latents': all_hr_latents,  # List of [1, 16, H/16, W/16]
            'lr_latents': all_lr_latents,  # List of [1, 16, H/16, W/16]
            'lr_pixels': all_lr_pixels,     # List of [1, 3, H, W] - 用于 PixelExtractor
            'num_crops': len(crops_to_save),
        }, output_path)
        
        # 定期清理
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # 统计
    total_files = len(list(output_dir.glob("*.pt"))) - 1  # 减去 config.pt
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.pt")) / (1024**3)
    
    print("\n" + "=" * 70)
    print("✅ Caching Complete!")
    print("=" * 70)
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size:.2f} GB")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # 创建索引文件
    index = {
        'files': [f.stem for f in sorted(output_dir.glob("*.pt")) if f.stem != 'config'],
        'resolution': args.resolution,
        'total': total_files,
    }
    torch.save(index, output_dir / 'index.pt')
    print(f"\nIndex saved to {output_dir / 'index.pt'}")


if __name__ == '__main__':
    main()
