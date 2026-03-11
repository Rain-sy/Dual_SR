"""
Generate LR images from HR images using bicubic downsampling

Usage:
    python create_lr_dataset.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_train_HR \
        --output_dir meanflow/Data/DIV2K/DIV2K_train_LR_bicubic_X8 \
        --scale 8

    python create_lr_dataset.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_valid_HR \
        --output_dir meanflow/Data/DIV2K/DIV2K_valid_LR_bicubic_X8 \
        --scale 8
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm


def create_lr_images(hr_dir, output_dir, scale=8):
    """
    Create LR images by bicubic downsampling
    
    Args:
        hr_dir: Directory containing HR images
        output_dir: Output directory for LR images
        scale: Downsampling scale factor
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    hr_files = sorted([f for f in os.listdir(hr_dir) 
                       if f.lower().endswith(extensions)])
    
    print(f"Found {len(hr_files)} HR images in {hr_dir}")
    print(f"Creating {scale}x downsampled LR images in {output_dir}")
    
    for filename in tqdm(hr_files, desc=f"Generating {scale}x LR"):
        hr_path = os.path.join(hr_dir, filename)
        
        # Load HR image
        hr_img = Image.open(hr_path).convert('RGB')
        hr_w, hr_h = hr_img.size
        
        # Calculate LR size (ensure divisible by scale)
        lr_w = hr_w // scale
        lr_h = hr_h // scale
        
        # Downscale using bicubic interpolation
        lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
        
        # Save LR image (use PNG to avoid compression artifacts)
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        lr_img.save(output_path)
    
    print(f"\nDone! Created {len(hr_files)} LR images")
    print(f"  HR size example: {hr_w}x{hr_h}")
    print(f"  LR size example: {lr_w}x{lr_h} ({scale}x downsampled)")


def main():
    parser = argparse.ArgumentParser(description='Create LR dataset from HR images')
    parser.add_argument('--hr_dir', type=str, required=True,
                        help='Directory containing HR images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for LR images')
    parser.add_argument('--scale', type=int, default=8,
                        help='Downsampling scale factor (default: 8)')
    
    args = parser.parse_args()
    
    create_lr_images(args.hr_dir, args.output_dir, args.scale)


if __name__ == '__main__':
    main()