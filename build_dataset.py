import os
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def process_dataset(source_dir, target_hr_dir, target_lr_dir, prefix, sample_count=None):
    valid_exts = {'.png', '.jpg', '.jpeg'}
    all_files = [f for f in os.listdir(source_dir) if Path(f).suffix.lower() in valid_exts]
    
    if not all_files:
        print(f"❌ [警告] 在 {source_dir} 中没有找到图片，请检查路径是否正确！")
        return 0

    if sample_count and sample_count < len(all_files):
        print(f"[{prefix}] 随机抽样 {sample_count} / {len(all_files)} 张图片...")
        selected_files = random.sample(all_files, sample_count)
    else:
        print(f"[{prefix}] 提取全部 {len(all_files)} 张图片...")
        selected_files = all_files

    processed_count = 0
    for i, filename in enumerate(tqdm(selected_files, desc=f"处理 {prefix}")):
        try:
            img_path = os.path.join(source_dir, filename)
            hr_img = Image.open(img_path).convert('RGB')
            
            # 生成新文件名，防止不同数据集重名
            new_filename = f"{prefix}_{i:05d}.png"
            hr_save_path = os.path.join(target_hr_dir, new_filename)
            lr_save_path = os.path.join(target_lr_dir, new_filename)
            
            # 1. 保存 HR
            hr_img.save(hr_save_path, format='PNG')
            
            # 2. 生成 LR (严格使用 Bicubic 降采样 4 倍)
            W, H = hr_img.size
            lr_img = hr_img.resize((W // 4, H // 4), Image.BICUBIC)
            lr_img.save(lr_save_path, format='PNG')
            
            processed_count += 1
        except Exception as e:
            print(f"跳过损坏图片 {filename}: {e}")
            
    return processed_count

if __name__ == "__main__":
    # ================= 配置区 =================
    # 🌟 极度重要：请确保这里的 path 指向的是真正装有 .png 图片的文件夹！
    # 如果解压后路径有变，请手动修改下方路径
    SOURCES = {
        "DIV2K":    {"path": "./Data/DIV2K/DIV2K_train_HR", "count": 800},
        "Flickr2K": {"path": "./Data/Flickr2K/Flickr2K", "count": 2650},
        "LSDIR":    {"path": "./Data/LSDIR/LSDIR_HR", "count": 8000},
        "FFHQ":     {"path": "./Data/FFHQ", "count": 3550}, # 15000 - 800 - 2650 - 8000 = 3550
    }
    
    # 输出的大数据集路径
    TARGET_HR = "./Data/Mix15K_HR"
    TARGET_LR = "./Data/Mix15K_LR_bicubic_X4"
    # ==========================================

    os.makedirs(TARGET_HR, exist_ok=True)
    os.makedirs(TARGET_LR, exist_ok=True)
    
    total_images = 0
    for prefix, info in SOURCES.items():
        if os.path.exists(info["path"]):
            processed = process_dataset(info["path"], TARGET_HR, TARGET_LR, prefix, info["count"])
            total_images += processed
        else:
            print(f"\n❌ [致命错误] 找不到文件夹: {info['path']}")
            print("请检查该数据集是否解压成功，或者文件夹名字是否多了一层（比如 ./Data/FFHQ/images）")
            
    print("\n" + "=" * 50)
    print(f"✅ Mix15K 数据集构建完毕！共计成功处理图片: {total_images} 张")
    print(f"HR 路径: {TARGET_HR}")
    print(f"LR 路径: {TARGET_LR}")
    print("=" * 50)