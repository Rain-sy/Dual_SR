import os
from datasets import load_dataset
from tqdm import tqdm

def extract_lsdir_parquet():
    # 你下载的 parquet 文件所在的路径
    input_pattern = "./Data/LSDIR/data/*.parquet"
    # 我们要提取出来的目标 HR 文件夹
    output_dir = "./Data/LSDIR/LSDIR_HR"
    # 只需要 8000 张
    target_count = 8000
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📦 正在读取 Parquet 文件 (这可能需要十几秒来建立索引)...")
    try:
        # 加载本地的 parquet 文件
        dataset = load_dataset("parquet", data_files=input_pattern, split="train")
    except Exception as e:
        print(f"❌ 加载失败，请检查路径: {e}")
        return
        
    # 检查总数据量以防万一
    total_available = len(dataset)
    print(f"📊 数据集中总共有 {total_available} 张图片。")
    
    actual_count = min(target_count, total_available)
    print(f"🚀 开始提取前 {actual_count} 张图片到 {output_dir} ...")
    
    # 自动获取图片列的名称（通常是 'image'）
    image_column = 'image' if 'image' in dataset.column_names else dataset.column_names[0]
    
    for i in tqdm(range(actual_count), desc="提取进度"):
        try:
            # Hugging Face 会自动将 parquet 数据解码为 PIL.Image 对象
            img = dataset[i][image_column]
            
            # 强制转换为 RGB（防止灰度图或带透明通道的图）
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 保存为 PNG
            save_path = os.path.join(output_dir, f"LSDIR_{i:05d}.png")
            img.save(save_path, format="PNG")
        except Exception as e:
            print(f"⚠️ 提取第 {i} 张图片时出错: {e}")
            
    print("\n" + "="*50)
    print(f"✅ 提取大功告成！你需要的 8000 张图已经安静地躺在 {output_dir} 里了。")
    print("="*50)

if __name__ == "__main__":
    extract_lsdir_parquet()