import os
import sys
import numpy as np
from PIL import Image

# 直接检查 ACDC 数据集的标签文件
data_dir = "/home/deng/datasets/acdc_mmseg"

# 找一张训练图像和对应的标签
image_dir = os.path.join(data_dir, "leftImg8bit", "train")
label_dir = os.path.join(data_dir, "gtFine", "train")

# 取第一张图像
for fname in sorted(os.listdir(image_dir)):
    if fname.endswith(".png"):
        image_path = os.path.join(image_dir, fname)
        # 构建标签路径
        base_name = fname.replace("_leftImg8bit", "")
        base_name = os.path.splitext(base_name)[0]
        mask_name = f"{base_name}_gtFine_labelTrainIds.png"
        mask_path = os.path.join(label_dir, mask_name)
        
        if os.path.exists(mask_path):
            print(f"Image: {fname}")
            print(f"Label: {mask_name}")
            
            # 读取标签
            mask = np.array(Image.open(mask_path))
            unique_vals = np.unique(mask)
            print(f"Unique values in label: {unique_vals}")
            print(f"Label min: {mask.min()}, max: {mask.max()}")
            print(f"Label shape: {mask.shape}")
            print(f"Total pixels: {mask.size}")
            
            # 统计有效像素（假设 255 是忽略标签）
            valid_count = (mask != 255).sum()
            print(f"Valid pixels (not 255): {valid_count}/{mask.size} ({100*valid_count/mask.size:.1f}%)")
            
            # 检查是否有 0-18 的标签
            for cls in range(19):
                count = (mask == cls).sum()
                if count > 0:
                    print(f"  Class {cls}: {count} pixels")
            
            break
        else:
            print(f"Label not found for {fname}")
            break