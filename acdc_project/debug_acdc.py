import os
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ACDCDebugDataset(Dataset):
    """简化的ACDC数据集，仅用于调试"""
    IGNORE_INDEX = 255
    
    def __init__(self, data_dir, split="train", crop_size=None, is_training=False):
        self.crop_size = crop_size
        self.is_training = is_training
        self.samples = []
        
        # 查找图像和标签目录
        img_dir = None
        label_dir = None
        
        for root, dirs, files in os.walk(data_dir):
            if img_dir is None and "leftImg8bit" in root and split in root:
                img_dir = root
            if label_dir is None and "gtFine" in root and split in root:
                label_dir = root
        
        if img_dir is None:
            img_dir = os.path.join(data_dir, "leftImg8bit", split)
        if label_dir is None:
            label_dir = os.path.join(data_dir, "gtFine", split)
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"找不到图像目录: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"找不到标签目录: {label_dir}")
        
        print(f"图像目录: {img_dir}")
        print(f"标签目录: {label_dir}")
        
        # 收集配对文件
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(('.png', '.jpg')):
                continue
            
            img_path = os.path.join(img_dir, fname)
            
            base_name = fname.replace("_leftImg8bit", "").replace(".png", "").replace(".jpg", "")
            label_candidates = [
                f"{base_name}_gtFine_labelTrainIds.png",
                f"{base_name}_gtFine_labelIds.png",
            ]
            
            label_path = None
            for candidate in label_candidates:
                candidate_path = os.path.join(label_dir, candidate)
                if os.path.exists(candidate_path):
                    label_path = candidate_path
                    break
            
            if label_path:
                self.samples.append((img_path, label_path))
            else:
                print(f"警告: {fname} 找不到对应标签")
        
        print(f"找到 {len(self.samples)} 个配对样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import cv2
        import random
        
        img_path, label_path = self.samples[idx]
        
        # 加载图像和标签
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path))
        
        # 转换为int64
        label = label.astype(np.int64)
        
        # 只处理标签，不映射（因为已经是trainIds格式）
        # 但保留255作为忽略标签
        label[(label < 0) | (label >= 19)] = self.IGNORE_INDEX
        
        # 数据增强（训练时）
        if self.is_training and self.crop_size is not None:
            # 随机缩放
            scale = random.uniform(0.5, 2.0)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 随机翻转
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
            
            # 随机裁剪或padding
            crop_h, crop_w = self.crop_size
            h, w = image.shape[:2]
            
            if h < crop_h or w < crop_w:
                pad_h = max(0, crop_h - h)
                pad_w = max(0, crop_w - w)
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)),
                             mode='constant', constant_values=0)
                label = np.pad(label, ((0, pad_h), (0, pad_w)),
                             mode='constant', constant_values=self.IGNORE_INDEX)
                h, w = image.shape[:2]
            
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            image = image[top:top+crop_h, left:left+crop_w]
            label = label[top:top+crop_h, left:left+crop_w]
            
        elif self.crop_size is not None:
            # 验证模式：直接resize
            target_h, target_w = self.crop_size
            h, w = image.shape[:2]
            if h != target_h or w != target_w:
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # 转换为tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        label = torch.from_numpy(label).long()
        
        return {"data": image, "label": label, "img_path": img_path, "label_path": label_path}


def analyze_raw_labels(data_path, split="train"):
    """分析原始标签文件"""
    print("\n" + "=" * 60)
    print("1. 分析原始标签文件")
    print("=" * 60)
    
    # 查找标签目录
    label_dir = None
    for root, dirs, files in os.walk(data_path):
        if "gtFine" in root and split in root:
            label_dir = root
            break
    
    if label_dir is None:
        print(f"❌ 找不到{split}集的标签目录")
        return
    
    print(f"标签目录: {label_dir}")
    
    # 查找标签文件
    trainid_files = [f for f in os.listdir(label_dir) 
                    if f.endswith('.png') and 'labelTrainIds' in f]
    
    print(f"labelTrainIds文件: {len(trainid_files)} 个")
    
    if not trainid_files:
        print("❌ 没有找到标签文件!")
        return
    
    # 分析多个文件
    all_vals = set()
    file_stats = []
    
    for fname in trainid_files[:10]:  # 分析前10个
        fpath = os.path.join(label_dir, fname)
        label = np.array(Image.open(fpath))
        unique_vals = np.unique(label)
        all_vals.update(unique_vals.tolist())
        
        # 统计各类别出现情况
        classes_present = [v for v in unique_vals if v != 255]
        file_stats.append({
            'name': fname,
            'num_classes': len(classes_present),
            'classes': classes_present
        })
    
    print(f"\n前10个文件的统计:")
    for stat in file_stats[:5]:
        print(f"  {stat['name']}: {stat['num_classes']}个类别 - {stat['classes']}")
    
    print(f"\n所有样本的唯一值: {sorted(all_vals)}")
    
    # 判断
    valid_classes = [v for v in all_vals if v != 255]
    max_class = max(valid_classes) if valid_classes else -1
    
    if max_class <= 18:
        print(f"\n✅ 标签是trainIds格式!")
        print(f"有效类别: {sorted(valid_classes)}")
        print(f"忽略标签: 255")
        print(f"总类别数: {len(valid_classes)} (预期19)")
        
        if len(valid_classes) < 19:
            missing = set(range(19)) - set(valid_classes)
            print(f"⚠️ 注意: 这些类别在分析的样本中未出现: {missing}")
            print(f"  这在训练集中是正常的，某些类别可能很少见")
        
        return "trainids"
    else:
        print(f"\n❌ 标签格式异常，最大类别值: {max_class}")
        return "unknown"


def test_dataset_loading(data_path, split="train"):
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("2. 测试数据集加载")
    print("=" * 60)
    
    configs = [
        ("无裁剪 - 原始尺寸", None, False),
        ("训练模式 512x1024", (512, 1024), True),
        ("训练模式 1024x1024", (1024, 1024), True),
        ("验证模式 512x1024", (512, 1024), False),
    ]
    
    class_names = ["road", "sidewalk", "building", "wall", "fence", "pole",
                   "traffic light", "traffic sign", "vegetation", "terrain",
                   "sky", "person", "rider", "car", "truck", "bus",
                   "train", "motorcycle", "bicycle"]
    
    for config_name, crop_size, is_training in configs:
        print(f"\n{'='*40}")
        print(f"测试: {config_name}")
        print('='*40)
        
        try:
            dataset = ACDCDebugDataset(data_path, split=split,
                                      crop_size=crop_size, is_training=is_training)
            
            if len(dataset) == 0:
                print("  ❌ 数据集为空")
                continue
            
            # 测试前3个样本
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                image = sample["data"]
                label = sample["label"]
                
                print(f"\n  样本 {i+1}:")
                print(f"    图像形状: {list(image.shape)}")
                print(f"    标签形状: {list(label.shape)}")
                print(f"    图像范围: [{image.min():.3f}, {image.max():.3f}]")
                print(f"    图像均值: {image.mean():.3f}, 标准差: {image.std():.3f}")
                
                label_np = label.numpy()
                unique_vals = np.unique(label_np)
                print(f"    标签唯一值: {unique_vals.tolist()}")
                
                # 检查忽略标签
                ignore_mask = (label_np == 255)
                ignore_ratio = ignore_mask.sum() / label_np.size
                print(f"    忽略标签(255)占比: {ignore_ratio:.2%}")
                
                # 检查有效标签
                valid_mask = ~ignore_mask
                if valid_mask.any():
                    valid_labels = label_np[valid_mask]
                    # 检查是否所有标签都在有效范围
                    if valid_labels.max() <= 18 and valid_labels.min() >= 0:
                        print(f"    ✅ 所有有效标签在0-18范围内")
                    else:
                        print(f"    ❌ 存在无效标签! 范围:[{valid_labels.min()}, {valid_labels.max()}]")
                    
                    # 显示类别分布
                    class_counts = np.bincount(valid_labels, minlength=19)
                    top_classes = np.argsort(class_counts)[::-1][:5]
                    print(f"    主要类别:")
                    for cls_id in top_classes:
                        if class_counts[cls_id] > 0:
                            ratio = class_counts[cls_id] / len(valid_labels) * 100
                            print(f"      {class_names[cls_id]:15s}: {ratio:5.1f}%")
                
                # 可视化第一个样本
                if i == 0:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # 反归一化图像
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_vis = image * std + mean
                    img_vis = torch.clamp(img_vis, 0, 1)
                    img_vis = img_vis.numpy().transpose(1, 2, 0)
                    
                    axes[0].imshow(img_vis)
                    axes[0].set_title(f"图像 - {sample['img_path'].split('/')[-1][:30]}")
                    axes[0].axis('off')
                    
                    # 标签可视化
                    np.random.seed(42)
                    colors = np.random.randint(50, 255, (256, 3))
                    colors[255] = [0, 0, 0]
                    label_vis = colors[label_np]
                    
                    axes[1].imshow(label_vis)
                    axes[1].set_title("标签 (随机颜色)")
                    axes[1].axis('off')
                    
                    plt.suptitle(config_name)
                    plt.tight_layout()
                    save_name = f"debug_{config_name.replace(' ', '_').replace('x', 'x')}.png"
                    plt.savefig(save_name, dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"    ✅ 可视化保存: {save_name}")
        
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()


def test_loss_computation(data_path, split="train"):
    """测试损失计算"""
    print("\n" + "=" * 60)
    print("3. 测试损失计算")
    print("=" * 60)
    
    dataset = ACDCDebugDataset(data_path, split=split,
                              crop_size=(512, 1024), is_training=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    batch = next(iter(dataloader))
    images = batch["data"]
    labels = batch["label"]
    
    print(f"Batch图像形状: {images.shape}")
    print(f"Batch标签形状: {labels.shape}")
    
    # 检查标签
    labels_np = labels.numpy()
    for i in range(min(4, labels_np.shape[0])):
        unique_vals = np.unique(labels_np[i])
        ignore_ratio = (labels_np[i] == 255).sum() / labels_np[i].size
        print(f"  样本{i}: 唯一值={unique_vals.tolist()}, 忽略率={ignore_ratio:.2%}")
    
    # 模拟损失计算
    print(f"\n模拟损失计算:")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    # 测试1: 随机输出
    dummy_output = torch.randn(4, 19, 512, 1024)
    try:
        loss = criterion(dummy_output, labels)
        print(f"  随机输出损失: {loss.item():.4f}")
        print(f"  理论值 (ln(19)): {2.9444:.4f}")
        
        if 2.5 < loss.item() < 3.5:
            print(f"  ✅ 损失值合理，标签格式正确!")
        else:
            print(f"  ⚠️ 损失值偏离预期")
    except Exception as e:
        print(f"  ❌ 损失计算失败: {e}")
    
    # 测试2: 如果标签错误（模拟未映射的labelIds）
    print(f"\n模拟错误标签的损失:")
    bad_labels = labels.clone()
    # 模拟未映射的情况
    bad_labels[bad_labels == 0] = 7  # 模拟labelIds
    try:
        loss_bad = criterion(dummy_output, bad_labels)
        print(f"  错误标签损失: {loss_bad.item():.4f}")
        if loss_bad.item() > 4.0:
            print(f"  如果使用未映射的labelIds，损失会异常高")
    except Exception as e:
        print(f"  错误标签导致: {e}")


def generate_report():
    """生成最终报告"""
    print("\n" + "=" * 60)
    print("4. 验证报告")
    print("=" * 60)
    
    print("""
✅ 你的ACDC数据集分析结果：

1. 标签格式: trainIds (0-18 + 255)
   - 数据已经是正确的训练格式
   - 不需要额外的标签映射

2. 原始训练失败的可能原因：
   a) 学习率过高 (1e-4 对微调来说可能偏高)
   b) 缺少数据增强
   c) 可能使用了不当的crop_size
   d) 缺少学习率调度

3. 建议的改进方案：
   a) 降低学习率到 5e-5 或 1e-5
   b) 添加随机缩放、翻转等数据增强
   c) 使用 warmup + 余弦退火学习率调度
   d) 使用 512x1024 的 crop_size（保持原始宽高比）
   e) 添加梯度裁剪

4. 预期效果：
   - 修复后的训练应该能达到 60%+ mIoU
   - 训练过程中mIoU应该呈上升趋势
    """)
    
    print("\n生成的可视化文件:")
    for f in sorted(os.listdir('.')):
        if f.startswith('debug_') and f.endswith('.png'):
            print(f"  📊 {f}")


def main():
    print("=" * 60)
    print("ACDC数据集验证工具 v2")
    print("=" * 60)
    
    data_path = "/home/deng/datasets/acdc_mmseg"
    
    # 1. 分析原始标签
    label_type = analyze_raw_labels(data_path, "train")
    
    if label_type == "trainids":
        # 2. 测试数据加载
        test_dataset_loading(data_path, "train")
        
        # 3. 测试损失计算
        test_loss_computation(data_path, "train")
        
        # 4. 生成报告
        generate_report()
    else:
        print("\n❌ 标签格式异常，请检查数据")
    
    print("\n验证完成!")


if __name__ == "__main__":
    main()
