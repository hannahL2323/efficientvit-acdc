import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


class ACDCDataset(Dataset):
    classes = (
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle"
    )
    
    # Cityscapes 原始标签 id 到训练标签 id 的映射
    label_map = np.array(
        (-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5,
         -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18)
    )
    
    # 统一使用 255 作为 ignore_index，与 CrossEntropyLoss 保持一致
    IGNORE_INDEX = 255

    def __init__(self, data_dir: str, split: str = "train", crop_size: Optional[tuple[int, int]] = None):
        self.crop_size = crop_size
        self.samples = []
        
        images_dir = os.path.join(data_dir, "leftImg8bit", split)
        labels_dir = os.path.join(data_dir, "gtFine", split)
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Cannot find images directory: {images_dir}")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Cannot find labels directory: {labels_dir}")
        
        for fname in sorted(os.listdir(images_dir)):
            if not fname.endswith((".png", ".jpg")):
                continue
            
            image_path = os.path.join(images_dir, fname)
            
            base_name = fname.replace("_leftImg8bit", "")
            base_name = os.path.splitext(base_name)[0]
            mask_name = f"{base_name}_gtFine_labelTrainIds.png"
            mask_path = os.path.join(labels_dir, mask_name)
            
            if not os.path.exists(mask_path):
                alt_mask_name = fname.replace("_leftImg8bit", "_gtFine_labelTrainIds")
                mask_path = os.path.join(labels_dir, alt_mask_name)
                if not os.path.exists(mask_path):
                    alt_mask_name2 = fname.replace("_leftImg8bit", "_gtFine_labelIds")
                    mask_path = os.path.join(labels_dir, alt_mask_name2)
                    if not os.path.exists(mask_path):
                        print(f"Warning: No label found for {fname}")
                        continue
            
            self.samples.append((image_path, mask_path))
        
        print(f"ACDC {split} set: found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        from PIL import Image
        
        image_path, mask_path = self.samples[idx]
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # ===== ACDC 的 labelTrainIds 已经是 0-18 格式 =====
        # 直接使用，只把 255 保留为忽略标签
        # mask 中的值已经是 0-18 和 255，不需要映射
        # 但为了安全，把超出 0-18 的值转为 255
        mask = mask.astype(np.int64)
        mask[(mask < 0) | (mask >= 19)] = 255
        # ===================================================
        
        if self.crop_size is not None:
            h, w, _ = image.shape
            target_h, target_w = self.crop_size
            if w != target_w or h != target_h:
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        mask = torch.from_numpy(mask).long()
        
        return {"data": image, "label": mask}


def evaluate(model, dataloader, num_classes=19, device="cuda"):
    model.eval()
    iou_per_class = np.zeros(num_classes)
    valid_class_count = np.zeros(num_classes)
    
    with torch.inference_mode():
        for feed_dict in tqdm(dataloader, desc="Evaluating"):
            images = feed_dict["data"].to(device)
            masks = feed_dict["label"].to(device)
            
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = resize(outputs, size=masks.shape[-2:])
            preds = outputs.argmax(dim=1)
            
            for cls in range(num_classes):
                pred_cls = (preds == cls)
                target_cls = (masks == cls)
                intersection = (pred_cls & target_cls).sum().item()
                union = (pred_cls | target_cls).sum().item()
                
                if union > 0:
                    iou_per_class[cls] += intersection / union
                    valid_class_count[cls] += 1
    
    avg_iou_per_class = iou_per_class / (valid_class_count + 1e-10)
    miou = np.nanmean(avg_iou_per_class)
    
    return miou * 100, avg_iou_per_class * 100

def train_one_epoch(model, dataloader, optimizer, criterion, device="cuda", max_iters=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    iterator = iter(dataloader)
    for _ in range(len(dataloader)):
        if max_iters is not None and num_batches >= max_iters:
            break
        try:
            feed_dict = next(iterator)
        except StopIteration:
            break
            
        images = feed_dict["data"].to(device)
        masks = feed_dict["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = resize(outputs, size=masks.shape[-2:])
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 1000 == 0:
            print(f"  Iteration {num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def train_with_iters(model, train_loader, val_loader, optimizer, criterion, 
                      total_iters, device="cuda", save_dir="./checkpoints_acdc"):
    model.train()
    global_step = 0
    total_loss = 0
    best_miou = 0.0
    
    print(f"Starting training for {total_iters} iterations...")
    start_time = time.time()
    
    train_iter = iter(train_loader)
    
    while global_step < total_iters:
        try:
            feed_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            feed_dict = next(train_iter)
        
        images = feed_dict["data"].to(device)
        masks = feed_dict["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = resize(outputs, size=masks.shape[-2:])
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        global_step += 1
        
        if global_step % 500 == 0:
            avg_loss = total_loss / global_step
            elapsed = time.time() - start_time
            print(f"Iteration {global_step}/{total_iters}, Avg Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
        
        if global_step % 5000 == 0 or global_step == total_iters:
            print(f"\n=== Evaluating at iteration {global_step} ===")
            miou, iou_per_class = evaluate(model, val_loader, device=device)
            print(f"Val mIoU: {miou:.2f}%")
            
            print("Per-class IoU:")
            for i, cls_name in enumerate(train_loader.dataset.classes):
                print(f"  {cls_name}: {iou_per_class[i]:.2f}%")
            
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"Best model saved! mIoU: {miou:.2f}%")
            
            model.train()
    
    return best_miou


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientViT on ACDC")
    parser.add_argument("--data_path", type=str, required=True, help="ACDC dataset root path")
    parser.add_argument("--model", type=str, default="efficientvit-seg-b1-cityscapes")
    parser.add_argument("--weight_url", type=str, default="./checkpoints/efficientvit_seg_b1_cityscapes.pt")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--iters", type=int, default=None, help="Total training iterations (overrides epochs)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size (height)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_acdc", help="Save directory")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- 加载模型 ---
    print(f"Loading model: {args.model}")
    model = create_efficientvit_seg_model(args.model, weight_url=args.weight_url)
    model = model.to(device)
    
    # --- 准备数据集 ---
    print("Loading ACDC dataset...")
    crop_size = (args.crop_size, args.crop_size * 2)
    
    train_dataset = ACDCDataset(args.data_path, split="train", crop_size=crop_size)
    val_dataset = ACDCDataset(args.data_path, split="val", crop_size=crop_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # --- 训练配置 ---
    # 修复：使用 ACDCDataset.IGNORE_INDEX (255) 保持一致
    criterion = nn.CrossEntropyLoss(ignore_index=ACDCDataset.IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # --- 开始训练 ---
    if args.iters is not None:
        print(f"\n=== Training for {args.iters} iterations ===")
        best_miou = train_with_iters(
            model, train_loader, val_loader, optimizer, criterion,
            total_iters=args.iters, device=device, save_dir=args.save_dir
        )
    else:
        print(f"\n=== Training for {args.epochs} epochs ===")
        best_miou = 0.0
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Train Loss: {train_loss:.4f}")
            
            miou, iou_per_class = evaluate(model, val_loader, device=device)
            print(f"Val mIoU: {miou:.2f}%")
            
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                print(f"Best model saved! mIoU: {miou:.2f}%")
    
    print(f"\nTraining complete. Best mIoU: {best_miou:.2f}%")


if __name__ == "__main__":
    main()