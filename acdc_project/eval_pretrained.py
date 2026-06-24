import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import cv2
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from efficientvit.models.utils.network import resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


class ACDCDataset(Dataset):
    classes = (
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle"
    )
    
    IGNORE_INDEX = 255

    def __init__(self, data_dir, split="val", crop_size=None):
        self.crop_size = crop_size
        self.samples = []
        
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
        
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(('.png', '.jpg')):
                continue
            img_path = os.path.join(img_dir, fname)
            base_name = fname.replace("_leftImg8bit", "").replace(".png", "").replace(".jpg", "")
            label_name = f"{base_name}_gtFine_labelTrainIds.png"
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))
        
        print(f"ACDC {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path))
        
        label = label.astype(np.int64)
        label[(label < 0) | (label >= 19)] = self.IGNORE_INDEX
        
        if self.crop_size is not None:
            target_h, target_w = self.crop_size
            h, w = image.shape[:2]
            if h != target_h or w != target_w:
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])(image)
        label = torch.from_numpy(label).long()
        
        return {"data": image, "label": label, "img_path": img_path}


def evaluate(model, dataloader, num_classes=19, device="cuda"):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.inference_mode():
        for feed_dict in tqdm(dataloader, desc="Evaluating"):
            images = feed_dict["data"].to(device)
            masks = feed_dict["label"].to(device)
            
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = resize(outputs, size=masks.shape[-2:])
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            
            for pred, target in zip(preds, targets):
                mask = (target != 255)
                pred = pred[mask]
                target = target[mask]
                confusion_matrix += np.bincount(
                    target * num_classes + pred, 
                    minlength=num_classes**2
                ).reshape(num_classes, num_classes)
    
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        intersection = confusion_matrix[cls, cls]
        union = (confusion_matrix[cls, :].sum() + 
                confusion_matrix[:, cls].sum() - 
                confusion_matrix[cls, cls])
        if union > 0:
            iou_per_class[cls] = intersection / union
    
    miou = np.nanmean(iou_per_class) if np.any(iou_per_class > 0) else 0.0
    return miou * 100, iou_per_class * 100


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained EfficientViT on ACDC (zero-shot)")
    parser.add_argument("--data_path", type=str, default="/home/deng/datasets/acdc_mmseg")
    parser.add_argument("--model", type=str, default="efficientvit-seg-b1-cityscapes")
    parser.add_argument("--weight_url", type=str, default="./checkpoints/efficientvit_seg_b1_cityscapes.pt")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--crop_height", type=int, default=512)
    parser.add_argument("--crop_width", type=int, default=1024)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print("Zero-shot Evaluation: Cityscapes Pretrained Model on ACDC")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Weights: {args.weight_url}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading pretrained model...")
    model = create_efficientvit_seg_model(args.model, weight_url=args.weight_url)
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading ACDC validation set...")
    crop_size = (args.crop_height, args.crop_width)
    val_dataset = ACDCDataset(args.data_path, split="val", crop_size=crop_size)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Evaluate
    print("\nRunning evaluation...")
    miou, iou_per_class = evaluate(model, val_loader, device=device)
    
    # Results
    print(f"\n{'='*60}")
    print("Results (Zero-shot, No Fine-tuning)")
    print(f"{'='*60}")
    print(f"mIoU: {miou:.2f}%\n")
    
    # Sort by IoU
    class_names = val_dataset.classes
    results = [(iou_per_class[i], class_names[i]) for i in range(19) if iou_per_class[i] > 0]
    results.sort(reverse=True)
    
    print("Per-class IoU (sorted):")
    print(f"{'Class':<18} {'IoU':>8}")
    print("-" * 28)
    for iou, name in results:
        print(f"{name:<18} {iou:>7.2f}%")
    
    # Breakdown
    print(f"\nGood (IoU > 60%):")
    good = [(iou, name) for iou, name in results if iou > 60]
    if good:
        for iou, name in good:
            print(f"  {name:<15} {iou:.1f}%")
    else:
        print("  None")
    
    print(f"\nMedium (30% < IoU <= 60%):")
    medium = [(iou, name) for iou, name in results if 30 < iou <= 60]
    if medium:
        for iou, name in medium:
            print(f"  {name:<15} {iou:.1f}%")
    else:
        print("  None")
    
    print(f"\nPoor (IoU <= 30%):")
    poor = [(iou, name) for iou, name in results if iou <= 30]
    if poor:
        for iou, name in poor:
            print(f"  {name:<15} {iou:.1f}%")
    else:
        print("  None")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


