import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import math

# Add EfficientViT root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def parse_args():
    parser = argparse.ArgumentParser(description='EfficientViT ACDC Evaluation')
    parser.add_argument('--data_root', type=str, default='/home/deng/datasets/acdc_mmseg')
    parser.add_argument('--weight_url', type=str, 
                        default='/home/deng/efficientvit/checkpoints/efficientvit_seg_b1_cityscapes.pt')
    parser.add_argument('--model', type=str, default='efficientvit-seg-b1-cityscapes',
                        help='Model name from model zoo')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def get_confusion_matrix(gt_label, pred_label, num_classes):
    """Calculate confusion matrix"""
    mask = (gt_label >= 0) & (gt_label < num_classes)
    hist = np.bincount(
        num_classes * gt_label[mask].astype(int) + pred_label[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def pad_to_multiple(image, multiple=32):
    """Pad image dimensions to be multiples of 'multiple'"""
    h, w = image.shape[:2]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    
    if h != new_h or w != new_w:
        if len(image.shape) == 3:
            # For RGB image
            pad_h = new_h - h
            pad_w = new_w - w
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        else:
            # For grayscale label
            pad_h = new_h - h
            pad_w = new_w - w
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
    
    return image, h, w  # Return original size for later cropping


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Build model using official API
    print(f"Loading model: {args.model}")
    model = create_efficientvit_seg_model(
        name=args.model,
        weight_url=args.weight_url
    )
    model = model.cuda()
    model.eval()
    
    # Setup data directories
    img_dir = os.path.join(args.data_root, 'leftImg8bit', 'val')
    label_dir = os.path.join(args.data_root, 'gtFine', 'val')
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Cannot find image directory: {img_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Cannot find label directory: {label_dir}")
    
    # Collect all image files
    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith('_leftImg8bit.png'):
            samples.append(fname)
    
    print(f"Found {len(samples)} images")
    
    # Group images by weather
    weather_images = defaultdict(list)
    for fname in samples:
        weather = fname.split('_')[0]  # Extract weather from filename
        weather_images[weather].append(fname)
    
    print(f"Weather distribution:")
    for weather, files in sorted(weather_images.items()):
        print(f"  {weather}: {len(files)} images")
    
    # Class names
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]
    
    # Store per-weather and overall confusion matrix
    weather_confusion = {}
    overall_confusion = np.zeros((args.num_classes, args.num_classes))
    
    # Evaluate per weather condition
    for weather, files in sorted(weather_images.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating: {weather} ({len(files)} images)")
        print(f"{'='*60}")
        
        confusion_matrix = np.zeros((args.num_classes, args.num_classes))
        
        with torch.no_grad():
            for fname in tqdm(files, desc=weather):
                # Load image
                img_path = os.path.join(img_dir, fname)
                image = np.array(Image.open(img_path).convert("RGB"))
                orig_h, orig_w = image.shape[:2]
                
                # Pad image to be multiple of 32
                image, _, _ = pad_to_multiple(image, multiple=32)
                
                # Load label - ACDC already provides trainIds
                label_name = fname.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                label_path = os.path.join(label_dir, label_name)
                label = np.array(Image.open(label_path), dtype=np.int64)
                
                # Set ignore index (255) to -1 for confusion matrix calculation
                label[label == 255] = -1
                
                # Preprocess image
                image = image / 255.0
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda()
                
                # Inference
                output = model(image)
                
                # Handle different output formats
                if isinstance(output, dict):
                    if 'out' in output:
                        output = output['out']
                    elif 'output' in output:
                        output = output['output']
                    else:
                        output = list(output.values())[0]
                elif isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Resize output back to original image size
                if output.shape[-2:] != (orig_h, orig_w):
                    output = resize(output, size=(orig_h, orig_w))
                
                # Get prediction
                pred = output.argmax(1).squeeze().cpu().numpy()
                
                # Ensure prediction is same size as label
                if pred.shape != label.shape:
                    pred = pred[:label.shape[0], :label.shape[1]]
                
                # Update confusion matrices
                hist = get_confusion_matrix(label, pred, args.num_classes)
                confusion_matrix += hist
                overall_confusion += hist
        
        weather_confusion[weather] = confusion_matrix
        
        # Compute metrics
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10)
        miou = np.nanmean(iou)
        
        print(f"\n{weather} - mIoU: {miou*100:.2f}%")
        print(f"\nPer-Class IoU ({weather}):")
        print("-" * 40)
        for i, name in enumerate(class_names):
            print(f"  {name:15s}: {iou[i]*100:.2f}%")
    
    # Overall results
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS (All Weather Conditions)")
    print(f"{'='*60}")
    
    overall_iou = np.diag(overall_confusion) / (overall_confusion.sum(axis=1) + overall_confusion.sum(axis=0) - np.diag(overall_confusion) + 1e-10)
    overall_miou = np.nanmean(overall_iou)
    total_images = sum(len(files) for files in weather_images.values())
    
    print(f"\nOverall mIoU: {overall_miou*100:.2f}%")
    print(f"Total Images Evaluated: {total_images}")
    print(f"\nOverall Per-Class IoU:")
    print("-" * 40)
    for i, name in enumerate(class_names):
        print(f"  {name:15s}: {overall_iou[i]*100:.2f}%")
    
    # Per-weather summary
    print(f"\n{'='*60}")
    print(f"SUMMARY BY WEATHER CONDITION")
    print(f"{'='*60}")
    print(f"\n{'Weather':<10s} {'mIoU':<10s} {'Images':<10s}")
    print("-" * 35)
    for weather in sorted(weather_images.keys()):
        conf = weather_confusion[weather]
        iou = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf) + 1e-10)
        miou = np.nanmean(iou)
        n_images = len(weather_images[weather])
        print(f"  {weather:<8s}: {miou*100:>8.2f}%  ({n_images} images)")
    
    # Detailed per-class per-weather summary
    print(f"\n{'='*60}")
    print(f"PER-CLASS IoU BY WEATHER CONDITION")
    print(f"{'='*60}")
    
    # Header
    header = f"{'Class':<15s}"
    for weather in sorted(weather_images.keys()):
        header += f" {weather:>10s}"
    header += f" {'Overall':>10s}"
    print(header)
    print("-" * len(header))
    
    # Per-class results
    for i, name in enumerate(class_names):
        row = f"{name:<15s}"
        for weather in sorted(weather_images.keys()):
            conf = weather_confusion[weather]
            iou = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf) + 1e-10)
            row += f" {iou[i]*100:>9.2f}%"
        row += f" {overall_iou[i]*100:>9.2f}%"
        print(row)


if __name__ == '__main__':
    main()