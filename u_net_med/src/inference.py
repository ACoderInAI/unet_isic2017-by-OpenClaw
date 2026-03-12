"""
Inference for U-Net medical segmentation
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from models import UNet


def load_model(ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = UNet(
        n_channels=cfg.get('train', {}).get('in_channels', 3),
        n_classes=cfg.get('train', {}).get('out_channels', 1),
        bilinear=True
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"Loaded: {ckpt_path} (val_dice={ckpt.get('best_val_dice', 'N/A')})")
    return model, cfg


def preprocess(img_path, size=256):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Can't read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor, img


def predict(model, tensor, device='cuda', thresh=0.5):
    with torch.no_grad():
        tensor = tensor.to(device)
        out = model(tensor)
        prob = torch.sigmoid(out)
        pred = (prob > thresh).cpu().numpy()
    return pred[0, 0], prob[0, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image")
    p.add_argument("--image-dir")
    p.add_argument("--output-dir", default="predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model, cfg = load_model(args.checkpoint, args.device)
    size = cfg.get('train', {}).get('img_size', 256)
    
    if args.image:
        print(f"\nPredicting: {args.image}")
        tensor, _ = preprocess(args.image, size)
        pred, _ = predict(model, tensor, args.device, args.threshold)
        save_path = out_dir / f"pred_{Path(args.image).stem}.png"
        cv2.imwrite(str(save_path), (pred * 255).astype(np.uint8))
        print(f"Saved: {save_path}")
    
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        print(f"\nBatch predicting {len(imgs)} images...")
        for img_path in imgs:
            print(f"  {img_path.name}...", end=" ")
            tensor, _ = preprocess(str(img_path), size)
            pred, _ = predict(model, tensor, args.device, args.threshold)
            save_path = out_dir / f"pred_{img_path.stem}.png"
            cv2.imwrite(str(save_path), (pred * 255).astype(np.uint8))
            print("✓")


if __name__ == "__main__":
    main()
