"""
Evaluation Script for ISIC 2017 Segmentation
Metrics: Dice, IoU, Precision, Recall, F1
"""

from pathlib import Path
import yaml
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from dataset import get_dataloaders
from models import UNet


def compute_metrics(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    inter = (pred_flat & target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - inter
    dice = (2. * inter + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    prec = precision_score(target_flat, pred_flat, zero_division=0)
    rec = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1}


def evaluate(model, loader, device):
    model.eval()
    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Eval"):
            imgs, masks = imgs.to(device), masks.to(device)
            outs = model(imgs)
            preds = (torch.sigmoid(outs) > 0.5).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for i in range(preds.shape[0]):
                m = compute_metrics(preds[i, 0], masks_np[i, 0])
                for k in metrics:
                    metrics[k].append(m[k])
    
    return {k: np.mean(v) for k, v in metrics.items()}


def main():
    cfg_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ckpt_dir = Path(cfg['train']['checkpoint_dir']) / cfg['logging']['name']
    ckpt_path = ckpt_dir / 'best_model.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(
        n_channels=cfg['train']['in_channels'],
        n_classes=cfg['train']['out_channels'],
        bilinear=True
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint (dice={ckpt.get('best_val_dice', 'N/A')})")
    
    _, val_loader = get_dataloaders(
        data_dir=cfg['train']['data_dir'],
        img_size=cfg['train']['img_size'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        val_split=cfg['train']['val_split'],
        augment=False
    )
    
    print(f"\nEvaluating on {len(val_loader.dataset)} samples...")
    metrics = evaluate(model, val_loader, device)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:12}: {v:.4f}")
    print("="*50)
    
    res_dir = Path('results')
    res_dir.mkdir(exist_ok=True)
    with open(res_dir / 'metrics.txt', 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Saved to: {res_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
