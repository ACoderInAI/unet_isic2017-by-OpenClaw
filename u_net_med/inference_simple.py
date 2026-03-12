"""
Simplified inference with no multiprocessing for quick results.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm  # Added
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent / 'src'))

from models import UNet
from dataset import ISICDataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import cv2

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_cfg = config['train']
    dataset_cfg = config['dataset']['isic2017']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = UNet(
        n_channels=train_cfg['in_channels'],
        n_classes=train_cfg['out_channels'],
        bilinear=True
    )
    checkpoint_path = 'checkpoints/u_net_med/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Loaded best model from {checkpoint_path}")
    print(f"Epoch {checkpoint['epoch']}, Best Val Dice: {checkpoint['best_val_dice']:.4f}")

    # Create validation dataset (same split logic as training)
    full_dataset = ISICDataset(
        data_dir=train_cfg['data_dir'],
        img_size=train_cfg['img_size'],
        augment=False,
        is_train=False
    )
    
    # Apply same 80/20 split with seed=42
    indices = list(range(len(full_dataset)))
    from sklearn.model_selection import train_test_split
    _, val_indices = train_test_split(
        indices, test_size=train_cfg['val_split'], random_state=42, shuffle=True
    )
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Validation dataset size: {len(val_dataset)}")

    # Simple transform for validation (already applied in dataset)
    # Inference loop - no DataLoader to avoid multiprocessing
    total_dice = 0.0
    num_samples = 0
    
    output_dir = Path('inference_results')
    output_dir.mkdir(exist_ok=True)
    
    sample_saved = 0
    max_samples = 5
    
    print("\nRunning inference on validation set...")
    
    with torch.no_grad():
        for idx in tqdm(val_indices, desc="Samples"):
            img, mask = full_dataset[idx]
            img = img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            outputs = model(img)
            probs = torch.sigmoid(outputs)
            
            # Dice
            pred = (probs > 0.5).float()
            smooth = 1e-6
            intersection = (pred * mask).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth)
            total_dice += dice.item()
            num_samples += 1
            
            # Save first few samples
            if sample_saved < max_samples:
                img_np = img[0].cpu().permute(1,2,0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                
                mask_np = mask[0].cpu().squeeze().numpy()
                pred_np = probs[0].cpu().squeeze().numpy()
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np)
                axes[0].set_title('Image')
                axes[0].axis('off')
                axes[1].imshow(mask_np, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                axes[2].imshow(pred_np, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(output_dir / f'sample_{sample_saved+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                sample_saved += 1
    
    avg_dice = total_dice / num_samples
    print(f"\n🎯 Validation Dice Score (on {num_samples} samples): {avg_dice:.6f}")
    
    # Save results
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"Best model checkpoint: {checkpoint_path}\n")
        f.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
        f.write(f"Checkpoint best val dice: {checkpoint['best_val_dice']:.6f}\n")
        f.write(f"\nTest Dice (on validation set): {avg_dice:.6f}\n")
        f.write(f"Number of samples: {num_samples}\n")
    
    print(f"✅ Results saved to {output_dir}")

if __name__ == '__main__':
    main()
