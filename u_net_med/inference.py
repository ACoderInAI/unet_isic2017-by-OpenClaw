"""
Run inference on the validation set using the best trained model.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix for OpenMP duplicate library issue on Windows

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models import UNet
from dataset import get_dataloaders


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

    # Get validation dataloader (using same split as training)
    # We set augment=False for validation/inference
    _, val_loader = get_dataloaders(
        data_dir=train_cfg['data_dir'],
        img_size=train_cfg['img_size'],
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg['num_workers'],
        val_split=train_cfg['val_split'],
        augment=False,  # No augmentation for validation/inference
        seed=42  # Same seed to ensure same split
    )

    print(f"Validation loader: {len(val_loader.dataset)} samples")

    # Dice metric
    def dice_coeff(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        target = target.float()
        smooth = 1e-6
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # Inference loop
    total_dice = 0.0
    num_samples = 0

    output_dir = Path('inference_results')
    output_dir.mkdir(exist_ok=True)

    # We'll also save some sample predictions
    sample_saved = 0
    max_samples_to_save = 5

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Running inference")
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass - use AMP if available
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
            else:
                outputs = model(imgs)

            probs = torch.sigmoid(outputs)

            # Compute Dice for each sample in batch
            batch_size = imgs.size(0)
            for i in range(batch_size):
                dice = dice_coeff(probs[i], masks[i])
                total_dice += dice.item()
                num_samples += 1

            # Save first few samples from first batch for visualization
            if sample_saved < max_samples_to_save:
                for i in range(min(batch_size, max_samples_to_save - sample_saved)):
                    img = imgs[i].cpu()
                    mask = masks[i].cpu()
                    pred = probs[i].cpu()

                    # Create figure
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    # Denormalize image for display
                    img_np = img.permute(1,2,0).numpy()
                    # Undo normalization (assuming ImageNet normalization)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)

                    axes[0].imshow(img_np)
                    axes[0].set_title('Image')
                    axes[0].axis('off')
                    axes[1].imshow(mask.squeeze().numpy(), cmap='gray')
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    axes[2].imshow(pred.squeeze().numpy(), cmap='gray')
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(output_dir / f'sample_{sample_saved+1}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    sample_saved += 1

    avg_dice = total_dice / num_samples
    print(f"\n🎯 Validation Dice Score (on {num_samples} samples): {avg_dice:.6f}")

    # Save score to a text file
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"Best model checkpoint: {checkpoint_path}\n")
        f.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
        f.write(f"Checkpoint best val dice: {checkpoint['best_val_dice']:.6f}\n")
        f.write(f"\nTest Dice (on validation set): {avg_dice:.6f}\n")
        f.write(f"Number of samples: {num_samples}\n")

    print(f"✅ Results saved to {output_dir}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)  # Necessary for Windows with CUDA
    main()
