"""
Save all inference predictions as images.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent / 'src'))

from models import UNet
from dataset import ISICDataset
from sklearn.model_selection import train_test_split

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

    # Create validation dataset (same split as training)
    full_dataset = ISICDataset(
        data_dir=train_cfg['data_dir'],
        img_size=train_cfg['img_size'],
        augment=False,
        is_train=False
    )
    
    indices = list(range(len(full_dataset)))
    _, val_indices = train_test_split(
        indices, test_size=train_cfg['val_split'], random_state=42, shuffle=True
    )
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Validation dataset size: {len(val_dataset)}")

    # Output directory
    output_dir = Path('inference_results_all')
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    (output_dir / 'predictions').mkdir(exist_ok=True)
    (output_dir / 'overlays').mkdir(exist_ok=True)

    print("\nRunning inference and saving all predictions...")
    
    dice_scores = []
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(val_indices, desc="Processing")):
            img, mask = full_dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            mask_tensor = mask.unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).float()
            
            # Compute Dice
            smooth = 1e-6
            intersection = (pred * mask_tensor).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + mask_tensor.sum() + smooth)
            dice_scores.append(dice.item())
            
            # Get image name
            img_path = full_dataset.image_paths[idx]
            img_name = img_path.stem
            
            # Convert tensors to numpy
            img_np = img.permute(1,2,0).numpy()
            # Undo normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            mask_np = mask.squeeze().numpy()
            pred_np = pred[0].cpu().squeeze().numpy()
            prob_np = probs[0].cpu().squeeze().numpy()
            
            # Save individual images
            plt.imsave(output_dir / 'images' / f'{img_name}.png', img_np)
            plt.imsave(output_dir / 'masks' / f'{img_name}_mask.png', mask_np, cmap='gray')
            plt.imsave(output_dir / 'predictions' / f'{img_name}_pred.png', pred_np, cmap='gray')
            plt.imsave(output_dir / 'predictions' / f'{img_name}_prob.png', prob_np, cmap='gray')
            
            # Create overlay (image + prediction in red)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(img_np)
            ax.imshow(pred_np, alpha=0.4, cmap='Reds')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / 'overlays' / f'{img_name}_overlay.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create comparison figure (image, mask, prediction)
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
            plt.suptitle(f'{img_name} - Dice: {dice.item():.4f}')
            plt.tight_layout()
            plt.savefig(output_dir / f'{img_name}.png', dpi=150, bbox_inches='tight')
            plt.close()

    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"\n🎯 Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
    
    # Save summary CSV
    import csv
    with open(output_dir / 'dice_scores.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'dice_score'])
        for idx, dice_val in zip(val_indices, dice_scores):
            img_name = full_dataset.image_paths[idx].stem
            writer.writerow([img_name, f'{dice_val:.6f}'])
    
    # Save summary statistics
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"Best model checkpoint: {checkpoint_path}\n")
        f.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
        f.write(f"Checkpoint best val dice: {checkpoint['best_val_dice']:.6f}\n\n")
        f.write(f"Number of validation samples: {len(val_dataset)}\n")
        f.write(f"Mean Dice Score: {avg_dice:.6f}\n")
        f.write(f"Std Dice Score: {std_dice:.6f}\n")
        f.write(f"Min Dice Score: {np.min(dice_scores):.6f}\n")
        f.write(f"Max Dice Score: {np.max(dice_scores):.6f}\n")
        f.write(f"Median Dice Score: {np.median(dice_scores):.6f}\n")
    
    print(f"✅ All results saved to {output_dir}")
    print(f"📊 Mean Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"📈 Min/Max: {np.min(dice_scores):.4f} / {np.max(dice_scores):.4f}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
