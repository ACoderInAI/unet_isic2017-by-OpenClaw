"""
Training Script for U-Net Medical Image Segmentation
ISIC 2017 Dataset, Adam Optimizer, 256x256, GTX 3090
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parent))
from models import UNet, count_parameters
from dataset import get_dataloaders


# ==================== Loss Functions ====================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        prob = torch.sigmoid(pred)
        dice = self.dice(prob, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + (1 - self.dice_weight) * bce


# ==================== Trainer ====================

class Trainer:
    def __init__(self, config, exp_name="u_net_med"):
        self.config = config
        self.exp_name = exp_name
        
        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config['train']['gpu_ids'][0]}")
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"VRAM: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print("WARNING: CUDA not available - using CPU")
        
        # Paths
        self.log_dir = Path(config['train']['log_dir']) / exp_name
        self.checkpoint_dir = Path(config['train']['checkpoint_dir']) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if config['train']['tensorboard'] else None
        
        # Model
        self.model = UNet(
            n_channels=config['train']['in_channels'],
            n_classes=config['train']['out_channels'],
            bilinear=True
        ).to(self.device)
        
        total, trainable = count_parameters(self.model)
        print(f"\nModel: U-Net")
        print(f"Params: {total:,} total, {trainable:,} trainable")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train'].get('weight_decay', 0)
        )
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Loss
        self.criterion = DiceBCELoss() if config['train']['loss'] == 'dice_bce' else getattr(nn, config['train']['loss'].upper())()

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if config['train']['use_amp'] and self.device.type == 'cuda' else None
        
        # State
        self.start_epoch = 0
        self.best_val_dice = 0.0
        self.early_stop_counter = 0
        
        if 'resume' in config and config['resume'].get('checkpoint'):
            self._load(config['resume']['checkpoint'])
    
    def _get_scheduler(self):
        sched_type = self.config['train']['scheduler']
        if sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['train']['epochs'])
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif sched_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)
        return None
    
    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_dice = ckpt.get('best_val_dice', 0.0)
        print(f"Resumed from epoch {self.start_epoch}, best dice: {self.best_val_dice:.4f}")
    
    def _save(self, epoch, val_dice, best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(ckpt, self.checkpoint_dir / 'latest.pth')
        if best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pth')
            print(f"\n  ✓ Saved best (dice={val_dice:.4f})")
    
    def _dice(self, preds, targets):
        smooth = 1e-6
        pred_flat = preds.view(-1).float()
        target_flat = targets.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, total_dice = 0.0, 0.0
        n_batches = len(loader)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config['train']['epochs']} [Train]")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                dice = self._dice(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
            
            if self.writer and batch_idx % 10 == 0:
                step = epoch * n_batches + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/dice', dice.item(), step)
        
        return {'loss': total_loss / n_batches, 'dice': total_dice / n_batches}
    
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss, total_dice = 0.0, 0.0
        n_batches = len(loader)
        
        pbar = tqdm(loader, desc="Val")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            preds = torch.sigmoid(outputs) > 0.5
            dice = self._dice(preds, masks)
            total_loss += loss.item()
            total_dice += dice.item()
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_dice': f'{dice.item():.4f}'})
        
        return {'loss': total_loss / n_batches, 'dice': total_dice / n_batches}
    
    def train(self, train_loader, val_loader):
        print("\n" + "="*60)
        print(f"Starting Training: {self.exp_name}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        epochs = self.config['train']['epochs']
        patience = self.config['train']['early_stopping_patience']
        
        for epoch in range(self.start_epoch, epochs):
            start = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/train_dice', train_metrics['dice'], epoch)
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/val_dice', val_metrics['dice'], epoch)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            elapsed = time.time() - start
            print(f"\nEpoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
            print(f"  Train - loss: {train_metrics['loss']:.4f}, dice: {train_metrics['dice']:.4f}")
            print(f"  Val   - loss: {val_metrics['loss']:.4f}, dice: {val_metrics['dice']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            if (epoch + 1) % self.config['train']['save_frequency'] == 0 or is_best:
                self._save(epoch, val_metrics['dice'], is_best)
            
            if self.early_stop_counter >= patience:
                print(f"\n⚠️ Early stop after {epoch+1} epochs")
                break
        
        print("\n" + "="*60)
        print(f"Training complete! Best val dice: {self.best_val_dice:.4f}")
        print("="*60)
        
        if self.writer:
            self.writer.close()


def main():
    cfg_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(
        data_dir=cfg['train']['data_dir'],
        img_size=cfg['train']['img_size'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        val_split=cfg['train']['val_split'],
        augment=cfg['dataset']['isic2017'].get('augmentations', True)
    )
    
    trainer = Trainer(cfg, cfg['logging']['name'])
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
