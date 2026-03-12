"""
Dataset Module for ISIC 2017 Skin Lesion Segmentation
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, GaussNoise, GaussianBlur, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class ISICDataset(Dataset):
    """
    ISIC 2017 Skin Lesion Segmentation Dataset.
    
    Expects structure:
        data_dir/
            images/  (containing .jpg files)
            masks/   (containing .png binary masks)
    """
    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        augment: bool = False,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        is_train: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.is_train = is_train
        
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        self.mask_paths = []
        
        for img_path in self.image_paths:
            mask_path = self.mask_dir / img_path.with_suffix(".png").name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                mask_path_alt = self.mask_dir / f"{img_path.stem}_segmentation.png"
                if mask_path_alt.exists():
                    self.mask_paths.append(mask_path_alt)
                else:
                    print(f"Warning: No mask for {img_path.name}")
                    self.image_paths.remove(img_path)
        
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch images/masks"
        print(f"Dataset loaded: {len(self.image_paths)} images")
        
        self._setup_transforms(normalize_mean, normalize_std)
    
    def _setup_transforms(self, mean, std):
        if self.augment and self.is_train:
            self.transform = Compose([
                Resize(self.img_size, self.img_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
                GaussNoise(p=0.2),
                GaussianBlur(blur_limit=(3, 7), p=0.2),
                Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                Resize(self.img_size, self.img_size),
                Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load: {mask_path}")
        
        mask = (mask > 127).astype(np.float32)
        mask = mask[..., np.newaxis]
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].permute(2, 0, 1)
        
        return image, mask


def get_dataloaders(
    data_dir: str,
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    augment: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    full_dataset = ISICDataset(
        data_dir=data_dir,
        img_size=img_size,
        augment=augment,
        is_train=True
    )
    
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=seed, shuffle=True
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    val_dataset.dataset.augment = False
    val_dataset.dataset.is_train = False
    
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    common_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing ISIC Dataset...")
    
    data_dir = "data"
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        img_size=256,
        batch_size=4,
        num_workers=2,
        val_split=0.2,
        augment=True
    )
    
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Range: [{images.min():.3f}, {images.max():.3f}]")
