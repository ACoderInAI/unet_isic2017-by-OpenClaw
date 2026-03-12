# U-Net_med: Medical Image Segmentation with U-Net

**Dataset:** ISIC 2017 (Skin Lesion Segmentation)  
**Architecture:** U-Net  
**Optimizer:** Adam (lr=0.001)  
**Resolution:** 256×256  
**GPU:** NVIDIA GTX 3090 (24GB)  
**Logging:** TensorBoard  

---

## Quick Start

```powershell
cd D:\PycharmProjects\u_net_med

# Install dependencies (CUDA 11.8)
pip install -r requirements.txt

# Put ISIC 2017 data in:
#   data/images/*.jpg
#   data/masks/*.png

# Train
python src/train.py

# Monitor
tensorboard --logdir logs/u_net_med

# Evaluate
python src/evaluate.py

# Predict
python src/inference.py --checkpoint checkpoints/u_net_med/best_model.pth --image-dir data/images
```

---

## Project Structure

```
u_net_med/
├── configs/
│   └── config.yaml        # Training configuration
├── src/
│   ├── models.py          # U-Net definition
│   ├── dataset.py         # ISIC 2017 loader with augmentations
│   ├── train.py           # Training loop with TensorBoard logging
│   ├── inference.py       # Prediction script
│   └── evaluate.py        # Compute Dice, IoU, Precision, Recall, F1
├── data/
│   ├── images/            # ISIC 2017 images (.jpg)
│   └── masks/             # Binary masks (.png)
├── checkpoints/           # Auto-created during training
├── logs/                  # TensorBoard logs
├── predictions/           # Inference outputs
├── results/               # Evaluation metrics
├── requirements.txt
├── README.md
└── train.bat              # Windows launcher
```

---

## Configuration Highlights

| Option | Value |
|--------|-------|
| `batch_size` | 32 (fits 3090 with AMP) |
| `img_size` | 256 |
| `optimizer` | Adam |
| `learning_rate` | 0.001 |
| `loss` | `dice_bce` |
| `use_amp` | `true` (mixed precision) |
| `epochs` | 100 |
| `early_stopping_patience` | 15 |
| `tensorboard` | `true` |

---

## GTX 3090 Optimizations

- **CUDA 11.8** in `requirements.txt`  
- **AMP (Mixed Precision):** `use_amp: true` → 2x speed, less VRAM  
- `pin_memory=True` for faster data transfer  
- `num_workers=8` for parallel loading  
- Batch size 32 fits comfortably (~8-10 GB VRAM)

---

## Logging

- **TensorBoard:** `logs/u_net_med/`  
  Run `tensorboard --logdir logs/u_net_med` and open http://localhost:6006  
  - Training/validation loss curves  
  - Dice score progression  
  - Learning rate chart  

- **Checkpoints:** `checkpoints/u_net_med/`
  - `latest.pth` (every `save_frequency` epochs)
  - `best_model.pth` (automatically on improvement)

- **Console:** tqdm progress bars with batch metrics

---

## Expected Performance

On ISIC 2017 with this setup:
- **Training time:** ~2-4 hours (100 epochs, GTX 3090)
- **VRAM usage:** ~8-10 GB (AMP enabled)
- **Expected Dice:** 0.90-0.92 on validation (baseline U-Net)

---

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` to 16 or 8 in `config.yaml`

**Slow training:**
- Ensure AMP is enabled
- Increase `num_workers` if CPU allows
- Store data on SSD

**No checkpoint found:**
- Let training complete at least 1 epoch first

---

## Notes

- Datasets automatically binarizes masks (threshold 127)
- Augmentations: flip, rotate, shift, scale, brightness, blur, noise
- Mixed precision training enabled for 3090
- Early stopping prevents overfitting

---

Cite ISIC 2017 and U-Net if used in research.
