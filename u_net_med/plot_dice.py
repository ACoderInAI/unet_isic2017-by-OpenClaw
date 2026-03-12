"""
Generate training progress figure: Train and Validation Dice over epochs.
"""
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# Path to TensorBoard log
log_path = r""

# Load data
ea = event_accumulator.EventAccumulator(log_path, size_guidance={'scalars': 0})
ea.Reload()

# Extract metrics
train_dice = [e.value for e in ea.Scalars('epoch/train_dice')]
val_dice = [e.value for e in ea.Scalars('epoch/val_dice')]
epochs = list(range(1, len(train_dice) + 1))

# Find best validation epoch
best_val_idx = np.argmax(val_dice)
best_val_dice = val_dice[best_val_idx]
best_epoch = epochs[best_val_idx]

# Create figure
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(epochs, train_dice, 'b-', linewidth=2, label='Train Dice', alpha=0.8)
plt.plot(epochs, val_dice, 'r-', linewidth=2, label='Validation Dice', alpha=0.8)

# Highlight best validation point
plt.scatter(best_epoch, best_val_dice, color='gold', s=150, zorder=5, label=f'Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})', edgecolors='black')

# Fill area between curves (optional, shows gap)
plt.fill_between(epochs, train_dice, val_dice, alpha=0.2, color='gray')

# Labels and title
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Dice Score', fontsize=12)
plt.title('U-Net Training Progress on ISIC 2017\nTrain vs Validation Dice', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')

# Annotations
plt.annotate(f'Best: {best_val_dice:.4f} @ epoch {best_epoch}',
             xy=(best_epoch, best_val_dice),
             xytext=(best_epoch, best_val_dice + 0.02),
             ha='center',
             arrowprops=dict(arrowstyle='->', color='gold', lw=1.5),
             fontsize=10,
             fontweight='bold')

# Set axis limits
plt.ylim([0.5, 1.0])
plt.xlim([1, len(epochs)])

# Add some stats text box
final_train = train_dice[-1]
final_val = val_dice[-1]
textstr = f'Final Metrics (Epoch {len(epochs)}):\n' \
          f'  Train Dice: {final_train:.4f}\n' \
          f'  Val Dice: {final_val:.4f}\n' \
          f'  Gap: {abs(final_train-final_val):.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

# Save figure
output_path = r"u_net_med\training_progress_dice.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved: {output_path}")

# Also show key stats
print(f"📊 Best Validation Dice: {best_val_dice:.4f} at epoch {best_epoch}")
print(f"📈 Final Train Dice: {final_train:.4f}")
print(f"📉 Final Val Dice: {final_val:.4f}")

plt.close()
