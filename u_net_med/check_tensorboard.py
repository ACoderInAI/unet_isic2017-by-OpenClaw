"""
Extract training metrics from TensorBoard event file.
"""
import sys
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# Path to the TensorBoard log
log_path = r""

# Load the event accumulator
ea = event_accumulator.EventAccumulator(log_path, size_guidance={'scalars': 0})
ea.Reload()

# Get available scalar tags
scalar_tags = ea.Tags()['scalars']
print("Available scalar tags:", scalar_tags)
print()

# Extract epoch-level metrics
epoch_metrics = {}
for tag in ['epoch/train_loss', 'epoch/train_dice', 'epoch/val_loss', 'epoch/val_dice', 'lr']:
    if tag in scalar_tags:
        events = ea.Scalars(tag)
        # Extract step (epoch) and value
        data = [(e.step, e.value) for e in events]
        epoch_metrics[tag] = data
        print(f"{tag}: {len(data)} entries")
    else:
        print(f"Tag not found: {tag}")

# Build a combined table
if epoch_metrics:
    # Determine number of epochs from any metric
    num_epochs = len(epoch_metrics['epoch/train_loss'])
    print(f"\nTotal epochs recorded: {num_epochs}\n")

    # Create a table
    rows = []
    for epoch in range(num_epochs):
        row = {'epoch': epoch + 1}
        for tag in ['epoch/train_loss', 'epoch/train_dice', 'epoch/val_loss', 'epoch/val_dice', 'lr']:
            if tag in epoch_metrics and epoch < len(epoch_metrics[tag]):
                row[tag.replace('epoch/', '')] = epoch_metrics[tag][epoch][1]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Rearrange columns
    cols = ['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'lr']
    df = df[cols]

    # Print full table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))

    # Identify best validation dice
    best_idx = df['val_dice'].idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    best_dice = df.loc[best_idx, 'val_dice']
    print(f"\nBest Validation Dice: {best_dice:.4f} at epoch {best_epoch}")

else:
    print("No epoch metrics found in the log file.")
