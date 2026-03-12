import torch
cp = torch.load('checkpoints/u_net_med/best_model.pth', map_location='cpu', weights_only=False)
print('Keys:', list(cp.keys()))
if 'epoch' in cp:
    print('Epoch:', cp['epoch'])
if 'val_dice' in cp:
    print('Val Dice:', cp['val_dice'])
if 'best_val_dice' in cp:
    print('Best Val Dice:', cp['best_val_dice'])
