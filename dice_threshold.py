import torchmetrics
import torch
import os
from tqdm import tqdm

"""
Finds the best dice threshold for a set of predictions.
"""


# Config
preds_dir = "./pred_test/"

def get_dice_score(preds_dir, threshold=0.5):

    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold = threshold)

    # Iterate over preds
    for t_path in os.listdir(preds_dir):
        
        # Load preds + truth
        loaded_tensor = torch.load(preds_dir + t_path, map_location=torch.device('cpu'))
        pred = loaded_tensor[0, ...]
        truth = loaded_tensor[1, ...].int()

        # Update metric
        metric.update(pred, truth)

    return metric.compute().item()

# ---------- Baseline ----------
base_dice = get_dice_score(preds_dir=preds_dir, threshold=0.5)
print("Dice: {:.6f}".format(base_dice))

# ---------- Find Best Threshold -----------
# Iterate over predictions
best_dice = 0
best_threshold = 0

for i in tqdm(range(10, 90, 1)):

    # Get dice score
    current_threshold = i/100
    current_dice = get_dice_score(preds_dir=preds_dir, threshold=current_threshold)

    # Update if score is better
    if current_dice > best_dice:
        best_dice = current_dice
        best_threshold = current_threshold

print("Best-Thresh: {:.2f} Best-Dice: {:.6f}".format(best_threshold, best_dice))
