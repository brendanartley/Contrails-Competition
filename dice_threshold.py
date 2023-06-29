import torchmetrics
import torch
import os
from tqdm import tqdm

"""
Finds the best dice threshold for a set of predictions.
"""


# Config
class config:
    preds_dir = "/data/bartley/gpu_test/preds"
    # all_pred_dirs = ["fresh-sunset-131", "elated-oath-130", "firm-butterfly-128", "pious-sun-127", "drawn-cloud-126"]
    all_pred_dirs = ["None"]
    device = torch.device("cpu")

def get_dice_score(threshold=0.5):

    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold = threshold)

    # Iterate over preds
    for fold_path in config.all_pred_dirs:
        for batch_path in os.listdir(os.path.join(config.preds_dir, fold_path)):
            
            # Load preds + truth
            loaded_tensor = torch.load(os.path.join(config.preds_dir, fold_path, batch_path), map_location=config.device)

            pred = loaded_tensor[0, ...]
            truth = loaded_tensor[1, ...].int()

            # Update metric
            metric.update(pred, truth)

    return metric.compute().item()

# ---------- Baseline ----------
base_dice = get_dice_score(threshold=0.5)
print("Dice: {:.6f}".format(base_dice))

# ---------- Find Best Threshold -----------
# Iterate over predictions
best_dice = 0
best_threshold = 0

for i in tqdm(range(-500, 100, 25)):

    # Get dice score
    current_threshold = i/100
    current_dice = get_dice_score(threshold=current_threshold)

    # Update if score is better
    if current_dice > best_dice:
        best_dice = current_dice
        best_threshold = current_threshold
    print("Thresh: {:.2f} Dice: {:.6f}".format(current_threshold, current_dice))

# for i in tqdm(range(10, 91, 2)):

#     # Get dice score
#     current_threshold = i/100
#     current_dice = get_dice_score(threshold=current_threshold)

#     # Update if score is better
#     if current_dice > best_dice:
#         best_dice = current_dice
#         best_threshold = current_threshold
#     print("Thresh: {:.2f} Dice: {:.6f}".format(current_threshold, current_dice))

print("Best-Thresh: {:.2f} Best-Dice: {:.6f}".format(best_threshold, best_dice))
