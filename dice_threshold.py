import torchmetrics
import torch
import os
from tqdm import tqdm
from types import SimpleNamespace
import argparse
import numpy as np

"""
Finds the best dice threshold for a set of predictions.
"""

config = SimpleNamespace(
    preds_dir = "/data/bartley/gpu_test/preds/",
    device = torch.device("cpu"),
    # Known Thresholds
    all_thresholds = {
        'bfg_1': -3.02, # Best-Dice: 0.667923 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'pretty-microwave-583': -4.50, # Best-Dice: 0.659503 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'twilight-sun-592': -2.10, # Best-Dice: 0.659122 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'treasured-waterfall-590': -2.62, # Best-Dice: 0.657126 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bright-water-589': -3.94, # Best-Dice: 0.653599 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'electric-haze-579': 0.34, # Best-Dice: 0.653147 - mit_b4
        'light-valley-599': -1.38, # Best-Dice: 0.652941 - mit_b4
        'olive-gorge-380': -4.1, # Best-Dice: 0.652049 - mit_b4
        'peachy-dream-519': -3.94, # Best-Dice: 0.649420 - mit_b4
    },
    all_pred_dirs = ["bfg_1", "bfg_1", 'pretty-microwave-583', 'twilight-sun-592', 'treasured-waterfall-590', 'bright-water-589', 'electric-haze-579'], # 0.677
)

# assert len(config.all_pred_dirs) > 2


def get_dice_score(model_name, threshold=0.5):
    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold = threshold)

    for img_idx in os.listdir(os.path.join(config.preds_dir, model_name)):
        
        # Load preds + truth
        loaded_tensor = torch.load(os.path.join(config.preds_dir, model_name, img_idx), map_location=config.device)

        pred = loaded_tensor[0, ...]
        truth = loaded_tensor[1, ...].int()

        # Update metric
        metric.update(pred, truth)

    return metric.compute().item()

def get_best_threshold(model_name):
    # Iterate over predictions
    best_dice = 0
    best_threshold = 0

    # Find general area of best threshold
    for current_threshold in tqdm(range(-10, 1, 1)):
        # Get dice score
        current_dice = get_dice_score(model_name=model_name, threshold=current_threshold)

        # Update if score is better
        if current_dice > best_dice:
            best_dice = current_dice
            best_threshold = current_threshold

    # Iterate over predictions
    middle = best_threshold
    best_dice = 0
    best_threshold = 0

    # Find more specific best threshold
    for current_threshold in tqdm(np.arange(middle-0.5, middle+0.5, 0.04)):
        # Get dice score
        current_dice = get_dice_score(model_name=model_name, threshold=current_threshold)

        # Update if score is better
        if current_dice > best_dice:
            best_dice = current_dice
            best_threshold = current_threshold

    print()
    print("'{}': {:.2f}, # Best-Dice: {:.6f}".format(model_name, best_threshold, best_dice))
    return best_threshold
    
def get_best_thresholds(all_thresholds):

    # Only return selected models
    res = {}

    # Iterate over all folders
    for model_name in config.all_pred_dirs:

        # Break if already computed
        if model_name in all_thresholds:
            res[model_name] = all_thresholds[model_name]
            continue

        best_threshold = get_best_threshold(model_name=model_name)
        res[model_name] = best_threshold
    
    return res

def dice_ensemble(all_thresholds):

    # Log threshold values
    print(all_thresholds)

    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold=0.5)

    for batch_path in tqdm(os.listdir(os.path.join(config.preds_dir, config.all_pred_dirs[0]))):

        cur_batch = []
        for fold_path in config.all_pred_dirs:
            
            # Load preds + truth
            loaded_tensor = torch.load(os.path.join(config.preds_dir, fold_path, batch_path), map_location=config.device)
            pred = loaded_tensor[0, ...]
            truth = loaded_tensor[1, ...].int()

            # Make predictions
            new_mask = torch.zeros_like(pred, dtype=torch.float32)
            new_mask[pred >= all_thresholds[fold_path]] = 1.0
            new_mask[pred < all_thresholds[fold_path]] = 0.0
            new_mask = new_mask.squeeze()

            # Add to pred ensemble
            cur_batch.append(new_mask)

        # Take the median of all preds
        cur_batch = torch.round(torch.stack(cur_batch).squeeze())
        cur_batch = torch.median(cur_batch, dim=0)[0]

        # Update metric
        metric.update(cur_batch, truth)

    return metric.compute().item()

def main():

    # # Simple dice scores
    # for model_name in config.all_pred_dirs:
    #     print("Model {}, Score: {:.6f}".format(model_name, get_dice_score(model_name, threshold=0.5)))

    # Get thresholds
    all_thresholds = config.all_thresholds
    all_thresholds = get_best_thresholds(all_thresholds)

    # Ensemble
    ensemble_score = dice_ensemble(all_thresholds)
    print("Final: ", ensemble_score)
    return

if __name__ == "__main__":
    main()
