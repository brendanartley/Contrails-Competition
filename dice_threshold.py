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
        'bfg_2': -3.90, # Best-Dice: 0.666979 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_874': -2.90, # Best-Dice: 0.663905 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_3': -5.90, # Best-Dice: 0.663819 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k (784)
        'bfg_122': -4.06, # Best-Dice: 0.663042 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_867': -2.22, # Best-Dice: 0.662304 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_416': -1.62, # Best-Dice: 0.662164 - tu-maxvit_base_tf_512.in21k_ft_in1k
        # 'silvery-plasma-621': -3.14, # Best-Dice: 0.662511 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'pretty-microwave-583': -4.50, # Best-Dice: 0.659503 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'twilight-sun-592': -2.10, # Best-Dice: 0.659122 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bfg_612': -3.30, # Best-Dice: 0.657971 - mit_b4 (800 img_size)
        # 'denim-blaze-658': -3.58, # Best-Dice: 0.657714 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'fresh-bee-660': -2.82, # Best-Dice: 0.657683 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'neat-wind-659': -2.82, # Best-Dice: 0.657312 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'treasured-waterfall-590': -2.62, # Best-Dice: 0.657126 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bfg_852': -2.62, # Best-Dice: 0.657135 - mit_b4 (800 img_size)
        'bfg_684': -5.06, # Best-Dice: 0.656235 - tu-resnest269e.in1k (800 img size)
        # 'bright-water-589': -3.94, # Best-Dice: 0.653599 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'iconic-field-657': -1.74, # Best-Dice: 0.653737 - mit_b4
        # 'electric-haze-579': 0.34, # Best-Dice: 0.653147 - mit_b4
        # 'light-valley-599': -1.38, # Best-Dice: 0.652941 - mit_b4
        # 'olive-gorge-380': -4.1, # Best-Dice: 0.652049 - mit_b4
        # 'peachy-dream-519': -3.94, # Best-Dice: 0.649420 - mit_b4
        # 'spring-sweep-2': -1.38, # Best-Dice: 0.636749 - mit_tiny
    },
    # models = ["bfg_1", "bfg_2", "bfg_416", 'pretty-microwave-583', 'treasured-waterfall-590', 'bfg_852', "bfg_684"], # 0.6812
    # models = ["bfg_1", "bfg_2", "bfg_3", "bfg_416", 'pretty-microwave-583', 'bfg_684', 'bfg_852'] # 0.6817
    models = ["bfg_1", "bfg_2", "bfg_3", "bfg_416", 'pretty-microwave-583', 'bfg_684', 'bfg_852']
)


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
    for model_name in config.models:

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

    for batch_path in tqdm(os.listdir(os.path.join(config.preds_dir, config.models[0]))):

        cur_batch = []
        for fold_path in config.models:
            
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
    # for model_name in config.models:
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
