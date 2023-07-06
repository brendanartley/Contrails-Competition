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
    # all_pred_dirs = ["astral-spaceship-387", "olive-dew-339", "olive-gorge-380"],
    all_pred_dirs = ["olive-gorge-380"],
    # all_pred_dirs = ["None_0", "None"],
    ensemble = False,
    device = torch.device("cpu"),
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

def get_best_thresholds():

    all_thresholds = {k:0 for k in config.all_pred_dirs}

    # Iterate over all folders
    for model_name in config.all_pred_dirs:
        
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
                all_thresholds[model_name] = np.round(best_threshold, 2)

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
                all_thresholds[model_name] = np.round(best_threshold, 2)
            # Get dice score
            current_dice = get_dice_score(model_name=model_name, threshold=current_threshold)

            # Update if score is better
            if current_dice > best_dice:
                best_dice = current_dice
                best_threshold = current_threshold
                all_thresholds[model_name] = np.round(best_threshold, 2)
        
        print()
        print("Model: {}  Best-Thresh: {:.2f} Best-Dice: {:.6f}".format(model_name, best_threshold, best_dice))
    
    return all_thresholds

def dice_ensemble(all_thresholds):

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
            new_mask = torch.zeros_like(pred)
            new_mask[pred >= all_thresholds[fold_path]] = 1
            new_mask[pred < all_thresholds[fold_path]] = 0
            new_mask = new_mask.squeeze()

            # Add to pred ensemble
            cur_batch.append(new_mask)

        # Take the median of all preds
        cur_batch = torch.stack(cur_batch).int()
        cur_batch = torch.median(cur_batch, dim=0)[0]

        # Update metric
        metric.update(pred, truth)

    return metric.compute().item()


def parse_args(config):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument('--ensemble', action='store_true', help='Process ensemble predictions.')
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config

def main():
    all_thresholds = get_best_thresholds()
    ensemble_score = dice_ensemble(all_thresholds)
    print("Final: ", ensemble_score)
    return

if __name__ == "__main__":
    main()
