import torchmetrics
import torch
import os
from tqdm import tqdm
from types import SimpleNamespace
import argparse

"""
Finds the best dice threshold for a set of predictions.
"""

config = SimpleNamespace(
    preds_dir = "/data/bartley/gpu_test/preds/",
    # all_pred_dirs = ["fresh-sunset-131", "elated-oath-130", "firm-butterfly-128", "pious-sun-127", "drawn-cloud-126"],
    all_pred_dirs = ["None_0", "None"],
    ensemble = True,
    device = torch.device("cpu"),
)


def get_dice_score(threshold=0.5):
    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold = threshold)

    if config.ensemble == False:
        # Iterate over preds
        for fold_path in config.all_pred_dirs:
            for batch_path in os.listdir(os.path.join(config.preds_dir, fold_path)):
                
                # Load preds + truth
                loaded_tensor = torch.load(os.path.join(config.preds_dir, fold_path, batch_path), map_location=config.device)

                pred = loaded_tensor[0, ...]
                truth = loaded_tensor[1, ...].int()
                
                # Update metric
                metric.update(pred, truth)

    elif config.ensemble == True:
            for batch_path in os.listdir(os.path.join(config.preds_dir, config.all_pred_dirs[0])):
                # Iterate over all preds
                all_preds = []
                for i, fold_path in enumerate(config.all_pred_dirs):
                    loaded_tensor = torch.load(os.path.join(config.preds_dir, fold_path, batch_path), map_location=config.device)
                    # extract label on first iteration
                    if i == 0:
                        labels = loaded_tensor[1, ...].int()
                        
                    all_preds.append(
                        loaded_tensor[0, ...]
                    )
                avg_preds = torch.sum(torch.stack(all_preds), dim=0) / len(all_preds)
                
                # Update metric
                metric.update(avg_preds, labels)

    return metric.compute().item()

def check_dice_scores():
    # ---------- Baseline ----------
    base_dice = get_dice_score(threshold=0.5)
    print("Dice: {:.6f}".format(base_dice))

    # ---------- Test different thresholds -----------
    # Iterate over predictions
    best_dice = 0
    best_threshold = 0
    xs = []
    ys = []

    for i in tqdm(range(-500, 100, 50)):
        # Get dice score
        current_threshold = i/100
        current_dice = get_dice_score(threshold=current_threshold)

        # Update if score is better
        if current_dice > best_dice:
            best_dice = current_dice
            best_threshold = current_threshold

        xs.append(current_threshold)
        ys.append(current_dice)
        print("Thresh: {:.2f} Dice: {:.6f}".format(current_threshold, current_dice))

    print("xs =", xs)
    print("ys =", ys)
    print()
    print("Best-Thresh: {:.2f} Best-Dice: {:.6f}".format(best_threshold, best_dice))
    return

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
    check_dice_scores()
    return

if __name__ == "__main__":
    main()
