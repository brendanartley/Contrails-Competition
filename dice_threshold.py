import torchmetrics
import torch
import os
from tqdm import tqdm
from types import SimpleNamespace
import argparse
import numpy as np
import pandas as pd
import itertools

"""
Finds the best dice threshold for a set of predictions.
"""

config = SimpleNamespace(
    preds_dir = "/data/bartley/gpu_test/preds/",
    device = torch.device("cpu"),
    # Known Thresholds
    all_thresholds = {
        'gpu_1': -4.14, # Best-Dice: 0.671748 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k (768, 15 epochs w/ SWA)
        'bfg_1': -3.02, # Best-Dice: 0.667923 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_2': -3.90, # Best-Dice: 0.666979 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_874': -2.90, # Best-Dice: 0.663905 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_3': -5.90, # Best-Dice: 0.663819 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k (784)
        'bfg_122': -4.06, # Best-Dice: 0.663042 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_867': -2.22, # Best-Dice: 0.662304 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_416': -1.62, # Best-Dice: 0.662164 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'silvery-plasma-621': -3.14, # Best-Dice: 0.662511 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bfg_6': -2.66, # Best-Dice: 0.661204 - mit_b4 (1024 img_size)
        'bfg_5': -6.90, # Best-Dice: 0.660798 - tu-resnest269e.in1k (1024 img size - didnt train long enough)
        'bfg_4': -1.50, # Best-Dice: 0.659906 - tu-maxvit_base_tf_512.in21k_ft_in1k (UnetPlusPlus)
        'pretty-microwave-583': -4.50, # Best-Dice: 0.659503 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'twilight-sun-592': -2.10, # Best-Dice: 0.659122 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bfg_612': -3.30, # Best-Dice: 0.657971 - mit_b4 (800 img_size)
        'denim-blaze-658': -3.58, # Best-Dice: 0.657714 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'fresh-bee-660': -2.82, # Best-Dice: 0.657683 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'neat-wind-659': -2.82, # Best-Dice: 0.657312 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'treasured-waterfall-590': -2.62, # Best-Dice: 0.657126 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bfg_852': -2.62, # Best-Dice: 0.657135 - mit_b4 (800 img_size)
        # 'bfg_684': -5.06, # Best-Dice: 0.656235 - tu-resnest269e.in1k (800 img size)
        # 'bright-water-589': -3.94, # Best-Dice: 0.653599 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'iconic-field-657': -1.74, # Best-Dice: 0.653737 - mit_b4
        # 'electric-haze-579': 0.34, # Best-Dice: 0.653147 - mit_b4
        # 'light-valley-599': -1.38, # Best-Dice: 0.652941 - mit_b4
        # 'olive-gorge-380': -4.1, # Best-Dice: 0.652049 - mit_b4
        # 'peachy-dream-519': -3.94, # Best-Dice: 0.649420 - mit_b4
        # 'spring-sweep-2': -1.38, # Best-Dice: 0.636749 - mit_tiny
    },
    # models = "bfg_1|bfg_2|bfg_416|bfg_5|bfg_852|gpu_1|pretty-microwave-583".split("|"), # 0.682
    models = [
        "bfg_1|bfg_416|gpu_1".split("|"),
        "bfg_2|bfg_5|gpu_1".split("|"),
        "bfg_416|bfg_5|gpu_1".split("|"),
    ]
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

def dice_ensemble(thresh_dict, models, log=False):

    # Log threshold values
    if log:print(thresh_dict)

    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold=0.5)

    for batch_path in os.listdir(os.path.join(config.preds_dir, models[0])):

        cur_batch = []
        for fold_path in models:
            
            # Load preds + truth
            loaded_tensor = torch.load(os.path.join(config.preds_dir, fold_path, batch_path), map_location=config.device)
            pred = loaded_tensor[0, ...]
            truth = loaded_tensor[1, ...].int()

            # Make predictions
            new_mask = torch.zeros_like(pred, dtype=torch.float32)
            new_mask[pred >= thresh_dict[fold_path]] = 1.0
            new_mask[pred < thresh_dict[fold_path]] = 0.0
            new_mask = new_mask.squeeze()

            # Add to pred ensemble
            cur_batch.append(new_mask)

        # Take the median of all preds
        cur_batch = torch.round(torch.stack(cur_batch).squeeze())
        cur_batch = torch.median(cur_batch, dim=0)[0]

        # Update metric
        metric.update(cur_batch, truth)

    return metric.compute().item()

def nested_dice_ensemble(thresh_dict, models, log=False):

    # Make nested prediction ensemble
    # Write to temp location
    # Combine the multiple nested predictions
    # Score against Dice

    # Log threshold values
    if log:print(thresh_dict)

    # iterate each nested ensemble
    final_ens = []
    for i, nest_ens in enumerate(models):

        # Define Metric
        metric = torchmetrics.Dice(average = 'micro', threshold=0.5)

        # Make temp directory for preds
        tmp_pred_dir = os.path.join(config.preds_dir, f"tmp_{i}/")
        if not os.path.exists(tmp_pred_dir):
            os.mkdir(tmp_pred_dir)
            print(f"Created {tmp_pred_dir}.")
        final_ens.append(f"tmp_{i}")

        for batch_path in tqdm(os.listdir(os.path.join(config.preds_dir, nest_ens[0]))):
            cur_batch = []
            for model in nest_ens:
            
                # Load preds + truth
                loaded_tensor = torch.load(os.path.join(config.preds_dir, model, batch_path), map_location=config.device)
                pred = loaded_tensor[0, ...]
                truth = loaded_tensor[1, ...].int()

                # Make predictions
                new_mask = torch.zeros_like(pred, dtype=torch.float32)
                new_mask[pred >= thresh_dict[model]] = 1.0
                new_mask[pred < thresh_dict[model]] = 0.0
                new_mask = new_mask.squeeze()

                # Add to pred ensemble
                cur_batch.append(new_mask)

            # Take the median of all preds
            cur_batch = torch.round(torch.stack(cur_batch).squeeze())
            if len(nest_ens) > 1:
                cur_batch = torch.median(cur_batch, dim=0)[0]

            # Update metric
            metric.update(cur_batch, truth)

            # Save to temp location
            # print(cur_batch.shape, truth.shape)
            cur_batch = torch.stack([cur_batch, truth], axis=0)
            # print(cur_batch.shape)
            # assert 1 > 2
            torch.save(cur_batch.half(), os.path.join(tmp_pred_dir, batch_path))
        print(metric.compute().item())

    final_score = dice_ensemble({x:0.5 for x in final_ens}, final_ens, log=True)
    return final_score

def ensemble_combinations(arr, size, all_thresholds):

    # Load known ensembles
    df = pd.read_csv("./ensemble.csv")
    known_ens = set(list(df["models"].values))
    print(df.head())

    new_ens = []
    combinations = list(itertools.combinations(arr, size))

    # Score each ensemble
    for i, comb in tqdm(enumerate(combinations), total=len(combinations)):
       
        # Check if ensemble already computed
        comb_string = "|".join(sorted(comb))
        if comb_string in known_ens:
            continue

        # Compute score    
        score = dice_ensemble(all_thresholds, comb)
        score = np.round(score, 6)
        new_ens.append({"models": comb_string, "score": score, "size": size})
    
    # Add new ensembles to dataframe
    new_df = pd.DataFrame(new_ens)
    df = pd.concat([df, new_df], axis=0).sort_values("score", ascending=False)
    df.to_csv("ensemble.csv", index=False)
    return

def main():

    # # Simple dice scores
    # for model_name in config.models:
    #     print("Model {}, Score: {:.6f}".format(model_name, get_dice_score(model_name, threshold=0.5)))

    # Get thresholds
    all_thresholds = config.all_thresholds
    # all_thresholds = get_best_thresholds(all_thresholds)

    # Ensemble
    # ensemble_score = dice_ensemble(all_thresholds, config.models, log=True)
    ensemble_score = nested_dice_ensemble(all_thresholds, config.models, log=True)
    print("Final: ", ensemble_score)

    # Testing all combinations
    # all_thresholds = config.all_thresholds
    # ensemble_combinations(
    #     arr = list(all_thresholds.keys()), 
    #     size = 3,
    #     all_thresholds = all_thresholds,
    #     )
    
    # ensemble_combinations(
    #     arr = list(all_thresholds.keys()), 
    #     size = 3,
    #     all_thresholds = all_thresholds,
    #     )

    return

if __name__ == "__main__":
    main()
