import torchmetrics
import torch
import os
from tqdm import tqdm
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Plots a correlation matrix to find models that are not correlated.
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
    models = ["bfg_1", "bfg_2", "bfg_416", 'bfg_3', 'treasured-waterfall-590', 'bfg_852', "bfg_684"], # 0.6812
)

known_corr =  {'bfg_1': {'bfg_2': 0.9995815, 'bfg_874': 0.9995744, 'bfg_3': 0.9993248, 'bfg_122': 0.9995198, 'bfg_867': 0.999516, 'bfg_416': 0.9995137, 'pretty-microwave-583': 0.9993104, 'bfg_612': 0.9993356, 'treasured-waterfall-590': 0.9993014, 'bfg_852': 0.9993196, 'bfg_684': 0.9993144, 'iconic-field-657': 0.9992986}, 'bfg_2': {'bfg_1': 0.9995815, 'bfg_874': 0.999568, 'bfg_3': 0.9993443, 'bfg_122': 0.9995651, 'bfg_867': 0.9995279, 'bfg_416': 0.99951, 'pretty-microwave-583': 0.9993357, 'bfg_612': 0.99937, 'treasured-waterfall-590': 0.9993243, 'bfg_852': 0.9993433, 'bfg_684': 0.9993336, 'iconic-field-657': 0.9993303}, 'bfg_874': {'bfg_1': 0.9995744, 'bfg_2': 0.999568, 'bfg_3': 0.9993559, 'bfg_122': 0.9995621, 'bfg_867': 0.9995204, 'bfg_416': 0.9995185, 'pretty-microwave-583': 0.9993286, 'bfg_612': 0.9993677, 'treasured-waterfall-590': 0.9993176, 'bfg_852': 0.9993438, 'bfg_684': 0.9993452, 'iconic-field-657': 0.9993291}, 'bfg_3': {'bfg_1': 0.9993248, 'bfg_2': 0.9993443, 'bfg_874': 0.9993559, 'bfg_122': 0.9993168, 'bfg_867': 0.9993118, 'bfg_416': 0.9993063, 'pretty-microwave-583': 0.9993899, 'bfg_612': 0.9993955, 'treasured-waterfall-590': 0.999369, 'bfg_852': 0.9993784, 'bfg_684': 0.9994094, 'iconic-field-657': 0.9993597}, 'bfg_122': {'bfg_1': 0.9995198, 'bfg_2': 0.9995651, 'bfg_874': 0.9995621, 'bfg_3': 0.9993168, 'bfg_867': 0.9995058, 'bfg_416': 0.9995029, 'pretty-microwave-583': 0.9993064, 'bfg_612': 0.999347, 'treasured-waterfall-590': 0.9992935, 'bfg_852': 0.9993292, 'bfg_684': 0.9993079, 'iconic-field-657': 0.9992972}, 'bfg_867': {'bfg_1': 0.999516, 'bfg_2': 0.9995279, 'bfg_874': 0.9995204, 'bfg_3': 0.9993118, 'bfg_122': 0.9995058, 'bfg_416': 0.9995645, 'pretty-microwave-583': 0.9992991, 'bfg_612': 0.9993254, 'treasured-waterfall-590': 0.999279, 'bfg_852': 0.9993115, 'bfg_684': 0.9992881, 'iconic-field-657': 0.9993068}, 'bfg_416': {'bfg_1': 0.9995137, 'bfg_2': 0.99951, 'bfg_874': 0.9995185, 'bfg_3': 0.9993063, 'bfg_122': 0.9995029, 'bfg_867': 0.9995645, 'pretty-microwave-583': 0.9992913, 'bfg_612': 0.9993178, 'treasured-waterfall-590': 0.9992793, 'bfg_852': 0.9992979, 'bfg_684': 0.9992872, 'iconic-field-657': 0.9993072}, 'pretty-microwave-583': {'bfg_1': 0.9993104, 'bfg_2': 0.9993357, 'bfg_874': 0.9993286, 'bfg_3': 0.9993899, 'bfg_122': 0.9993064, 'bfg_867': 0.9992991, 'bfg_416': 0.9992913, 'bfg_612': 0.9993626, 'treasured-waterfall-590': 0.9994658, 'bfg_852': 0.9993449, 'bfg_684': 0.9993337, 'iconic-field-657': 0.9993536}, 'bfg_612': {'bfg_1': 0.9993356, 'bfg_2': 0.99937, 'bfg_874': 0.9993677, 'bfg_3': 0.9993955, 'bfg_122': 0.999347, 'bfg_867': 0.9993254, 'bfg_416': 0.9993178, 'pretty-microwave-583': 0.9993626, 'treasured-waterfall-590': 0.999341, 'bfg_852': 0.9995348, 'bfg_684': 0.9993851, 'iconic-field-657': 0.9994278}, 'treasured-waterfall-590': {'bfg_1': 0.9993014, 'bfg_2': 0.9993243, 'bfg_874': 0.9993176, 'bfg_3': 0.999369, 'bfg_122': 0.9992935, 'bfg_867': 0.999279, 'bfg_416': 0.9992793, 'pretty-microwave-583': 0.9994658, 'bfg_612': 0.999341, 'bfg_852': 0.9993212, 'bfg_684': 0.9993272, 'iconic-field-657': 0.9993266}, 'bfg_852': {'bfg_1': 0.9993196, 'bfg_2': 0.9993433, 'bfg_874': 0.9993438, 'bfg_3': 0.9993784, 'bfg_122': 0.9993292, 'bfg_867': 0.9993115, 'bfg_416': 0.9992979, 'pretty-microwave-583': 0.9993449, 'bfg_612': 0.9995348, 'treasured-waterfall-590': 0.9993212, 'bfg_684': 0.9993811, 'iconic-field-657': 0.9994293}, 'bfg_684': {'bfg_1': 0.9993144, 'bfg_2': 0.9993336, 'bfg_874': 0.9993452, 'bfg_3': 0.9994094, 'bfg_122': 0.9993079, 'bfg_867': 0.9992881, 'bfg_416': 0.9992872, 'pretty-microwave-583': 0.9993337, 'bfg_612': 0.9993851, 'treasured-waterfall-590': 0.9993272, 'bfg_852': 0.9993811, 'iconic-field-657': 0.9993508}, 'iconic-field-657': {'bfg_1': 0.9992986, 'bfg_2': 0.9993303, 'bfg_874': 0.9993291, 'bfg_3': 0.9993597, 'bfg_122': 0.9992972, 'bfg_867': 0.9993068, 'bfg_416': 0.9993072, 'pretty-microwave-583': 0.9993536, 'bfg_612': 0.9994278, 'treasured-waterfall-590': 0.9993266, 'bfg_852': 0.9994293, 'bfg_684': 0.9993508}}

def get_dice_correlation(m1, m2):
    # Define Metric
    metric = torchmetrics.Dice(average = 'micro', threshold=0.5)

    for img_idx in os.listdir(os.path.join(config.preds_dir, m1)):
        
        # Load 2 models
        # Model 1
        mask1 = torch.load(os.path.join(config.preds_dir, m1, img_idx), map_location=config.device)[0, ...]
        new_mask = torch.zeros_like(mask1, dtype=torch.int32)
        new_mask[mask1 >= config.all_thresholds[m1]] = 1
        new_mask[mask1 < config.all_thresholds[m1]] = 0
        mask1 = new_mask.squeeze()

        # Model 2
        mask2 = torch.load(os.path.join(config.preds_dir, m2, img_idx), map_location=config.device)[0, ...]
        new_mask = torch.zeros_like(mask2, dtype=torch.int32)
        new_mask[mask2 >= config.all_thresholds[m2]] = 1
        new_mask[mask2 < config.all_thresholds[m2]] = 0
        mask2 = new_mask.squeeze()

        # Update metric
        metric.update(mask1, mask2)

    return np.round(metric.compute().item(), 7)

def get_correlation_dict(models):

    # Store correlation scores
    corr_dict = {}
    for val in models:
        corr_dict[val] = {}

    # Iterate over every combination
    for i in tqdm(range(len(models)-1)):
        m1 = models[i]
        for j in range(i+1, len(models)):
            m2 = models[j]

            # Check score not calculated already
            if m2 in known_corr and m1 in known_corr[m2]:
                dice_corr = known_corr[m2][m1]
            elif m1 in known_corr and m2 in known_corr[m1]:
                dice_corr = known_corr[m1][m2]
            # Calculate Dice Corr
            else:
                dice_corr = get_dice_correlation(m1, m2)
            corr_dict[m1][m2] = dice_corr
            corr_dict[m2][m1] = dice_corr

    print("known_corr = ", corr_dict)
    return corr_dict

def save_correlation_plot(corr_dict, models):

    # Initialize an empty correlation matrix
    corr_matrix = pd.DataFrame(index=models, columns=models)

    # Fill in the correlation values
    for key1, values in corr_dict.items():
        for key2, value in values.items():
            corr_matrix.at[key1, key2] = value
            corr_matrix.at[key2, key1] = value
    corr_matrix = corr_matrix.fillna(1)

    # Create a new figure and axes using fig, ax
    fig, ax = plt.subplots(figsize=(12,12))

    # Display the correlation matrix as a heatmap using matplotlib
    im = ax.imshow(corr_matrix, cmap='RdBu_r', interpolation='nearest')

    # Add a color bar
    cbar = fig.colorbar(im, ax=ax)

    # Add axis labels and ticks
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.index)

    # Display the correlation values on the heatmap
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.5f}", ha='center', va='center', color='white', fontsize=5)

    # Add a title
    ax.set_title("Correlation Matrix")

    # Show the plot
    plt.tight_layout()
    plt.savefig('correlation.png')
    return


def main():

    # Model list
    models = list(config.all_thresholds.keys())
    # models = config.models

    # Store correlation scores
    corr_dict = get_correlation_dict(models)

    # Save correlation matrix
    save_correlation_plot(corr_dict, models)

if __name__ == "__main__":
    main()
