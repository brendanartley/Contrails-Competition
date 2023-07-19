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
        'bfg_122': -4.06, # Best-Dice: 0.663042 - tu-maxvit_base_tf_512.in21k_ft_in1k
        'bfg_867': -2.22, # Best-Dice: 0.662304 - tu-maxvit_base_tf_512.in21k_ft_in1k
        # 'silvery-plasma-621': -3.14, # Best-Dice: 0.662511 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'pretty-microwave-583': -4.50, # Best-Dice: 0.659503 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'twilight-sun-592': -2.10, # Best-Dice: 0.659122 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'denim-blaze-658': -3.58, # Best-Dice: 0.657714 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'fresh-bee-660': -2.82, # Best-Dice: 0.657683 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'neat-wind-659': -2.82, # Best-Dice: 0.657312 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'treasured-waterfall-590': -2.62, # Best-Dice: 0.657126 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        # 'bright-water-589': -3.94, # Best-Dice: 0.653599 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'iconic-field-657': -1.74, # Best-Dice: 0.653737 - mit_b4
        # 'electric-haze-579': 0.34, # Best-Dice: 0.653147 - mit_b4
        # 'light-valley-599': -1.38, # Best-Dice: 0.652941 - mit_b4
        # 'olive-gorge-380': -4.1, # Best-Dice: 0.652049 - mit_b4
        # 'peachy-dream-519': -3.94, # Best-Dice: 0.649420 - mit_b4
        # 'spring-sweep-2': -1.38, # Best-Dice: 0.636749 - mit_tiny
    },
    # models = ["bfg_1", "bfg_2", 'pretty-microwave-583', 'treasured-waterfall-590', 'iconic-field-657', "azure-elevator-669"], # 0.679
    models = ["bfg_1", "bfg_2", "bfg_867", 'pretty-microwave-583', 'treasured-waterfall-590', 'iconic-field-657'], # 0.6791
)

known_corr = {'bfg_1': {'bfg_2': 0.9995815, 'bfg_874': 0.9995744, 'bfg_122': 0.9995198, 'bfg_867': 0.999516, 'pretty-microwave-583': 0.9993104, 'treasured-waterfall-590': 0.9993014, 'iconic-field-657': 0.9992986}, 'bfg_2': {'bfg_1': 0.9995815, 'bfg_874': 0.999568, 'bfg_122': 0.9995651, 'bfg_867': 0.9995279, 'pretty-microwave-583': 0.9993357, 'treasured-waterfall-590': 0.9993243, 'iconic-field-657': 0.9993303}, 'bfg_874': {'bfg_1': 0.9995744, 'bfg_2': 0.999568, 'bfg_122': 0.9995621, 'bfg_867': 0.9995204, 'pretty-microwave-583': 0.9993286, 'treasured-waterfall-590': 0.9993176, 'iconic-field-657': 0.9993291}, 'bfg_122': {'bfg_1': 0.9995198, 'bfg_2': 0.9995651, 'bfg_874': 0.9995621, 'bfg_867': 0.9995058, 'pretty-microwave-583': 0.9993064, 'treasured-waterfall-590': 0.9992935, 'iconic-field-657': 0.9992972}, 'bfg_867': {'bfg_1': 0.999516, 'bfg_2': 0.9995279, 'bfg_874': 0.9995204, 'bfg_122': 0.9995058, 'pretty-microwave-583': 0.9992991, 'treasured-waterfall-590': 0.999279, 'iconic-field-657': 0.9993068}, 'pretty-microwave-583': {'bfg_1': 0.9993104, 'bfg_2': 0.9993357, 'bfg_874': 0.9993286, 'bfg_122': 0.9993064, 'bfg_867': 0.9992991, 'treasured-waterfall-590': 0.9994658, 'iconic-field-657': 0.9993536}, 'treasured-waterfall-590': {'bfg_1': 0.9993014, 'bfg_2': 0.9993243, 'bfg_874': 0.9993176, 'bfg_122': 0.9992935, 'bfg_867': 0.999279, 'pretty-microwave-583': 0.9994658, 'iconic-field-657': 0.9993266}, 'iconic-field-657': {'bfg_1': 0.9992986, 'bfg_2': 0.9993303, 'bfg_874': 0.9993291, 'bfg_122': 0.9992972, 'bfg_867': 0.9993068, 'pretty-microwave-583': 0.9993536, 'treasured-waterfall-590': 0.9993266}}

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

    print(corr_dict)
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
