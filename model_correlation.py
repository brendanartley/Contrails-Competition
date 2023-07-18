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
        'silvery-plasma-621': -3.14, # Best-Dice: 0.662511 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'pretty-microwave-583': -4.50, # Best-Dice: 0.659503 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'twilight-sun-592': -2.10, # Best-Dice: 0.659122 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'denim-blaze-658': -3.58, # Best-Dice: 0.657714 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'fresh-bee-660': -2.82, # Best-Dice: 0.657683 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'neat-wind-659': -2.82, # Best-Dice: 0.657312 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'treasured-waterfall-590': -2.62, # Best-Dice: 0.657126 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'bright-water-589': -3.94, # Best-Dice: 0.653599 - tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k
        'iconic-field-657': -1.74, # Best-Dice: 0.653737 - mit_b4
        'electric-haze-579': 0.34, # Best-Dice: 0.653147 - mit_b4
        'light-valley-599': -1.38, # Best-Dice: 0.652941 - mit_b4
        'olive-gorge-380': -4.1, # Best-Dice: 0.652049 - mit_b4
        'peachy-dream-519': -3.94, # Best-Dice: 0.649420 - mit_b4
        'spring-sweep-2': -1.38, # Best-Dice: 0.636749 - mit_tiny
    },
    all_pred_dirs = ["bfg_1", "bfg_2", 'pretty-microwave-583', 'treasured-waterfall-590', 'iconic-field-657'], # 0.6791
)

known_corr = {'bfg_1': {'bfg_2': 0.9995815, 'silvery-plasma-621': 0.9993443, 'pretty-microwave-583': 0.9993104, 'twilight-sun-592': 0.9993215, 'denim-blaze-658': 0.9993446, 'fresh-bee-660': 0.9993213, 'neat-wind-659': 0.9993383, 'treasured-waterfall-590': 0.9993014, 'bright-water-589': 0.9992713, 'iconic-field-657': 0.9992986, 'electric-haze-579': 0.9993026, 'light-valley-599': 0.9992918, 'olive-gorge-380': 0.9992849, 'peachy-dream-519': 0.9992982, 'spring-sweep-2': 0.9992754}, 'bfg_2': {'bfg_1': 0.9995815, 'silvery-plasma-621': 0.9993612, 'pretty-microwave-583': 0.9993357, 'twilight-sun-592': 0.9993443, 'denim-blaze-658': 0.99936, 'fresh-bee-660': 0.9993673, 'neat-wind-659': 0.9993602, 'treasured-waterfall-590': 0.9993243, 'bright-water-589': 0.9992912, 'iconic-field-657': 0.9993303, 'electric-haze-579': 0.9993216, 'light-valley-599': 0.9993186, 'olive-gorge-380': 0.9993107, 'peachy-dream-519': 0.9993335, 'spring-sweep-2': 0.9992954}, 'silvery-plasma-621': {'bfg_1': 0.9993443, 'bfg_2': 0.9993612, 'pretty-microwave-583': 0.9995232, 'twilight-sun-592': 0.9995189, 'denim-blaze-658': 0.9995631, 'fresh-bee-660': 0.9995469, 'neat-wind-659': 0.9995528, 'treasured-waterfall-590': 0.9994922, 'bright-water-589': 0.9994931, 'iconic-field-657': 0.9993731, 'electric-haze-579': 0.9993705, 'light-valley-599': 0.9993786, 'olive-gorge-380': 0.9993666, 'peachy-dream-519': 0.9993744, 'spring-sweep-2': 0.9993216}, 'pretty-microwave-583': {'bfg_1': 0.9993104, 'bfg_2': 0.9993357, 'silvery-plasma-621': 0.9995232, 'twilight-sun-592': 0.9995003, 'denim-blaze-658': 0.9995638, 'fresh-bee-660': 0.9995282, 'neat-wind-659': 0.9995471, 'treasured-waterfall-590': 0.9994658, 'bright-water-589': 0.9994729, 'iconic-field-657': 0.9993536, 'electric-haze-579': 0.9993345, 'light-valley-599': 0.9993444, 'olive-gorge-380': 0.9993352, 'peachy-dream-519': 0.9993458, 'spring-sweep-2': 0.9992739}, 'twilight-sun-592': {'bfg_1': 0.9993215, 'bfg_2': 0.9993443, 'silvery-plasma-621': 0.9995189, 'pretty-microwave-583': 0.9995003, 'denim-blaze-658': 0.9995355, 'fresh-bee-660': 0.9995364, 'neat-wind-659': 0.9995785, 'treasured-waterfall-590': 0.99947, 'bright-water-589': 0.9994709, 'iconic-field-657': 0.9993587, 'electric-haze-579': 0.999345, 'light-valley-599': 0.9993334, 'olive-gorge-380': 0.9993479, 'peachy-dream-519': 0.9993535, 'spring-sweep-2': 0.999274}, 'denim-blaze-658': {'bfg_1': 0.9993446, 'bfg_2': 0.99936, 'silvery-plasma-621': 0.9995631, 'pretty-microwave-583': 0.9995638, 'twilight-sun-592': 0.9995355, 'fresh-bee-660': 0.9995607, 'neat-wind-659': 0.9995717, 'treasured-waterfall-590': 0.9995104, 'bright-water-589': 0.9995188, 'iconic-field-657': 0.9993762, 'electric-haze-579': 0.9993681, 'light-valley-599': 0.9993552, 'olive-gorge-380': 0.9993596, 'peachy-dream-519': 0.9993632, 'spring-sweep-2': 0.9992979}, 'fresh-bee-660': {'bfg_1': 0.9993213, 'bfg_2': 0.9993673, 'silvery-plasma-621': 0.9995469, 'pretty-microwave-583': 0.9995282, 'twilight-sun-592': 0.9995364, 'denim-blaze-658': 0.9995607, 'neat-wind-659': 0.999566, 'treasured-waterfall-590': 0.9994983, 'bright-water-589': 0.999522, 'iconic-field-657': 0.9993676, 'electric-haze-579': 0.9993457, 'light-valley-599': 0.9993585, 'olive-gorge-380': 0.9993524, 'peachy-dream-519': 0.9993715, 'spring-sweep-2': 0.9992995}, 'neat-wind-659': {'bfg_1': 0.9993383, 'bfg_2': 0.9993602, 'silvery-plasma-621': 0.9995528, 'pretty-microwave-583': 0.9995471, 'twilight-sun-592': 0.9995785, 'denim-blaze-658': 0.9995717, 'fresh-bee-660': 0.999566, 'treasured-waterfall-590': 0.9994977, 'bright-water-589': 0.9995102, 'iconic-field-657': 0.9993852, 'electric-haze-579': 0.9993758, 'light-valley-599': 0.9993688, 'olive-gorge-380': 0.9993678, 'peachy-dream-519': 0.9993735, 'spring-sweep-2': 0.99931}, 'treasured-waterfall-590': {'bfg_1': 0.9993014, 'bfg_2': 0.9993243, 'silvery-plasma-621': 0.9994922, 'pretty-microwave-583': 0.9994658, 'twilight-sun-592': 0.99947, 'denim-blaze-658': 0.9995104, 'fresh-bee-660': 0.9994983, 'neat-wind-659': 0.9994977, 'bright-water-589': 0.9994519, 'iconic-field-657': 0.9993266, 'electric-haze-579': 0.9993107, 'light-valley-599': 0.9993013, 'olive-gorge-380': 0.9993065, 'peachy-dream-519': 0.9993047, 'spring-sweep-2': 0.9992471}, 'bright-water-589': {'bfg_1': 0.9992713, 'bfg_2': 0.9992912, 'silvery-plasma-621': 0.9994931, 'pretty-microwave-583': 0.9994729, 'twilight-sun-592': 0.9994709, 'denim-blaze-658': 0.9995188, 'fresh-bee-660': 0.999522, 'neat-wind-659': 0.9995102, 'treasured-waterfall-590': 0.9994519, 'iconic-field-657': 0.9993122, 'electric-haze-579': 0.9993069, 'light-valley-599': 0.999298, 'olive-gorge-380': 0.9993081, 'peachy-dream-519': 0.999307, 'spring-sweep-2': 0.99924}, 'iconic-field-657': {'bfg_1': 0.9992986, 'bfg_2': 0.9993303, 'silvery-plasma-621': 0.9993731, 'pretty-microwave-583': 0.9993536, 'twilight-sun-592': 0.9993587, 'denim-blaze-658': 0.9993762, 'fresh-bee-660': 0.9993676, 'neat-wind-659': 0.9993852, 'treasured-waterfall-590': 0.9993266, 'bright-water-589': 0.9993122, 'electric-haze-579': 0.9995362, 'light-valley-599': 0.9995065, 'olive-gorge-380': 0.9995136, 'peachy-dream-519': 0.9995197, 'spring-sweep-2': 0.9992903}, 'electric-haze-579': {'bfg_1': 0.9993026, 'bfg_2': 0.9993216, 'silvery-plasma-621': 0.9993705, 'pretty-microwave-583': 0.9993345, 'twilight-sun-592': 0.999345, 'denim-blaze-658': 0.9993681, 'fresh-bee-660': 0.9993457, 'neat-wind-659': 0.9993758, 'treasured-waterfall-590': 0.9993107, 'bright-water-589': 0.9993069, 'iconic-field-657': 0.9995362, 'light-valley-599': 0.9995027, 'olive-gorge-380': 0.999528, 'peachy-dream-519': 0.9995257, 'spring-sweep-2': 0.9992964}, 'light-valley-599': {'bfg_1': 0.9992918, 'bfg_2': 0.9993186, 'silvery-plasma-621': 0.9993786, 'pretty-microwave-583': 0.9993444, 'twilight-sun-592': 0.9993334, 'denim-blaze-658': 0.9993552, 'fresh-bee-660': 0.9993585, 'neat-wind-659': 0.9993688, 'treasured-waterfall-590': 0.9993013, 'bright-water-589': 0.999298, 'iconic-field-657': 0.9995065, 'electric-haze-579': 0.9995027, 'olive-gorge-380': 0.9995114, 'peachy-dream-519': 0.9995138, 'spring-sweep-2': 0.9992915}, 'olive-gorge-380': {'bfg_1': 0.9992849, 'bfg_2': 0.9993107, 'silvery-plasma-621': 0.9993666, 'pretty-microwave-583': 0.9993352, 'twilight-sun-592': 0.9993479, 'denim-blaze-658': 0.9993596, 'fresh-bee-660': 0.9993524, 'neat-wind-659': 0.9993678, 'treasured-waterfall-590': 0.9993065, 'bright-water-589': 0.9993081, 'iconic-field-657': 0.9995136, 'electric-haze-579': 0.999528, 'light-valley-599': 0.9995114, 'peachy-dream-519': 0.9995239, 'spring-sweep-2': 0.9992963}, 'peachy-dream-519': {'bfg_1': 0.9992982, 'bfg_2': 0.9993335, 'silvery-plasma-621': 0.9993744, 'pretty-microwave-583': 0.9993458, 'twilight-sun-592': 0.9993535, 'denim-blaze-658': 0.9993632, 'fresh-bee-660': 0.9993715, 'neat-wind-659': 0.9993735, 'treasured-waterfall-590': 0.9993047, 'bright-water-589': 0.999307, 'iconic-field-657': 0.9995197, 'electric-haze-579': 0.9995257, 'light-valley-599': 0.9995138, 'olive-gorge-380': 0.9995239, 'spring-sweep-2': 0.9992912}, 'spring-sweep-2': {'bfg_1': 0.9992754, 'bfg_2': 0.9992954, 'silvery-plasma-621': 0.9993216, 'pretty-microwave-583': 0.9992739, 'twilight-sun-592': 0.999274, 'denim-blaze-658': 0.9992979, 'fresh-bee-660': 0.9992995, 'neat-wind-659': 0.99931, 'treasured-waterfall-590': 0.9992471, 'bright-water-589': 0.99924, 'iconic-field-657': 0.9992903, 'electric-haze-579': 0.9992964, 'light-valley-599': 0.9992915, 'olive-gorge-380': 0.9992963, 'peachy-dream-519': 0.9992912}}

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
            if m1 in known_corr[m2]:
                dice_corr = known_corr[m2][m1]
            elif m2 in known_corr[m1]:
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
    # models = config.all_pred_dirs

    # Store correlation scores
    corr_dict = get_correlation_dict(models)

    # Save correlation matrix
    save_correlation_plot(corr_dict, models)

if __name__ == "__main__":
    main()
