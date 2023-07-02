from contrails_pl.train import train
from contrails_pl.validate import validate

import argparse
from types import SimpleNamespace
import json

# Load environment variables (stored in config.json file)
with open('./config.json') as f:
    data = json.load(f)
DATA_DIR = data["DATA_DIR"]

# default configuration parameters
config = SimpleNamespace(
    project = "Contrails-ICRGW",
    # data_dir = DATA_DIR + "/data/bartley/gpu_test/contrails-images-ash-color/",    
    data_dir = DATA_DIR + "bartley/gpu_test/my-ash-contrails-data/",
    model_save_dir = DATA_DIR + "bartley/gpu_test/models/segmentation/",
    preds_dir = DATA_DIR + "bartley/gpu_test/preds/",
    torch_cache = DATA_DIR + "bartley/gpu_test/TORCH_CACHE/",
    model_name = "efficientnetv2_rw_t.ra2_in1k",
    model_weights = None, # Used for validation run
    decoder_type = "Unet",
    img_size = 256,
    rand_scale_min = 0.9,
    batch_size = 32,
    epochs = 5,
    val_fold = 0,
    num_folds = 5,
    all_folds = False,
    lr = 2e-4,
    lr_min = 1e-5,
    num_cycles = 5,
    scheduler = "CosineAnnealingLR",
    interpolate = "nearest",
    # -- Trainer Config --
    accelerator = "gpu",
    fast_dev_run = False,
    overfit_batches = 0,
    devices = 1,
    precision = "16-mixed", # No accuracy should be lost w/ 16-mixed
    accumulate_grad_batches = 1,
    val_check_interval = None,
    num_workers = 2,
    seed = 0,
    verbose = 2,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ----- End RGB tests -----
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Learning rate scheduler for the model to use.")
    parser.add_argument("--interpolate", type=str, default=config.interpolate, help="Interpolation method for the decoder.")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Encoder model to use for training.")
    parser.add_argument("--decoder_type", type=str, default=config.decoder_type, help="Model type (seg/timm).")
    parser.add_argument("--model_weights", type=str, default=config.model_weights, help="Model weights file location (used for validation run).")
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument('--train_all', action='store_true', help='Indicator wether to train on all the data.')
    parser.add_argument('--comp_val', action='store_true', help='Indicator wether to train on all the data.')
    parser.add_argument('--save_weights', action='store_true', help='Indicator wether to save model weights.')
    parser.add_argument('--fast_dev_run', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument('--save_preds', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument("--val_fold", type=int, default=config.val_fold, help="OOF validation fold.")
    parser.add_argument('--all_folds', action='store_true', help='Do full K-Fold validation.')
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Num of batches to overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true', help='Wether to log with weights and biases.')
    parser.add_argument('--no_transform', action='store_true', help='Wether to apply transformations to training data.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed for reproducability.")
    parser.add_argument("--precision", type=str, default=config.precision, help="Precision to use (AMP).")
    parser.add_argument("--img_size", type=int, default=config.img_size, help="Interpolates to an image size (orig = 256x256).")
    parser.add_argument("--rand_scale_min", type=float, default=config.rand_scale_min, help="Lower bound of random crop augmentation.")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Num data points per batch.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches, help="Number of steps before each optimizer step.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Starting learning rate for the model.")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Lowest allowed learning rate for the model.")
    parser.add_argument("--num_cycles", type=int, default=config.num_cycles, help="Number of cycles for the cyclical cosine annealing LR.")
    parser.add_argument("--val_check_interval", type=float, default=config.val_check_interval, help="Number of batches between validation checks.")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of CPU cores to use.")
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config

def main(config):

    # Train Run
    if config.model_weights == None:
        module = train(config)
    # Validation Run
    else:
        module = validate(config)
    pass

if __name__ == "__main__":
    config = parse_args()
    main(config)