from contrails_segmentation.train import train
import argparse
from types import SimpleNamespace

# default configuration parameters
config = SimpleNamespace(
    data_dir = "/data/bartley/gpu_test/contrails-images-ash-color/",    
    model_save_dir = "/data/bartley/gpu_test/models/segmentation/",
    model_name = "efficientnet-b0",
    project = "Contrails-ICRGW",
    save_model = False,
    batch_size = 32,
    epochs = 1,
    val_fold = 0,
    num_folds = 5,
    lr = 2e-4,
    lr_min = 1e-8,
    num_cycles = 5,
    scheduler = "CosineAnnealingLR",
    # -- Trainer Config --
    accelerator = "gpu",
    fast_dev_run = False,
    overfit_batches = 0,
    devices = 1,
    precision = 32,
    log_every_n_steps = 10,
    accumulate_grad_batches = 1,
    val_check_interval = 0.10,
    num_workers = 2,
    seed = 0,
    verbose = 2,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Learning rate scheduler for the model to use.")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Encoder model to use for training.")
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument('--train_all', action='store_true', help='Indicator wether to train on all the data.')
    parser.add_argument('--fast_dev_run', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument('--all_folds', action='store_true', help='Do full K-Fold validation.')
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Num of batches to overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true', help='Wether to log with weights and biases.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed for reproducability.")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Num data points per batch.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches, help="Number of steps before each optimizer step.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Starting learning rate for the model.")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Lowest allowed learning rate for the model.")
    parser.add_argument("--num_cycles", type=int, default=config.num_cycles, help="Number of cycles for the cyclical cosine annealing LR.")
    parser.add_argument("--val_check_interval", type=float, default=config.val_check_interval, help="Number of batches between validation checks.")
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config

def main(config):
    module = train(config)
    pass

if __name__ == "__main__":
    config = parse_args()
    main(config)