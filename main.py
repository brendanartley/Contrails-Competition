from contrails_pl.train import train
import argparse
from types import SimpleNamespace

# default configuration parameters
config = SimpleNamespace(
    project = "Contrails-ICRGW",
    # data_dir = "/data/bartley/gpu_test/contrails-images-ash-color/",    
    data_dir = "/data/bartley/gpu_test/my-ash-contrails-data/",
    model_save_dir = "/data/bartley/gpu_test/models/segmentation/",
    torch_cache = "/data/bartley/gpu_test/TORCH_CACHE/",
    model_name = "efficientnetv2_rw_t.ra2_in1k",
    model_type = "timm",
    save_model = False,
    batch_size = 32,
    epochs = 5,
    val_fold = 0,
    num_folds = 5,
    all_folds = False,
    lr = 2e-4,
    lr_min = 1e-8,
    num_cycles = 5,
    scheduler = "CosineAnnealingLR",
    # -- RGB Recipe Tests --
    r1 = -4,
    r2 = 2,
    g1 = -4,
    g2 = 5,
    b1 = 243,
    b2 = 303,
    gamma1 = 1.0,
    gamma2 = 1.0,
    gamma3 = 1.0,
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
    # ----- RGB tests -----
    parser.add_argument("--r1", type=float, default=config.r1, help=".")
    parser.add_argument("--r2", type=float, default=config.r2, help=".")
    parser.add_argument("--g1", type=float, default=config.g1, help=".")
    parser.add_argument("--g2", type=float, default=config.g2, help=".")
    parser.add_argument("--b1", type=float, default=config.b1, help=".")
    parser.add_argument("--b2", type=float, default=config.b2, help=".")
    parser.add_argument("--gamma1", type=float, default=config.gamma1, help=".")
    parser.add_argument("--gamma2", type=float, default=config.gamma2, help=".")
    parser.add_argument("--gamma3", type=float, default=config.gamma3, help=".")
    # ----- End RGB tests -----
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Learning rate scheduler for the model to use.")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Encoder model to use for training.")
    parser.add_argument("--model_type", type=str, default=config.model_type, help="Model type (seg/timm).")
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument('--train_all', action='store_true', help='Indicator wether to train on all the data.')
    parser.add_argument('--fast_dev_run', action='store_true', help='Check PL modules are set up correctly.')
    # parser.add_argument('--all_folds', action='store_true', help='Do full K-Fold validation.')
    parser.add_argument('--all_folds', type=bool, help='Do full K-Fold validation.')
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Num of batches to overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true', help='Wether to log with weights and biases.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed for reproducability.")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Num data points per batch.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches, help="Number of steps before each optimizer step.")
    parser.add_argument("--log_every_n_steps", type=int, default=config.log_every_n_steps, help="Logs every N steps.")
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