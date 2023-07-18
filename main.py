from pl_scripts.train import train
from pl_scripts.validate import validate

import argparse
from types import SimpleNamespace
import json, os

# Load environment variables (stored in config.json file)
with open('./config.json') as f:
    data = json.load(f)
DATA_DIR = data["DATA_DIR"]

# default configuration parameters
config = SimpleNamespace(
    project = "Contrails-ICRGW",
    hf_cache = os.path.join(DATA_DIR, "bartley/gpu_test/HF_CACHE/"),
    torch_cache = os.path.join(DATA_DIR, "bartley/gpu_test/TORCH_CACHE/"),
    data_dir = os.path.join(DATA_DIR, "bartley/gpu_test/ct_numpy_data/"),
    model_save_dir = os.path.join(DATA_DIR, "bartley/gpu_test/models/segmentation/"),
    preds_dir = os.path.join(DATA_DIR, "bartley/gpu_test/preds/"),
    model_name = "tu-maxvit_rmlp_tiny_rw_256.sw_in1k",
    model_weights = None, # Used for validation run
    save_model=True,
    decoder_type = "Unet",
    img_size = 256,
    rand_scale_min = 0.95,
    rand_scale_prob = 0.5,
    batch_size = 32,
    epochs = 11,
    lr = 2e-4,
    lr_min = 1e-5,
    num_cycles = 5,
    scheduler = "CosineAnnealingLR",
    dice_threshold = 0.5,
    loss = "Dice",
    smooth = 0.20,
    mask_downsample="BILINEAR",
    swa = False,
    swa_epochs = 3,
    # -- Trainer Config --
    accelerator = "gpu",
    fast_dev_run = False,
    overfit_batches = 0,
    strategy = "auto",
    precision = "16-mixed", # No accuracy should be lost w/ 16-mixed
    accumulate_grad_batches = 1,
    val_check_interval = 0.10,
    num_workers = 2,
    seed = 0,
    verbose = 2,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--swa', action='store_true', help='Stochastic weight average (starts 1/2 way through training).')
    parser.add_argument('--swa', type=bool, default=config.swa, help='Stochastic weight average (starts 1/2 way through training).')
    parser.add_argument("--swa_epochs", type=int, default=config.swa_epochs, help="Number of SWA epochs to do.")
    parser.add_argument("--strategy", type=str, default=config.strategy, help="Training strategy (auto, ddp).")
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Learning rate scheduler for the model to use.")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Encoder model to use for training.")
    parser.add_argument("--decoder_type", type=str, default=config.decoder_type, help="Model type (seg/timm).")
    parser.add_argument("--model_weights", type=str, default=config.model_weights, help="Model weights file location (used for validation run).")
    parser.add_argument("--data_dir", type=str, default=config.data_dir, help="Data directory path.")
    parser.add_argument('--fast_dev_run', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument('--save_preds', action='store_true', help='Check PL modules are set up correctly.')
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Num of batches to overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true', help='Wether to log with weights and biases.')
    parser.add_argument('--transform', action='store_false', help='Wether to apply transformations to training data.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed for reproducability.")
    parser.add_argument("--precision", type=str, default=config.precision, help="Precision to use (AMP).")
    parser.add_argument("--img_size", type=int, default=config.img_size, help="Interpolates to an image size (orig = 256x256).")
    parser.add_argument("--rand_scale_min", type=float, default=config.rand_scale_min, help="Lower bound of random crop augmentation.")
    parser.add_argument("--rand_scale_prob", type=float, default=config.rand_scale_prob, help="Pct chance of random crop augmentation.")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Num data points per batch.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches, help="Number of steps before each optimizer step.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Starting learning rate for the model.")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Lowest allowed learning rate for the model.")
    parser.add_argument("--num_cycles", type=int, default=config.num_cycles, help="Number of cycles for the cyclical cosine annealing LR.")
    parser.add_argument("--val_check_interval", type=float, default=config.val_check_interval, help="Number of batches between validation checks.")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of CPU cores to use.")
    parser.add_argument("--loss", type=str, default=config.loss, help="Loss function to use.")
    parser.add_argument("--smooth", type=float, default=config.smooth, help="Smoothing factor on Dice Loss function.")
    parser.add_argument("--dice_threshold", type=float, default=config.dice_threshold, help="Threshold for the GlobalDiceCoefficient.")
    parser.add_argument("--mask_downsample", type=str, default=config.mask_downsample, help="Type of downsample used for the mask (only used if img_size >= 256).")
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

# (START) THIS IS NEED TO LOAD MODELS FOR VALIDATION..
from torchvision import transforms
import segmentation_models_pytorch as smp

class Unet(smp.Unet):
    def __init__(self, inter_type=transforms.InterpolationMode.NEAREST, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True, interpolation=inter_type)])

    def forward(self, x):
        
        # Original forward function
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        # Add a resize mask to match label size
        resized_masks = self.resize_transform(masks)

        return resized_masks
    
class UnetPlusPlus(smp.UnetPlusPlus):
    def __init__(self, inter_type=transforms.InterpolationMode.NEAREST, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True, interpolation=inter_type)])

    def forward(self, x):
        
        # Original forward function
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        # Add a resize mask to match label size
        resized_masks = self.resize_transform(masks)

        return resized_masks

class MAnet(smp.MAnet):
    def __init__(self, inter_type=transforms.InterpolationMode.NEAREST, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True, interpolation=inter_type)])

    def forward(self, x):
        
        # Original forward function
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        # Add a resize mask to match label size
        resized_masks = self.resize_transform(masks)

        return resized_masks

# (END) THIS IS NEED TO LOAD MODELS FOR VALIDATION..

if __name__ == "__main__":
    config = parse_args()
    main(config)