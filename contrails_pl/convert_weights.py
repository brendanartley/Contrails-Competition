import torch
import torchinfo
import segmentation_models_pytorch as smp
from torchvision import transforms
import argparse

"""
This file takes in saved_weights and reinstantiates them
w/ an SMP Class.

Note: This is done to make to model deserializable w/ out torch lightning modules.
"""

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

class config:
    seg_models = {
        "Unet": Unet,
        "UnetPlusPlus": UnetPlusPlus,
        "MAnet": MAnet,
    }

    transforms_map = {
        "BILINEAR": transforms.InterpolationMode.BILINEAR,
        "BICUBIC": transforms.InterpolationMode.BICUBIC,
        "NEAREST": transforms.InterpolationMode.NEAREST,
        "NEAREST_EXACT": transforms.InterpolationMode.NEAREST_EXACT,
    }
    weights_path = ""
    model_name = ""
    decoder_type = ""
    mask_downsample = ""
    experiment_name = ""

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights_path", type=str, default=config.weights_path, help="Model weights file location (used for validation run).")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Encoder model to use for training.")
    parser.add_argument("--decoder_type", type=str, default=config.decoder_type, help="Model type (seg/timm).")
    parser.add_argument("--mask_downsample", type=str, default=config.mask_downsample, help="Type of downsample used for the mask (only used if img_size >= 256).")
    parser.add_argument("--experiment_name", type=str, default=config.experiment_name, help="Experiment name.")
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)
    return config

def load_and_save(config):
    print("----- Converting to SMP Class -----")

    # Create SMP Class
    smp_model = config.seg_models[config.decoder_type](
            encoder_name = config.model_name,
            in_channels = 3,
            classes = 1,
            inter_type=config.transforms_map[config.mask_downsample],
    )

    # Load weights
    smp_model.load_state_dict(torch.load(config.weights_path, map_location=torch.device('cpu')))

    # Save full model
    torch.save(smp_model, config.weights_path)
    print("----- Converted -----")

def main():
    config = parse_args()
    load_and_save(config)
    return

if __name__ == "__main__":
    main()