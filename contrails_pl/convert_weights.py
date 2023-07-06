import torch
import torchinfo
import segmentation_models_pytorch as smp
from torchvision import transforms

"""
This file takes in saved_weights and reinstantiates them
w/ an SMP Class.

Note: We define the class in the script to avoid pickling errors.

This is done so that we can use `model = torch.load(PATH)` in
another env without using pytorch lightning.
"""

class Unet(smp.Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True, interpolation=transforms.InterpolationMode.NEAREST)])

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True, interpolation=transforms.InterpolationMode.NEAREST)])

    def forward(self, x):
        
        # Original forward function
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        # Add a resize mask to match label size
        resized_masks = self.resize_transform(masks)
        return resized_masks

seg_models = {
    "Unet": Unet,
    "UnetPlusPlus": UnetPlusPlus,
}

def load_and_save(config, experiment_name):
    print("----- Converting to SMP Class -----")

    # Weights path
    weights_path =  "{}{}.pt".format(config.model_save_dir, experiment_name)

    # Load weights into SMP Class
    smp_model = seg_models[config.decoder_type](
            encoder_name = config.model_name,
            in_channels = 3,
            classes = 1,
    )
    smp_model.load_state_dict(torch.load(weights_path))

    # Save full model (need to do this for timm encoders)
    torch.save(smp_model, weights_path)
    print("----- Converted -----")