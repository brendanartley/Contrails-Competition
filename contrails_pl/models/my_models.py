from torchvision import transforms
import segmentation_models_pytorch as smp

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
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True)])

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = transforms.Compose([transforms.Resize(256, antialias=True)])

    def forward(self, x):
        
        # Original forward function
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        # Add a resize mask to match label size
        resized_masks = self.resize_transform(masks)
        return resized_masks