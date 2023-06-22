from contrails_pl.models.timm_unet import TimmUnet
import torchinfo

model = TimmUnet(
    backbone="efficientnet_b0.ra_in1k", 
    in_chans=3,
    num_classes=1,
    
)
torchinfo.summary(model, input_size=(1, 3, 256, 256))