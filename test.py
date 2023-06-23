from contrails_pl.models.timm_unet import TimmUnet
import torchinfo
import torch

model = TimmUnet(
    backbone="efficientnet_b0.ra_in1k", 
    in_chans=3,
    num_classes=1,
)

