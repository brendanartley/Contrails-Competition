from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model

"""
Modified Unet implementation.
Source: https://gist.github.com/rwightman/f8b24f4e6f5504aba03e999e02460d31
"""

class CustomUnet(nn.Module):
    """Unet is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        center: if ``True`` add ``Conv2dReLU`` block on encoder head
    NOTE: This is based off an old version of Unet in https://github.com/qubvel/segmentation_models.pytorch

    ex.

    Find all available encoders w/ `timm.list_models(pretrained=True)`

    ```
    model = CustomUnet(
        backbone="tf_efficientnetv2_s.in1k", 
        in_channels=3,
        classes=1,
    )
    ```

    """

    def __init__(
            self,
            encoder_name='resnet50',
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1,
            center=False,
            norm_layer=nn.BatchNorm2d,
            inter_type="nearest",
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}

        # # ----- FEATURE ALIGNMENT FOR SPECIFIC MODELS -----
        # if backbone == "swinv2_small_window8_256.ms_in1k":
        #     decoder_channels = (16, 32, 64, 128)

        # Using segmentation_models naming convention
        if encoder_name.startswith("tu-"):
            encoder_name = encoder_name[3:]
            
        encoder = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=in_channels,
            pretrained=True, **backbone_kwargs)
        
        # TESTING
        encoder2 = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=in_channels,
            pretrained=True, **backbone_kwargs)
        
        # Adding temporal convs to combine encoders
        s = encoder2.feature_info.channels()

        self.c1 = nn.Conv2d(s[0]*2, int(s[0]), kernel_size=(3,3), padding=1)
        self.c2 = nn.Conv2d(s[1]*2, int(s[1]), kernel_size=(3,3), padding=1)
        self.c3 = nn.Conv2d(s[2]*2, int(s[2]), kernel_size=(3,3), padding=1)
        self.c4 = nn.Conv2d(s[3]*2, int(s[3]), kernel_size=(3,3), padding=1)
        self.c5 = nn.Conv2d(s[4]*2, int(s[4]), kernel_size=(3,3), padding=1)
        # End Testing
        
        encoder_channels = encoder.feature_info.channels()[::-1]
        self.encoder = encoder
        self.encoder2 = encoder2

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=classes,
            norm_layer=norm_layer,
            center=center,
            inter_type=inter_type,
        )

    def forward(self, x: torch.Tensor):
        x1 = self.encoder(x)
        x2 = self.encoder2(x)

        x3 = self.c1(torch.cat([x1[0], x2[0]], dim=1))
        x4 = self.c2(torch.cat([x1[1], x2[1]], dim=1))
        x5 = self.c3(torch.cat([x1[2], x2[2]], dim=1))
        x6 = self.c4(torch.cat([x1[3], x2[3]], dim=1))
        x7 = self.c5(torch.cat([x1[4], x2[4]], dim=1))

        x3 = [x3, x4, x5, x6, x7]
        x3.reverse()  # torchscript doesn't work with [::-1]
        x = self.decoder(x3)
        # x.reverse()  # torchscript doesn't work with [::-1]
        # x = self.decoder(x)
        return x


class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels, eps=1e-05)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inter_type="nearest"):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        self.inter_type = inter_type
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.inter_type)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        super().__init__(conv2d)    

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=False,
            inter_type="nearest",
    ):
        super().__init__()
        self.inter_type = inter_type

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(channels, channels, scale_factor=1.0, norm_layer=norm_layer, inter_type=inter_type)
        else:
            self.center = nn.Identity()

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer, inter_type=inter_type))

        # self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))
        self.final_conv = SegmentationHead(out_channels[-1], final_channels)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)

        # Iterate decoder blocks
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)

        # Reshape to mask shape (256)
        if x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear")
        return x