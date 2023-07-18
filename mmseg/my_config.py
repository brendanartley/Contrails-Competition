
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(512, 512),
    )
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# # model settings
# norm_cfg = dict(type='BN', requires_grad=True)  # Segmentation usually uses SyncBN
# model = dict(
#     type='EncoderDecoder',  # Name of segmentor
#     data_preprocessor=None,
#     pretrained='open-mmlab://resnet50_v1c',  # The ImageNet pretrained backbone to be loaded
#     backbone=dict(
#         type='ResNetV1c',  # The type of backbone. Please refer to mmseg/models/backbones/resnet.py for details.
#         depth=50,  # Depth of backbone. Normally 50, 101 are used.
#         num_stages=4,  # Number of stages of backbone.
#         out_indices=(0, 1, 2, 3),  # The index of output feature maps produced in each stages.
#         dilations=(1, 1, 2, 4),  # The dilation rate of each layer.
#         strides=(1, 2, 1, 1),  # The stride of each layer.
#         norm_cfg=norm_cfg,  # The configuration of norm layer.
#         norm_eval=False,  # Whether to freeze the statistics in BN
#         style='pytorch',  # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 'caffe' means stride 2 layers are in 1x1 convs.
#         contract_dilation=True),  # When dilation > 1, whether contract first layer of dilation.
#     decode_head=dict(
#         type='PSPHead',  # Type of decode head. Please refer to mmseg/models/decode_heads for available options.
#         in_channels=2048,  # Input channel of decode head.
#         in_index=3,  # The index of feature map to select.
#         channels=512,  # The intermediate channels of decode head.
#         pool_scales=(1, 2, 3, 6),  # The avg pooling scales of PSPHead. Please refer to paper for details.
#         dropout_ratio=0.1,  # The dropout ratio before final classification layer.
#         num_classes=1,  # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
#         norm_cfg=norm_cfg,  # The configuration of norm layer.
#         align_corners=False,  # The align_corners argument for resize in decoding.
#         loss_decode=dict(  # Config of loss function for the decode_head.
#             type='DiceLoss',  # Type of loss used for segmentation.
#             loss_weight=1.0)),  # Loss weight of decode_head.
#     # model training and testing settings
#     train_cfg=dict(),  # train_cfg is just a place holder for now.
#     test_cfg=dict(mode='whole'))  # The test mode, options are 'whole' and 'slide'. 'whole': whole image fully-convolutional test. 'slide': sliding crop window on the image.