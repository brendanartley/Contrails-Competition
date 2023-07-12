# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

Can a better interpolation method for upsampling improve model?

- Try MSNet? Will have to edit, but could be useful
    - https://github.com/taochx/MSNet/blob/main/msnet.py

Add model checkpoint callback to save best weights. (this will be useful for saving the best weights according to val_loss)
    - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html

- Use CV2 line detector (or at least try it. Winning solution in scroll competition uses something similar)
- SOTA Medical Segmentation Competition Solutions: https://github.com/JunMa11/SOTA-MedSeg

- Automatic contrail tracking paper. https://amt.copernicus.org/articles/3/1089/2010/amt-3-1089-2010.pdf

### Preprint Notes / Findings

- Best per-pixel threshold is 0.4 (optimize this by writing prediction to disk and comparing score across thresholds)

### OpenMMLab Notes

conda activate /data/bartley/gpu_test/openmmlab

Downloading a config + checkpoint
```
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```

### GPU Efficiency Notes

EffnetV2_t
| Resolution | Batch Size | Precision    | GPU     |
|------------|------------|--------------|---------|
| 256        | 32         | 32           | 8.2GB   |
| 384        | 32         | 32           | 17.7GB  |
| 384        | 16         | 32           | 9.8GB   |
| 384        | 32         | 16-mixed     | 9.8GB   |
| 384        | 16         | 16-mixed     | 5.7GB   |
| 384        | 16         | 16-mixed     | 5.5GB   | <- w/ bits and bytes


### Attempted

- Efficientnetv2, DPT
- Losses: Tversky, LogCoshDice
- Downsampling Interpolation Methods
- Removing Islands
- Openmmlab (upernet, swin)

### Sample Workflow

1. Train Model

`CUDA_VISIBLE_DEVICES=2 python main.py --train_all --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --save_weights`

2. Evaluate on Validation

`CUDA_VISIBLE_DEVICES=2 python main.py --comp_val --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --model_weights="/data/bartley/gpu_test/models/segmentation/golden-water-149.pt" --save_preds`

3. Find best threshold

`CUDA_VISIBLE_DEVICES="" python dice_threshold.py`

### Commands

os.system("CUDA_VISIBLE_DEVICES python ./contrails_pl/convert_weights.py")

CUDA_VISIBLE_DEVICES="" python dice_threshold.py

CUDA_VISIBLE_DEVICES=1 python main.py --no_transform --val_check_interval=0.10
CUDA_VISIBLE_DEVICES=2 python main.py --val_check_interval=0.10

CUDA_VISIBLE_DEVICES=2 python main.py --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=15 --val_check_interval=0.10
CUDA_VISIBLE_DEVICES=3 python main.py --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=15 --val_check_interval=0.10
CUDA_VISIBLE_DEVICES=2 python main.py --model_name="nvidia/segformer-b3-finetuned-ade-512-512" --decoder_type="hf" --img_size=512 --lr=1e-4 --batch_size=16 --val_check_interval=0.10 --no_wandb


CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="Upernet" --model_name="openmmlab/upernet-convnext-tiny" --val_check_interval=0.10

CUDA_VISIBLE_DEVICES=1 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/expert-donkey-570.pt" --save_preds
CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/vibrant-night-521.pt" --save_preds --model_name=tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k --img_size=384
CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/zany-dust-522.pt" --save_preds --model_name=tu-maxxvit_rmlp_small_rw_256.sw_in1k --decoder_type="UnetPlusPlus"

#### Train on all Training Data (Add --save_preds if you dont want to manually run validation after training)
CUDA_VISIBLE_DEVICES=2 python main.py --model_name=mit_b5 --img_size=512 --lr=8e-4 --batch_size=12 --save_preds


#### Save preds (if forgot above)
CUDA_VISIBLE_DEVICES=0 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/lyric-fire-428.pt" --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --save_preds

CUDA_VISIBLE_DEVICES=1 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/spring-valley-429.pt" --model_name="tu-maxxvit_rmlp_small_rw_256.sw_in1k" --decoder_type="UnetPlusPlus" --save_preds

#### Competition Validation
CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/olive-gorge-380.pt" --img_size=512 --dice_threshold=-4.1

#### Check dice threshold (edit model preds dir in code)
CUDA_VISIBLE_DEVICES="" python dice_threshold.py

#### Submit.

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=0 --epochs=13
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=1 --epochs=13
CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=2 --epochs=13