# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

- Train larger model on Sunny
- 

- Make Unet Extension
    - Toms winning Unet: https://www.kaggle.com/competitions/hubmap-kidney-segmentation/discussion/238198
- Add deep supervision to non-empty masks?
    - https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/68435
    - Tom used 0.1 here..


    - https://github.com/frgfm/Holocron/blob/f78c6c58c0007e3d892fcaa1f1ff786cdbb5195f/holocron/models/segmentation/unet3p.py#L95
    - https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5

- Might be some good tips here. https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview

- Dynamic threshold. (if the image before has more predicted pixels, decrease threshold,.If there is less, increase?)
    - Look at PCT of predicted pixels when doing dice_threshold.py

- Stage 2. Use frozen backbone to combine three prediction maps

- Use CV2 line detector (or at least try it. Winning solution in scroll competition uses something similar)
- SOTA Medical Segmentation Competition Solutions: https://github.com/JunMa11/SOTA-MedSeg
- Automatic contrail tracking paper. https://amt.copernicus.org/articles/3/1089/2010/amt-3-1089-2010.pdf

### Preprint Notes / Findings

- Best per-pixel threshold is 0.4 (optimize this by writing prediction to disk and comparing score across thresholds)

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
- Deepsupervision

### Sample Workflow

1. Train Model

`CUDA_VISIBLE_DEVICES=2 python main.py --train_all --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --save_weights`

2. Evaluate on Validation

`CUDA_VISIBLE_DEVICES=2 python main.py --comp_val --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --model_weights="/data/bartley/gpu_test/models/segmentation/golden-water-149.pt" --save_preds`

3. Find best threshold

`CUDA_VISIBLE_DEVICES="" python dice_threshold.py`

### Commands

CUDA_VISIBLE_DEVICES="" python dice_threshold.py


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --lr=8e-4 --batch_size=3 --precision="32" --swa=True --no_wandb 

CUDA_VISIBLE_DEVICES=0 python main.py --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=15 --seed=0 --swa=True --seed=0
CUDA_VISIBLE_DEVICES=1 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --swa=True --seed=0
CUDA_VISIBLE_DEVICES=2 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --swa=True --seed=1
CUDA_VISIBLE_DEVICES=3 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --swa=True --seed=2


CUDA_VISIBLE_DEVICES=0 python main.py --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=15 --seed=0

CUDA_VISIBLE_DEVICES=0 python main.py --deep --decoder_type="CustomUnet" --batch_size=64
CUDA_VISIBLE_DEVICES=1 python main.py --deep --decoder_type="CustomUnet"
CUDA_VISIBLE_DEVICES=2 python main.py --deep --decoder_type="CustomUnet"
CUDA_VISIBLE_DEVICES=3 python main.py --deep --decoder_type="CustomUnet" --batch_size=32


CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Contrails-ICRGW/uwchrqjz
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Contrails-ICRGW/uwchrqjz
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Contrails-ICRGW/uwchrqjz
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Contrails-ICRGW/uwchrqjz

CUDA_VISIBLE_DEVICES=0 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/spring-sweep-2.pt"  --save_preds


CUDA_VISIBLE_DEVICES=0 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/light-valley-599.pt" --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=15 --val_check_interval=0.10 --seed=0 --val_check_interval=0.10 --save_preds

CUDA_VISIBLE_DEVICES=1 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/pretty-microwave-583.pt" --save_preds --model_name=tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k --img_size=384

#### Train on all Training Data (Add --save_preds if you dont want to manually run validation after training)
CUDA_VISIBLE_DEVICES=2 python main.py --model_name=mit_b5 --img_size=512 --lr=8e-4 --batch_size=12 --save_preds

#### Save preds (if Trained in FP32 MUST HAVE --precision=32, and for everything --save_preds)
CUDA_VISIBLE_DEVICES=0 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/electric-haze-579.pt" --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --save_preds --precision=32

CUDA_VISIBLE_DEVICES=1 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/spring-valley-429.pt" --model_name="tu-maxxvit_rmlp_small_rw_256.sw_in1k" --decoder_type="UnetPlusPlus" --save_preds --precision=32

#### Competition Validation
CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/olive-gorge-380.pt" --img_size=512 --dice_threshold=-4.1

#### Check dice threshold (edit model preds dir in code)
CUDA_VISIBLE_DEVICES="" python dice_threshold.py

#### Submit.

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=0 --epochs=13
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=1 --epochs=13
CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=2 --epochs=13


### Big Notes

- Editor Config
- Edit callbacks header (remove no_wandb in get_callbacks(), or just run --no_wandb)
- Run w/ no_wandb

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=2e-4 --val_check_interval=0.10 --precision="32" --swa=True --seed=0 --no_wandb

CUDA_VISIBLE_DEVICES=1 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=2e-4 --val_check_interval=0.10 --precision="32" --swa=True --seed=1 --no_wandb

CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/bfg_1.pt" --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=16 --save_preds