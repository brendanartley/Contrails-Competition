# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### TODO

- When save_preds=True, preds are written on every validation step.. (can I write preds at the end of training?)

### Ideas


Look at using `in_22k` versions as they are pre-trained on more data (test this)

models = [
    'caformer_b36.sail_in1',
    'convformer_b36.sail_in1k',
    'convnextv2_tiny.fcmae_ft_in1k (2023 model)',
    'efficientformerv2_l.snap_dist_in1k',
    'maxvit_small_tf_224.in1k (lots of models here)',
    'poolformer_m36.sail_in1k',
    'poolformerv2_m36.sail_in1k',
    'swin_small_patch4_window7_224.ms_in1k',
    'swinv2_small_window8_256.ms_in1k'
]

big_models = [
    'tf_efficientnetv2_xl.in21k',
    'maxvit_base_tf_224.in21k'
]


- Fix KFold w/ Sweeps Error
- Read into the multi-frame model approach in the paper (Resnet3D?)

- Write a script that computes score / correlation of model predictions, and assign weights for weighted ensemble?
    - Make this portable to other competitions as well.

### Preprint Notes / Findings

- Using 3 frames before yields the best results for resnet3D
    - 0B-0A = 71.4
    - 1B-0A = 71.7
    - 3B-0A = 72.7
    - 4B-0A = 72.0

- Best per-pixel threshold is 0.4 (optimize this by writing prediction to disk and comparing score across thresholds)

### Models

Timm pre-trained models
`['efficientnetv2_rw_t.ra2_in1k','efficientnetv2_rw_s.ra2_in1k','efficientnetv2_rw_m.agc_in1k']`

Segmentation Models
`["timm-regnetx_016", "timm-regnetx_032", "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]`

### EfficientNet Sizes

| Model                | Img Size | Params |
|----------------------|----------|--------|
| EfficientNetB0       | 224      | 4M     |
| EfficientNetB1       | 240      | 6M     |
| EfficientNetB2       | 260      | 7M     |
| EfficientNetB3       | 300      | 10M    |
| EfficientNetB4       | 380      | 17M    |
| EfficientNetB5       | 456      | 28M    |
| EfficientNetB6       | 528      | 40M    |
| EfficientNetB7       | 600      | 63M    |
| efficientnetv2_rw_t  | -        | 13M    |
| efficientnetv2_rw_s  | -        | 23M    |
| efficientnetv2_rw_m  | -        | 53M    |
| efficientnetv2_s     | -        | 23M    |
| efficientnetv2_m     | -        | 53M    |
| mit_b1               | -        | 13M    |
| mit_b2               | -        | 24M    |
| mit_b3               | -        | 44M    |
| mit_b4               | -        | 60M    |
| mit_b5               | -        | 81M    |



### Links

Segmentation Models Documentation: https://segmentation-modelspytorch.readthedocs.io/en/latest/


### Sample Workflow

1. Train Model

`CUDA_VISIBLE_DEVICES=2 python main.py --train_all --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --save_weights`

2. Evaluate on Validation

`CUDA_VISIBLE_DEVICES=2 python main.py --comp_val --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --model_weights="/data/bartley/gpu_test/models/segmentation/golden-water-149.pt" --save_preds`

3. Find best threshold

`CUDA_VISIBLE_DEVICES="" python dice_threshold.py`

### Commands

CUDA_VISIBLE_DEVICES="" python dice_threshold.py

CUDA_VISIBLE_DEVICES=2 python main_copy.py --save_preds --no_wandb --model_type="seg" --model_name="mit_b0"

CUDA_VISIBLE_DEVICES="" python contrails_pl/models/my_models.py

CUDA_VISIBLE_DEVICES=3 python main.py --save_weights --save_preds --all_folds --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

CUDA_VISIBLE_DEVICES=2 python main.py --all_folds

"bilinear", "nearest", "bicubic", "area", "nearest-exact"

CUDA_VISIBLE_DEVICES=3 python main.py --all_folds --interpolate="bilinear" && CUDA_VISIBLE_DEVICES=3 python main.py --all_folds --interpolate="nearest" && CUDA_VISIBLE_DEVICES=3 python main.py --all_folds --interpolate="bicubic" && CUDA_VISIBLE_DEVICES=3 python main.py --all_folds --interpolate="area" && CUDA_VISIBLE_DEVICES=3 python main.py --all_folds --interpolate="nearest-exact" && CUDA_VISIBLE_DEVICES=3 python main.py --all_folds --interpolate="nearest"

CUDA_VISIBLE_DEVICES=3 python main.py --fast_dev_run --interpolate="linear"
CUDA_VISIBLE_DEVICES=3 python main.py --fast_dev_run --interpolate="bilinear"
CUDA_VISIBLE_DEVICES=3 python main.py --fast_dev_run --interpolate="area"

CUDA_VISIBLE_DEVICES=3 python main.py --fast_dev_run --interpolate="nearest-exact"


#### Check dice threshold
CUDA_VISIBLE_DEVICES="" python dice_threshold.py

#### KFold on All Training
CUDA_VISIBLE_DEVICES=3 python main.py --save_weights --save_preds --all_folds --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Train on all Training Data (except comp validation)
CUDA_VISIBLE_DEVICES=2 python main.py --save_weights --train_all --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Competition Validation
CUDA_VISIBLE_DEVICES=2 python main.py --comp_val --save_preds --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --model_weights="/data/bartley/gpu_test/models/segmentation/deft-sun-159.pt" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"