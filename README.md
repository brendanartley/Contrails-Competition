# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

- Post processing in paper (D. Converting binary masks to line segments)
    - They use OpenCVs LineSegmentDetector in the paper..

- Look at AMP w/ Lightning 
    - https://lightning.ai/docs/pytorch/1.5.7/advanced/mixed_precision.html
    - https://github.com/TimDettmers/bitsandbytes

- Train a 512 img model. +2% from 256 -> 384
'maxvit_tiny_tf_512.in1k'?

- Automatic contrail tracking paper. https://amt.copernicus.org/articles/3/1089/2010/amt-3-1089-2010.pdf

- Try and use SOTA models that are top of the imagenetLB
    - https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-real.csv

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

EffnetV2_t
| Resolution | Batch Size | Precision    | GPU     |
|------------|------------|--------------|---------|
| 256        | 32         | 32           | 8.2GB   |
| 384        | 32         | 32           | 17.7GB  |
| 384        | 16         | 32           | 9.8GB   |
| 384        | 32         | 16-mixed     | 9.8GB   |
| 384        | 16         | 16-mixed     | 5.7GB   |
| 384        | 16         | 16-mixed     | 5.5GB   | <- w/ bits and bytes


Timm pre-trained models
`['efficientnetv2_rw_t.ra2_in1k','efficientnetv2_rw_s.ra2_in1k','efficientnetv2_rw_m.agc_in1k', 'maxxvit_rmlp_small_rw_256.sw_in1k']`

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

### Decoder Sizes

| Type       | Params
---------------------------
| Unet         | 31.8 M
| UnetPlusPlus | 33.4 M
| MAnet        | 39.1 M
| FPN          | 30.5 M

MAnet uses less GPU memory than Unet++.

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

CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="Unet" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=15 --fast_dev_run &&
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="UnetPlusPlus" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=15 --fast_dev_run &&
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="MAnet" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=15 --fast_dev_run &&
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="FPN" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=15 --fast_dev_run

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="UnetPlusPlus" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=30
CUDA_VISIBLE_DEVICES=1 python main.py --decoder_type="UnetPlusPlus" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=30
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="Unet" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=15
CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="UnetPlusPlus" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --epochs=15

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="CustomUnet" --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --epochs=30

CUDA_VISIBLE_DEVICES=1 python main.py --decoder_type="Unet" --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --epochs=30

CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="CustomUnet" --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --epochs=15 --no_transform

CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="CustomUnet" --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=1e-4 --batch_size=16 --epochs=15


#### KFold on All Training
CUDA_VISIBLE_DEVICES=3 python main.py --save_weights --save_preds --all_folds --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Train on all Training Data (except comp validation)
CUDA_VISIBLE_DEVICES=2 python main.py --save_weights --train_all --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Competition Validation
CUDA_VISIBLE_DEVICES=2 python main.py --comp_val --save_preds --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --model_weights="/data/bartley/gpu_test/models/segmentation/deft-sun-159.pt" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Check dice threshold (edit model preds dir in code)
CUDA_VISIBLE_DEVICES="" python dice_threshold.py

#### Submit.