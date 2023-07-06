# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

- Might be able to use some stuff here: https://github.com/open-mmlab

- Try some different loss functions?
    - https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py

Add model checkpoint callback to save best weights.
    - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html


- Use CV2 line detector (or at least try it. Winning solution in scroll competition uses something similar)

- Efficient UnetPlusPlus: https://github.com/jlcsilva/EfficientUNetPlusPlus
- SOTA Medical Segmentation Competition Solutions: https://github.com/JunMa11/SOTA-MedSeg

- See why UnetPlusPlus is so slow w/ 384 images
    - Computation complexity is too high?

- Automatic contrail tracking paper. https://amt.copernicus.org/articles/3/1089/2010/amt-3-1089-2010.pdf

### Preprint Notes / Findings

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


Models to try
`['tu-efficientnetv2_rw_t.ra2_in1k','tu-efficientnetv2_rw_s.ra2_in1k','tu-efficientnetv2_rw_m.agc_in1k', 'maxxvit_rmlp_small_rw_256.sw_in1k', "mit_b0", "mit_b1", "mit_b2", "mit_b3"]`


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

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Contrails-ICRGW/eyzk70zy
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Contrails-ICRGW/eyzk70zy

CUDA_VISIBLE_DEVICES=0 python main.py --loss="Tversky" --tversky_pair="0.6_0.4"
CUDA_VISIBLE_DEVICES=1 python main.py --loss="Tversky" --tversky_pair="0.4_0.6"
CUDA_VISIBLE_DEVICES=1 python main.py --smooth=0.2
CUDA_VISIBLE_DEVICES=2 python main.py --smooth=0.3
CUDA_VISIBLE_DEVICES=3 python main.py --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=16

CUDA_VISIBLE_DEVICES=2 python main.py --model_name="tu-maxxvit_rmlp_small_rw_256.sw_in1k" --decoder_type="UnetPlusPlus"
CUDA_VISIBLE_DEVICES=3 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=384 --lr=7e-5 --batch_size=16

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Contrails-ICRGW/qf8h8sb9
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Contrails-ICRGW/qf8h8sb9
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Contrails-ICRGW/qf8h8sb9
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Contrails-ICRGW/qf8h8sb9

brendanartley/Contrails-ICRGW/ccqo954

#### Train on all Training Data (Add --save_preds if you dont want to manually run validation after training)
CUDA_VISIBLE_DEVICES=2 python main.py --model_name=mit_b4 --img_size=512 --lr=1e-4 --batch_size=16 --save_preds

#### Competition Validation
CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/olive-gorge-380.pt" --img_size=512 --dice_threshold=-4.1

#### Check dice threshold (edit model preds dir in code)
CUDA_VISIBLE_DEVICES="" python dice_threshold.py

#### Submit.

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=0 --epochs=13
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=1 --epochs=13
CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=2 --epochs=13