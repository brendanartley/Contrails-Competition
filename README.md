# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

- Save Model By Default
- Save Model After Every Epoch (in case training crashes)


- Efficient UnetPlusPlus: https://github.com/jlcsilva/EfficientUNetPlusPlus
- SOTA Medical Segmentation Competition Solutions: https://github.com/JunMa11/SOTA-MedSeg
- How come bigger images worked before? Check broken code and see why.

Different input img sizes works with MIT_B's, EffnetV2's
- Its because the shapes are aligned in the decoder? How could I do this for maxxvit?

- See why UnetPlusPlus is so slow w/ 384 images
    - Computation complexity is too high

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

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-efficientnetv2_rw_t.ra2_in1k" --fast_dev_run
CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-efficientnetv2_rw_t.ra2_in1k" --fast_dev_run
CUDA_VISIBLE_DEVICES=2 python main.py --model_name="tu-efficientnetv2_rw_m.agc_in1k" --img_size=512 --lr=1e-4 --batch_size=16 --save_model
CUDA_VISIBLE_DEVICES=3 python main.py --model_name="mit_b4" --img_size=512 --lr=1e-4 --batch_size=16 --save_model
CUDA_VISIBLE_DEVICES=2 python main.py --model_name="mit_b5"


`['tu-efficientnetv2_rw_t.ra2_in1k','tu-efficientnetv2_rw_s.ra2_in1k','tu-efficientnetv2_rw_m.agc_in1k', 'maxxvit_rmlp_small_rw_256.sw_in1k', "mit_b5", "mit_b4"]`


CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Contrails-ICRGW/8jbtyog0
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Contrails-ICRGW/8jbtyog0
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Contrails-ICRGW/8jbtyog0
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Contrails-ICRGW/8jbtyog0

brendanartley/Contrails-ICRGW/ccqo954v

#### KFold on All Training
CUDA_VISIBLE_DEVICES=3 python main.py --save_weights --save_preds --all_folds --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Train on all Training Data (MUST HAVE -> --train_all --save_preds --save_weights --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/")
CUDA_VISIBLE_DEVICES=2 python main.py --save_weights --train_all --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Competition Validation
CUDA_VISIBLE_DEVICES=3 python main.py --comp_val --save_preds --decoder_type="UnetPlusPlus" --model_name="tu-maxvit_rmlp_tiny_rw_256.sw_in1k" --model_weights="/data/bartley/gpu_test/models/segmentation/volcanic-morning-341.pt" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/"

#### Check dice threshold (edit model preds dir in code)
CUDA_VISIBLE_DEVICES="" python dice_threshold.py

#### Submit.

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=0 --epochs=13
CUDA_VISIBLE_DEVICES=2 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=1 --epochs=13
CUDA_VISIBLE_DEVICES=3 python main.py --decoder_type="UnetPlusPlus" --data_dir="/data/bartley/gpu_test/contrails-images-ash-color/" --save_preds --save_weights --seed=2 --epochs=13