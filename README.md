# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas / TODO

- DepthwiseConv2D layer using the two neigbhouring frames
- Create different color maps. Ash false-color might not be optimal.. (should be able to automate this)
    - use 1000k img subset, and V2_tiny. Should be quick enough.
- Write predictions to a file and look at correlation between model predictions.
- Fix KFold w/ Sweeps Error
- Set up auto color generation process (should be a dataloader not preprocessing)

Other potential color schemes
- NGFS Microphysics RGB (NOAA)


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


### Commands

CUDA_VISIBLE_DEVICES=0 python main.py --model_type="timm" --model_name="efficientnetv2_rw_s.ra2_in1k" --all_folds=True && \

CUDA_VISIBLE_DEVICES=2 python main.py --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --all_folds=True && \
CUDA_VISIBLE_DEVICES=2 python main.py --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --all_folds=True

CUDA_VISIBLE_DEVICES=3 python main.py --model_type="seg" --model_name="mit_b4" --all_folds=True

CUDA_VISIBLE_DEVICES=3 python main.py --model_type="seg" --model_name="mit_b1" --all_folds=True && \
CUDA_VISIBLE_DEVICES=3 python main.py --model_type="seg" --model_name="timm-regnetx_032" --all_folds=True && \

CUDA_VISIBLE_DEVICES=1 python main.py --model_type="timm" --model_name="convnextv2_tiny.fcmae_ft_in1k" --fast_dev_run

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/Contrails-ICRGW-RGB/vqyx2gfk
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Contrails-ICRGW-RGB/vqyx2gfk
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Contrails-ICRGW-RGB/vqyx2gfk
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Contrails-ICRGW-RGB/vqyx2gfk

CUDA_VISIBLE_DEVICES=0 python main.py --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --data_dir="/data/bartley/gpu_test/my-raw-contrails-data/"
CUDA_VISIBLE_DEVICES=1 python main.py --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --data_dir="/data/bartley/gpu_test/my-ash-contrails-data/"
CUDA_VISIBLE_DEVICES=2 python main.py --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --data_dir="/data/bartley/gpu_test/my-raw-contrails-data/" --fast_dev_run

