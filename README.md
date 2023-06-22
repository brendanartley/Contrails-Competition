# Notes

Code for ICRGW.

EfficientNetB6 is the max size for efficientnet models (without fancy tricks).

Test w/ only contrails data, and train final model on all the data.

### Ideas

- DepthwiseConv2D layer using the the two neigbhouring frames
- Look at RegNext
- Mix Vision Transformer 
    - B1 == effnetb3-4
    - B2 == effnetb5
    - B3 == effnetb6
    - B4 == effnetb7
    - B5 == effnetb8

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

CUDA_VISIBLE_DEVICES=0 python main.py --model_type="timm" --model_name="efficientnetv2_rw_t.ra2_in1k" --all_folds=True && \
CUDA_VISIBLE_DEVICES=0 python main.py --model_type="timm" --model_name="efficientnetv2_rw_s.ra2_in1k" --all_folds=True && \
CUDA_VISIBLE_DEVICES=0 python main.py --model_type="timm" --model_name="efficientnetv2_rw_m.agc_in1k" --all_folds=True


CUDA_VISIBLE_DEVICES=0 python main.py --model_type="seg" --model_name="mit_b6" --no_wandb
CUDA_VISIBLE_DEVICES=1 python main.py --model_type="seg" --model_name="mit_b4" --no_wandb
CUDA_VISIBLE_DEVICES=2 python main.py --model_type="seg" --model_name="mit_b5" --no_wandb

CUDA_VISIBLE_DEVICES=0 python main.py --epochs=5 --model_name="efficientnet-b0" --all_folds
CUDA_VISIBLE_DEVICES=1 python main.py --epochs=5 --model_name="efficientnet-b1" --all_folds
CUDA_VISIBLE_DEVICES=2 python main.py --epochs=5 --model_name="efficientnet-b2" --all_folds
CUDA_VISIBLE_DEVICES=3 python main.py --epochs=5 --model_name="efficientnet-b3" --all_folds

CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/Contrails-ICRGW/hwsqaxzv
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/Contrails-ICRGW/hwsqaxzv
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/Contrails-ICRGW/hwsqaxzv

CUDA_VISIBLE_DEVICES=3 python main.py --epochs=5 --model_name="efficientnet-b0" --all_folds
CUDA_VISIBLE_DEVICES=3 python main.py --epochs=5 --model_name="efficientnet-b7" --all_folds --batch_size=16 --lr=1e-4 --log_every_n_steps=20
