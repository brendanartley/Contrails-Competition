# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas


- When doing accumulate_grad_batches, multiply the learning rate by accumulate_grad_batches
- Freeze BN layers in encoder?

- Train larger model on Sunny
- Use MAnet in ensemble

- Uniformer could be very useful
    - https://github.com/Sense-X/UniFormer

- Make Unet Extension (Unet3+ is the move!)
    - UNeXt? https://github.com/jeya-maria-jose/UNeXt-pytorch
    - DoubleUnet .

### GPU Efficiency Notes

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

### Running training scripts on remote machines

Use nohup. This means the script runs even when SSH is closed.

nohup CUDA_VISIBLE_DEVICES=0 python main.py &

### Commands

CUDA_VISIBLE_DEVICES="" python dice_threshold.py
CUDA_VISIBLE_DEVICES="" python model_correlation.py

CUDA_VISIBLE_DEVICES=0 nohup python main.py --model_name="tu-resnest50d.in1k" --decoder_type="DeepLabV3Plus" > nohup3.out &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_name="tu-resnest50d.in1k" --decoder_type="MAnet" --batch_size=16 --lr=1e-4 --precision=32 > nohup4.out &
CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-resnest269e.in1k" --batch_size=3 --img_size=1024 --lr=1e-4 --no_wandb

CUDA_VISIBLE_DEVICES=0 nohup python main.py --model_name="tu-resnest269e.in1k" --epochs=12 --batch_size=16 --lr=1e-4 --img_size=800 --no_wandb > nohup.out
CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-resnest269e.in1k" --batch_size=16 --lr=1e-4 --img_size=800 --no_wandb --fast_dev_run

CUDA_VISIBLE_DEVICES=1 nohup python main.py > nohup4.out &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_name="tu-resnest50d.in1k" > nohup.out &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --model_name="tu-resnest269e.in1k" > nohup1.out &
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="tu-resnest50d.in1k" --img_size=512 --batch_size=16 --lr=1e-4 > nohup2.out &

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-resnest50d.in1k" --img_size=512 --batch_size=1 --lr=1e-4 --no_wandb


CUDA_VISIBLE_DEVICES=1 nohup python main.py --scheduler="CosineAnnealingLRCyclic" --epochs=15 --lr=1e-3 &
CUDA_VISIBLE_DEVICES=1 python main.py --scheduler="CosineAnnealingLRCyclic" --epochs=15 --lr=1e-3 --fast_dev_run

CUDA_VISIBLE_DEVICES=2,3 nohup python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --decoder_type="MAnet" --img_size=384 --lr=2e-4 --batch_size=7 --swa=True --seed=420 --precision=32 --accumulate_grad_batches=2 --epochs=13 > nohup1.out &

CUDA_VISIBLE_DEVICES=0,1 nohup python main.py --model_name=tu-maxvit_base_tf_512.in21k_ft_in1k --img_size=512 --lr=6e-4 --batch_size=3 --accumulate_grad_batches=5 --precision=32 --swa=True --seed=0 > nohup2.out &


CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="timm-efficientnet-b4" --decoder_type="DeepLabV3Plus" > nohup.out &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="mit_b4" --img_size=1024 --batch_size=16 --lr=4e-4 --val_check_interval=0.10 &

CUDA_VISIBLE_DEVICES=2 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/bfg_612.pt" --model_name="mit_b4" --img_size=800 --batch_size=16 --save_preds --val

CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/bfg_852.pt" --model_name="mit_b4" --img_size=800 --batch_size=16 --save_preds --val

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

### Big Notes

- Editor Config
- Edit callbacks header (remove no_wandb in get_callbacks(), or just run --no_wandb)
- Run w/ no_wandb
- Change experiment name line to this
```
    # Get run metadata
    try: experiment_name = logger._experiment.name
    except: experiment_name = str(np.random.randint(0,1000))
```

CUDA_VISIBLE_DEVICES=0 nohup python main.py --model_name="mit_b4" --img_size=800 --batch_size=16 --lr=1e-4 --val_check_interval=0.10 --seed=111 --no_wandb &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_name="mit_b4" --img_size=800 --batch_size=16 --lr=1e-4 --val_check_interval=0.10 --seed=112 --no_wandb &

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=1e-4 --val_check_interval=0.10 --precision="32" --seed=5 --no_wandb

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=1e-4 --val_check_interval=0.10 --precision="32" --seed=312 --no_wandb --epochs=12 > nohup1.out &

CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/bfg_1.pt" --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=16 --save_preds