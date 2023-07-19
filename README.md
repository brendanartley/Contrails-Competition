# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

- When doing accumulate_grad_batches, multiply the learning rate by accumulate_grad_batches

- Train larger model on Sunny
- Use MAnet in ensemble

- Hacking MMSEGMENTATION
    - https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/425175

- Make Unet Extension (Unet3+ is the move!)
    - https://github.com/frgfm/Holocron/blob/f78c6c58c0007e3d892fcaa1f1ff786cdbb5195f/holocron/models/segmentation/unet3p.py#L95
    - https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5

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

### Running training scripts on remote machines

Use nohup. This means the script runs even when SSH is closed.

nohup CUDA_VISIBLE_DEVICES=0 python main.py &

### Commands

CUDA_VISIBLE_DEVICES=0 python main.py --decoder_type="CustomUnet" --fast_dev_run
CUDA_VISIBLE_DEVICES=1 python main.py --decoder_type="CustomUnetV2" --fast_dev_run

CUDA_VISIBLE_DEVICES=1 None.pt python main.py --fast_dev_run


CUDA_VISIBLE_DEVICES="" python dice_threshold.py
CUDA_VISIBLE_DEVICES="" python model_correlation.py

CUDA_VISIBLE_DEVICES=2,3 nohup python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --decoder_type="MAnet" --img_size=384 --lr=2e-4 --batch_size=7 --swa=True --seed=420 --precision=32 --accumulate_grad_batches=2 --epochs=13 > nohup1.out &

CUDA_VISIBLE_DEVICES=1,2 python main.py --model_name=tu-maxvit_base_tf_512.in21k_ft_in1k --img_size=512 --lr=5e-4 --batch_size=3 --accumulate_grad_batches=5 --precision=32 --swa=True --seed=0
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="mit_b4" --img_size=1024 --batch_size=16 --lr=4e-4 --val_check_interval=0.10 &

CUDA_VISIBLE_DEVICES=2 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/None.pt" --save_preds --model_name=tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k --img_size=384 --fast_dev_run

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

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=1e-4 --val_check_interval=0.10 --precision="32" --seed=5 --no_wandb

CUDA_VISIBLE_DEVICES=1 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=1e-4 --val_check_interval=0.10 --precision="32" --seed=4 --no_wandb

CUDA_VISIBLE_DEVICES=3 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/bfg_1.pt" --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=16 --save_preds