# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Ideas

- Train model on 9 folds
    - 9 maxvit 384 models (768 img_size, trained on small GPU, w/ SWA, 15 epochs)
    - Done: 1
    """
    CUDA_VISIBLE_DEVICES=1,2,3 nohup python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=2 > nohup.out &
    """

    - 9 tu-maxvit_base_tf_512.in21k_ft_in1k models (512 img_size, 13 epochs, no SWA, bfg server)
    - Done: 1,2,3,4,5
    """
    CUDA_VISIBLE_DEVICES=2 nohup python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=15e-5 --val_check_interval=0.10 --precision="32" --seed=5 --epochs=13 --no_wandb --val_fold=2 > nohup.out &
    """

    - 9 tu-resnest269e.in1k models (1024 img_size, 14 epochs, no SWA, bfg server)
    - Done: 1
    """
    CUDA_VISIBLE_DEVICES=3 python main.py --model_name="tu-resnest269e.in1k" --img_size=1024 --batch_size=14 --lr=15e-5 --epochs=14 --val_fold=1 --no_wandb
    """

    - 9 mit_b4 models (800 img_size, 14 epochs, SWA?)
    - Done: 1
    """
    CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=16 --lr=1e-4 --val_fold=1 --no_wandb > nohup.out &
    """

- Train larger model on Sunny

### Final Ensemble Ideas

- Nested 5 model ensemble
    - V1 - [maxvit512s (512), maxvit512s (512), maxvit384s (784), resnests (1024), mit_b4s (800)]
    - V2 - [maxvit512s (512), maxvit384s (784), maxvit384s (784), resnests (1024), mit_b4s (800)]

- Nested 9 Fold ensemble
    - V3 - [Fold-1, Fold-2, Fold-3, Fold-4, Fold-5, Fold-6, Fold-7]
    - V4 - [Fold-1, Fold-2, Fold-3, Fold-4, Fold-5, Fold-6, Fold-7]

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

`nohup CUDA_VISIBLE_DEVICES=0 python main.py &`

For chaining with nohup, write commands in a shell script and call with `NAME.sh &!`

```
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="tu-resnest269e.in1k" --fast_dev_run > nohup.out
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="tu-resnest269e.in1k" --fast_dev_run > nohup1.out
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="tu-resnest269e.in1k" --fast_dev_run > nohup2.out
```

### Improvenments

Need to use checkpointing for long training runs.
    - This will enable the usage of pausing and restarting long training runs
    - https://pytorch-lightning.readthedocs.io/en/1.6.2/common/checkpointing.html

### Commands

CUDA_VISIBLE_DEVICES="" python dice_threshold.py
CUDA_VISIBLE_DEVICES="" python model_correlation.py

CUDA_VISIBLE_DEVICES="" nohup python dice_threshold.py > nohup2.out &


CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=16 --lr=1e-4 --val_fold=1 --no_wandb > nohup.out &




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

- Edit Config
- Run w/ no_wandb
- Change experiment name line to this
```
    # Get run metadata
    try: experiment_name = logger._experiment.name
    except: experiment_name = "fold_" + config.val_fold + "_" + str(np.random.randint(0,1000))
```

CUDA_VISIBLE_DEVICES=1,2,3 nohup python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=5 > nohup.out &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=16 --lr=1e-4 --val_fold=1 --no_wandb > nohup.out &
CUDA_VISIBLE_DEVICES=0 python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=4 --accumulate_grad_batches=4 --lr=2e-4 --val_fold=1 --no_wandb

## Sat Morning

CUDA_VISIBLE_DEVICES=1,2,3 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=6 && CUDA_VISIBLE_DEVICES=1,2,3 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=7