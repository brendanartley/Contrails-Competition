# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### Final Submission Options

V1. Each fold is nested ([f1], [f2], [f3], ..., [f9])
V2. Overweight 512 ([m1f1-5], [m1f6-10], [m2], [m3], [m4])
V3. Overweight 768 ([m1], [m2f1-5], [m2f-10], [m3], [m4])

### Ideas

- Train model on 9 folds
    - 9 maxvit 384 models (768 img_size, trained on small GPU, w/ SWA, 15 epochs)
    - Done: 8,9,(1 w/ new seed)
    """
    CUDA_VISIBLE_DEVICES=1,2,3 nohup python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=2 > nohup.out &
    """

    - 9 tu-maxvit_base_tf_512.in21k_ft_in1k models (512 img_size, 13 epochs, no SWA, bfg server)
    - Done
    """
    CUDA_VISIBLE_DEVICES=2 nohup python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --img_size=512 --batch_size=14 --lr=15e-5 --val_check_interval=0.10 --precision="32" --seed=5 --epochs=13 --no_wandb --val_fold=2 > nohup.out &

    CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-maxvit_base_tf_512.in21k_ft_in1k" --model_weights="/data/bartley/gpu_test/models/backup_final_models/fold_10_867_tu-maxvit_base_tf_512.in21k_ft_in1k.pt" --img_size=512 --batch_size=8 --precision="32" --save_preds --val
    """

    C:\dev\gpu_test\backup_final_models\fold_10_867_tu-maxvit_base_tf_512.in21k_ft_in1k.pt

    - 9 tu-resnest269e.in1k models (1024 img_size, 14 epochs, no SWA, bfg server)
    - TODO: 7,8,9
    """
    CUDA_VISIBLE_DEVICES=3 python main.py --model_name="tu-resnest269e.in1k" --img_size=1024 --batch_size=14 --lr=15e-5 --epochs=14 --val_fold=1 --no_wandb
    """

    - 9 mit_b4 models (800 img_size, 14 epochs, SWA?)
    - TODO: 8, 9
    """
    CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=16 --lr=1e-4 --val_fold=1 --no_wandb > nohup.out &
    """

- Train larger model on Sunny

### Final Ensemble Ideas

- Nested 5 model ensemble (need to figure out what to do with even number here..)
    - V1 - [5 maxvit512s (512), 4 maxvit512s (512), 9 maxvit384s (784), 9 resnests (1024), 9 mit_b4s (800)]
    - V2 - [9 maxvit512s (512), 5 maxvit384s (784), 4 maxvit384s (784), 9 resnests (1024), 9 mit_b4s (800)]
    - Use another pre-trained maxvit512, train another seed for 384. This will get to odd number for both.

- Nested 9-nested Fold ensemble [(fold 1 ens), (fold 2 ens)... (fold 9 ens)]
    - V3 - [9 maxvit512s (512), 9 maxvit384s (784), 9 resnests (1024)]

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

CUDA_VISIBLE_DEVICES=0 nohup python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=16 --lr=1e-4 --val_fold=2 --no_wandb > nohup.out &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=16 --lr=1e-4 --val_fold=3 --no_wandb > nohup.out &

CUDA_VISIBLE_DEVICES=1 python main.py --model_name="mit_b4" --img_size=800 --epochs=13 --batch_size=4 --accumulate_grad_batches=4 --lr=2e-4 --val_fold=1 --no_wandb

## Sat Morning

CUDA_VISIBLE_DEVICES=1,2,3 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=8 && CUDA_VISIBLE_DEVICES=1,2,3 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=2e-4 --batch_size=4 --accumulate_grad_batches=4 --epochs=15 --swa=True --swa_epochs=5 --val_fold=9 && 

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k" --img_size=768 --lr=1e-4 --batch_size=15 --epochs=15 --swa=True --swa_epochs=5 --val_fold=1 --seed=971

CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-resnest269e.in1k" --img_size=1024 --batch_size=14 --lr=15e-5 --epochs=14 --val_fold=7 --no_wandb &&
CUDA_VISIBLE_DEVICES=0 python main.py --model_name="tu-resnest269e.in1k" --img_size=1024 --batch_size=14 --lr=15e-5 --epochs=14 --val_fold=8 --no_wandb

CUDA_VISIBLE_DEVICES=1 python main.py --model_name="tu-resnest269e.in1k" --img_size=1024 --batch_size=14 --lr=15e-5 --epochs=14 --val_fold=9 --no_wandb