import lightning.pytorch as pl
import torch
import os
import numpy as np

from pl_scripts.modules import CustomModule, CustomDataModule
from pl_scripts.callbacks.helpers import load_logger_and_callbacks

def train(
        config,
):
    # Seed
    pl.seed_everything(config.seed, workers=True)

    # Set torch cache location
    torch.hub.set_dir(config.torch_cache)

    # Limit CPU if doing dev run
    if config.fast_dev_run == True:
        config.num_workers = 1

    # Get logger 1st so we can use run_name
    logger, callbacks = load_logger_and_callbacks(
        fast_dev_run = config.fast_dev_run,
        metrics = {
            "val_loss": "min", 
            "train_loss": "min",
            "val_dice": "last",
            "val_dice": "max",
            },
        overfit_batches = config.overfit_batches,
        no_wandb = config.no_wandb,
        project = config.project,
        swa = config.swa,
        swa_epochs = config.swa_epochs,
        epochs = config.epochs,
        lr_min = config.lr_min,
    )

    # Get run metadata
    try: experiment_name = logger._experiment.name
    except: experiment_name = None

    # Create directory for saving predictions
    if config.save_preds:
        if not os.path.exists(config.preds_dir + str(experiment_name)):
            os.mkdir(config.preds_dir + str(experiment_name))

    data_module = CustomDataModule(
        data_dir = config.data_dir,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        img_size = config.img_size,
        rand_scale_min = config.rand_scale_min,
        rand_scale_prob = config.rand_scale_prob,
        transform = config.transform,
        seed = config.seed,
        )

    module = CustomModule(
        lr = config.lr,
        lr_min = config.lr_min,
        hf_cache = config.hf_cache,
        model_save_dir = config.model_save_dir,
        model_name = config.model_name,
        preds_dir = config.preds_dir,
        decoder_type = config.decoder_type,
        model_weights = config.model_weights,
        experiment_name = experiment_name,
        save_model = config.save_model,
        save_preds = config.save_preds,
        epochs = config.epochs,
        scheduler = config.scheduler,
        fast_dev_run = config.fast_dev_run,
        num_cycles = config.num_cycles,
        loss = config.loss,
        smooth = config.smooth,
        dice_threshold = config.dice_threshold,
        mask_downsample = config.mask_downsample,
        swa = config.swa,
    )

    # Trainer Args: https://lightning.ai/docs/pytorch/stable/common/trainer.html#benchmark
    trainer = pl.Trainer(
        accelerator = config.accelerator,
        benchmark = True, # set to True if input size does not change (increases speed)
        fast_dev_run = config.fast_dev_run,
        max_epochs = config.epochs,
        num_sanity_val_steps = 0,
        overfit_batches = config.overfit_batches,
        precision = config.precision,
        callbacks = callbacks,
        logger = logger,
        log_every_n_steps = (32 // config.batch_size) * 10,
        accumulate_grad_batches = config.accumulate_grad_batches,
        val_check_interval = config.val_check_interval,
        enable_checkpointing = False,
        gradient_clip_val = 1.0,
        strategy = config.strategy,
    )
    trainer.fit(module, datamodule=data_module)
    trainer.validate(module, datamodule=data_module)
    return