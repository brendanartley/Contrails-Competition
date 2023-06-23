import lightning.pytorch as pl
import torch
import os

from contrails_pl.modules import ContrailsModule, ContrailsDataModule

def validate(
        config,
):
    # Seed
    pl.seed_everything(config.seed, workers=True)

    # Set torch cache location
    torch.hub.set_dir(config.torch_cache)

    # Limits CPU if doing dev run
    if config.fast_dev_run == True:
        config.num_workers = 1

    #TODO: NEED TO FIX DIRECTORY NAME HERE..
    # # Create directory for saving predictions
    # if not os.path.exists(config.preds_dir):
    #     os.mkdir(config.preds_dir)

    data_module = ContrailsDataModule(
        data_dir = config.data_dir,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        val_fold = config.val_fold,
        )

    module = ContrailsModule(
        lr = config.lr,
        lr_min = config.lr_min,
        model_save_dir = config.model_save_dir,
        model_name = config.model_name,
        preds_dir = config.preds_dir,
        model_type = config.model_type,
        model_weights = config.model_weights,
        run_name = None,
        save_weights = config.save_weights,
        save_preds = config.save_preds,
        epochs = config.epochs,
        scheduler = config.scheduler,
        fast_dev_run = config.fast_dev_run,
        num_cycles = config.num_cycles,
        val_fold = config.val_fold,
    )

    # Trainer Args: https://lightning.ai/docs/pytorch/stable/common/trainer.html#benchmark
    trainer = pl.Trainer(
        accelerator = config.accelerator,
        benchmark = True, # set to True if input size does not change (increases speed)
        devices = config.devices,
        fast_dev_run = config.fast_dev_run,
        max_epochs = config.epochs,
        num_sanity_val_steps = 1,
        overfit_batches = config.overfit_batches,
        precision = config.precision,
        callbacks = None,
        logger = None,
        log_every_n_steps = config.log_every_n_steps,
        accumulate_grad_batches = config.accumulate_grad_batches,
        val_check_interval = config.val_check_interval,
        enable_checkpointing = False,
        gradient_clip_val = 1.0,
    )
    trainer.validate(module, datamodule=data_module)

    return