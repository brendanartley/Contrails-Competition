import lightning.pytorch as pl
import wandb
import uuid
import torch

from contrails_pl.modules import ContrailsModule, ContrailsDataModule
from contrails_pl.helpers import load_logger_and_callbacks


def train(
        config,
):
    # Seed
    pl.seed_everything(config.seed, workers=True)

    # Set torch cache location
    torch.hub.set_dir(config.torch_cache)

    # Limits CPU if doing dev run
    if config.fast_dev_run == True:
        config.num_workers = 1

    # Optional: Full K-Fold cross validation
    if config.all_folds == True:
        num_iters = config.num_folds
        group = config.model_name + "_" + uuid.uuid4().hex[:8] # Create random group name
    else:
        num_iters = 1
        group = None
    
    for _ in range(num_iters):

        data_module = ContrailsDataModule(
            data_dir = config.data_dir,
            batch_size = config.batch_size,
            num_workers = config.num_workers,
            val_fold = config.val_fold,
            )

        logger, callbacks = load_logger_and_callbacks(
            fast_dev_run = config.fast_dev_run,
            metrics = {
                "val_loss": "min", 
                "train_loss": "min",
                "val_dice": "last",
                },
            overfit_batches = config.overfit_batches,
            no_wandb = config.no_wandb,
            project = config.project,
            group = group,
        )

        module = ContrailsModule(
            lr = config.lr,
            lr_min = config.lr_min,
            model_save_dir = config.model_save_dir,
            model_name = config.model_name,
            model_type = config.model_type,
            run_name = logger._experiment.name if logger else None,
            save_model = config.save_model,
            epochs = config.epochs,
            scheduler = config.scheduler,
            fast_dev_run = config.fast_dev_run,
            num_cycles = config.num_cycles,
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
            callbacks = callbacks,
            logger = logger,
            log_every_n_steps = config.log_every_n_steps,
            accumulate_grad_batches = config.accumulate_grad_batches,
            val_check_interval = config.val_check_interval,
            enable_checkpointing = False,
            gradient_clip_val = 1.0,
        )
        trainer.fit(module, datamodule=data_module)

        # Need to finish run when doing multiple runs in the same process
        # Source: https://docs.wandb.ai/ref/python/finish
        if config.all_folds == True:
            wandb.finish()

    return