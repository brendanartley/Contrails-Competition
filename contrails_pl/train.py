import lightning.pytorch as pl
import wandb
import uuid
import torch
import os

from contrails_pl.modules import ContrailsModule, ContrailsDataModule
from contrails_pl.helpers import load_logger_and_callbacks


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

    # Optional: Full K-Fold cross validation
    if config.all_folds == True:
        num_iters = config.num_folds
        group = config.model_name + "_" + uuid.uuid4().hex[:8] # Create group name for preds_dir
    
        for val_fold in range(num_iters):
            
            # Get logger 1st so we can use run_name
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

            # Get run metadata
            experiment_name = logger._experiment.name if logger else None

            # Create directory for saving predictions
            if config.save_preds:
                if not os.path.exists(config.preds_dir + str(experiment_name)):
                    os.mkdir(config.preds_dir + str(experiment_name))

            data_module = ContrailsDataModule(
                data_dir = config.data_dir,
                batch_size = config.batch_size,
                num_workers = config.num_workers,
                val_fold = val_fold,
                train_all = config.train_all,
                comp_val = config.comp_val,
                img_size = config.img_size,
                )

            module = ContrailsModule(
                lr = config.lr,
                lr_min = config.lr_min,
                model_save_dir = config.model_save_dir,
                model_name = config.model_name,
                preds_dir = config.preds_dir,
                model_type = config.model_type,
                model_weights = config.model_weights,
                run_name = experiment_name,
                save_weights = config.save_weights,
                save_preds = config.save_preds,
                epochs = config.epochs,
                scheduler = config.scheduler,
                fast_dev_run = config.fast_dev_run,
                num_cycles = config.num_cycles,
                val_fold = val_fold,
                interpolate = config.interpolate,
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
            trainer.validate(module, datamodule=data_module)

            # Need to finish run when doing multiple runs in the same process
            # Source: https://docs.wandb.ai/ref/python/finish
            if config.all_folds == True:
                wandb.finish()
    
    # Single fold validation
    else:
        group = config.model_name # used for preds_dir

        # Get logger 1st so we can use run_name
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
            group = None,
        )

        # Get run metadata
        experiment_name = logger._experiment.name if logger else None

        # Create directory for saving predictions
        if config.save_preds:
            if not os.path.exists(config.preds_dir + str(experiment_name)):
                os.mkdir(config.preds_dir + str(experiment_name))

        data_module = ContrailsDataModule(
            data_dir = config.data_dir,
            batch_size = config.batch_size,
            num_workers = config.num_workers,
            val_fold = config.val_fold,
            train_all = config.train_all,
            comp_val = config.comp_val,
            img_size = config.img_size,
            )

        module = ContrailsModule(
            lr = config.lr,
            lr_min = config.lr_min,
            model_save_dir = config.model_save_dir,
            model_name = config.model_name,
            preds_dir = config.preds_dir,
            model_type = config.model_type,
            model_weights = config.model_weights,
            run_name = experiment_name,
            save_weights = config.save_weights,
            save_preds = config.save_preds,
            epochs = config.epochs,
            scheduler = config.scheduler,
            fast_dev_run = config.fast_dev_run,
            num_cycles = config.num_cycles,
            val_fold = config.val_fold,
            interpolate = config.interpolate,
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
        trainer.validate(module, datamodule=data_module)

    return