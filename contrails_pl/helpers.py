import lightning.pytorch as pl
import torch

def load_logger_and_callbacks(
    fast_dev_run,
    metrics,
    overfit_batches,
    no_wandb,
    project,
):
    """
    Function that loads logger and callbacks.
    
    Returns:
        logger: lighting logger
        callbacks: lightning callbacks
    """
    # Params used to check for Bugs/Errors in Implementation
    if fast_dev_run or overfit_batches > 0 or no_wandb == True:
        logger, callbacks = None, None
    else:
        logger, id_ = get_logger(
            metrics = metrics, 
            project = project,
            )
        callbacks = [
            pl.callbacks.LearningRateMonitor(),
            CustomWeightSaver(),
        ]
    return logger, callbacks

def get_logger(metrics, project):
    """
    Function to load logger.
    
    Returns:
        logger: lighting logger
        id_: experiment id
    """
    logger = pl.loggers.WandbLogger(
        project = project, 
        save_dir = None,
        )
    id_ = logger.experiment.id
    
    # Wandb metric summary options (min,max,mean,best,last,none): https://docs.wandb.ai/ref/python/run#define_metric
    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)
    
    return logger, id_

class CustomWeightSaver(pl.Callback):
  """
  Saves model weights if best val_score so far.
  """
  def __init__(self):
    super().__init__()
    self.best_val_dice = 0

  def on_validation_epoch_end(self, trainer, module):
    if module.hparams.fast_dev_run == False and module.hparams.save_model == True:

        # Update best val dice
        val_dice = trainer.logged_metrics["val_dice"].item()
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice

            # Save weights
            module._save_weights(val_dice)
    return