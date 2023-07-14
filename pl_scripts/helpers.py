import lightning.pytorch as pl
import torch

class CustomWeightSaver(pl.Callback):
  """
  Saves model weights if the val_dice is best so far.
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

def get_logger(metrics, project, no_wandb):
    """
    Function to load wandb logger.
    """
    if no_wandb == True:
       return None, None

    # Get logger
    logger = pl.loggers.WandbLogger(
        project = project, 
        save_dir = None,
        )
    id_ = logger.experiment.id
    
    # Wandb metric summary options (min,max,mean,best,last,none): https://docs.wandb.ai/ref/python/run#define_metric
    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)
    
    return logger, id_

def get_SWA(epochs, lr):
   """
   Stochastic weight averaging.
   """
   mid_epoch = epochs - (epochs//2)
   return pl.callbacks.StochasticWeightAveraging(swa_lrs=lr, annealing_epochs=10-mid_epoch, swa_epoch_start=mid_epoch, annealing_strategy="cos")

def load_logger_and_callbacks(
    fast_dev_run,
    metrics,
    overfit_batches,
    no_wandb,
    project,
    swa,
    epochs,
    lr
):
    callbacks = []

    # Test Runs.
    if fast_dev_run or overfit_batches > 0:
        return None, None
    
    # Stochastic weight averaging
    if swa == True:
       swa_callback = get_SWA(
          epochs = epochs,
          lr = lr,
       )
       callbacks.append(swa_callback)

    # Other Callbacks
    callbacks.extend([
        pl.callbacks.LearningRateMonitor(),
        CustomWeightSaver(),
    ])

    # Logger
    logger, id_ = get_logger(
        metrics = metrics, 
        project = project,
        no_wandb = no_wandb,
        )

    return logger, callbacks