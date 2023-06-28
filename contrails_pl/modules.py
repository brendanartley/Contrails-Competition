import lightning.pytorch as pl
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchinfo

import timm

import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp

from .models.timm_unet import TimmUnet
from .models.my_models import CustomUnet

class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_fold, train_all, comp_val, train=True, transform=None):
        self.data_dir = data_dir
        self.trn = train
        self.val_fold = val_fold
        self.records = self.load_records(train_all, comp_val)

    def load_records(self, train_all, comp_val):

        # COMPETITION validation set
        if comp_val == True:
            if self.trn == False:
                df = pd.read_csv(self.data_dir + "valid_df.csv")
                print("Comp val length: ", len(df))
                return df["record_id"].values
        
        df = pd.read_csv(self.data_dir + "train_df.csv")

        # Train on all data
        if train_all == True:
            if self.trn == False:
                return [], []
            else:
                return df["record_id"].values
            
        # OOF Validation
        else:
            if self.trn == True:
                df = df[df.fold != self.val_fold]
            else:
                df = df[df.fold == self.val_fold]
            return df["record_id"].values
    
    def __getitem__(self, index):
        fpath = str(self.records[index])
        con_path = self.data_dir + "contrails/" + fpath + ".npy"
        con = np.load(str(con_path))

        # 4th dimension is the binary mask (label)
        img = con[..., :-1]
        label = con[..., -1]
        
        img = torch.tensor(img)
        label = torch.tensor(label)

        # (256, 256, 3) -> (3, 256, 256)
        img = img.permute(2, 0, 1)
            
        return img.float(), label.int()
    
    def __len__(self):
        return len(self.records)

class ContrailsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        val_fold: int,
        train_all: bool,
        comp_val: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        img_transforms = None
        return img_transforms, img_transforms
    
    def setup(self, stage):        
        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train=True, transform=self.train_transform)
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)

        elif stage == "validate":
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)
            
    def _dataset(self, train, transform):
        return ContrailsDataset(
            data_dir=self.hparams.data_dir, 
            val_fold=self.hparams.val_fold,
            train=train,
            train_all=self.hparams.train_all,
            comp_val=self.hparams.comp_val,
            )
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self):
        if self.hparams.train_all == False:
            return self._dataloader(self.val_dataset, train=False)
        else:
            return []

    def _dataloader(self, dataset, train=False):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle = train,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory = True, # True for when processing is done on CPU
        )

class ContrailsModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        model_save_dir: str,
        model_name: str,
        preds_dir: str,
        model_type: str,
        model_weights: str,
        run_name: str,
        save_weights: bool,
        save_preds: bool,
        epochs: int,
        scheduler: str,
        fast_dev_run: bool,
        lr_min: float,
        num_cycles: int,
        val_fold: int,
        interpolate: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.metrics = self._init_metrics()
        self.loss_fn = self._init_loss_fn()

    def _init_model(self):

        # Training Run
        if self.hparams.model_type == "timm":
            # Timm Encoders
            model = TimmUnet(
                backbone=self.hparams.model_name, 
                in_chans=3,
                num_classes=1,
                interpolate=self.hparams.interpolate,
            )
        elif self.hparams.model_type == "seg":
            # Segmentation Models
            model = smp.Unet(
                encoder_name = self.hparams.model_name, 
                encoder_weights = "imagenet", 
                decoder_use_batchnorm = True,
                classes = 1, 
                activation =  None,
            )
        elif self.hparams.model_type == "mine":
            model = CustomUnet(
                num_classes=1,
            )

        # Validation: Load saved weights
        if self.hparams.model_weights != None:
            model.load_state_dict(torch.load(self.hparams.model_weights))
        return model
    
    def _init_optimizer(self):
        return optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            )

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.estimated_stepping_batches,
                eta_min = self.hparams.lr_min,
                )
        elif self.hparams.scheduler == "CosineAnnealingLRWarmup":
            return timm.scheduler.cosine_lr.CosineLRScheduler(
                optimizer, 
                t_initial = self.trainer.estimated_stepping_batches,
                warmup_t = self.trainer.estimated_stepping_batches//25,
                lr_min = self.hparams.lr_min,
                warmup_lr_init = self.hparams.lr * 1e-2,
                )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not a valid scheduler.")
        
    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step(
            epoch=self.global_step
        )
    
    def _init_loss_fn(self):
        return smp.losses.DiceLoss(
            mode = 'binary',
            )
    
    def _init_metrics(self):
        metrics = {
            "dice": torchmetrics.Dice(
                average = 'micro',
            )
            }
        metric_collection = torchmetrics.MetricCollection(metrics)
        return torch.nn.ModuleDict(
            {
                # "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, stage, batch_idx):
        x, y = batch
        y_logits = self(x) # Raw logits
        loss = self.loss_fn(y_logits, y)

        # Compute Metric
        if stage == "val":
            self.metrics[f"{stage}_metrics"](y_logits, y)
            
            # Save Preds: Dice Threshold, Ensemble, etc
            if self.hparams.save_preds:
                save_preds = torch.stack([
                    y_logits.squeeze(dim=1),
                    y.squeeze(),
                ])
                torch.save(save_preds, "{}{}/fold_{}_pred_{}.pt".format(
                    self.hparams.preds_dir, 
                    self.hparams.run_name,
                    self.hparams.val_fold, 
                    batch_idx,
                    ))

        # Log Loss
        self._log(stage, loss, batch_size=len(x))
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val", batch_idx)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)
    
    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        if stage == "val":
            self.log_dict(self.metrics[f"{stage}_metrics"], prog_bar=True, batch_size=batch_size)

    def on_train_end(self):
        if self.hparams.fast_dev_run == False and self.hparams.save_weights == True:
            torch.save(self.model.state_dict(), "{}{}.pt".format(self.hparams.model_save_dir, self.hparams.run_name))
        return