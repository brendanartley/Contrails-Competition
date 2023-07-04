import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchinfo
import torchvision
# import bitsandbytes as bnb

import cv2
import albumentations as A

import timm
from timm.scheduler.cosine_lr import CosineLRScheduler

import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp

# Models
from .models.my_models import Unet, UnetPlusPlus, MAnet
from .models.timm_unet import CustomUnet

class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_size, train=True, transform=None):
        self.data_dir = data_dir
        self.trn = train
        self.transform = transform
        self.records = self.load_records()
        # Final mask shape must be 256x256
        if img_size != 256:
            self.mask_transform = A.Compose([
                A.Resize(height=256, width=256, interpolation=cv2.INTER_NEAREST)
            ])
        else:	
            self.mask_transform = None

    def load_records(self):
        train_df = pd.read_csv(self.data_dir + "train_df.csv")
        valid_df = pd.read_csv(self.data_dir + "valid_df.csv")

        # Train on all data
        if self.trn == True:
            return train_df["record_id"].values
        else:
            return valid_df["record_id"].values
    
    def __getitem__(self, index):
        fpath = str(self.records[index])
        con_path = self.data_dir + "contrails/" + fpath + ".npy"
        con = np.load(str(con_path)).astype("float")

    	# 4th dimension is the binary mask (label)	
        img = con[..., :-1]	
        mask = con[..., -1]	

        if self.transform:
            augs = self.transform(image=img, mask=mask)
            img = augs["image"]
            mask = augs["mask"]

        if self.mask_transform:
            augs = self.mask_transform(image=mask, mask=mask)
            mask = augs["mask"]
        
        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask)
            
        return img.float(), mask.int(), int(fpath)
    
    def __len__(self):
        return len(self.records)

class ContrailsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        img_size: int,
        rand_scale_min: float,
        rand_scale_prob: float,
        no_transform: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        # No transforms: (except reshape)
        if self.hparams.no_transform == True:
            train_transforms = A.Compose([
                    A.Resize(height=self.hparams.img_size, width=self.hparams.img_size)
                ])
            valid_transforms = A.Compose([
                    A.Resize(height=self.hparams.img_size, width=self.hparams.img_size),
                ])
            
        # w/ Transformations
        else:
            train_transforms = A.Compose([
                # A.RandomSizedCrop(min_max_height=(int(256*self.hparams.rand_scale_min), 256), height=256, width=256, p=self.hparams.rand_scale_prob),
                A.Resize(height=self.hparams.img_size, width=self.hparams.img_size)
            ])
            valid_transforms = A.Compose([
                A.Resize(height=self.hparams.img_size, width=self.hparams.img_size),
            ])

        return train_transforms, valid_transforms


    def setup(self, stage):        
        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train=True, transform=self.train_transform)
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)

        elif stage == "validate":
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)
            
    def _dataset(self, train, transform):
        return ContrailsDataset(
            data_dir=self.hparams.data_dir, 
            train=train,
            img_size=self.hparams.img_size,
            transform=transform
            )
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self):
        return self._dataloader(self.val_dataset, train=False)

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
        decoder_type: str,
        model_weights: str,
        run_name: str,
        save_model: bool,
        save_preds: bool,
        epochs: int,
        scheduler: str,
        fast_dev_run: bool,
        lr_min: float,
        num_cycles: int,
        interpolate: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.metrics = self._init_metrics()
        self.loss_fn = self._init_loss_fn()

    def _init_model(self):

        # Decoder Options
        seg_models = {
            "Unet": Unet,
            "UnetPlusPlus": UnetPlusPlus,
            "MAnet": MAnet,
            "CustomUnet": CustomUnet,
        }

        # Training Run
        if self.hparams.decoder_type in seg_models.keys():
            model = seg_models[self.hparams.decoder_type](
                encoder_name=self.hparams.model_name, 
                in_channels=3,
                classes=1,
            )
        else:
            raise ValueError(f"{self.hparams.decoder_type} not recognized.")

        # Validation: Load saved weights
        if self.hparams.model_weights != None:
            model.load_state_dict(torch.load(self.hparams.model_weights))
        return model
    
    def _init_optimizer(self):
        return optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            )
        # Bits + Bytes ()
        # return bnb.optim.Adam8bit(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.estimated_stepping_batches,
                eta_min = self.hparams.lr_min,
                )
        elif self.hparams.scheduler == "CosineAnnealingLRWarmup":
            return CosineLRScheduler(
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
            "dice": torchmetrics.Dice(average = 'micro', threshold=0.5),
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
        x, y, fpath = batch
        y_logits = self(x) # Raw logits
        loss = self.loss_fn(y_logits, y)

        # Compute Metric
        if stage == "val":
            self.metrics[f"{stage}_metrics"](y_logits, y)
            
            # Save Preds: Dice Threshold, Ensemble, etc
            if self.hparams.save_preds:
                
                # Save each pred as its own tensor
                for i, img_idx in enumerate(fpath):
                    save_preds = torch.stack([
                        y_logits[i, 0, :, :],
                        y[i]
                    ])

                    torch.save(save_preds, "{}{}/{}.pt".format(
                        self.hparams.preds_dir, 
                        self.hparams.run_name,
                        img_idx, 
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
        if self.hparams.fast_dev_run == False and self.hparams.save_model == True:
            torch.save(self.model, "{}{}.pt".format(self.hparams.model_save_dir, self.hparams.run_name))
        return