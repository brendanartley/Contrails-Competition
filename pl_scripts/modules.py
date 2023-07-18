import lightning.pytorch as pl
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
# import torchinfo
from torchvision import transforms
import os
# import bitsandbytes as bnb
import cv2
import albumentations as A

from timm.scheduler.cosine_lr import CosineLRScheduler

import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp

# Models
from .models.my_models import Unet, UnetPlusPlus, MAnet
from .models.custom_models import CustomUnet

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_size, train=True, transform=None):
        self.data_dir = data_dir
        self.trn = train
        self.transform = transform
        self.img_size = img_size
        
        # Resize transform for IMG ONLY (so mask is resized only once)
        if img_size != 256:
            self.resize_transform = A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
            ])
        else:	
            self.resize_transform = None
        self.records = self.load_records()

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
        con_path = self.data_dir + "imgs/" + fpath + ".npy"
        con = np.load(str(con_path)).astype("float")

    	# 4th dimension is the binary mask (label)	
        img = con[..., :-1]	
        mask = con[..., -1]	

        # RandomResizeCrop
        if self.transform:
            augs = self.transform(image=img, mask=mask)
            img = augs["image"]
            mask = augs["mask"]

        # Resize img (not mask)
        if self.img_size != 256:
            img = self.resize_transform(image=img)["image"]
        
        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask)
            
        return img.float(), mask.int(), int(fpath)
    
    def __len__(self):
        return len(self.records)

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        img_size: int,
        rand_scale_min: float,
        rand_scale_prob: float,
        transform: bool,
        seed: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        if self.hparams.transform == True:
            train_transforms = A.Compose([
                A.RandomSizedCrop(min_max_height=(int(256*self.hparams.rand_scale_min), 256), height=256, width=256, p=self.hparams.rand_scale_prob),
            ])
            valid_transforms = None
            
        else:
            train_transforms = None
            valid_transforms = None

        return train_transforms, valid_transforms


    def setup(self, stage):        
        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train=True, transform=self.train_transform)
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)

        elif stage == "validate":
            self.val_dataset = self._dataset(train=False, transform=self.val_transform)
            
    def _dataset(self, train, transform):
        return CustomDataset(
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

class CustomModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_min: float,
        hf_cache: str,
        model_save_dir: str,
        model_name: str,
        preds_dir: str,
        decoder_type: str,
        model_weights: str,
        experiment_name: str,
        save_model: bool,
        save_preds: bool,
        epochs: int,
        scheduler: str,
        fast_dev_run: bool,
        num_cycles: int,
        loss: str,
        smooth: float,
        dice_threshold: float,
        mask_downsample: str,
        swa: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.metrics = self._init_metrics()
        self.loss_fn = self._init_loss_fn()
        self.val_dice_best = 0

    def _init_model(self):

        # SMP Decoder Options
        seg_models = {
            "Unet": Unet,
            "UnetPlusPlus": UnetPlusPlus,
            "MAnet": MAnet,
        }
        # Transformation Options (downsampling)
        transforms_map = {
            "BILINEAR": transforms.InterpolationMode.BILINEAR,
            "BICUBIC": transforms.InterpolationMode.BICUBIC,
            "NEAREST": transforms.InterpolationMode.NEAREST,
            "NEAREST_EXACT": transforms.InterpolationMode.NEAREST_EXACT,
        }

        # Validation Runs
        if self.hparams.model_weights != None:
            model = seg_models[self.hparams.decoder_type](
                encoder_name=self.hparams.model_name, 
                in_channels=3,
                classes=1,
                inter_type=transforms_map[self.hparams.mask_downsample],
            )
            tmp_model = torch.load(self.hparams.model_weights)
            model.load_state_dict(tmp_model.state_dict())
            
        # Training Runs
        elif self.hparams.decoder_type in seg_models.keys():
            model = seg_models[self.hparams.decoder_type](
                encoder_name=self.hparams.model_name, 
                in_channels=3,
                classes=1,
                inter_type=transforms_map[self.hparams.mask_downsample],
            )
        elif self.hparams.decoder_type == "CustomUnet":
            model = CustomUnet(
                encoder_name=self.hparams.model_name, 
                in_channels=3,
                classes=1,
            )
        # elif self.hparams.decoder_type == "Test":
        #     # ---- Hacking MMSegmentation ----
        #     from mmseg.registry import MODELS
        #     from mmengine.model.utils import revert_sync_batchnorm
        #     from mmseg.utils import register_all_modules
        #     from mmengine.config import Config
        #     register_all_modules()

        #     cfg_file = '/home/bartley/gpu_test/ICRGW/mmseg/my_config.py'
        #     mmconfig = Config.fromfile(cfg_file)
        #     model = revert_sync_batchnorm(MODELS.build(mmconfig.model))
        #     model.init_weights() # initialize the model with pretrained: mmengine - INFO - load model from: ...

        else:
            raise ValueError(f"{self.hparams.decoder_type} not recognized.")
        return model
    
    def _init_optimizer(self):
        return optim.AdamW(self.trainer.model.parameters(), lr=self.hparams.lr)
        # Bits + Bytes ()
        # return bnb.optim.Adam8bit(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.estimated_stepping_batches,
                eta_min = self.hparams.lr_min,
                )
        elif self.hparams.scheduler == "CosineAnnealingLRCyclic":
            # Otherwise fails on fast_dev_run
            if self.hparams.fast_dev_run == True:
                num_cycles = 1
            else:
                num_cycles = self.hparams.num_cycles
            return CosineLRScheduler(
                optimizer, 
                t_initial = self.trainer.estimated_stepping_batches // num_cycles,
                cycle_decay = 0.75,
                cycle_limit = num_cycles,
                lr_min = self.hparams.lr_min,
                )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not a valid scheduler.")
        
    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step(
            epoch=self.global_step
        )
    
    def _init_loss_fn(self):
        if self.hparams.loss == "Dice":
            return smp.losses.DiceLoss(
                mode = 'binary',
                smooth = self.hparams.smooth,
                )
        elif self.hparams.loss == "Tversky":
            return smp.losses.TverskyLoss(
                mode = "binary",
                alpha = 0.45,
                beta = 0.55,
                gamma = 1.25,
                smooth = self.hparams.smooth,
            )
        else:
            raise ValueError(f"{self.hparams.loss} is not a recognized loss function.")
    
    def _init_metrics(self):
        metrics = {
            "dice": torchmetrics.Dice(average = 'micro', threshold=self.hparams.dice_threshold),
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
        y_logits = self(x)

        # # For MMSeg Tests        
        # if self.hparams.decoder_type.startswith("Test"):
        #     if batch_idx == 0: print(y_logits.shape, y.shape)
        #     y_logits = F.interpolate(y_logits, 256)
        #     y_logits = y_logits.squeeze(dim=1)
        #     if batch_idx == 0: print(y_logits.shape, y.shape)

        loss = self.loss_fn(y_logits, y)

        # Compute Metric
        if stage == "val":
            self.metrics[f"{stage}_metrics"](y_logits, y)
            
            # Save Preds: Dice Threshold, Ensemble, etc
            if self.hparams.save_preds:
                
                # Save each pred as its own tensor (as fp1)
                for i, img_idx in enumerate(fpath):
                    save_preds = torch.stack([
                        y_logits[i, 0, :, :],
                        y[i]
                    ])

                    torch.save(save_preds.half(), "{}{}/{}.pt".format(
                        self.hparams.preds_dir, 
                        self.hparams.experiment_name,
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
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        if stage == "val":
            self.log_dict(self.metrics[f"{stage}_metrics"], prog_bar=True, batch_size=batch_size, sync_dist=True)

    def _save_weights(self, val_dice):
        # Saving model weights
        if self.hparams.fast_dev_run == False and self.hparams.save_model == True:
            weights_path = "{}{}.pt".format(self.hparams.model_save_dir, self.hparams.experiment_name)
            torch.save(self.model.state_dict(), weights_path)
            print("\nSaved weights. val_dice: {:.4f}".format(val_dice))
        return
    
    # --- TEMP SWA FIX ---
    # Source: https://github.com/Lightning-AI/lightning/issues/17245
    def on_train_epoch_start(self) -> None:
        if self.hparams.swa == True and self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False

    
    def on_train_end(self):
        # Makes model deserializable w/out PL module
        if self.hparams.fast_dev_run == False and self.hparams.save_model == True:
            weights_path = "{}{}.pt".format(self.hparams.model_save_dir, self.hparams.experiment_name)  
            os.system("CUDA_VISIBLE_DEVICES="" python ./pl_scripts/convert_weights.py --weights_path={} --model_name={} --decoder_type={} --mask_downsample={} --experiment_name={}"
                    .format(weights_path, self.hparams.model_name, self.hparams.decoder_type, self.hparams.mask_downsample, self.hparams.experiment_name))
        return