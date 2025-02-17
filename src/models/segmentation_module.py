import torch
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
import numpy as np
import os
from datetime import datetime
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms.functional import rgb_to_grayscale
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig
from src.models.support_functions.evaluation import *
import tqdm
from diffusers.models import UNet2DModel

class SegmentationLitModule(LightningModule):
    def __init__(
        self, 
        unet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        Seg_param: DictConfig,
        compile,
        paths: DictConfig,
    ):
        """Flow matching.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet'])

        self.unet              = unet.to(self.device)

        # Configure Segmentation related parameters dict
        self.latent             = Seg_param.latent
        self.pretrained         = Seg_param.pretrained
        self.batch_size         = Seg_param.batch_size
        self.save_reconstructs  = Seg_param.save_reconstructs
        self.plot_n_epoch       = Seg_param.plot_n_epoch
        self.target             = Seg_param.target
        self.mode               = Seg_param.mode
        self.plot               = Seg_param.plot
        self.plot_ids           = Seg_param.plot_ids

        if self.latent:
            self.vae =  AutoencoderKL.from_pretrained(self.pretrained,
                                                      local_files_only=True,
                                                      use_safetensors=True
                                                     ).to(self.device)
            # Make sure to freeze parameters 
            for param in self.vae.parameters():
                param.requires_grad= False
        else:
            self.vae = None

        # Specify fontsize for plots
        self.fs = 16

        self.log_dir = paths.log_dir
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        if self.save_reconstructs:
            time = datetime.today().strftime('%Y-%m-%d')
            self.reconstruct_dir = os.path.join(self.image_dir, time + "_reconstructs.h5")
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Used for inspecting learning curve
        self.train_epoch_loss   = []
        self.val_epoch_loss     = []

        # Used for classification 
        self.test_losses = []
        self.test_labels = []

    def forward(self, x):
        return self.unet(x)
    
    def select_mode(self, batch, mode):
        if mode == "both":
            x = torch.cat((batch[0], batch[1]), dim=1).to(torch.float)
        elif mode == "height":
            x = batch[1].to(torch.float)
        elif mode == "rgb":
            x = batch[0].to(torch.float)
        
        # Whether to use class information or not
        y = batch[2]
        return x, y
    
    def encode_data(self, batch, mode):
        if self.latent:
            if mode == "both":
                x1 = batch[0]
                x2 = torch.cat((batch[1], batch[1], batch[1]), dim=1)
                with torch.no_grad():
                    x1 = self.vae.encode(x1).latent_dist.sample().mul_(0.18215)
                    x2 = self.vae.encode(x2).latent_dist.sample().mul_(0.18215)
                x = torch.cat((x1, x2), dim=1)
            elif mode == "height":
                x = torch.cat((batch[1], batch[1], batch[1]), dim=1)
                with torch.no_grad():
                    x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            elif mode == "rgb":
                x = batch[0]
                with torch.no_grad():
                    x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            x, _ = self.select_mode(batch, mode)
        return x
    
    def decode_data(self, z, mode):
        if mode=="both":
            z1, z2 = z[:,:4], z[:,4:]
            x1 = self.vae.decode(z1/0.18215).sample
            
            x2 = self.vae.decode(z2/0.18215).sample
            # Extract only 1 channel
            x2 = x2[:,0].unsqueeze(1)
            return torch.cat((x1,x2), dim=1)
        elif mode=="rgb":
            x = self.vae.decode(z/0.18215).sample
            return x
        elif mode=="height":
            x = self.vae.decode(z/0.18215).sample
            x = x[:,0].unsqueeze(1)
            return x
        
    def training_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.mode)
        x, y    = self.select_mode(batch, self.mode)     
        y_pred  = self.unet(x)
        loss    = self.segmentation_loss(y, y_pred)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.mode)
        x, y    = self.select_mode(batch, self.mode)
        y_pred  = self.unet(x)
        loss    = self.segmentation_loss(y, y_pred)
        self.log("val/loss", loss, prog_bar=True)

         # Only sample every n epochs
        if (self.current_epoch % self.plot_n_epoch == 0) \
            & (self.current_epoch != 0):
            # Pick the second last batch (which is full)
            if (x.shape[0] == self.batch_size) or (batch_idx == 0):
                y_pred = self.unet(x)
                self.last_val_batch = [y_pred, y]

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        if (self.current_epoch % self.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample every n epochs
            plot_loss(self, skip=2)

            y_pred, y = self.last_val_batch

            if self.plot:
                if self.latent:
                    self.last_val_batch[0] = self.decode_data(y_pred, mode="height")
                    self.last_val_batch[1] = self.decode_data(y, mode="height")  
                        
                # if self.mode == "both":
                #     class_reconstructs_2ch(self, 
                #                            self.last_val_batch[0],
                #                            self.last_val_batch[1],
                #                            self.last_val_batch[2], 
                #                            self.plot_ids)
         
    def test_step(self, batch, batch_idx):
        x, y    = self.select_mode(batch, self.mode)
        y_pred  = self.unet(x)
        loss    = self.segmentation_loss(y, y_pred)
        self.log("test/loss", loss, prog_bar=True)

        # Pick the last full batch or
        if (x.shape[0] == self.batch_size) or (batch_idx == 0):
            self.last_test_batch = [y_pred, y]
                
        if self.save_reconstructs:
            if self.latent:
                self.last_test_batch[0] = self.decode_data(y_pred, mode="height").cpu()
                self.last_test_batch[1] = self.decode_data(y, mode="height").cpu()  
            save_reconstructions_to_h5(self.reconstruct_dir, self.last_test_batch, cfg=True)
     
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Visualizations

        plot_loss(self, skip=1)
        
        if self.plot:
            if self.latent and not(self.save_reconstructs):
                self.last_test_batch[0] = self.decode_data(self.last_test_batch[0], self.mode)
                self.last_test_batch[1] = self.decode_data(self.last_test_batch[1], self.mode)
            
            # if self.mode == "both":
            #     class_reconstructs_2ch(self, 
            #                         self.last_test_batch[0],
            #                         self.last_test_batch[1],
            #                         self.last_test_batch[2],
            #                         self.plot_ids)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def segmentation_loss(self, y, y_pred):
        '''
        Conditional flow matching loss
        :param x: input image
        '''
        return (y-y_pred).square().mean()
    
    def min_max_normalize(self, x, dim=(0,2,3)):
        min_val = x.amin(dim=dim, keepdim=True)
        max_val = x.amax(dim=dim, keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-8)
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.unet = torch.compile(self.unet)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': 
                {'scheduler': scheduler, 
                 'interval': 'epoch', 
                 'frequency': 1
                },
            }
        return {"optimizer": optimizer}
    
if __name__ == "__main__":
    _ = SegmentationLitModule(None, None, None, None, None)