import torch
import numpy as np
import os
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig
from src.models.components.utils.evaluation import *


class DeepSVDDLitModule(LightningModule):
    def __init__(
        self, 
        net: torch.nn.Module,
        center: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        dSVDD_param: DictConfig,
        compile,
        paths: DictConfig,
    ):
        """ImageFlow.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.net.to(self.device)

        # Configure dSVDD related parameters dict
        self.dSVDD_param = dSVDD_param

        self.center = center
        self.R = self.dSVDD_param.R

        if self.dSVDD_param.latent:
            self.vae =  AutoencoderKL.from_pretrained(self.dSVDD_param.pretrained,
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
        
        self.train_loss     = MeanMetric()
        self.val_loss       = MeanMetric()
        self.test_loss      = MeanMetric()

        # Used for inspecting learning curve
        self.train_epoch_loss   = []
        self.val_epoch_loss     = []

        # Used for classification 
        self.test_losses = []
        self.test_labels = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.dSVDD_param.mode)
        x_rep = self(x)
        # Compute distance to the representation center
        loss = self.compute_loss(x_rep)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def compute_loss(self, encoded_vectors):
        dist = torch.norm(encoded_vectors - self.center.c, dim=1)
        return dist.mean()
        
    def validation_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.dSVDD_param.mode)
        x_rep = self(x)
        # Compute distance to the representation center
        loss = self.compute_loss(x_rep)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        if (self.current_epoch % self.dSVDD_param.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample once per 5 epochs
            plot_loss(self, skip=2)

    def test_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.dSVDD_param.mode)
        y = batch[self.dSVDD_param.target]
        x_rep = self(x)
        # Compute distance to the representation center
        loss = self.compute_loss(x_rep)
        self.log("test/loss", loss, prog_bar=True)

        if self.dSVDD_param.ood:
            # Calculate reconstruction loss used for OOD-detection
            self.test_losses.append(torch.norm(x_rep - self.center.c, dim=1))
            self.test_labels.append(y)
     
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        plot_loss(self, skip=1)

        if self.dSVDD_param.ood:
            plot_histogram(self)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def select_mode(self, batch, mode):
        if mode == "both":
                x = torch.cat((batch[0], batch[1]), dim=1).to(torch.float)
        elif mode == "height":
            x = batch[1].to(torch.float)
        elif mode == "rgb":
            x = batch[0].to(torch.float)
        return x

    def encode_data(self, batch, mode):
        if self.dSVDD_param.latent:
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
            x = self.select_mode(batch,mode)
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
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

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
    _ = DeepSVDDLitModule(None, None, None, None, None)