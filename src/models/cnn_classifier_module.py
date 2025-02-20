import torch
from diffusers.models import AutoencoderKL
import numpy as np
import os
from datetime import datetime
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig
from src.models.support_functions.evaluation import *
import tqdm

class CNNClassifierLitModule(LightningModule):
    def __init__(
        self, 
        cnn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cnn_param: DictConfig,
        compile,
        paths: DictConfig,
    ):
        """Flow matching.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet'])

        self.cnn              = cnn.to(self.device)

        # Configure FM related parameters dict
        self.n_classes      = cnn_param.n_classes
        self.mode           = cnn_param.mode 
        self.target         = cnn_param.target
        self.ood            = cnn_param.ood
        self.plot_n_epoch   = cnn_param.plot_n_epoch
        self.batch_size     = cnn_param.batch_size

        # if self.latent:
        #     self.vae =  AutoencoderKL.from_pretrained(self.pretrained,
        #                                               local_files_only=True,
        #                                               use_safetensors=True
        #                                              ).to(self.device)
        #     # Make sure to freeze parameters 
        #     for param in self.vae.parameters():
        #         param.requires_grad= False
        # else:
        #     self.vae = None

        # Specify fontsize for plots
        self.fs = 16

        self.log_dir = paths.log_dir
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Define loss criterion
        self.criterion = torch.nn.BCELoss().to(self.device)

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
        # Convert class labels to Long Tensor for embedding
        return self.cnn(x)
    
    def select_mode(self, batch, mode):
        if mode == "both":
            x = torch.cat((batch[0], batch[1]), dim=1).to(torch.float)
        elif mode == "height":
            x = batch[1].to(torch.float)
        elif mode == "rgb":
            x = batch[0].to(torch.float)
        
        y = self.one_hot_encode(batch[self.target])

        return x, y
    
    def one_hot_encode(self, y):
        binary_labels = y.long()
        one_hot = torch.zeros((y.shape[0],2), dtype=torch.float32, device=self.device)
        one_hot[torch.arange(y.shape[0]), binary_labels] = 1
        return one_hot
        
    def training_step(self, batch, batch_idx):
        x, y  = self.select_mode(batch, self.mode)     
        y_pred = self.cnn(x)
        loss   = self.criterion(y_pred, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.select_mode(batch, self.mode)
        y_pred = self.cnn(x)
        loss   = self.criterion(y_pred, y)
        self.log("val/loss", loss, prog_bar=True)

         # Only sample every n epochs
        if (self.current_epoch % self.plot_n_epoch == 0) \
            & (self.current_epoch != 0):
            # Pick the second last batch (which is full)
            if (x.shape[0] == self.batch_size) or (batch_idx == 0):
                self.last_val_batch = [x, y]

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        if (self.current_epoch % self.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample every n epochs
            plot_loss(self, skip=2)
         
    def test_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.mode)
        x, y = self.select_mode(batch, self.mode)
        y_pred = self.cnn(x)
        loss   = self.criterion(y_pred, y)
        self.log("test/loss", loss, prog_bar=True)

        # Pick the last full batch or
        if (x.shape[0] == self.batch_size) or (batch_idx == 0):
            # Reconstruct twice: with both 0 and 1 label

            if self.ood:
                # Append scores
                self.test_losses.append(y_pred)
                self.test_labels.append(y)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Visualizations
        # Save last batch for visualization

        plot_loss(self, skip=1)

        if self.ood:
            y_score = np.concatenate([t for t in self.test_losses]) # use t.cpu().numpy() if Tensor)
            y_true  = np.concatenate([t.cpu().numpy() for t in self.test_labels]).astype(int)
            save_loc = os.path.join(self.log_dir, "classification_metrics.txt")
            plot_confusion_matrix()
            # plot_histogram(y_score, y_true, save_loc, self=self)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()
    
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
            self.cnn = torch.compile(self.cnn)

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
    _ = CNNClassifierLitModule(None, None, None, None, None)