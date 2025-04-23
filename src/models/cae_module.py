from typing import Any, Dict, Tuple
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig

# Local imports
import src.models.components.utils.evaluation as evaluation
import src.models.components.utils.visualization as visualization

class ConvAutoEncoderLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        CAE_param: DictConfig,
        compile,
        paths: DictConfig,
    ):
        """ImageFlow.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model              = net
        self.criterion          = criterion

        self.CAE_param          = CAE_param

        self.target_index   = CAE_param.target_index
        self.wh             = CAE_param.wh
        self.mode           = CAE_param.mode
        self.plot_ids       = CAE_param.plot_ids
        self.ood            = CAE_param.ood

        # Specify fontsize for plots
        self.fs = 16

        self.log_dir = paths.log_dir
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
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
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.mode)
        x_hat = self(x)
        loss = self.criterion(x_hat, x, self.device)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.mode)
        x_hat = self(x)
        loss = self.criterion(x_hat, x, self.device)
        self.log("val/loss", loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
            
    def test_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.mode)
        y = batch[self.target_index]
        x_hat = self(x)
        loss = self.criterion(x_hat, x, self.device)
        self.log("test/loss", loss, prog_bar=True)

        losses = self.reconstruction_loss(x, x_hat, reduction='batch')
        self.last_test_batch = [x, x_hat, y]
        # In case you want to evaluate on just the MSE from the Unet
        # losses = self.criterion(residual, noise, self.device, reduction='none')

        self.test_losses.append(losses)
        self.test_labels.append(y)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        
        # Visualizations
        evaluation.plot_loss(self, skip=1)
        if self.mode == "both":
            visualization.visualize_reconstructs_2ch(self.last_test_batch[0], self.last_test_batch[1], self.last_test_batch[2], self.plot_ids)
        else:
            visualization.visualize_reconstructs_1ch(self.last_test_batch[0], self.last_test_batch[1], self.plot_ids)
        
        if self.ood:
            y_score = np.argmax(np.concatenate([t.cpu().numpy() for t in self.test_losses]), axis=1) # use t.cpu().numpy() if Tensor)
            y_true  = np.argmax(np.concatenate([t.cpu().numpy() for t in self.test_labels]).astype(int), axis=1)

            # Save OOD-scores and true labels for later use
            np.savez(os.path.join(self.log_dir, "0_labelsNscores"), y_true=y_true, y_scores=y_score)
            
            evaluation.plot_histogram(y_score, y_true, save_dir = self.log_dir)
            evaluation.plot_classification_metrics(y_score, y_true, save_dir=self.log_dir)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def select_mode(self, batch, mode):
        if mode == "both":
            x = torch.cat((batch[0], batch[1]), dim=1)
        elif mode == "height":
            x = batch[1]
        elif mode == "rgb":
            x = batch[0]
        return x
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
if __name__ == "__main__":
    _ = ConvAutoEncoderLitModule(None, None, None, None, None)