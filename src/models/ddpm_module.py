import torch
import torch.nn as nn
import diffusers
import numpy as np
import os
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig

class DenoisingDiffusionLitModule(LightningModule):
    def __init__(
        self, 
        unet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        noise_scheduler,
        compile,
        paths: DictConfig,
    ):
        """ImageFlow.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet'])

        self.model              = unet
        self.noise_scheduler    = noise_scheduler

        self.log_dir = paths.log_dir

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.test_losses = []
        self.test_labels = []

    def forward(self, x):
        noise = torch.randn(x.shape, device=self.device)
        steps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch[0])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch[0])
        self.log("val/loss", loss, prog_bar=True)

    @torch.no_grad()
    def sample(self, num_samples):
        img_size = self.model.config.sample_size
        channels = self.model.config.in_channels
        num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        x = torch.randn(num_samples, channels, img_size, img_size).to(self.device)

        for timestep in range(num_inference_steps, 0, -1):
            t = torch.tensor([timestep] * num_samples, device=self.device)
            noise_pred = self.model(x, t)
            x = x - (noise_pred[0] / num_inference_steps)
            
        x = (x + 1.) / 2.
        return torch.clamp(x, min=0, max=1)

    @torch.no_grad()
    def visualize_samples(self):
        # Sample
        x = self.sample(num_samples=16)
        # Create figure
        grid = make_grid(x, nrow=int(np.sqrt(x.shape[0])))
        plt.figure(figsize=(12,12))
        plt.imshow(grid.permute(1,2,0).cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_epoch_sample.png")
        plt.savefig(plt_dir)
        plt.close()
        # Send figure as artifact to logger
        if self.logger.__class__.__name__ == "MLFlowLogger":
            self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)
        # os.remove(image_path)
    
    def test_step(self, batch, batch_idx):
        loss = self(batch[0])
        self.log("test/loss", loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if (self.current_epoch % 3 == 0) & (self.current_epoch != 0): # Only sample once per 5 epochs
            self.visualize_samples()

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.visualize_samples()
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.unet = torch.compile(self.unet)
        
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

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
    _ = DenoisingDiffusionLitModule(None, None, None, None, None)