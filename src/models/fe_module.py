import torch
import torch.nn as nn
from diffusers import UNet2DModel
from torchvision import transforms
import numpy as np
import os
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig

class FeatureExtractorLitModule(LightningModule):
    def __init__(
        self, 
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        compile,
        unet: DictConfig,
        noise_scheduler,
        target: int,
        num_condition_steps: int,
        condition_weight: float,
        paths: DictConfig,
    ):
        """Feature extractor.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['net', 'criterion'])

        # Unet dict
        self.unet                       = unet

        # Pre-trained wide ResNet
        self.feature_extractor          = net
        self.frozen_feature_extractor   = net

        # Trained Unet model
        self.unet_model                 = UNet2DModel.from_pretrained(self.unet.ckpt_path)
        self.noise_scheduler            = noise_scheduler
        self.criterion                  = criterion

        # Enable training
        self.feature_extractor.train()

        # Freeze models
        self.frozen_feature_extractor.eval()
        self.unet.eval()

        self.transform = transforms.Compose([
                    transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        # Define which dimension has the target
        self.target = target

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
        half_batch_size = batch[0].shape[0] //2

        target = batch[0][:half_batch_size]
        input = batch[0][half_batch_size:]  
        
        x0 = self.reconstruction(input, target)[-1]
        x0 = self.transform(x0)
        target = self.transform(target)

        reconst_fe  = self.feature_extractor(x0)
        target_fe   = self.feature_extractor(target)

        target_frozen_fe = self.frozen_feature_extractor(target)
        reconst_frozen_fe = self.frozen_feature_extractor(x0)

        loss = self.criterion(reconst_fe, target_fe, target_frozen_fe,reconst_frozen_fe)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass
            
    def test_step(self, batch, batch_idx):
        pass
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()
    
    def reconstruction(self, x, y):
        Tc = self.unet.num_condition_steps

        # Start with adding noise at timestep Tc
        t = torch.tensor([Tc] * x.shape[0], device=self.device)
        xt = self.noise_scheduler.add_noise(x, torch.randn_like(x), t)

        # Implementation from Mousakha et al, 2023
        for timestep in range(Tc, 0, -1):
            t = torch.tensor([timestep] * x.shape[0], device=self.device)
            e = self.unet_model(xt, t)['sample']
            
            var         = self.noise_scheduler._get_variance(timestep)
            alpha_prod       = self.noise_scheduler.alphas_cumprod[timestep]
            alpha_prod_prev  = self.noise_scheduler.alphas_cumprod[timestep-1]
            
            yt = self.noise_scheduler.add_noise(y, e, t)
            e_hat = e - self.unet.condition_weight * torch.sqrt(1-alpha_prod) * (yt-xt)
            ft = (xt - torch.sqrt(1-alpha_prod)*e_hat) / torch.sqrt(alpha_prod)
            xt = torch.sqrt(alpha_prod_prev) * ft + torch.sqrt(1-alpha_prod_prev-var) * e_hat + torch.sqrt(var) * torch.randn_like(xt)

        return xt
        
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
    _ = FeatureExtractorLitModule(None, None, None, None, None)