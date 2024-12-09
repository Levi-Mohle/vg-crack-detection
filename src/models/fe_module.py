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
        FE: torch.nn.Module,
        frozen_FE: torch.nn.Module,
        unet: torch.nn.Module,
        unet_dict: DictConfig,
        noise_scheduler,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        compile,
        target: int,
        paths: DictConfig,
    ):
        """Feature extractor.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['net', 'criterion'])        

        # Pre-trained wide ResNet
        self.feature_extractor          = FE
        self.criterion                  = criterion

        # Freeze models
        self.frozen_feature_extractor   = frozen_FE
        self.frozen_feature_extractor.eval()
        for param in self.frozen_feature_extractor.parameters():
            param.requires_grad= False
            
        # Trained Unet model
        self.unet_dict                  = unet_dict
        self.unet_model                 = unet 
        checkpoint = torch.load(self.unet_dict.ckpt_path)

        # Fix issue with dict names
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint["state_dict"].items()}
        self.unet_model.load_state_dict(state_dict)
        
        self.noise_scheduler            = noise_scheduler
        self.unet_model.eval() 
        for param in self.unet_model.parameters():
            param.requires_grad= False   
        
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
        
        x0 = self.reconstruction(input, target)
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
        Tc = self.unet_dict.num_condition_steps

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
            e_hat = e - self.unet_dict.condition_weight * torch.sqrt(1-alpha_prod) * (yt-xt)
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
            self.unet_model = torch.compile(self.unet_model)

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