import torch
import torch.nn as nn
import diffusers
import numpy as np
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig
from src.models.support_functions.evaluation import *

class DenoisingDiffusionLitModule(LightningModule):
    def __init__(
        self, 
        unet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        noise_scheduler,
        DDPM_param: DictConfig,
        compile,
        paths: DictConfig,
    ):
        """ImageFlow.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet', 'criterion'])

        self.model              = unet
        self.noise_scheduler    = noise_scheduler
        self.criterion          = criterion

        # Configure DDPM related parameters dict
        self.DDPM_param = DDPM_param

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

    def forward(self, x, steps=None):
        noise = torch.randn(x.shape, device=self.device)
        if steps == None:
            steps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        else:
            steps = torch.tensor([steps] * x.shape[0], device=self.device)
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps)
        residual = self.model(noisy_images, steps).sample
        
        return residual, noise

    def select_mode(batch, mode):
        if mode == "both":
            x = torch.cat(batch[0], batch[1])
        elif mode == "height":
            x = batch[1]
        elif mode == "rgb":
            x = batch[0]
        return x
        
    def training_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.DDPM_param.mode)
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.DDPM_param.mode)
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("val/loss", loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        # if (self.current_epoch % 3 == 0) & (self.current_epoch != 0): # Only sample once per 5 epochs
        #     x_hat = self.sample(num_samples=16)
        #     self.visualize_samples(x_hat)

    def reconstruction_loss(self, x, reconstruct, reduction=None):
        if reduction == None:
            chl_loss = (x - reconstruct)**2
        elif reduction == 'batch':
            chl_loss = torch.mean((x - reconstruct)**2, dim=(2,3))

        if self.DDPM_param.mode == "both":
            return (chl_loss[:,0] + self.DDPM_param.wh * chl_loss[:,1]).unsqueeze(1)
        else:
            return chl_loss
        
    def test_step(self, batch, batch_idx):
        x = self.select_mode(batch, self.DDPM_param.mode)
        y = batch[self.DDPM_param.target]
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("test/loss", loss, prog_bar=True)

        # Reconstruct test samples
        x, reconstruct = self.partial_diffusion(x, self.DDPM_param.reconstruct)
        # Calculate reconstruction loss used for OOD-detection
        losses = self.reconstruction_loss(x, reconstruct, reduction='batch')
        # Save last batch for visualization
        self.last_test_batch = [x, reconstruct, y]

        self.test_losses.append(losses)
        self.test_labels.append(y)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Sample from gaussian nosie
        # x_hat = self.sample(num_samples=16)
        
        # Visualizations
        # self.visualize_samples(x_hat)
        plot_loss(self)
        if self.DDPM_param.mode == "both":
            self.visualize_reconstructs_2ch(self.last_test_batch[0], self.last_test_batch[1], self.last_test_batch[2], self.DDPM_param.plot_ids)
        else:
            self.visualize_reconstructs_1ch(self.last_test_batch[0], self.last_test_batch[1], self.last_test_batch[2])
        plot_histogram(self)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def partial_diffusion(self, x, coef):
        # Define noise
        noise = torch.randn(x.shape, device=self.device)
        # Take fraction of steps for forward diffusion process
        Tc = int(coef * self.noise_scheduler.config.num_train_timesteps)
        Tc_tensor = torch.tensor([Tc] * x.shape[0], device=self.device)
        # Noise the samples
        xt = self.noise_scheduler.add_noise(x, noise, Tc_tensor)
        # Reconstruct the samples
        for timestep in range(Tc, 0, -1):
                t = torch.tensor([timestep] * x.shape[0], device=self.device)
                e = self.model(xt, t).sample
                
                alpha            = self.noise_scheduler.alphas[timestep]
                alpha_prod       = self.noise_scheduler.alphas_cumprod[timestep]
                alpha_prod_prev  = self.noise_scheduler.alphas_cumprod[timestep-1]
                sigma = torch.sqrt((1 - alpha_prod / alpha_prod_prev) * (1 - alpha_prod_prev) / (1 - alpha_prod))

                if timestep > 1:
                    xt = 1 / torch.sqrt(alpha) * (xt - (1-alpha)/torch.sqrt(1-alpha_prod) * e) \
                    + sigma * torch.randn_like(xt)
                else:
                    xt = 1 / torch.sqrt(alpha) * (xt - (1-alpha)/torch.sqrt(1-alpha_prod) * e) 
            
        reconstruct = xt
        
        return x, reconstruct
    
    def min_max_normalize(self, x):
        min_val = x.amin(dim=(0,2,3), keepdim=True)
        max_val = x.amax(dim=(0,2,3), keepdim=True)
        x_norm = x
        for i in range(x.shape[1]):
            x_norm[:,i,:,:] = (x[:,i,:,:] - min_val[i]) / (max_val[i] - min_val[i] + 1e-8)
        return x_norm
        
    def visualize_reconstructs_1ch(self, x, reconstruct, labels):
        # Convert back to [0,1] for plotting
        x = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        # Calculate pixel-wise squared error + normalize
        error = self.reconstruction_loss(x, reconstruct, reduction=None)
        error = self.min_max_normalize(error)

        # For plotting reasons
        x = x.permute(0,2,3,1).cpu()
        reconstruct = reconstruct.permute(0,2,3,1).cpu()
        
        img = [x, reconstruct, error]

        title = ["Original sample", "Reconstructed Sample", "Anomaly map"]
        vmax_e = torch.max(error).item()
        vmax_list = [1, 1, 1]
        for i in range(4):
            fig = plt.figure(constrained_layout=True, figsize=(11,9))
            # create 3x1 subfigs
            subfigs = fig.subfigures(nrows=3, ncols=1)
            for row, subfig in enumerate(subfigs):
                subfig.suptitle(title[row], fontsize = self.fs)
                # create 1x3 subplots per subfig
                axs = subfig.subplots(nrows=1, ncols=4)
                for col, ax in enumerate(axs):
                    im = ax.imshow(img[row][col+4*i], vmin=0, vmax=vmax_list[row])
                    ax.axis("off")
                    ax.set_title(f"Label: {labels[col+4*i]}")
                    if (row == 2) & (col == 0):
                        plt.colorbar(im, ax=ax)
                
                        
            plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs_{i}.png")
            fig.savefig(plt_dir)
            plt.close()
            # Send figure as artifact to logger
            # if self.logger.__class__.__name__ == "MLFlowLogger":
            #     self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)
    
    def visualize_reconstructs_2ch(self, x, reconstruct, labels, plot_ids):
        # Convert back to [0,1] for plotting
        x = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        # Calculate pixel-wise squared error per channel + normalize
        error_idv = (x - reconstruct)**2
        error_idv = self.min_max_normalize(error_idv)

        # Calculate pixel-wise squared error combined + normalize
        error_comb = self.reconstruction_loss(x, reconstruct, reduction=None)
        error_comb = self.min_max_normalize(error_comb)

        # For plotting reasons
        x = x.permute(0,2,3,1).cpu()
        reconstruct = reconstruct.permute(0,2,3,1).cpu()
        
        img = [x, reconstruct, error_idv, error_comb]

        for i in plot_ids:
            fig = plt.figure(constrained_layout=True, figsize=(11,11))
            gs = GridSpec(2, 4, figure=fig, width_ratios=[1,1,1,0.8])
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[0,2])
            ax4 = fig.add_subplot(gs[1,0])
            ax5 = fig.add_subplot(gs[1,1])
            ax6 = fig.add_subplot(gs[1,2])
            # Span whole column
            ax7 = fig.add_subplot(gs[:,3])
            axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            # Plot
            im1 = ax1.imshow(img[0][i,0])
            ax1.set_title("Original sample", fontsize =self.fs)
            ax1.text(-0.3, 0.5, "Gray-scale", fontsize= self.fs, rotation=90, va="center", ha="center", transform=ax1.transAxes)
            im2 = ax2.imshow(img[1][i,0])
            ax2.set_title("Reconstructed sample", fontsize =self.fs)
            im3 = ax3.imshow(img[2][i,0])
            ax3.set_title("Anomaly map individual", fontsize =self.fs)
            plt.colorbar(im3, ax=ax3)
            im4 = ax4.imshow(img[0][i,1])
            ax4.text(-0.3, 0.5, "Height", fontsize= self.fs, rotation=90, va="center", ha="center", transform=ax4.transAxes)
            im5 = ax5.imshow(img[1][i,1])
            im6 = ax6.imshow(img[2][i,1])
            plt.colorbar(im6, ax=ax6)
            # Span whole column
            im7 = ax7.imshow(img[3][i])
            ax7.set_title("Anomaly map combined", fontsize =self.fs)
            plt.colorbar(im7, ax=ax7)

            for ax in axs:
                ax.axis("off")
                        
            plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs_{i}.png")
            fig.savefig(plt_dir)
            plt.close()
            # Send figure as artifact to logger
            # if self.logger.__class__.__name__ == "MLFlowLogger":
            #     self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)

        
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
    _ = DenoisingDiffusionLitModule(None, None, None, None, None)