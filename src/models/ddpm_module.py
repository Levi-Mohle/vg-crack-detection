import torch
import numpy as np
import os
from datetime import datetime
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        self.mode               = DDPM_param.mode
        self.target             = DDPM_param.target
        self.reconstruct        = DDPM_param.reconstruct
        self.wh                 = DDPM_param.wh
        self.plot_n_epoch       = DDPM_param.plot_n_epoch
        self.plot_ids           = DDPM_param.plot_ids
        self.encode             = DDPM_param.encode
        self.pretrained         = DDPM_param.pretrained
        self.ood                = DDPM_param.ood
        self.max_epochs         = DDPM_param.max_epochs
        self.batch_size         = DDPM_param.batch_size
        self.use_cond           = DDPM_param.use_cond
        self.condition_weight   = DDPM_param.condition_weight
        self.skip_steps         = DDPM_param.skip_steps
        self.eta                = DDPM_param.eta
        self.save_reconstructs  = DDPM_param.save_reconstructs
        self.plot               = DDPM_param.plot
        self.win_size           = DDPM_param.win_size

        if self.encode:
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

    def forward(self, x, steps=None, y=None):
        noise = torch.randn(x.shape, device=self.device)
        if steps == None:
            steps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        else:
            steps = torch.tensor([steps] * x.shape[0], device=self.device)
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps)
        residual = self.model(noisy_images, steps, y=y)
        
        return residual, noise

    def select_mode(self, batch, mode):
        if mode == "both":
                x = torch.cat((batch[0], batch[1]), dim=1).to(torch.float)
        elif mode == "height":
            x = batch[1].to(torch.float)
        elif mode == "rgb":
            x = batch[0].to(torch.float)

        # Return true label
        y = batch[self.target]
        return x,y
        
    def encode_data(self, batch, mode):
        if self.encode:
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
            x, _ = self.select_mode(batch,mode)
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
        x, _ = self.select_mode(batch, self.mode)
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.mode)
        x, y = self.select_mode(batch, self.mode)
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("val/loss", loss, prog_bar=True)

        if (self.current_epoch % self.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample once per 5 epochs
            # Pick the second last batch (which is full)
            if (x.shape[0] == self.batch_size) or (batch_idx == 0):
                self.last_val_batch = [x, 0, y]

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        if (self.current_epoch % self.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample once per 5 epochs
            plot_loss(self, skip=2)
            
            # x, y = self.last_val_batch
            # _, reconstruct = self.partial_diffusion(x, self.reconstruct)
            # if self.encode:
            #     self.last_val_batch[0] = self.decode_data(x, self.mode)    
            #     self.last_val_batch[1] = self.decode_data(reconstruct, self.mode)    
            # else:
            #     self.last_val_batch[1] = reconstruct
            
            # if self.mode == "both":
            #     visualize_reconstructs_2ch(self, 
            #                                    self.last_test_batch[0], 
            #                                    self.last_test_batch[1],
            #                                    self.last_test_batch[2],
            #                                    self.plot_ids
            #                                    )
            # else:
            #     # self.visualize_reconstructs_1ch(self.last_val_batch[0], 
            #     #                                 self.last_val_batch[1], 
            #     #                                 self.plot_ids)
            #     pass
        
    def reconstruction_loss(self, x, reconstruct, reduction=None):
        if reduction == None:
            chl_loss = (x - reconstruct)**2
        elif reduction == 'batch':
            chl_loss = torch.mean((x - reconstruct)**2, dim=(2,3))

        if self.mode == "both":
            return (chl_loss[:,0] + self.wh * chl_loss[:,1]).unsqueeze(1)
        else:
            return chl_loss
        
    def test_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.mode)
        x, y = self.select_mode(batch, self.mode)
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("test/loss", loss, prog_bar=True)

        # Reconstruct test samples
        _, reconstruct = self.partial_diffusion(x, self.reconstruct)

        if self.ood:
            # Calculate reconstruction loss used for OOD-detection
            x0 = self.decode_data(x, self.mode)
            x1 = self.decode_data(reconstruct, self.mode) # Only pick non-crack reconstructions
 
            # Convert rgb channels to grayscale and revert normalization to [0,1]
            x0, x1          = to_gray_0_1(x0), to_gray_0_1(x1)
            _, ood_score    = OOD_score(x0=x0, x1=x0, x2=x1)

            # Append scores
            self.test_losses.append(ood_score)
            self.test_labels.append(y)

        # Pick the last full batch or
        if (x.shape[0] == self.batch_size) or (batch_idx == 0):
            self.last_test_batch = [x, reconstruct, y]

        if self.save_reconstructs:
            if self.encode:
                self.last_test_batch[0] = self.decode_data(x, self.mode).cpu()
                self.last_test_batch[1] = self.decode_data(reconstruct, self.mode).cpu()
                self.last_test_batch[2] = y.cpu()
            save_reconstructions_to_h5(self.reconstruct_dir, self.last_test_batch, cfg=False) # TODO make cfg related to self.n_classes
            
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        
        plot_loss(self, skip=1)

        if self.ood:
            y_score = np.concatenate([t for t in self.test_losses]) # use t.cpu().numpy() if Tensor)
            y_true = np.concatenate([t.cpu().numpy() for t in self.test_labels]).astype(int)
            
            # Save OOD-scores and true labels for later use
            np.savez(os.path.join(self.log_dir, "labelsNscores"), y_true=y_true, y_scores=y_score)
            
            plot_histogram(y_score, y_true, save_dir = self.log_dir)
            plot_classification_metrics(y_score, y_true, save_dir=self.log_dir)
            
        if self.plot:
            if self.encode and not(self.save_reconstructs):
                self.last_test_batch[0] = self.decode_data(self.last_test_batch[0], self.mode)
                self.last_test_batch[1] = self.decode_data(self.last_test_batch[1], self.mode)
            
            if self.mode == "both":
                visualize_reconstructs_2ch(self = self, 
                                            x = self.last_test_batch[0], 
                                            reconstruct = self.last_test_batch[1],
                                            target = self.last_test_batch[2],
                                            plot_ids = self.plot_ids,
                                            ood = self.test_losses[-1] if self.test_losses[-1].shape[0] == self.batch_size else self.test_losses[-2], 
                                            )
            else:
                # self.visualize_reconstructs_1ch(self.last_test_batch[0], 
                #                                 self.last_test_batch[1], 
                #                                 self.plot_ids)
                pass # TODO fix 1ch plot for newer versions

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

        # Use conditional DDIM
        if self.use_cond:
            # Implementation from Mousakha et al, 2023

            # Define target
            y = x

            seq = range(1 , Tc+1, self.skip_steps)
            seq_next = [0] + list(seq[:-1])
            for index, (i,j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = torch.tensor([i] * x.shape[0], device=self.device)
                
                e, _ = self(xt, i)
                
                alpha_prod       = self.noise_scheduler.alphas_cumprod[i]
                alpha_prod_prev  = self.noise_scheduler.alphas_cumprod[j]
                sigma = self.eta * torch.sqrt((1 - alpha_prod / alpha_prod_prev) * (1 - alpha_prod_prev) / (1 - alpha_prod))
                
                yt = self.noise_scheduler.add_noise(y, e, t)
                
                e_hat = e - self.condition_weight * torch.sqrt(1-alpha_prod) * (yt-xt)
                ft = (xt - torch.sqrt(1-alpha_prod)*e_hat) / torch.sqrt(alpha_prod)
                
                xt = torch.sqrt(alpha_prod_prev) * ft + torch.sqrt(1-alpha_prod_prev-sigma**2) * e_hat + sigma * torch.randn_like(xt)

        else:
            for timestep in range(Tc, 0, -1):
                    t = torch.tensor([timestep] * x.shape[0], device=self.device)
                    e, _ = self(xt, timestep)
                    
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
    _ = DenoisingDiffusionLitModule(None, None, None, None, None)