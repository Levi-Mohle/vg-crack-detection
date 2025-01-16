import torch
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
import numpy as np
import os
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

class FlowMatchingLitModule(LightningModule):
    def __init__(
        self, 
        unet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        FM_param: DictConfig,
        compile,
        paths: DictConfig,
    ):
        """Flow matching.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet'])

        self.unet              = unet

        # Configure FM related parameters dict
        self.FM_param          = FM_param

        if self.FM_param.latent:
            self.vae =  AutoencoderKL.from_pretrained(self.FM_param.pretrained,
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
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Used for inspecting learning curve
        self.train_epoch_loss   = []
        self.val_epoch_loss     = []

        # Used for classification 
        self.test_losses = []
        self.test_labels = []

    def forward(self, x, t):
        return self.unet(x, t)
    
    def select_mode(self, batch, mode):
        if mode == "both":
            x = torch.cat((batch[0], batch[1]), dim=1)
        elif mode == "height":
            x = batch[1]
        elif mode == "rgb":
            x = batch[0]
        return x
    
    def encode_data(self, batch, mode):
        if self.FM_param.latent:
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
            x = self.select_mode(batch, mode)
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
        x = self.encode_data(batch, self.FM_param.mode)
        
        loss = self.conditional_flow_matching_loss(x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.encode_data(batch, self.FM_param.mode)
        loss = self.conditional_flow_matching_loss(x)
        self.log("val/loss", loss, prog_bar=True)

        # Reconstruct test samples
        reconstruct = self.reconstruction(x)

        # Pick the second last batch (which is full)
        if (x.shape[0] == self.FM_param.batch_size) or (batch_idx == 0):
            x = self.select_mode(batch, self.FM_param.mode)
            self.last_val_batch = [x, reconstruct]

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        if (self.current_epoch % self.FM_param.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample every n epochs
            plot_loss(self, skip=2)
            if self.FM_param.latent:
                self.last_val_batch[1] = self.decode_data(self.last_val_batch[1], 
                                                           self.FM_param.mode)    
            if self.FM_param.mode == "both":
                self.visualize_reconstructs_2ch(self.last_val_batch[0], 
                                                self.last_val_batch[1],  
                                                self.FM_param.plot_ids)
            else:
                self.visualize_reconstructs_1ch(self.last_val_batch[0], 
                                                self.last_val_batch[1], 
                                                self.last_val_batch[2])
                
    def reconstruction_loss(self, x, reconstruct, reduction=None):
        if reduction == None:
            chl_loss = (x - reconstruct)**2
        elif reduction == 'batch':
            chl_loss = torch.mean((x - reconstruct)**2, dim=(2,3))

        if self.FM_param.mode == "both":
            return (chl_loss[:,0] + self.FM_param.wh * chl_loss[:,1]).unsqueeze(1)
        else:
            return chl_loss
                
    def test_step(self, batch, batch_idx):
        x = self.encode_data(batch, self.FM_param.mode)
        y = batch[self.FM_param.target]
        self.shape = x.shape
        loss    = self.conditional_flow_matching_loss(x)
        self.log("test/loss", loss, prog_bar=True)

        reconstruct = self.reconstruction(x)
        losses = self.reconstruction_loss(x, reconstruct, reduction="batch")
        self.last_test_batch = [x, reconstruct, y]

        if self.FM_param.ood:
            # Calculate reconstruction loss used for OOD-detection
            losses = self.reconstruction_loss(x, reconstruct, reduction='batch')
            self.test_losses.append(losses)
            self.test_labels.append(y)

        # Pick the last full batch or
        if (x.shape[0] == self.FM_param.batch_size) or (batch_idx == 0):
            x = self.select_mode(batch, self.FM_param.mode)
            self.last_test_batch = [x, reconstruct, y]
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Visualizations
        # Save last batch for visualization
        if self.FM_param.latent:
            self.last_test_batch[1] = self.decode_data(self.last_test_batch[1], self.FM_param.mode)
            
        plot_loss(self, skip=1)
        if self.FM_param.mode == "both":
            self.visualize_reconstructs_2ch(self.last_test_batch[0], 
                                            self.last_test_batch[1], 
                                            self.FM_param.plot_ids)
        else:
            self.visualize_reconstructs_1ch(self.last_test_batch[0], 
                                            self.last_test_batch[1], 
                                            self.last_test_batch[2])

        if self.FM_param.ood:
            plot_histogram(self)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def conditional_flow_matching_loss(self, x):
        '''
        Conditional flow matching loss
        :param x: input image
        '''
        sigma_min = self.FM_param.sigma_min
        t = torch.rand(x.shape[0], device=self.device)
        noise = torch.randn_like(x)

        x_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * noise + t[:, None, None, None] * x
        optimal_flow = x - (1 - sigma_min) * noise
        predicted_flow = self(x_t, t).sample

        return (predicted_flow - optimal_flow).square().mean()

    @torch.no_grad()
    def sample(self, n_samples):
        '''
        Sample images
        :param n_samples: number of samples
        '''
        x_0 = torch.randn(n_samples, self.shape[1], self.shape[2], self.shape[3], device=self.device)

        def f(t: float, x):
            return self(x, torch.full(x.shape[:1], t, device=self.device)).sample
        
        if self.FM_param.solver_lib == 'torchdiffeq':
            if self.FM_param.solver == 'euler' or self.FM_param.solver == 'rk4' or self.FM_param.solver == 'midpoint' or self.FM_param.solver == 'explicit_adams' or self.FM_param.solver == 'implicit_adams':
                samples = odeint(f, x_0, t=torch.linspace(0, 1, 2).to(self.device), options={'step_size': self.FM_param.step_size}, method=self.FM_param.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_0, t=torch.linspace(0, 1, 2).to(self.device), method=self.FM_param.solver, options={'max_num_steps': 1//self.FM_param.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[1]
        else:
            t=0
            for i in tqdm(range(int(1/self.FM_param.step_size)), desc='Sampling', leave=False):
                v = self(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                x_0 = x_0 + self.FM_param.step_size * v
                t += self.FM_param.step_size
            samples = x_0
        
        if self.vae is not None:
            samples = self.vae.decode(samples / 0.18215).sample
        samples = samples*0.5 + 0.5
        samples = samples.clamp(0, 1)

        return samples
    
    @torch.no_grad()
    def reconstruction(self, x):
        
        sigma_min = self.FM_param.sigma_min
        tstart = 1 - self.FM_param.reconstruct
        e = torch.rand_like(x, device=self.device)
        
        xt = (1-(1-sigma_min)*tstart)*e + x*tstart
        
        def f(t: float, x):
            return self(x, torch.full(x.shape[:1], t, device=self.device)).sample
        
        if self.FM_param.solver_lib == 'torchdiffeq':
            if self.FM_param.solver == 'euler' or self.FM_param.solver == 'rk4' or self.FM_param.solver == 'midpoint' \
            or self.FM_param.solver == 'explicit_adams' or self.FM_param.solver== 'implicit_adams':
                
                reconstruct = odeint(f, xt, t=torch.linspace(tstart, 1, 2).to(self.device), options={'step_size': self.FM_param.step_size}, \
                                 method=self.FM_param.solver, rtol=1e-5, atol=1e-5)
            else:
                reconstruct = odeint(f, xt, t=torch.linspace(tstart, 1, 2).to(self.device), method=self.FM_param.solver, \
                                 options={'max_num_steps': 1//self.FM_param.step_size}, rtol=1e-5, atol=1e-5)
            reconstruct = reconstruct[1]
        else:
            t=tstart
            for i in range(int(self.FM_param.reconstruct*(1/self.FM_param.step_size))):
                v = self(xt, torch.full(xt.shape[:1], t, device=self.device)).sample
                xt = xt + self.FM_param.step_size * v
                t += self.FM_param.step_size
            reconstruct = xt
        
        # if self.vae is not None:
        #     reconstruct = self.vae.decode(reconstruct / 0.18215).sample
        # reconstruct = reconstruct*0.5 + 0.5
        # reconstruct = reconstruct.clamp(0, 1)

        return reconstruct
    
    @torch.no_grad()
    def visualize_samples(self, x):
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
    
    def min_max_normalize(self, x, dim=(0,2,3)):
        min_val = x.amin(dim=dim, keepdim=True)
        max_val = x.amax(dim=dim, keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-8)
        
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
    
    def visualize_reconstructs_2ch(self, x, reconstruct, plot_ids):
        # Convert back to [0,1] for plotting
        x = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        if self.FM_param.latent:
            x_gray = rgb_to_grayscale(x[:,:3])
            x = torch.cat((x_gray, x[:,3:]), dim=1)
            
            reconstruct_gray = rgb_to_grayscale(reconstruct[:,:3])
            reconstruct = torch.cat((reconstruct_gray, reconstruct[:,3:]), dim=1)
            
        # Calculate pixel-wise squared error per channel + normalize
        error_idv = ((x - reconstruct)**2).cpu()
        # error_idv = self.min_max_normalize(error_idv, dim=(2,3))

        # Calculate pixel-wise squared error combined + normalize
        error_comb = self.reconstruction_loss(x, reconstruct, reduction=None).cpu()
        # error_comb = self.min_max_normalize(error_comb, dim=(2,3))
        
        img = [self.min_max_normalize(x, dim=(2,3)).cpu(), self.min_max_normalize(reconstruct, dim=(2,3)).cpu(), error_idv, error_comb]
        extent = [0,4,0,4]
        for i in plot_ids:
            fig = plt.figure(constrained_layout=True, figsize=(15,7))
            gs = GridSpec(2, 4, figure=fig, width_ratios=[1.08,1,1.08,1.08], height_ratios=[1,1], hspace=0.05, wspace=0.2)
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
            im1 = ax1.imshow(img[0][i,0], extent=extent, vmin=0, vmax=1)
            ax1.set_yticks([0,1,2,3,4])
            ax1.tick_params(axis='both', which='both', labelbottom=False, labelleft=True)
            ax1.set_title("Original sample", fontsize =self.fs)
            ax1.set_ylabel("Y [mm]")
            ax1.text(-0.3, 0.5, "Gray-scale", fontsize= self.fs, rotation=90, va="center", ha="center", transform=ax1.transAxes)
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im1, cax=cax1)

            im2 = ax2.imshow(img[1][i,0], extent=extent, vmin=0, vmax=1)
            ax2.set_yticks([0,1,2,3,4])
            ax2.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            ax2.set_title("Reconstructed sample", fontsize =self.fs)
            
            im3 = ax3.imshow(img[2][i,0], extent=extent, vmin=0)
            ax3.set_yticks([0,1,2,3,4])
            ax3.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            ax3.set_title("Anomaly map individual", fontsize =self.fs)
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im3, cax=cax3)

            im4 = ax4.imshow(img[0][i,1], extent=extent, vmin=0, vmax=1)
            ax4.set_yticks([0,1,2,3,4])
            ax4.set_xlabel("X [mm]")
            ax4.set_ylabel("Y [mm]")
            ax4.text(-0.3, 0.5, "Height", fontsize= self.fs, rotation=90, va="center", ha="center", transform=ax4.transAxes)
            divider = make_axes_locatable(ax4)
            cax4 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im4, cax=cax4)

            im5 = ax5.imshow(reconstruct[i,1].cpu(), extent=extent)
            ax5.set_yticks([0,1,2,3,4])
            ax5.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax5.set_xlabel("X [mm]")

            im6 = ax6.imshow(img[2][i,1], extent=extent, vmin=0)
            ax6.set_yticks([0,1,2,3,4])
            ax6.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax6.set_xlabel("X [mm]")
            divider = make_axes_locatable(ax6)
            cax6 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im6, cax=cax6)

            # Span whole column
            im7 = ax7.imshow(img[3][i,0], extent=extent, vmin=0)
            ax7.set_title("Anomaly map combined", fontsize =self.fs)
            ax7.set_yticks([0,1,2,3,4])
            ax7.set_xlabel("X [mm]")
            ax7.set_ylabel("Y [mm]")

            # for ax in axs:
            #     ax.axis("off")

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
    _ = FlowMatchingLitModule(None, None, None, None, None)