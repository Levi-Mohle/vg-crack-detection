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

# For flowmatching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

class ClassConditionedFlowMatchingLitModule(LightningModule):
    def __init__(
        self, 
        unet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        FM_param: DictConfig,
        OT,
        compile,
        paths: DictConfig,
    ):
        """Flow matching.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet'])

        self.vf              = unet.to(self.device)

        # Configure FM related parameters dict
        self.FM_param          = FM_param

        # instantiate an affine path object
        self.path = AffineProbPath(scheduler=CondOTScheduler())

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

        if self.FM_param.save_reconstructs:
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

    def forward(self, x, t, y=None):
        # Convert class labels to Long Tensor for embedding
        if y != None:
            y = y.type(torch.LongTensor).to(self.device)
        return self.vf(x=x, timesteps=t, y=y)
    
    def select_mode(self, batch, mode):
        if mode == "both":
            x = torch.cat((batch[0], batch[1]), dim=1).to(torch.float)
        elif mode == "height":
            x = batch[1].to(torch.float)
        elif mode == "rgb":
            x = batch[0].to(torch.float)
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
        # x = self.encode_data(batch, self.FM_param.mode)
        x = self.select_mode(batch, self.FM_param.mode)   
        y = batch[self.FM_param.target]     
        loss = self.conditional_flow_matching_loss(x, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.FM_param.mode)
        x = self.select_mode(batch, self.FM_param.mode)
        y = batch[self.FM_param.target] 
        loss = self.conditional_flow_matching_loss(x, y)
        self.log("val/loss", loss, prog_bar=True)

         # Only sample every n epochs
        if (self.current_epoch % self.FM_param.plot_n_epoch == 0) \
            & (self.current_epoch != 0):
            # Pick the second last batch (which is full)
            if (x.shape[0] == self.FM_param.batch_size) or (batch_idx == 0):
                self.last_val_batch = x

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])
        if (self.current_epoch % self.FM_param.plot_n_epoch == 0) \
            & (self.current_epoch != 0): # Only sample every n epochs
            plot_loss(self, skip=2)
            
            reconstructs = self.get_labeled_reconstructions(self.last_val_batch)
            self.last_val_batch = [self.last_val_batch, reconstructs]
            
            if self.FM_param.latent:
                self.last_val_batch[0] = self.decode_data(self.last_val_batch[0], 
                                                           self.FM_param.mode) 
                for i in range(2): 
                    self.last_val_batch[1][i] = self.decode_data(self.last_val_batch[1][i], self.FM_param.mode)
                    
            if self.FM_param.mode == "both":
                class_reconstructs_2ch(self, 
                                       self.last_val_batch[0],
                                       self.last_val_batch[1], 
                                       self.FM_param.plot_ids)
                
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
        # x = self.encode_data(batch, self.FM_param.mode)
        x = self.select_mode(batch, self.FM_param.mode)
        y = batch[self.FM_param.target]
        self.shape  = x.shape
        loss        = self.conditional_flow_matching_loss(x, y)
        self.log("test/loss", loss, prog_bar=True)

        if self.FM_param.ood:
            # Calculate reconstruction loss used for OOD-detection
            # losses = self.reconstruction_loss(x, reconstruct, reduction='batch')
            # self.test_losses.append(losses)
            # self.test_labels.append(y)
            pass

        # Pick the last full batch or
        if (x.shape[0] == self.FM_param.batch_size) or (batch_idx == 0):
            reconstructs = self.get_labeled_reconstructions(x)     
            self.last_test_batch = [x, reconstructs, y]

        if self.FM_param.save_reconstructs:
            if self.FM_param.latent:
                # self.vae.to("cpu")
                # self.last_test_batch = [self.last_test_batch[0].cpu(),
                #                        self.last_test_batch[1].cpu(),
                #                        self.last_test_batch[2].cpu()]
                self.last_test_batch[0] = self.decode_data(self.last_test_batch[0], self.FM_param.mode).cpu()
                self.last_test_batch[1] = self.decode_data(self.last_test_batch[1], self.FM_param.mode).cpu()
                self.last_test_batch[2] = self.last_test_batch[2].cpu()
            save_anomaly_maps(self.reconstruct_dir, self.last_test_batch)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Visualizations
        # Save last batch for visualization

        plot_loss(self, skip=1)
        
        if self.FM_param.latent:
            self.last_test_batch[0] = self.decode_data(self.last_test_batch[0], self.FM_param.mode)
            for i in range(2): 
                self.last_test_batch[1][i] = self.decode_data(self.last_test_batch[1][i], self.FM_param.mode)
        
        if self.FM_param.mode == "both":
            class_reconstructs_2ch(self, 
                                   self.last_test_batch[0],
                                   self.last_test_batch[1], 
                                   self.FM_param.plot_ids)

        if self.FM_param.ood:
            plot_histogram(self)

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def conditional_flow_matching_loss(self, x_1, y):
        '''
        Conditional flow matching loss
        :param x: input image
        '''
        t           = torch.rand(x_1.shape[0], device=self.device)
        
        # Randomly change class labels to n + 1 for Classifier Free Guidance
        indices     = torch.randperm(x_1.shape[0])[:int(self.FM_param.dropout_prob*x_1.shape[0])]
        y[indices]  = self.FM_param.n_classes - 1

        # Generate noise
        x_0       = torch.randn_like(x_1, device=self.device)

        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        
        # Predict vector field
        vt = self(path_sample.x_t, path_sample.t, y)
        # Actual vector field 
        ut = path_sample.dx_t

        return (vt - ut).square().mean()

    @torch.no_grad()
    def sample(self, n_samples, y):
        '''
        Sample images
        :param n_samples: number of samples
        '''
        x_0 = torch.randn(n_samples, self.shape[1], self.shape[2], self.shape[3], device=self.device)
        
        # Configure guidance_strength for Classifier Free Guidance
        w = self.FM_param.guidance_strength

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **extras):
                # Define label vector for unconditional case
                y_unconditional = (self.FM_param.n_classes - 1) * torch.ones(x_0.shape[0], device=self.device)
                return (1-w) * self.model(x, t, y_unconditional) + w * self.model(x, t, y)

        wrapped_vf = WrappedModel(self.vf)
        
        solver = ODESolver(velocity_model=wrapped_vf)
        sol    = solver.sample(x_init=x_0,
                                method="midpoint",
                                step_size=self.FM_param.step_size,
                                return_intermediates=False)

        return sol
    
    @torch.no_grad()
    def reconstruction(self, x_1, y):
        
        # Required for class embedding
        y = y.type(torch.LongTensor).to(self.device)

        tstart = 1 - self.FM_param.reconstruct
        T = torch.linspace(tstart, 1, 10).to(self.device)

        x_0 = torch.rand_like(x_1, device=self.device)
        x_t = x_1 * tstart + x_0 * (1-tstart)
        
        # Configure guidance_strength for Classifier Free Guidance
        w = self.FM_param.guidance_strength
        unknown_class = self.FM_param.n_classes - 1

        device = self.device
        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.LongTensor, **extras):
                # Define label vector for unconditional case
                t = t * torch.ones(x_1.shape[0], device=device)
                y_unconditional = (unknown_class * torch.ones(x_1.shape[0])).type(torch.LongTensor).to(device)
                return (1-w) * self.model(x, t, y_unconditional) + w * self.model(x, t, y)
        
        wrapped_vf = WrappedModel(self.vf)
        
        solver = ODESolver(velocity_model=wrapped_vf)
        sol    = solver.sample(time_grid=T,
                                x_init=x_t,
                                method="midpoint",
                                step_size=self.FM_param.step_size,
                                return_intermediates=False,
                                y = y)

        return sol

    def get_labeled_reconstructions(self, x):
        reconstructs = []
        reconstructs.append(self.reconstruction(x, y=torch.zeros(x.shape[0], 
                                                                device=self.device)))
        reconstructs.append(self.reconstruction(x, y=torch.ones(x.shape[0], 
                                                                device=self.device)))
        return reconstructs
        
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
            self.vf = torch.compile(self.vf)

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
    _ = ClassConditionedFlowMatchingLitModule(None, None, None, None, None)