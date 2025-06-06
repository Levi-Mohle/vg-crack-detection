import torch
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
import numpy as np
import os
from datetime import datetime
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms.functional import rgb_to_grayscale
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig

# Local imports
import src.models.components.utils.evaluation as evaluation
import src.models.components.utils.post_process as post_process
import src.models.components.utils.visualization as visualization
import src.models.components.utils.h5_support as h5_support


# For flowmatching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

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
        """Flow matching models, implemented from https://github.com/facebookresearch/flow_matching

        Args:
            unet (torch.nn.Module) : unet architecture, configured with its own parameters 
                                    under model.unet in config file
            optimizer (torch.optim.Optimizer) : optimizer for training neural network
            scheduler (torch.optim.lr_scheduler) : scheduler of the learning rate
            FM_param (DictConfig) : Flow matching related parameters
            compile (Boolean) : For faster training if True
            paths (DictConfig) : Config file containing relative paths for saving images/models etc.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet'])

        # Configure NN for predicting vector field
        self.vf              = unet.to(self.device)

        # Configure FM related parameters dict
        self.FM_param           = FM_param

        self.n_classes          = FM_param.n_classes
        self.encode             = FM_param.encode
        self.pretrained_dir     = FM_param.pretrained_dir
        self.step_size          = FM_param.step_size
        self.dropout_prob       = FM_param.dropout_prob
        self.guidance_strength  = FM_param.guidance_strength
        self.reconstruct        = FM_param.reconstruct
        self.wh                 = FM_param.wh
        self.batch_size         = FM_param.batch_size
        self.save_reconstructs  = FM_param.save_reconstructs
        self.plot_n_epoch       = FM_param.plot_n_epoch
        self.target_index       = FM_param.target_index
        self.solver_lib         = FM_param.solver_lib
        self.solver             = FM_param.solver
        self.mode               = FM_param.mode
        self.plot               = FM_param.plot
        self.plot_ids           = FM_param.plot_ids
        self.ood                = FM_param.ood
        self.win_size           = FM_param.win_size

        # instantiate an affine path object for flowmatching
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # In case when data is pre-encoded
        if self.encode:
            self.vae =  AutoencoderKL.from_pretrained(self.pretrained_dir,
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

        # Specify log and image directories
        self.log_dir = paths.log_dir
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Define file name for saving reconstructs
        if self.save_reconstructs:
            time = datetime.today().strftime('%Y-%m-%d')
            self.reconstruct_dir = os.path.join(self.image_dir, time + "_reconstructs.h5")
        
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.test_loss  = MeanMetric()

        # Used for inspecting learning curve
        self.train_epoch_loss   = []
        self.val_epoch_loss     = []

        # Used for classification 
        self.test_losses = []
        self.test_labels = []

    def forward(self, x, t, y=None):
        # Convert class labels to Long Tensor for embedding
        if self.n_classes != None:
            y = y.type(torch.LongTensor).to(self.device)
        return self.vf(x=x, timesteps=t, y=y)
    
    def select_mode(self, batch, mode):
        if mode == "both":
            x = torch.cat((batch[0], batch[1]), dim=1).to(torch.float)
        elif mode == "height":
            x = batch[1].to(torch.float)
        elif mode == "rgb":
            x = batch[0].to(torch.float)
        
        # Whether to use class information or not
        if self.n_classes != None:
            y = batch[self.target_index]
        else:
            y = None
        return x, y
    
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
            x, _ = self.select_mode(batch, mode)
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
        x, y  = self.select_mode(batch, self.mode)     
        loss = self.conditional_flow_matching_loss(x, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = self.encode_data(batch, self.mode)
        x, y = self.select_mode(batch, self.mode)
        loss = self.conditional_flow_matching_loss(x, y)
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
            evaluation.plot_loss(self, skip=2)

            x, y = self.last_val_batch
            if self.n_classes!=None:
                reconstructs = []
                reconstructs.append(self.reconstruction(x, y=torch.zeros(x.shape[0], 
                                                                        device=self.device)))
                reconstructs.append(self.reconstruction(x, y=torch.ones(x.shape[0], 
                                                                        device=self.device)))
            else:
                reconstructs = self.reconstruction(x, y)
                                
            self.last_val_batch = [x, reconstructs, y]

            if self.plot:
                if self.encode:
                    self.last_val_batch[0] = self.decode_data(self.last_val_batch[0], 
                                                               self.mode) 
                    if self.n_classes!=None:
                        for i in range(2): 
                            self.last_val_batch[1][i] = self.decode_data(self.last_val_batch[1][i], self.mode)
                        # self.last_val_batch[1] = self.decode_data(self.last_val_batch[1], self.mode)
                    else:
                        self.last_val_batch[1] = self.decode_data(self.last_val_batch[1], self.mode)

                if self.mode == "both":
                    visualization.class_reconstructs_2ch(self, 
                                                        self.last_val_batch[0],
                                                        self.last_val_batch[1],
                                                        self.last_val_batch[2], 
                                                        self.plot_ids)
         
    def test_step(self, batch, batch_idx):

        if batch_idx == 0:
            self.start_time = time.time()
        # x = self.encode_data(batch, self.mode)
        x, y = self.select_mode(batch, self.mode)
        loss        = self.conditional_flow_matching_loss(x, y)
        self.log("test/loss", loss, prog_bar=True)

        # Reconstruct twice: with both 0 and 1 label
        if self.n_classes!=None:
            reconstructs = []
            reconstructs.append(self.reconstruction(x, y=torch.zeros(x.shape[0], 
                                                                device=self.device)))
            reconstructs.append(self.reconstruction(x, y=torch.ones(x.shape[0], 
                                                                    device=self.device)))
        else:
            reconstructs = self.reconstruction(x, y)
                
        # Pick the last full batch or first 
        if (x.shape[0] == self.batch_size) or (batch_idx == 0):
            self.last_test_batch = [x, reconstructs, batch[self.target_index]]

        if self.ood:
            # Calculate reconstruction loss used for OOD-detection
            x0 = self.decode_data(x, self.mode)
            if self.n_classes!=None:
                x1 = self.decode_data(reconstructs[0], self.mode) # Only pick non-crack reconstructions
            else:
                x1 = self.decode_data(reconstructs, self.mode) # Only pick non-crack reconstructions   
            
            # Convert rgb channels to grayscale and revert normalization to [0,1]
            x0, x1          = post_process.to_gray_0_1(x0), post_process.to_gray_0_1(x1)
            ood_score       = post_process.get_OOD_score(x0=x0, x1=x1)

            # Append scores
            self.test_losses.append(ood_score)
            self.test_labels.append(batch[self.target_index])
                
        if self.save_reconstructs:
            if self.encode:
                x = self.decode_data(x, self.mode).cpu()
                if self.n_classes != None:
                    for i in range(2): 
                        reconstructs[i] = self.decode_data(reconstructs[i], self.mode).cpu()
                    # reconstructs = self.decode_data(reconstructs, self.mode).cpu()
                else:
                    reconstructs = self.decode_data(reconstructs, self.mode).cpu()
            
            y = batch[self.target_index].cpu()
            h5_support.save_reconstructions_to_h5(self.reconstruct_dir, [x, reconstructs, y], cfg=(self.n_classes != None))
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Visualizations
        # Save last batch for visualization
        evaluation.plot_loss(self, skip=1)

        if self.ood:
            y_score = np.concatenate([t for t in self.test_losses]) # use t.cpu().numpy() if Tensor
            y_true = np.concatenate([t.cpu().numpy() for t in self.test_labels]).astype(int)
            
            # Save OOD-scores and true labels for later use
            np.savez(os.path.join(self.log_dir, "0_labelsNscores"), y_true=y_true, y_scores=y_score)
            
            evaluation.plot_histogram(y_score, y_true, save_dir = self.log_dir)
            evaluation.plot_classification_metrics(y_score, y_true, save_dir=self.log_dir)

        if self.plot:
            if self.encode and not(self.save_reconstructs):
                self.last_test_batch[0] = self.decode_data(self.last_test_batch[0], self.mode)
                if self.n_classes!=None:
                    for i in range(2): 
                        self.last_test_batch[1][i] = self.decode_data(self.last_test_batch[1][i], self.mode)
                    # self.last_test_batch[1] = self.decode_data(self.last_test_batch[1], self.mode)
                else:
                    self.last_test_batch[1] = self.decode_data(self.last_test_batch[1], self.mode)
            
            if self.mode == "both":
                if self.n_classes!=None:
                    visualization.class_reconstructs_2ch(self, 
                                            self.last_test_batch[0],
                                            self.last_test_batch[1],
                                            self.last_test_batch[2],
                                            self.plot_ids,
                                            self.test_losses[-1] if self.test_losses[-1].shape[0] == self.batch_size else self.test_losses[-2],
                                            )
                else:
                    visualization.visualize_reconstructs_2ch(self, 
                                               self.last_test_batch[0], 
                                               self.last_test_batch[1],
                                               self.last_test_batch[2],
                                               self.plot_ids,
                                               self.test_losses[-1] if self.test_losses[-1].shape[0] == self.batch_size else self.test_losses[-2], 
                                               )

        self.end_time = time.time()
        inference_time = self.end_time - self.start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        
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
        
        if self.n_classes != None:
            y1          = y.detach().clone()
            # Randomly change class labels to n + 1 for Classifier Free Guidance
            indices     = torch.randperm(x_1.shape[0])[:int(self.dropout_prob*x_1.shape[0])]
            y1[indices]  = self.n_classes - 1
        else:
            y1 = None

        # Generate noise
        x_0       = torch.randn_like(x_1, device=self.device)

        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        
        # Predict vector field
        vt = self(path_sample.x_t, path_sample.t, y1)
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
        
        if self.n_classes!= None:
            # Configure guidance_strength for Classifier Free Guidance
            omega = self.guidance_strength
            unknown_class = self.n_classes - 1
            y_unconditional = (unknown_class * torch.ones(x_0.shape[0], device=self.device)).type(torch.LongTensor)
        else:
            w = 0
            y_unconditional = None

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **extras):
                # Define label vector for unconditional case
                return (1-omega) * self.model(x, t, y_unconditional) + omega * self.model(x, t, y)

        wrapped_vf = WrappedModel(self.vf)
        
        solver = ODESolver(velocity_model=wrapped_vf)
        sol    = solver.sample(x_init=x_0,
                                method=self.solver,
                                step_size=self.step_size,
                                return_intermediates=False)

        return sol
    
    @torch.no_grad()
    def reconstruction(self, x_1, y):

        tstart = 1 - self.reconstruct
        T = torch.linspace(tstart, 1, 10).to(self.device)

        x_0 = torch.rand_like(x_1, device=self.device)
        x_t = x_1 * tstart + x_0 * (1-tstart)
        
        if self.n_classes!= None:
            # Required for class embedding
            y = y.type(torch.LongTensor).to(self.device)
            # Configure guidance_strength for Classifier Free Guidance
            w = self.guidance_strength
            unknown_class = self.n_classes - 1
            y_unconditional = (unknown_class * torch.ones(x_1.shape[0])).type(torch.LongTensor).to(self.device)
        else:
            w = 0
            y_unconditional = None

        device = self.device
        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.LongTensor, **extras):
                # Define label vector for unconditional case
                t = t * torch.ones(x_1.shape[0], device=device)
                return (1-w) * self.model(x, t, y_unconditional) + w * self.model(x, t, y)

        wrapped_vf = WrappedModel(self.vf)
        
        solver = ODESolver(velocity_model=wrapped_vf)
        sol    = solver.sample(time_grid=T,
                                x_init=x_t,
                                method=self.solver,
                                step_size=self.step_size,
                                return_intermediates=False,
                                y = y)

        return sol
    
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
    _ = FlowMatchingLitModule(None, None, None, None, None, None, None)