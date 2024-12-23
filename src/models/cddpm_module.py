import torch
import torch.nn.functional as F
import diffusers
import numpy as np
import os
import math
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig
from torchvision.models import resnet18
from src.models.support_functions.evaluation import *

class ConditionalDenoisingDiffusionLitModule(LightningModule):
    def __init__(
        self, 
        unet: torch.nn.Module,
        criterion: torch.nn.Module,
        unet_dict: DictConfig,
        fe_dict: DictConfig,
        noise_scheduler,
        compile,
        target: int,
        paths: DictConfig,
    ):
        """ImageFlow.

        Args:

        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['unet', 'criterion'])

        # Import pretrained unet
        self.unet_dict                  = unet_dict
        self.unet_model                 = unet 
        checkpoint = torch.load(self.unet_dict.ckpt_path)
        # Fix issue with dict names
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint["state_dict"].items()}
        self.unet_model.load_state_dict(state_dict)
        self.unet_model.eval() 
        for param in self.unet_model.parameters():
            param.requires_grad= False   

        self.noise_scheduler            = noise_scheduler

        self.fe_dict                  = fe_dict
        if self.fe_dict.use_FE:
        # Import pretrained feature extractor
            self.feature_extractor = resnet18()
            checkpoint = torch.load(self.fe_dict.ckpt_path)
            # Fix issue with dict names
            state_dict = {
                key.replace("feature_extractor.model.", ""): value 
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("feature_extractor.")
            }
            self.feature_extractor.load_state_dict(state_dict)
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad= False

        self.criterion          = criterion

        # Define which dimension contains the target
        self.target = target

        # Define start and weightof conditioning 
        self.num_condition_steps = self.unet_dict.num_condition_steps
        self.condition_weight    = self.unet_dict.condition_weight
        self.v = fe_dict.v

        # Specify fontsize for plots
        self.fs = 16

        self.log_dir = paths.log_dir
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.test_loss = MeanMetric()

        self.test_losses = []
        self.test_labels = []

    def forward(self, x, steps=None):
        noise = torch.randn(x.shape, device=self.device)
        if steps == None:
            steps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        else:
            steps = torch.tensor([steps] * x.shape[0], device=self.device)
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps)
        residual = self.unet_model(noisy_images, steps).sample
        
        return residual, noise
            
    def test_step(self, batch, batch_idx):
        x = batch[0]
        residual, noise = self(x)
        loss = self.criterion(residual, noise, self.device)
        self.log("test/loss", loss, prog_bar=True)

        reconstruct = self.reconstruction(x)
        losses = self.criterion(x,reconstruct, self.device, reduction='batch')
        self.last_test_batch = [x, reconstruct, batch[2]]

        self.test_losses.append(losses)
        self.test_labels.append(batch[self.target])
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Sample from gaussian nosie
        # x_hat = self.sample(num_samples=16)
        
        # Visualizations
        # self.visualize_samples(x_hat)
        self.visualize_reconstructs(self.last_test_batch[0], self.last_test_batch[1], self.last_test_batch[2])
        plot_histogram(self)

        # Clear variables
        self.test_losses.clear()
        self.test_labels.clear()
    
    @torch.no_grad()
    def reconstruction(self, x):
        Tc = self.num_condition_steps
        skip = self.unet_dict.skip
        eta = self.unet_dict.eta

        # Define target
        y = x

        # Start with adding noise at timestep Tc
        t = torch.tensor([Tc] * x.shape[0], device=self.device)
        xt = self.noise_scheduler.add_noise(x, torch.randn_like(x), t)

        
        if self.fe_dict.use_DDIM:
            # Implementation from Mousakha et al, 2023
            seq = range(1 , Tc+1, skip)
            seq_next = [0] + list(seq[:-1])
            for index, (i,j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = torch.tensor([i] * x.shape[0], device=self.device)
                
                e = self.unet_model(xt, t)['sample']
                
                alpha_prod       = self.noise_scheduler.alphas_cumprod[i]
                alpha_prod_prev  = self.noise_scheduler.alphas_cumprod[j]
                sigma = eta * torch.sqrt((1 - alpha_prod / alpha_prod_prev) * (1 - alpha_prod_prev) / (1 - alpha_prod))
                
                yt = self.noise_scheduler.add_noise(y, e, t)
                
                e_hat = e - self.unet_dict.condition_weight * torch.sqrt(1-alpha_prod) * (yt-xt)
                ft = (xt - torch.sqrt(1-alpha_prod)*e_hat) / torch.sqrt(alpha_prod)
                
                xt = torch.sqrt(alpha_prod_prev) * ft + torch.sqrt(1-alpha_prod_prev-sigma**2) * e_hat + sigma * torch.randn_like(xt)

        else:
            # Implementation of DDPM from Ho et al, 2020
            for timestep in range(Tc, 0, -1):
                t = torch.tensor([timestep] * x.shape[0], device=self.device)
                e = self.unet_model(xt, t)['sample']
                
                # var         = self.noise_scheduler._get_variance(timestep)
                alpha            = self.noise_scheduler.alphas[timestep]
                alpha_prod       = self.noise_scheduler.alphas_cumprod[timestep]
                alpha_prod_prev  = self.noise_scheduler.alphas_cumprod[timestep-1]
                sigma = eta * torch.sqrt((1 - alpha_prod / alpha_prod_prev) * (1 - alpha_prod_prev) / (1 - alpha_prod))
                
                yt = self.noise_scheduler.add_noise(y, e, t)
                e_hat = e - self.condition_weight * torch.sqrt(1-alpha_prod) * (yt-xt)
                xt = 1 / torch.sqrt(alpha) * (xt - (1-alpha)/torch.sqrt(1-alpha_prod) * e_hat) + sigma * torch.randn_like(xt)

        return xt
    
    # @torch.no_grad()
    # def sample(self, num_samples, x=None, steps=None):
    #     img_size = self.model.config.sample_size
    #     channels = self.model.config.in_channels
    #     if steps == None:
    #         steps = self.noise_scheduler.config.num_train_timesteps
    #         x = torch.randn(num_samples, channels, img_size, img_size).to(self.device)

    #     for timestep in range(steps-1, 0, -1):
    #         t = torch.tensor([timestep] * num_samples, device=self.device)
    #         noise_pred = self.model(x, t)['sample']
    #         x = self.noise_scheduler.step(noise_pred, timestep , x, generator=None)['prev_sample']
            
    #     x = (x + 1.) / 2.
    #     return torch.clamp(x, min=0, max=1)
        
    # @torch.no_grad()
    # def visualize_samples(self, x):
    #     # Create figure
    #     grid = make_grid(x, nrow=int(np.sqrt(x.shape[0])))
    #     plt.figure(figsize=(12,12))
    #     plt.imshow(grid.permute(1,2,0).cpu().squeeze(), cmap='gray')
    #     plt.axis('off')
    #     plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_epoch_sample.png")
    #     plt.savefig(plt_dir)
    #     plt.close()
    #     # Send figure as artifact to logger
    #     if self.logger.__class__.__name__ == "MLFlowLogger":
    #         self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)
    #     # os.remove(image_path)
    
    def min_max_normalize(self, x):
        min_val = x.amin(dim=(0,1,2), keepdim=True)
        max_val = x.amax(dim=(0,1,2), keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-8)
        
    def visualize_reconstructs(self, x, reconstruct, labels):
        # # Convert back to [0,1] for plotting
        x           = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        if self.fe_dict.use_FE:
            # Calculate error based on L1 + feature based
            error = self.heat_map(output=reconstruct, target=x)
        else:
            # Calculate pixel-wise error + convert to grey-scale
            error = (x - reconstruct)**2
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(self.device)
            error = torch.tensordot(error, rgb_weights, dims=([1],[0])).unsqueeze(1)

        # Normalize over all batches
        error = self.min_max_normalize(error)

        # Necessary permutation for plotting
        error       = error.permute(0,2,3,1).cpu()
        x           = x.permute(0,2,3,1).cpu()
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

    def heat_map(self, output, target):
        '''
        Compute the anomaly map
        :param output: the output of the reconstruction
        :param target: the target image
        :param FE: the feature extractor
        :param sigma: the sigma of the gaussian kernel
        :param i_d: the pixel distance
        :param f_d: the feature distance
        '''
        sigma = 4
        kernel_size = 2 * int(4 * sigma + 0.5) +1
        anomaly_map = 0

        output = output
        target = target

        i_d = self.pixel_distance(output, target)
        f_d = self.feature_distance((output),  (target))
        f_d = torch.Tensor(f_d)

        gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        anomaly_map += f_d + self.v * (torch.max(f_d)/ torch.max(i_d)) * i_d  
        anomaly_map = gaussian_blur(anomaly_map)
        anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
        return anomaly_map

    def feature_maps_resnet(self, input, module_name):

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        if module_name == 'layer2':
            self.feature_extractor.layer2.register_forward_hook(get_activation(module_name))
        if module_name == 'layer3':
            self.feature_extractor.layer3.register_forward_hook(get_activation(module_name))

        self.feature_extractor(input)

        return activation[module_name]

    def pixel_distance(self, output, target):
        '''
        Pixel distance between image1 and image2
        '''
        distance_map = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)
        return distance_map

    def feature_distance(self, output, target):
        '''
        Feature distance between output and target
        '''
        self.feature_extractor.eval()
        transform = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / (2)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target = transform(target)
        output = transform(output)

        input_features = [self.feature_maps_resnet(target, 'layer2'), 
                           self.feature_maps_resnet(target, 'layer3')]
        
        output_features = [self.feature_maps_resnet(output, 'layer2'), 
                           self.feature_maps_resnet(output, 'layer3')]

        out_size = output.shape[-1]
        anomaly_map = torch.zeros([input_features[0].shape[0] ,1 ,out_size, out_size]).to(self.device)
        for i in self.fe_dict.layer:
            a_map = 1 - F.cosine_similarity(self.patchify(input_features[i]), self.patchify(output_features[i]))
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            anomaly_map += a_map
        return anomaly_map 

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        patchsize = 3
        stride = 1
        padding = int((patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=patchsize, stride=stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (patchsize - 1) - 1
            ) / stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], patchsize, patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        max_features = torch.mean(unfolded_features, dim=(3,4))
        features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return features
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.unet = torch.compile(self.unet)

    
if __name__ == "__main__":
    _ = ConditionalDenoisingDiffusionLitModule(None, None, None, None, None)