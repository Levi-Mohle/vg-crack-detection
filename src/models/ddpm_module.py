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
        unet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        noise_scheduler,
        compile,
        target: int,
        reconstruct_coef: float,
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

        # Fraction of forward diffusion for reconstructing test images
        self.reconstruct_coef = reconstruct_coef

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

    def forward(self, x, steps=None):
        noise = torch.randn(x.shape, device=self.device)
        if steps == None:
            steps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        else:
            steps = torch.tensor([steps] * x.shape[0], device=self.device)
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps)
        residual = self.model(noisy_images, steps).sample
        
        return residual, noise
    
    def training_step(self, batch, batch_idx):
        residual, noise = self(batch[0])
        loss = self.criterion(residual, noise, self.device)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        residual, noise = self(batch[0])
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
            
    def test_step(self, batch, batch_idx):
        residual, noise = self(batch[0])
        loss = self.criterion(residual, noise, self.device)
        self.log("test/loss", loss, prog_bar=True)

        x, reconstruct = self.partial_diffusion(batch[0], self.reconstruct_coef)
        losses = self.criterion(x,reconstruct, self.device, reduction='none')
        self.last_test_batch = [x, reconstruct, batch[2]]
        # In case you want to evaluate on just the MSE from the Unet
        # losses = self.criterion(residual, noise, self.device, reduction='none')

        self.test_losses.append(losses)
        self.test_labels.append(batch[self.target])
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Sample from gaussian nosie
        # x_hat = self.sample(num_samples=16)
        
        # Visualizations
        # self.visualize_samples(x_hat)
        self.plot_loss()
        self.visualize_reconstructs(self.last_test_batch[0], self.last_test_batch[1], self.last_test_batch[2])
        self._log_histogram()

        # Clear variables
        self.train_epoch_loss.clear()
        self.val_epoch_loss.clear()
        self.test_losses.clear()
        self.test_labels.clear()

    def partial_diffusion(self, x, coef):
        # Define noise
        noise = torch.randn(x.shape, device=self.device)
        # Take fraction of steps for forward diffusion process
        steps = int(coef * self.noise_scheduler.config.num_train_timesteps)
        steps_tensor = torch.tensor([steps] * x.shape[0], device=self.device)
        # Noise the samples
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps_tensor)
        # Reconstruct the samples
        reconstruct = self.sample(x.shape[0], noisy_images, steps)

        return x, reconstruct
        
    @torch.no_grad()
    def sample(self, num_samples, x=None, steps=None):
        img_size = self.model.config.sample_size
        channels = self.model.config.in_channels
        if steps == None:
            steps = self.noise_scheduler.config.num_train_timesteps
            x = torch.randn(num_samples, channels, img_size, img_size).to(self.device)

        for timestep in range(steps-1, 0, -1):
            t = torch.tensor([timestep] * num_samples, device=self.device)
            noise_pred = self.model(x, t)['sample']
            x = self.noise_scheduler.step(noise_pred, timestep , x, generator=None)['prev_sample']
            
        x = (x + 1.) / 2.
        return torch.clamp(x, min=0, max=1)
        
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
    
    def min_max_normalize(self, x):
        min_val = x.amin(dim=(0,1,2), keepdim=True)
        max_val = x.amax(dim=(0,1,2), keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-8)
        
    def visualize_reconstructs(self, x, reconstruct, labels):
        x = x.permute(0,2,3,1).cpu()
        reconstruct = reconstruct.permute(0,2,3,1).cpu()

        # Calculate pixel-wise squared error + normalize + convert to grey-scale
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        error = self.min_max_normalize((x - reconstruct)**2)
        error = torch.tensordot(error, rgb_weights, dims=([-1],[0]))

        img = [x, reconstruct, error]

        title = ["Original sample", "Reconstructed Sample", "Pixel-wise Squared Error"]

        for i in range(4):
            fig = plt.figure(constrained_layout=True, figsize=(11,9))
            # create 3x1 subfigs
            subfigs = fig.subfigures(nrows=3, ncols=1)
            for row, subfig in enumerate(subfigs):
                subfig.suptitle(title[row], fontsize = self.fs)
                # create 1x3 subplots per subfig
                axs = subfig.subplots(nrows=1, ncols=4)
                for col, ax in enumerate(axs):
                    ax.imshow(img[row][col+4*i])
                    ax.axis("off")
                    ax.set_title(f"Label: {labels[col+4*i]}")
                
                        
            plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs_{i}.png")
            fig.savefig(plt_dir)
            plt.close()
            # Send figure as artifact to logger
            # if self.logger.__class__.__name__ == "MLFlowLogger":
            #     self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)
    # def visualize_reconstructs(self, x, reconstruct):
    #     x = x[:8].permute(0,2,3,1).cpu()
    #     reconstruct = reconstruct[:8].permute(0,2,3,1).cpu()
        
    #     fig, axes = plt.subplots(4, 4, figsize=(12,12))
        
    #     for i in range(8):
    #         if i > 3:
    #             axes[i-4,2].imshow(x[i], cmap='grey')
    #             axes[i-4,2].axis("off")
    #             axes[i-4,3].imshow(reconstruct[i], cmap='grey')
    #             axes[i-4,3].axis("off")
    #         else:
    #             axes[i,0].imshow(x[i], cmap='grey')
    #             axes[i,0].axis("off")
    #             axes[i,1].imshow(reconstruct[i], cmap='grey')
    #             axes[i,1].axis("off")
                
    #     axes[0,0].set_title("Original Sample", fontsize = self.fs)
    #     axes[0,1].set_title("Reconstructed Sample", fontsize = self.fs)
    #     axes[0,2].set_title("Original Sample", fontsize = self.fs)
    #     axes[0,3].set_title("Reconstructed Sample", fontsize = self.fs)
                
    #     plt.tight_layout()          
    #     plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs.png")
    #     plt.savefig(plt_dir)
    #     plt.close()
    #     # Send figure as artifact to logger
    #     if self.logger.__class__.__name__ == "MLFlowLogger":
    #         self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)

    def plot_loss(self):

        epochs = [i for i in range(1, self.current_epoch + 1)]
        plt.plot(epochs, [t.cpu().numpy() for t in self.train_epoch_loss], marker='o', linestyle = '-', label = "Training")
        plt.plot(epochs, [t.cpu().numpy() for t in self.val_epoch_loss][1:], marker='o', linestyle = '-', label = "Validation")
        plt.xlabel('Epochs', fontsize = self.fs)
        plt.ylabel('Loss [-]', fontsize = self.fs)
        plt.legend()
        plt.title('Training and Validation Loss', fontsize = self.fs)

        plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_loss.png")
        plt.savefig(plt_dir)
        plt.close()

    def _log_histogram(self):

        y_score = np.concatenate([t.cpu().numpy() for t in self.test_losses])
        y_true = np.concatenate([t.cpu().numpy() for t in self.test_labels])

        auc_score = roc_auc_score(y_true, y_score)
        if auc_score < 0.2:
            auc_score = 1. - auc_score
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]
            
        y_id = y_score[np.where(y_true == 0)]
        y_ood = y_score[np.where(y_true != 0)]

        fig, axes= plt.subplots(2,1, figsize=(10, 10))
        
        # plot histograms of scores in same plot
        axes[0].hist(y_id, bins=50, alpha=0.5, label='In-distribution', density=True)
        axes[0].hist(y_ood, bins=50, alpha=0.5, label='Out-of-distribution', density=True)
        axes[0].legend()
        axes[0].set_title('Outlier Detection', fontsize = self.fs)
        axes[0].set_ylabel('Counts', fontsize = self.fs)
        axes[0].set_xlabel('Loss', fontsize = self.fs)

        # plot roc
        axes[1].plot(fpr, tpr)
        axes[1].set_title('ROC', fontsize = self.fs)
        axes[1].set_ylabel('True Positive Rate', fontsize = self.fs)
        axes[1].set_xlabel('False Positive Rate', fontsize = self.fs)
        axes[1].legend([f"AUC {auc_score:.2f}"], fontsize = 12)
        axes[1].set_box_aspect(1)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        
        plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_hist_ROC.png")
        fig.savefig(plt_dir)
        
        # Logging plot as figure to mlflow
        if self.logger.__class__.__name__ == "MLFlowLogger":
            self.logger.experiment.log_artifact(local_path = self.image_dir,
                                                run_id=self.logger.run_id)
        # Remove image from folder (saved to logger)
        # os.remove(image_path)
        
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