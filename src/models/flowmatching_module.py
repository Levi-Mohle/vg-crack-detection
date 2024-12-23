import torch
import torch.nn as nn
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
import numpy as np
import os
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from pandas import DataFrame
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig
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
        self.FM_param          = FM_param

        if self.FM_param.latent:
            self.vae =  AutoencoderKL.from_pretrained(self.FM_param.pretrained, local_files_only=True).to(self.device) 
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
    
    def training_step(self, batch, batch_idx):

        x = batch[0]
        if self.vae is not None:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.1821)
        
        loss = self.conditional_flow_matching_loss(x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.conditional_flow_matching_loss(x)
        self.log("val/loss", loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_epoch_loss.append(self.trainer.callback_metrics['train/loss'])
        
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_epoch_loss.append(self.trainer.callback_metrics['val/loss'])

            
    def test_step(self, batch, batch_idx):
        x       = batch[0]
        self.shape = x.shape
        loss    = self.conditional_flow_matching_loss(x)
        self.log("test/loss", loss, prog_bar=True)

        reconstruct = self.reconstruction(x)
        losses = torch.mean((x-reconstruct)**2, dim=(1,2,3))
        self.last_test_batch = [x, reconstruct, batch[2]]
        # In case you want to evaluate on just the MSE from the Unet
        # losses = self.criterion(residual, noise, self.device, reduction='none')

        self.test_losses.append(losses)
        self.test_labels.append(batch[self.FM_param.target])
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Sample from gaussian nosie
        # x_hat = self.sample(n_samples=16)
        
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
        
        if self.vae is not None:
            reconstruct = self.vae.decode(reconstruct / 0.18215).sample
        reconstruct = reconstruct*0.5 + 0.5
        reconstruct = reconstruct.clamp(0, 1)

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
    
    def min_max_normalize(self, x):
        min_val = x.amin(dim=(0,1,2), keepdim=True)
        max_val = x.amax(dim=(0,1,2), keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-8)
        
    def visualize_reconstructs(self, x, reconstruct, labels):
        x = x.permute(0,2,3,1).cpu()
        reconstruct = reconstruct.permute(0,2,3,1).cpu()

        # Convert back to [0,1] for plotting
        x = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        # Calculate pixel-wise squared error + normalize + convert to grey-scale
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        error = self.min_max_normalize((x - reconstruct)**2)
        # error = (x-reconstruct)**2
        error = torch.tensordot(error, rgb_weights, dims=([-1],[0]))

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

    def confusion_matrix(y_scores, y_true, thresholds):

        accuracies = []
        for th in thresholds:
            y_pred = (y_scores >= th).astype(int)
            acc = (y_pred == y_true).sum() / len(y_true)
            accuracies.append(acc)

        best_index = np.argmax(accuracies) 
        best_th = thresholds[best_index]
        best_acc = accuracies[best_index]
        y_pred = (y_scores >= best_th).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        class_names = ["No crack", "Crack"]
        cm_df = DataFrame(cm, index=class_names, columns=class_names)

        print(f"Confusion Matrix for best accuracy {best_acc:.3f}:")
        print(cm_df)

    def _log_histogram(self):

        y_score = np.concatenate([t.cpu().numpy() for t in self.test_losses])
        y_true = np.concatenate([t.cpu().numpy() for t in self.test_labels])

        auc_score = roc_auc_score(y_true, y_score)
        if auc_score < 0.2:
            auc_score = 1. - auc_score
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]
        
        # Print confusion matrix
        self.confusion_matrix(y_score, y_true, thresholds)

        # Separate ID and OOD samples
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

        axes[1].plot([0,1], [0,1], ls="--")

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
    _ = FlowMatchingLitModule(None, None, None, None, None)