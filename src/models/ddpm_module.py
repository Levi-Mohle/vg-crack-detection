import torch
import torch.nn as nn
import diffusers
import numpy as np
import os
import time
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from lightning import LightningModule

class DenoisingDiffusionLitModule(LightningModule):
    def __init__(
        self, 
        unet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        noise_scheduler,
        compile,
    ):
        """ImageFlow.

        Args:
            flows: A list of flows (each a nn.Module) that should be applied on the images.
            import_samples: Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model              = unet
        self.noise_scheduler    = noise_scheduler


        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.test_losses = []
        self.test_labels = []


    def training_step(self, batch, batch_idx):
        x = batch[0]
        noise = torch.randn(x.shape)
        steps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        noisy_images = self.noise_scheduler.add_noise(x, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

        # if batch_idx == 0: # Only sample once per validation step (not every batch)
        #     # Generate samples
        #     z = self.sample(img_shape=(16,1,28,28))
        #     # Define image path
        #     image_path= self.logger._artifact_location + '/' + time.strftime("%H_%M_%S", time.localtime()) + '_sample.png'
        #     # Create figure
        #     grid = make_grid(z, nrow=int(np.sqrt(z.shape[0])))
        #     plt.figure(figsize=(12,12))
        #     plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(image_path)
        #     plt.close()
        #     # Send figure as artifact to logger
        #     self.logger.experiment.log_artifact(local_path=image_path, run_id=self.logger.run_id)
        #     os.remove(image_path)

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def _log_histogram(self):

        y_score = np.concatenate([t.cpu().numpy() for t in self.test_losses])
        y_true = np.concatenate([t.cpu().numpy() for t in self.test_labels])

        auc_score = roc_auc_score(y_true, y_score)
        if auc_score < 0.2:
            auc_score = 1. - auc_score
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]
            
        # print with 4 decimal places
        # print(f"ROC AUC: {auc_score:.4f}, FPR at 95% TPR: {fpr95:.4f}")
        y_id = y_score[np.where(y_true == 0)]
        y_ood = y_score[np.where(y_true != 0)]

        fig = plt.figure(figsize=(10, 10))
        axes = fig.subplots(2,1)

        # plot histograms of scores in same plot
        axes[0].hist(y_id, bins=50, alpha=0.5, label='In-distribution', density=True)
        axes[0].hist(y_ood, bins=50, alpha=0.5, label='Out-of-distribution', density=True)
        axes[0].legend()
        axes[0].set_title('Outlier Detection')
        axes[0].set_ylabel('Counts')
        axes[0].set_xlabel('Loss')

        disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score,
                                      estimator_name='Model')
        disp_roc.plot(ax=axes[1])
        axes[1].set_title('ROC')
        
        # Logging plot as figure to mlflow
        image_path = self.logger._artifact_location + '/' \
                                                    + time.strftime("%H_%M_%S", time.localtime()) \
                                                    + '_hist_ROC.png'
        fig.savefig(image_path)
        self.logger.experiment.log_artifact(local_path = image_path,
                                            run_id=self.logger.run_id)
        # Remove image from folder (saved to logger)
        os.remove(image_path)
        
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