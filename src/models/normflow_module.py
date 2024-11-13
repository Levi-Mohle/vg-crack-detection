import torch
import torch.nn as nn
import numpy as np
import os
import time
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from lightning import LightningModule
from omegaconf import DictConfig

class NormalizingFlowLitModule(LightningModule):
    def __init__(
        self, 
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        import_samples: int,
        paths: DictConfig,
        ) -> None:
        """ImageFlow.

        Args:
            flows: A list of flows (each a nn.Module) that should be applied on the images.
            import_samples: Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.flows = net.flows
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        self.log_dir = paths.log_dir

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.test_losses = []
        self.test_labels = []

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        self.img_size_after_flow = torch.tensor(z.shape[1:])
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """Given a batch of images, return the likelihood of those.

        If return_ll is True, this function returns the log likelihood of the input. Otherwise, the output metric is
        bits per dimension (scaled negative log likelihood)

        """
        z, ldj = self.encode(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """Sample a batch of images from the flow."""
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=torch.cat((torch.tensor([16]), self.img_size_after_flow))).to(self.device)
        else:
            z = z_init.to(self.device)
        # print(f"After sampling: min: {torch.min(z)}, max: {torch.max(z)}")
        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=self.device) 
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        # print(f"After reconstructing: min: {torch.min(z)}, max: {torch.max(z)}")
        return z

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch[0])
        self.train_loss(loss)
        self.log("train/loss", self.train_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch[0])
        self.val_loss(loss)
        self.log("val/loss", self.val_loss)

    def on_validation_epoch_end(self):
        
        if self.current_epoch % 5 == 0: # Only sample once per validation step (not every batch)
            self.generate_samples()

    def generate_samples(self):
        # Generate samples
        z = self.sample(img_shape=(16,1,28,28))

        # Create figure
        grid = make_grid(z, nrow=int(np.sqrt(z.shape[0])))
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

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image
        x, y = batch
        samples = []
        for _ in range(self.import_samples):
            img_ll = self._get_likelihood(x, return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # To average the probabilities, we need to go from log-space to exp, and back to log.
        # Logsumexp provides us a stable implementation for this
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(x.shape[1:])
        self.test_losses.append(bpd)
        self.test_labels.append(y)

        bpd = bpd.mean()
        self.test_loss(bpd)

        self.log("test/loss", self.test_loss)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.generate_samples()
        self._log_histogram()
        self.test_losses.clear()
        self.test_labels.clear()

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
            self.net = torch.compile(self.net)

        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None, None, None)