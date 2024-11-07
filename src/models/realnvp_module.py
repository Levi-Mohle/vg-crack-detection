from typing import Any, Dict, Tuple

import torch
import numpy as np
from lightning import LightningModule
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from torchmetrics import MeanMetric
# from models.components.loss_functions.realnvp_loss import RealNVPLoss


class RealNVPLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST OOD with RealNVP.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss     = MeanMetric()
        self.val_loss       = MeanMetric()
        self.test_loss      = MeanMetric()

        self.val_losses = []
        self.test_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, _ = batch
        x.requires_grad_(True)

        x, z, sldj = self.forward(x)
        return x, z, sldj

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, z, sldj = self.model_step(batch)

        # update and log metrics
        loss = self.criterion(x, z, sldj)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.enable_grad():
            x, z, sldj = self.model_step(batch)

            # update and log metrics
            loss = self.criterion(x, z, sldj)
            self.val_loss(loss)
            self.val_losses.append(loss)
            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Sampling a new instance
        # TODO create different naming when sampling after each epoch
        if batch_idx == 0:
            noise = torch.randn((1, z.shape[1], z.shape[2], z.shape[3]), 
                            dtype=torch.float32)
            image_path = self.logger._artifact_location + "/sample_0.png"
            _, x_hat, _ = self.net(noise, reverse=True)
            save_image(x_hat, image_path)
            self.logger.experiment.log_artifact(local_path=image_path, run_id=self.logger.run_id)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        last_losses = self.val_losses
        self.last_val_losses = last_losses
        self.val_losses = []
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.enable_grad():
            x, z, sldj = self.model_step(batch)

            # update and log metrics
            loss = self.criterion(x, z, sldj)
            self.test_loss(loss)
            self.test_losses.append(loss)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self._log_histogram()
        self.test_losses.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def _log_histogram(self):

        # TODO issue when running eval.py: does not run validation step and therefore does not have val_scores
        val_scores = np.array(self.last_val_losses)
        test_scores = np.array(self.test_losses)

        y_true = np.concatenate([np.zeros_like(val_scores), np.ones_like(test_scores)], axis=0)
        y_score = np.concatenate([val_scores, test_scores], axis=0)
        auc_score = roc_auc_score(y_true, y_score)
        if auc_score < 0.2:
            auc_score = 1. - auc_score
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]
            
        # print with 4 decimal places
        # print(f"ROC AUC: {auc_score:.4f}, FPR at 95% TPR: {fpr95:.4f}")
        
        fig = plt.figure(figsize=(10, 10))
        axes = fig.subplots(2,1)

        # plot histograms of scores in same plot
        axes[0].hist(val_scores, bins=50, alpha=0.5, label='In-distribution', density=True)
        axes[0].hist(test_scores, bins=50, alpha=0.5, label='Out-of-distribution', density=True)
        axes[0].legend()
        axes[0].set_title('Outlier Detection')
        axes[0].set_ylabel('Counts')
        axes[0].set_xlabel('Loss')

        disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score,
                                      estimator_name='Model')
        disp_roc.plot(ax=axes[1])
        axes[1].set_title('ROC')
        
        # Logging plot as figure to mlflow
        image_path = self.logger._artifact_location + "/hist_ROC.png"
        fig.savefig(image_path)
        self.logger.experiment.log_artifact(local_path = image_path,
                                            run_id=self.logger.run_id)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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
    _ = RealNVPLitModule(None, None, None, None)
