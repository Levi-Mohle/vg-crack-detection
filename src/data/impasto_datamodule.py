from typing import Any, Dict, Optional, Tuple

import os
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from src.data.components.loading import HDF5PatchesDatasetCustom
from src.data.components.path_setup import impasto_dataset_variant


class IMPASTO_DataModule(LightningDataModule):
    """`LightningDataModule` for the impasto dataset.

    The impasto dataset contains both RGB and height images, also called mini-patches, from different datasets, 
    depending on the .h5 that is loaded. The RGB images are 3 channel 512x512 px, while the height images are 1 channel 512x512 px. 
    If an encoded version of the impasto dataset is used, both RGB and height latents have dimensions 4x64x64.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/impasto",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        variant: str = "32x32",
        crack: str = "synthetic",
        transform = None,
        # rgb_transform: transforms.Compose = None,
        # height_transform: transforms.Compose = None,
    ) -> None:
        """Initialize a `IMPASTO_DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # data transformations
        # TODO create selective transform function
        # self.rgb_transform = rgb_transform
        # self.height_transform = height_transform
        self.transform = transform

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir

        self.variant = variant
        self.crack = crack
    # @property
    # def num_classes(self) -> int:
    #     """Get the number of classes.

    #     :return: The number of MNIST classes (10).
    #     """
    #     return 10
    
    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        IMPASTO_train_dir, IMPASTO_val_dir, IMPASTO_test_dir = impasto_dataset_variant(self.variant, self.crack)
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = HDF5PatchesDatasetCustom(hdf5_file_path   = os.path.join(self.data_dir, IMPASTO_train_dir),
                                                       transform = self.transform)
                                                    #    rgb_transform    = self.rgb_transform,
                                                    #    height_transform = self.height_transform)
            self.data_val   = HDF5PatchesDatasetCustom(hdf5_file_path   = os.path.join(self.data_dir, IMPASTO_val_dir),
                                                       transform = self.transform)
                                                    #    rgb_transform    = self.rgb_transform,
                                                    #    height_transform = self.height_transform)
            self.data_test  = HDF5PatchesDatasetCustom(hdf5_file_path   = os.path.join(self.data_dir, IMPASTO_test_dir),
                                                       transform = self.transform)
                                                    #    rgb_transform    = self.rgb_transform,
                                                    #    height_transform = self.height_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = IMPASTO_DataModule()
