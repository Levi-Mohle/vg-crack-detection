# %%
import sys
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from torchvision.transforms.functional import rgb_to_grayscale
# from torchvision.transforms import GaussianBlur

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from src.models.support_functions.evaluation import ssim_for_batch
# %% Functions


# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto",
                                    batch_size = 16,
                                    variant = "Enc_mix_512x512",
                                    # rgb_transform = diffuser_normalize(),
                                    # height_transform = diffuser_normalize_height_idv()
                                    )
lightning_data.setup()
test_loader = lightning_data.test_dataloader()

for _,_, label in test_loader:
    print(label)
