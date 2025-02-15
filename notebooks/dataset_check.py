# %% Imports

import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.components.transforms import *
from notebooks.preprocess_latent_space.dataset import HDF5PatchesDatasetCustom
from notebooks.preprocess_latent_space.cracks import *

# %% Load the data & model
# data_dir = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto"
# data_dir = r"C:\Users\lmohle\Documents\2_Coding\data\input\Training_data\512x512"
data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto"
IMPASTO_train_dir = "2024-11-26_Enc_synthetic_mix_512x512_train.h5"
# IMPASTO_train_dir = "2024-11-26_512x512_val.h5"
data_train = HDF5PatchesDatasetCustom(hdf5_file_path = os.path.join(data_dir, IMPASTO_train_dir))

dataloader_train = DataLoader(  dataset    = data_train,
                                batch_size = 16,
                                shuffle    =False,
                            )

print(dataloader_train.__len__())
for rgb, height, target in dataloader_train:
    print(rgb.shape)
    print(height.shape)
    print(target.shape)
    break
