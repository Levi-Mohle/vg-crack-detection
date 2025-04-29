"""
Script create, add and augment synthetic cracks to dataset 
and save as HDF5 file.

    Source Name : adding_synthetic_cracks.py
    Contents    : Functions to apply synthetic cracks to existing data samples
                    and save into new HDF5 files
    Date        : 2025

 """
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
from notebooks.utils.dataset import HDF5PatchesDatasetCustom
from notebooks.utils.synthetic_cracks import *
from notebooks.utils.latent_space import *

# %% Load the data & model

# Define directories
local       = False
data_dir    = os.path.join(wd, "data", "impasto")
img_dir     = os.path.join(wd, "notebooks", "images")
if local:
    model_dir = r"C:\Users\lmohle\Documents\2_Coding\data\Trained_Models\AutoEncoderKL"
    MPEG_path   = r"C:\Users\lmohle\Documents\2_Coding\data\Datasets\MPEG400"
    device="cpu"
else:
    model_dir = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"
    MPEG_path   = r"/data/storage_crack_detection/datasets/MPEG400"
    device = "cuda"

# IMPASTO_train_dir = "2024-11-26_512x512_train.h5"
IMPASTO_train_dir = "2024-11-26_512x512_val.h5"
# IMPASTO_train_dir = "2025-02-18_Real_Cracks512x512_test.h5"
data_train = HDF5PatchesDatasetCustom(hdf5_file_path = os.path.join(data_dir, IMPASTO_train_dir))

dataloader_train = DataLoader(  dataset    = data_train,
                                batch_size = 16,
                                shuffle    = False,
                            )
# %% Adding cracks + encoding

# Get binary shape masks
cat_name    = 'brick'
masks       = get_shapes(MPEG_path, cat_name, plot=False)

# %%
output_dir = data_dir
output_filename = r"2024-11-26_mix_512x512_val.h5"
output_filename_full_h5 = os.path.join(output_dir, output_filename)
# For adding p percentage of cracks + encoding
# add_enc_synthetic_cracks_to_h5(dataloader   = dataloader_train, 
    #                            masks        = masks, 
    #                            p            = 1, 
    #                            filename     = output_filename_full_h5, 
    #                            vae          = vae,
    #                            add_cracks   = False,
    #                            segmentation = False, 
    #                            device       = device
    #                            )

# For only encoding dataset
# encode_and_add2h5(dataloader = dataloader_train, 
#                   filename   = output_filename_full_h5, 
#                   vae        = vae,
#                   device     = device
#                  )

# # For only encoding AND augmenting dataset
# encode_and_augment2h5(dataloader = dataloader_train, 
#                       filename   = output_filename_full_h5, 
#                       vae        = vae,
#                       device     = device
#                      )

# For adding p percentage of cracks
add_synthetic_cracks_to_h5( dataloader   = dataloader_train, 
                            masks        = masks, 
                            p            = 1, 
                            filename     = output_filename_full_h5, 
                            add_cracks   = True,
                            segmentation = False, 
                            )