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
data_dir = r"C:\Users\lmohle\Documents\2_Coding\data\input\Training_data\512x512"
IMPASTO_train_dir = "2025-01-07_Real_Cracks512x512_test.h5"
data_train = HDF5PatchesDatasetCustom(hdf5_file_path = os.path.join(data_dir, IMPASTO_train_dir))

dataloader_train = DataLoader(  dataset=data_train,
                                batch_size= 16,
                                shuffle=False,
                            )

device      = "cpu" 
# model_dir  = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"
model_dir   = r"C:\Users\lmohle\Documents\2_Coding\data\Trained_Models\AutoEncoderKL"
add_cracks  = False

with torch.no_grad():
    vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True).to(device)

# %% Adding cracks + encoding

# Get binary shape masks
# MPEG_path   = r"/data/storage_crack_detection/datasets/MPEG400"
MPEG_path   = r"C:\Users\lmohle\Documents\2_Coding\data\Datasets\MPEG400"
cat_name    = 'brick'
masks       = get_shapes(MPEG_path, cat_name, plot=False)

# %%
output_dir = data_dir
# output_filename = r"2025-01-07_Enc_Real_Crack512x512_test.h5"
output_filename = "test.h5"
output_filename_full_h5 = os.path.join(output_dir, output_filename)
add_synthetic_cracks_to_h5(dataloader   = dataloader_train, 
                           masks        = masks, 
                           p            = 0.5, 
                           filename     = output_filename_full_h5, 
                           vae          = vae,
                           add_cracks   = True, 
                           )

# ### TEMPORARY ###
# for i, (rgb, height, id) in tqdm(enumerate(dataloader_train)):
#     height_cracks, rgb_cracks, _ = Create_cracks_with_lifted_edges(height, 
#                                                                 rgb, 
#                                                                 masks=masks, 
#                                                                 decay_rate=2)
#     break

# fig, axes = plt.subplots(4,4)
# for i, ax in enumerate(axes.flatten()):
#     # ax.imshow(rgb_cracks[i].permute(1,2,0))
#     ax.imshow(height_cracks[i,0])
#     ax.axis("off")
# %%

