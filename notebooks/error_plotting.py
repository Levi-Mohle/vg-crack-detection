# %% Load libraries + pretrained VAE
import torch
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from skimage.metrics import structural_similarity
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import skimage.morphology as morhpology
from tqdm import tqdm

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.components.transforms import *
from notebooks.preprocess_latent_space.dataset import HDF5PatchesDatasetReconstructs

# %% Load data

input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\input\Training_data\512x512\2025-01-22_reconstructs.h5"

reconstruct_dataset = HDF5PatchesDatasetReconstructs(input_file_name)

# Plot some mini-patches
dataloader = DataLoader(reconstruct_dataset, batch_size=3, shuffle=False)
# %%

for rgb, height, r_rgb, r_height, id in dataloader:
    print(rgb.shape)
    print(height.shape)
    print(r_rgb.shape)
    print(r_height.shape)
    print(id.shape)
    break

def to_grayscale(x):
    x = (x + 1) / 2
    return rgb_to_grayscale(x)

def to_0_1(x):
    return (x + 1) / 2

rgb         = to_grayscale(rgb)
r_rgb       = to_grayscale(r_rgb)
height      = to_0_1(height)
r_height    = to_0_1(r_height)

# rgb         = rgb.numpy()
# r_rgb       = r_rgb.numpy()
# height      = height.numpy()
# r_height    = r_height.numpy()
# %%
win_size =3

# _, img_ssim_rgb = structural_similarity(rgb[0,0], 
#                             r_rgb[0,0],
#                             win_size=win_size,
#                             data_range=1,
#                             full=True)

# _, img_ssim_height = structural_similarity(height[0,0], 
#                             r_height[0,0],
#                             win_size=win_size,
#                             data_range=1,
#                             full=True)

def ssim_for_batch(batch, r_batch):
    batch   = batch.numpy()
    r_batch = r_batch.numpy()
    bs = batch.shape[0]
    ssim_batch = np.zeros_like(batch)
    for i in range(bs):
        _,  img_ssim = structural_similarity(batch[i,0], 
                            r_batch[i,0],
                            win_size=3,
                            data_range=1,
                            full=True)
        ssim_batch[i, 0] = (img_ssim - img_ssim.max()) * -1
    
    return ssim_batch

ssim_rgb_batch = ssim_for_batch(rgb, r_rgb)
ssim_height_batch = ssim_for_batch(height, r_height)

# %%   

# ssim = SSIM(gaussian_kernel=False,
#             data_range=1,
#             kernel_size=5,
#             return_full_image=True)
# _, img_ssim_rgb = ssim(rgb, r_rgb)
# _, img_ssim_height = ssim(height, r_height)

# invert ssim
img_ssim_rgb    = (img_ssim_rgb - img_ssim_rgb.max()) * -1
img_ssim_height = (img_ssim_height- img_ssim_height.max()) * -1

# Morphology filter
img_ssim_rgb    = morhpology.white_tophat(img_ssim_rgb)
img_ssim_height = morhpology.white_tophat(img_ssim_height)

# %%
i, j = 2, 0
fig, axes = plt.subplots(2, 3, figsize=(15,12))
axes[0,0].imshow(rgb[i,0])
axes[0,0].set_title(f"Original mini patch", fontsize=16)
axes[0,0].axis('off')

axes[0,1].imshow(r_rgb[i,0])
axes[0,1].set_title(f"Reconstructed", fontsize=16)
axes[0,1].axis('off')

im3 = axes[0,2].imshow(ssim_rgb_batch[i,0])
axes[0,2].set_title(f"Reconstructed", fontsize=16)
divider = make_axes_locatable(axes[0,2])
cax1 = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im3, cax=cax1)
axes[0,2].axis('off')

axes[1,0].imshow(height[i,0])
axes[1,0].set_title(f"Original mini patch", fontsize=16)
axes[1,0].axis('off')

axes[1,1].imshow(r_height[i,0])
axes[1,1].set_title(f"Reconstructed", fontsize=16)
axes[1,1].axis('off')

im5 = axes[1,2].imshow(ssim_height_batch[i,0])
axes[1,2].set_title(f"Reconstructed", fontsize=16)
divider = make_axes_locatable(axes[1,2])
cax1 = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im5, cax=cax1)
axes[1,2].axis('off')


plt.tight_layout()
plt.show()

# %%
