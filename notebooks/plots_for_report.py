# %% Load libraries
import torch
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import skimage.morphology as morhpology
from tqdm import tqdm

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *

# %% Load the data

# Choose if run from local machine or SURF cloud
local = True

if local:
    data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto"
else:
    pass

lightning_data = IMPASTO_DataModule(data_dir           = data_dir,
                                    batch_size         = 18,
                                    variant            = "512x512_local",
                                    crack              = "synthetic"
                                    )

lightning_data.setup()
loader = lightning_data.test_dataloader()
# %% Plot a batch

for i, (rgb, height, id) in enumerate(loader):
     if i==1:
        break

idx = 2
x0_rgb      = torch.randn((3,512,512))
x0_height   = torch.randn((1,512,512))

rgb_noise       = []
height_noise    = []
T               = [0, .25, .5, 1]
for t in T:
    xt_rgb      = torch.clamp((1-t)*rgb[idx] / 255 + t*x0_rgb, 0, 1)
    xt_height   = torch.clamp((1-t)*height[idx] / 2**16 + t*x0_height, 0 ,1)

    rgb_noise.append(xt_rgb)
    height_noise.append(xt_height)

extent = [0,4,0,4]
fs = 16
fig, axes = plt.subplots(1,2, figsize=(10,15))
for i, ax in enumerate(axes.flatten()):
    ax.set_title(f"t={T[i]:.2f}")
    ax.imshow(rgb_noise[i].permute(1,2,0), extent=extent)
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

fig, axes = plt.subplots(1,2, figsize=(10,15))
for i, ax in enumerate(axes.flatten()):
    ax.set_title(f"t={T[i]:.2f}")
    ax.imshow(height_noise[i].permute(1,2,0), extent=extent)
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

# %% 