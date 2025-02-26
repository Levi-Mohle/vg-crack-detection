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

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from notebooks.utils.crack_detection import crack_detection_total
from src.models.support_functions.evaluation import classify_metrics
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
                                    crack              = "realBI"
                                    )

lightning_data.setup()
loader = lightning_data.test_dataloader()
# %% Plot crack detection results
idx = 9
for i, (rgb, height, id) in enumerate(loader):
    
    crack_mask, _ = crack_detection_total(rgb[idx].permute(1,2,0).numpy(), \
                                    height[idx,0].numpy())

    if i==0:
        break

extent = [0,4,0,4]
fs = 16
fig, axes = plt.subplots(1,2, figsize=(10,15))
axes[0].imshow(rgb[idx].permute(1,2,0), extent=extent)
axes[1].imshow(crack_mask, extent=extent)
for ax in axes.flatten():
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

# %% Run crack detection algorithm over all data samples

y_pred = []
y_true = []
for i, (rgb, height, id) in enumerate(loader):
    for j in range(rgb.shape[0]): 
        crack_mask, _ = crack_detection_total(rgb[j].permute(1,2,0).numpy(), \
                                                height[j,0].numpy())
        ood_score = np.sum(crack_mask.astype(int))
        y_pred.append(ood_score)
    y_true.append(id)

y_pred = np.array(y_pred)
y_true = np.concatenate([y.numpy() for y in y_true]).astype(int)
# %% Get classification metrics

save_loc = r"C:\Users\lmohle\Downloads\result.txt"
classify_metrics(y_pred, y_true, save_loc)