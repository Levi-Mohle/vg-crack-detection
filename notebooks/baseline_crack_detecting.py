"""
Script to apply the baseline crack detection algorithm to the 
impasto dataset, together with plotting intermediate results

    Source Name : baseline_crack_detecting.py
    Contents    : Functions to run detection algorithm iteratively and 
                    obtain classification metrics
    Date        : 2025

 """
# %% Load libraries
import torch
import numpy as np
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from skimage.morphology import skeletonize, binary_dilation, binary_erosion

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from notebooks.utils.baseline_crack_detection import *
from src.models.components.utils.evaluation import *
# %% Load the data

# Define directories
local       = False
data_dir    = os.path.join(wd, "data", "impasto")
img_dir     = os.path.join(wd, "notebooks", "images")

lightning_data = IMPASTO_DataModule(data_dir           = data_dir,
                                    batch_size         = 16,
                                    variant            = "512x512",
                                    crack              = "synthetic"
                                    )

lightning_data.setup()
loader = lightning_data.test_dataloader()

# %% Plot crack detection results
idx = 0
for i, (rgb, height, id) in enumerate(loader):
    data_color  = rgb[idx].permute(1,2,0).numpy()
    data_height = height[idx,0].numpy()
    crack_mask, data_nearest = crack_detection_total(data_color, data_height)

    if i==0:
        break

extent = [0,4,0,4]
fs = 16
fig, axes = plt.subplots(1,2, figsize=(10,15))
axes[0].imshow(height[idx].permute(1,2,0), extent=extent)
axes[1].imshow(crack_mask, extent=extent)
for ax in axes.flatten():
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

# %% Analysis of intermediate steps
data_gray = cv2.cvtColor(data_color, cv2.COLOR_BGR2YCR_CB)[:,:,0] #get brightness
mask_filter_gray_1, tophat_img = morphology_only(data_gray)

filtered_image, Rb_total, S2_total  = frangi_filter(data_height * 0.25, scale_range=(3, 10))
frangi_segmentation                 = (filtered_image >= 0.01)

mask_filter_frangi = np.zeros_like(data_gray)
mask_filter_frangi[binary_dilation(frangi_segmentation)] = binary_dilation(tophat_img)[binary_dilation(frangi_segmentation)] #combine color and height data
filter_shape = filter_eccentricity(mask_filter_frangi)

detected_mask = binary_dilation(binary_erosion(binary_dilation(filter_shape)))

fig, axes = plt.subplots(1,2, figsize=(10,15))
axes[0].imshow(mask_filter_gray_1, extent=extent)
axes[1].imshow(tophat_img, extent=extent)
for ax in axes.flatten():
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

fig, axes = plt.subplots(1,2, figsize=(10,15))
axes[0].imshow(filtered_image, extent=extent)
axes[1].imshow(frangi_segmentation, extent=extent)
for ax in axes.flatten():
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

fig, axes = plt.subplots(1,2, figsize=(10,15))
axes[0].imshow(mask_filter_frangi, extent=extent)
axes[1].imshow(detected_mask, extent=extent)
for ax in axes.flatten():
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
fig.tight_layout()

# %% Run crack detection algorithm over all data samples
start_time = time.time()

y_score = []
y_true = []
for i, (rgb, height, id) in enumerate(tqdm(loader)):
    for j in range(rgb.shape[0]): 
        crack_mask, _ = crack_detection_total(rgb[j].permute(1,2,0).numpy(), \
                                                height[j,0].numpy())
        ood_score = np.sum(crack_mask.astype(int))
        y_score.append(ood_score)
    y_true.append(id)

y_score = np.array(y_score)
y_true = np.concatenate([y.numpy() for y in y_true]).astype(int)

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")
# %% Get classification metrics

classify_metrics(y_score, y_true)
plot_histogram(y_score, y_true)
plot_classification_metrics(y_score, y_true)

