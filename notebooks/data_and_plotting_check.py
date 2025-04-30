"""
Script to plot data samples from PyTorch Lightning datamodule and test transform
functions 

    Source Name : dataset.py
    Contents    : Data loading and plotting functions
    Date        : 2025

 """
# %%
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import Augmentation, CNNTransform
# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"C:\Users\lmohle\Documents\2_Coding\ml-crack-detection-van-gogh\data\impasto",
                                    batch_size         = 16,
                                    variant            = "512x512",
                                    transform          = None,
                                    crack              = "realAB"
                                    )
lightning_data.setup()
loader = lightning_data.test_dataloader()

# %% Load first batch + check size

for i, (rgb, height, id) in enumerate(loader):
     print(f"{'Shape of RGB: ':<25}{rgb.shape}")
     print(f"{'Shape of Height: ':<25}{height.shape}")
     print(f"{'Shape of True label: ':<25}{id.shape}")
     break

# %% Plot mini-patches (ONLY USE WHEN IMAGE DATASET IS LOADED)

# Plot 16 patches
fig, axes = plt.subplots(4,4)
for i, ax in enumerate(axes.flatten()):
    # ax.imshow(rgb_cracks[i].permute(1,2,0))
    ax.imshow(rgb[i].permute(1,2,0))
    ax.axis("off")
fig.tight_layout()

fig, axes = plt.subplots(4,4)
for i, ax in enumerate(axes.flatten()):
    # ax.imshow(rgb_cracks[i].permute(1,2,0))
    ax.imshow(height[i,0])
    ax.axis("off")
fig.tight_layout()

# Plot individual patch
idx  = 0
data = [rgb[idx], height[idx]]

fig, axes = plt.subplots(1,2, figsize=(15,10))
for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(data[i].permute(1,2,0))
    # ax.axis("off")

# %% Check range of values (only relevant in case of [0,1] or [-1,1] normalization)
avg_diff_gray   = []
avg_diff_height = []

for i, (gray, height, id) in enumerate(loader):
    avg_diff_gray.append((gray.max() - gray.min()).item())
    avg_diff_height.append((height.max() - height.min()).item())
print(f"mean diff gray: {torch.mean(torch.tensor(avg_diff_gray).to(torch.float32)):.2f}")
print(f"mean diff height: {torch.mean(torch.tensor(avg_diff_height).to(torch.float32)):.2f}")

# %% Plot encoder outputs (ONLY USE WHEN ENCODED DATASET IS LOADED)

fig, axes = plt.subplots(1,4, figsize=(20,80))
for i, ax in enumerate(axes.flatten()):
     ax.imshow(rgb[0][i:i+1].permute(1,2,0))

fig, axes = plt.subplots(1,4, figsize=(20,80))
for i, ax in enumerate(axes.flatten()):
     ax.imshow(height[0][i:i+1].permute(1,2,0))

# %%
