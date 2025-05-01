"""
Script to post-process decoded reconstructions obtained 
from the machine learning model 

    Source Name : post_processing.py
    Contents    : Post processing + plotting functions
    Date        : 2025

 """

# %% Load libraries + pretrained VAE
import torch
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

# Local imports
from src.data.components.transforms import *
import src.models.components.utils.evaluation as evaluation
import src.models.components.utils.post_process as post_process
from src.models.components.utils.visualization import visualize_post_processing
from notebooks.utils.dataset import HDF5PatchesDatasetReconstructs

# %% Load data

# Choose input files with saved input and reconstructed samples
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-11_synthetic_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-11_real_reconstructs.h5"
# input_file_name = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto/2025-02-17_real_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-28_cDDPM_0.8_realBI_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-27_gc_FM_0.4_realBI_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-03-14_cDDIM2_0.4_realBI_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-04-16_cDDM_0.4_realAB_reconstructs.h5"
input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-04-28_cDDPM_0.4_realBI_1.4_reconstructs.h5"

# True if classifier free guidance FM is used
cfg = False

# Load dataset
reconstruct_dataset = HDF5PatchesDatasetReconstructs(input_file_name,
                                                     cfg= cfg,
                                                     rgb_transform=None,
                                                     height_transform=None)

# Create dataloader
dataloader = DataLoader(reconstruct_dataset, batch_size=80, shuffle=False)

# %% Load 1 batch of data

if cfg:
    for rgb, height, r0_rgb, r0_height, r1_rgb, r1_height, target in dataloader:
        x = torch.concat([rgb, height], dim=1)
        reconstructs = [torch.concat([r0_rgb, r0_height], dim=1), 
                        torch.concat([r1_rgb, r1_height], dim=1)]
        break
else:
    for rgb, height, r_rgb, r_height, target in dataloader:
        x = torch.concat([rgb, height], dim=1)
        reconstructs = torch.concat([r_rgb, r_height], dim=1)
        break    

# %% Post processing reconstructions

# Function to loop over all batches and apply post-processing
def OOD_predictions(dataloader, cfg, r=0):
    targets     = []
    predictions = []
    if cfg:
        for rgb, height, r0_rgb, r0_height, r1_rgb, r1_height, target in dataloader:

            x   = torch.concat([rgb,height], dim=1)
            r0  = torch.concat([r0_rgb,r0_height], dim=1)
            r1  = torch.concat([r1_rgb,r1_height], dim=1)

            if r == 0: r=r0
            else: r=r1
            x0, x1      = post_process.to_gray_0_1(x), post_process.to_gray_0_1(r)
            ood_score   = post_process.get_OOD_score(x0=x0, x1=x1)

            predictions.append(ood_score)
            targets.append(target)
    else:
        for rgb, height, r_rgb, r_height, target in dataloader:

            x   = torch.concat([rgb,height], dim=1)
            r0  = torch.concat([r_rgb,r_height], dim=1)

            x0, x1      = post_process.to_gray_0_1(x), post_process.to_gray_0_1(r0)
            ood_score   = post_process.get_OOD_score(x0=x0, x1=x1)

            predictions.append(ood_score)
            targets.append(target)

    y_true  = np.concatenate([t.numpy() for t in targets])
    y_score = np.concatenate([t for t in predictions])

    return y_score, y_true

def get_rectangle(mask):
    rows, cols = np.where(mask>0)
    xmin, xmax = cols.min(), cols.max()
    ymin, ymax = rows.min(), rows.max()
    w = xmax - xmin
    h = ymax - ymin
    rect = patches.Rectangle((xmin,ymin), w, h, linewidth=2, edgecolor='r', facecolor="none")
    return rect

def plotting_lifted_edge(x, recon, ano_maps, idx=0):
    fig, axes = plt.subplots(1,4, figsize=(12,6))
    axes[0].imshow(x[idx,1])
    rect = get_rectangle(ano_maps[idx])
    axes[0].add_patch(rect)
    
    axes[1].imshow(recon[idx,1])
    axes[2].imshow(ano_maps[idx])

    sobel = skimage.filters.sobel(x[idx,1].numpy())
    sobel = (sobel > .02).astype(int)
    axes[3].imshow(sobel)
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
    plt.show()

# %% Classifying + plotting results

y_score, y_true = OOD_predictions(dataloader, cfg)
evaluation.plot_classification_metrics(y_score, y_true)
evaluation.plot_histogram(y_score, y_true)

# %% Extra results 

x0          = post_process.to_gray_0_1(x)
x1          = post_process.to_gray_0_1(reconstructs)

# Review intermediate post processing results + plotting
ssim, filt1, filt2, ano_map = post_process.individual_post_processing(x0,x1,idx=61)
visualize_post_processing(ssim, filt1, filt2, ano_map)
# plotting_lifted_edge(x, reconstructs, ano_maps, idx=1)


# %% Check correct/wrong predictions

# Convenient if you want to plot all patches for review
fig, axes   = plt.subplots(9,9, figsize=(40,40))
x0          = post_process.to_gray_0_1(x)
y_check = (y_score >= 401.0) == y_true
for i, ax in enumerate(axes.flatten()):
    label = "correct" if y_check[i] else "wrong"
    ax.imshow(x0[i,0])
    ax.set_title(f"ID: {i}, Predcition: {label}")
    if i == x0.shape[0] - 1:
        break
fig.tight_layout()   

# %%
