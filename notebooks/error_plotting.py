# %% Load libraries + pretrained VAE
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
from skimage.metrics import structural_similarity
from torchvision.transforms import transforms
import skimage.morphology as morhpology
from tqdm import tqdm

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.components.transforms import *
from src.models.support_functions.evaluation import *
from notebooks.preprocess_latent_space.dataset import HDF5PatchesDatasetReconstructs

# %% Load data

# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-11_synthetic_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-11_real_reconstructs.h5"
input_file_name = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto/2025-02-17_real_reconstructs.h5"

reconstruct_dataset = HDF5PatchesDatasetReconstructs(input_file_name,
                                                     cfg= True,
                                                     rgb_transform=revert_normalize_rgb(),
                                                     height_transform= revert_normalize_height())

# Plot some mini-patches
dataloader = DataLoader(reconstruct_dataset, batch_size=16, shuffle=False)
# %% Load 1 batch of data

for rgb, height, r0_rgb, r0_height, r1_rgb, r1_height, target in dataloader:
    x = torch.concat([rgb, height], dim=1)
    reconstructs = [torch.concat([r0_rgb, r0_height], dim=1), 
                    torch.concat([r1_rgb, r1_height], dim=1)]
    break

# %% Visualize reconstructs
def class_reconstructs_2ch(x, reconstructs, target, plot_ids, win_size=5, fs=12):

    ssim_orig_vs_reconstruct = []
    for i, reconstruct in enumerate(reconstructs):
        # Calculate SSIM between original sample and all reconstructed labels
        _, ssim_img = ssim_for_batch(x, reconstructs[i], win_size)
        ssim_orig_vs_reconstruct.append(ssim_img) # (ssim_img > -0.1).astype(int)
        
    _, ssim_l0_vs_l1 = ssim_for_batch(reconstructs[0], reconstructs[1], win_size)

    extent = [0,4,0,4]
    for i in plot_ids:
        fig = plt.figure(constrained_layout=False, figsize=(15,17))
        gs = GridSpec(4, 4, figure=fig, width_ratios=[1.08,1,1.08,1.08], height_ratios=[1,1,1,1], hspace=0.2, wspace=0.2)
        
        # RGB images
        # Span whole column
        ax1 = fig.add_subplot(gs[0:2,0])
        ax6 = fig.add_subplot(gs[0:2,3])

        # Regular grid
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[0,2])
        ax5 = fig.add_subplot(gs[1,2])

        # Height images
        # Span whole column
        ax7 = fig.add_subplot(gs[2:4,0])
        ax12 = fig.add_subplot(gs[2:4,3])

        # Regular grid
        ax8  = fig.add_subplot(gs[2,1])
        ax9  = fig.add_subplot(gs[3,1])
        ax10 = fig.add_subplot(gs[2,2])
        ax11 = fig.add_subplot(gs[3,2])

        # Plot rgb
        im1 = ax1.imshow(x[i,0], extent=extent, vmin=0, vmax=1)
        ax1.set_yticks([0,1,2,3,4])
        # ax1.tick_params(axis='both', which='both', labelbottom=False, labelleft=True)
        ax1.set_title("Original sample", fontsize =fs)
        ax1.set_ylabel("Y [mm]")
        ax1.set_xlabel("X [mm]")
        ax1.text(-0.3, 0.5, f"Gray-scale {target[i]}", fontsize= fs*2, rotation=90, va="center", ha="center", transform=ax1.transAxes)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax1)

        for j, ax in enumerate([ax2, ax3]):
            ax.imshow(reconstructs[j][i,0], extent=extent, vmin=0, vmax=1)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax.set_title(f"Reconstructed sample label {j}", fontsize =fs)

        for j, ax in enumerate([ax4, ax5]):
            im = ax.imshow(ssim_orig_vs_reconstruct[j][i,0], extent=extent)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_title(f"SSIM label {j} recon vs orig", fontsize =fs)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        im6 = ax6.imshow(ssim_l0_vs_l1[i,0], extent=extent, vmin=0)
        ax6.set_yticks([0,1,2,3,4])
        ax6.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
        ax6.set_xlabel("X [mm]")
        ax6.set_title(f"SSIM label 0 vs label 1 recon", fontsize =fs)
        divider = make_axes_locatable(ax6)
        cax6 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im6, cax=cax6)

        # Plot height
        im7 = ax7.imshow(x[i,1], extent=extent, vmin=0, vmax=1)
        ax7.set_yticks([0,1,2,3,4])
        ax7.set_title("Original sample", fontsize =fs)
        ax7.set_ylabel("Y [mm]")
        ax7.set_xlabel("X [mm]")
        ax7.text(-0.3, 0.5, f"Height {target[i]}", fontsize= fs*2, rotation=90, va="center", ha="center", transform=ax7.transAxes)
        divider = make_axes_locatable(ax7)
        cax7 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im7, cax=cax7)

        for j, ax in enumerate([ax8, ax9]):
            ax.imshow(reconstructs[j][i,1], extent=extent, vmin=0, vmax=1)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax.set_title(f"Reconstructed sample label {j}", fontsize =fs)

        for j, ax in enumerate([ax10, ax11]):
            im = ax.imshow(ssim_orig_vs_reconstruct[j][i,1], extent=extent)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_title(f"SSIM label {j} recon vs orig", fontsize =fs)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        im12 = ax12.imshow(ssim_l0_vs_l1[i,1], extent=extent)
        ax12.set_yticks([0,1,2,3,4])
        ax12.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
        ax12.set_xlabel("X [mm]")
        ax12.set_title(f"SSIM label 0 vs label 1 recon", fontsize =fs)
        divider = make_axes_locatable(ax12)
        cax12 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im12, cax=cax12)

# class_reconstructs_2ch(x, reconstructs, target, plot_ids=[1])

# %% Get classification metrics

def OOD_proxy(batch, r_batch, win_size=5):
    batch   = batch.cpu().numpy()
    r_batch = r_batch.cpu().numpy()
    bs = batch.shape[0]
    ssim_batch     = np.zeros((batch.shape[0],batch.shape[1]))
    ssim_batch_img = np.zeros_like(batch)
    for i in range(bs):
        for j in range(batch.shape[1]):
            ssim,  img_ssim = structural_similarity(batch[i,j], 
                                r_batch[i,j],
                                win_size=win_size,
                                data_range=1,
                                full=True)
            ssim_batch_img[i, j] = img_ssim * -1
            ssim_batch[i, j]     = np.sum(ssim_batch_img[i, j] > -.7)
    
    OOD_mask = (ssim_batch_img[:,0] > -0.7) & (ssim_batch_img[:,1] > -0.95)
    ssim_comb = np.sum(OOD_mask, axis=(1,2))
    return ssim_comb, ssim_batch, ssim_batch_img

# %% Post processing SSIM results

def filter_eccentricity(image):
    regions = skimage.measure.regionprops(skimage.measure.label(image))
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    for region in regions:
        if region.eccentricity > 0.85:
            filtered_mask[region.coords[:,0], region.coords[:,1]] = 1
    return filtered_mask

def post_process_ssim(x, ssim_img):
    ssim_filt = np.zeros_like(ssim_img)

    # Sobel filter
    sobel = skimage.filters.sobel(x[:,1].numpy())
    sobel = (sobel > .02).astype(int)

    for idx in range(ssim_img.shape[0]):
        for i in range(ssim_img.shape[1]):

            # Thresholding
            ssim_filt[idx,i] = (ssim_img[idx,i] > np.percentile(ssim_img[idx,i], q=95)).astype(int)

            # # Variation denoising
            # ssim_filt[idx, i] = skimage.restoration.denoise_tv_chambolle(ssim_img[idx,i],
            #                                                              weight=0.1 
            #                                                             )
            # # Gaussian filter to remove noise
            # ssim_filt[idx,i] = skimage.filters.gaussian(ssim_filt[idx,i],
            #                                             sigma=1.5 )
            
            # Morphology filters
            ssim_filt[idx,i] = skimage.morphology.erosion(ssim_filt[idx,i],
                                                            )
            
            # ssim_filt[idx,i] = filter_eccentricity(ssim_filt[idx,i]
            #                                                 )
            
        ssim_filt[idx,i] = ((ssim_filt[idx,0]   == 1) & 
                            (ssim_filt[idx,1]   == 1) &
                            (sobel[idx]         == 1)
                            ).astype(int)
        
        # Opening (Erosion + Dilation) to remove noise + connect shapes
        ssim_filt[idx,i] = skimage.morphology.opening(ssim_filt[idx,i])

        # ssim_filt[idx,i] = skimage.morphology.dilation(ssim_filt[idx,i])
        # ssim_filt[idx,i] = skimage.morphology.dilation(ssim_filt[idx,i])
    
    

    ssim_sum = np.sum(ssim_filt[:,i], axis=(1,2))
                
    return ssim_filt, ssim_sum

def OOD_proxy_filtered(x1, x2):
    _, ssim_img     = ssim_for_batch(x1, x2)
    y               = post_process_ssim(x1, ssim_img)
    return y

def classify(dataloader):
    targets     = []
    predictions = []
    for rgb, height, r0_rgb, r0_height, r1_rgb, r1_height, target in dataloader:

        x   = torch.concat([rgb,height], dim=1)
        r0  = torch.concat([r0_rgb,r0_height], dim=1)
        r1  = torch.concat([r1_rgb,r1_height], dim=1)

        # ssim, _, _ = OOD_proxy(r0, r1)
        _, ssim = OOD_proxy_filtered(x, r0)

        predictions.append(ssim)
        targets.append(target)

    y_true  = np.concatenate([t.numpy() for t in targets])
    y_score = np.concatenate([t for t in predictions])

    # classify_metrics(y_score, y_true)
    plot_histogram(y_score, y_true)

def plotting(x, y, idx=0):
    z = np.concatenate([x, y], axis=1)
    fig, axes = plt.subplots(2,2)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(z[idx,i])
        ax.axis("off")
    plt.show()

def get_rectangle(mask):
    rows, cols = np.where(mask>0)
    xmin, xmax = cols.min(), cols.max()
    ymin, ymax = rows.min(), rows.max()
    w = xmax - xmin
    h = ymax - ymin
    rect = patches.Rectangle((xmin,ymin), w, h, linewidth=2, edgecolor='r', facecolor="none")
    return rect

def plotting_lifted_edge(x, recon, y, idx=0):
    fig, axes = plt.subplots(1,4, figsize=(12,6))
    axes[0].imshow(x[idx,1])
    rect = get_rectangle(y[idx,1])
    axes[0].add_patch(rect)
    
    axes[1].imshow(recon[idx,1])
    axes[2].imshow(y[idx,1])

    sobel = skimage.filters.sobel(x[idx,1].numpy())
    sobel = (sobel > .02).astype(int)
    axes[3].imshow(sobel)
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
    plt.show()

    return sobel

# Plotting post process results
_, ssim_img     = ssim_for_batch(x, reconstructs[1])
y, _            = post_process_ssim(x, ssim_img)
# plotting(ssim_img, y, idx=0)
# sobel = plotting_lifted_edge(x, ssim_img, y, idx=9)

# Classifying
classify(dataloader)
# %%
