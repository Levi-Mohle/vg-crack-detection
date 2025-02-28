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
from scipy.signal import convolve2d
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
# input_file_name = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto/2025-02-17_real_reconstructs.h5"
# input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-28_cDDPM_0.8_realBI_reconstructs.h5"
input_file_name = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-27_gc_FM_0.4_realBI_reconstructs.h5"

cfg = True
reconstruct_dataset = HDF5PatchesDatasetReconstructs(input_file_name,
                                                     cfg= cfg,
                                                     rgb_transform=revert_normalize_rgb(),
                                                     height_transform= revert_normalize_height())

# Plot some mini-patches
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
        plt.suptitle(f"{target[i]}")
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

class_reconstructs_2ch(x, reconstructs, target, plot_ids=[1])

# %% Post processing SSIM results

def filter_eccentricity(image):
    regions = skimage.measure.regionprops(skimage.measure.label(image))
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    for region in regions:
        if region.eccentricity > 0.85:
            filtered_mask[region.coords[:,0], region.coords[:,1]] = 1
    return filtered_mask

def post_process_ssim2(x0, ssim_img):
    """
    Given the input sample x0 and anomaly maps produced with SSIM,
    this function filters the anomaly maps of noise and non-crack 
    related artifacts. It derives an OOD-score from the filtered map.

    Args:
        x0 (2D tensor) : input sample. 2 channels contain grayscale
                            and height (Bx2xHxW)
        ssim_img (2D tensor) : reconstruction of x0 (Bx2xHxW)
        
    Returns:
        ano_maps (2D tensor) : filtered anomaly map (Bx1xHxW)
        ood_score (1D tensor) : out-of-distribution scores (Bx1)
    """
    # Create empty tensor for filtered ssim and anomaly maps
    ssim_filt = np.zeros_like(ssim_img)
    ano_maps  = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))

    # Sobel filter on height map
    sobel_filt = sobel(x0[:,1].cpu().numpy())
    sobel_filt = (sobel_filt > .02).astype(int)

    # Loop over images in batch and both channels. Necessary since
    # skimage has no batch processing
    for idx in range(ssim_img.shape[0]):
        for i in range(ssim_img.shape[1]):

            # Thresholding
            ssim_filt[idx,i] = (ssim_img[idx,i] > np.percentile(ssim_img[idx,i], q=95)).astype(int)
            
            # Morphology filters
            ssim_filt[idx,i] = morhpology.binary_erosion(ssim_filt[idx,i])

            # ssim_filt[idx,i] = morhpology.binary_opening(ssim_filt[idx,i])
            # ssim_filt[idx,i] = morhpology.dilation(ssim_filt[idx,i])

        # Boolean masks: if pixel is present in ssim height, ssim rgb
        # and sobel filter, it is accounted as crack pixel  
        # for layer in [ssim_filt[idx,0], ssim_filt[idx,1], sobel_filt[idx]]:
        # #     # ano_maps[idx] += convolve2d(layer, kernel, mode = "same")
        #     ano_maps[idx] += layer
        

        # ano_maps[idx] = (ano_maps[idx] >= 2).astype(int)
        ano_maps[idx] = ((ssim_filt[idx,0]   == 1) & 
                        (ssim_filt[idx,1]   == 1) &
                        (sobel_filt[idx]    == 1)
                        ).astype(int)
        
        # Opening (Erosion + Dilation) to remove noise + connect shapes
        ano_maps[idx] = morhpology.opening(ano_maps[idx])
    
    # Calculate OOD-score, based on total number of crack pixels
    ood_score = np.sum(ano_maps, axis=(1,2))
                
    return ano_maps, ood_score

def get_ood_scores(dataloader):
    targets     = []
    predictions = []
    if cfg:
        for rgb, height, r0_rgb, r0_height, r1_rgb, r1_height, target in dataloader:

            x   = torch.concat([rgb,height], dim=1)
            r0  = torch.concat([r0_rgb,r0_height], dim=1)
            r1  = torch.concat([r1_rgb,r1_height], dim=1)

            # ssim, _, _ = OOD_proxy(r0, r1)
            # _, ssim = OOD_proxy_filtered(x, r0)
            _, ood_score = OOD_score(x0=x, x1=x, x2=r0)

            predictions.append(ood_score)
            targets.append(target)
    else:
        for rgb, height, r_rgb, r_height, target in dataloader:

            x   = torch.concat([rgb,height], dim=1)
            r0  = torch.concat([r_rgb,r_height], dim=1)

            # ssim, _, _ = OOD_proxy(r0, r1)
            # _, ssim = OOD_proxy_filtered(x, r0)
            _, ood_score = OOD_score(x0=x, x1=x, x2=r0)

            predictions.append(ood_score)
            targets.append(target)

    y_true  = np.concatenate([t.numpy() for t in targets])
    y_score = np.concatenate([t for t in predictions])

    return y_score, y_true

def classify(dataloader):
    
    y_score, y_true = get_ood_scores(dataloader)

    # classify_metrics(y_score, y_true)
    # plot_histogram(y_score, y_true)
    _, _, thresholds    = roc_curve(y_true, y_score)
    print_confusion_matrix(y_score, y_true, thresholds)

def plotting(x, ssim, post, idx=0):
    z = np.concatenate([x, ssim, post], axis=1)
    fig, axes = plt.subplots(1,5, figsize=(12,6))
    titles = ['RGB', 'height', 'SSIM rgb', 'SSIM height', 'Post']
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(z[idx,i])
        ax.axis("off")
        ax.set_title(titles[i])
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

def threshold_mover(y_score, y_true, step_backward=0):

    auc_score           = roc_auc_score(y_true, y_score)
    _, _, thresholds    = roc_curve(y_true, y_score)
    np.append(thresholds, -np.inf)

    best_accuracy = 0
    best_threshold = None

    for i, th in enumerate(thresholds):
        y_pred      = (y_score >= th).astype(int)
        accuracy    = np.mean(y_pred == y_true)

        if accuracy > best_accuracy:
            best_y_pred     = y_pred
            best_accuracy   = accuracy
            best_threshold  = th
            best_i          = i

    y_pred      = (y_score >= thresholds[best_i+step_backward]).astype(int)
    accuracy    = np.mean(y_pred == y_true)
    
    cm = confusion_matrix(y_true, y_pred)
    name_true = ["No crack true", "Crack true"]
    name_pred = ["No crack pred", "Crack pred"]
    cm_df = DataFrame(cm, index=name_true, columns=name_pred)

    print("##############################################")
    print(f"Confusion Matrix for best accuracy {accuracy:.3f}:")
    print(cm_df)
    print("")
    print(f"Given best threshold value: {thresholds[best_i+step_backward]}")
    print(f"AUC score: {auc_score:.3f}")
    print(f"Recall: {cm[1,1]/(cm[1,0]+cm[1,1])}")
    print("##############################################")

def OOD_score(x0, x1, x2):
    """
    Given the original sample x0 and its reconstructions x1 and x2, 
    this function returns the filtered anomaly map and OOD-score to be
    used in classification. If comparison is made between x0 and x1 or x2,
    provide x1 = x0.

    Args:
        x0 (2D tensor) : input sample (Bx2xHxW)
        x1 (2D tensor) : reconstruction of x0 (Bx2xHxW)
        x2 (2D tensor) : reconstruction of x0 (Bx2xHxW)
        

    Returns:
        ano_maps (2D tensor) : filtered anomaly map (Bx1xHxW)
        ood_score (1D tensor) : out-of-distribution scores (Bx1)
    
    """
    # Obtain SSIM between x1 and x2
    _, ssim_img             = ssim_for_batch(x1, x2)
    # Calculate anomaly maps and OOD-score
    ano_maps, ood_score     = post_process_ssim2(x0, ssim_img)
    return ano_maps, ood_score# Classifying
# classify(dataloader)
y_score, y_true = get_ood_scores(dataloader)

# %%
plot_classification_metrics(y_score, y_true)
# plot_histogram(y_score, y_true)
print(y_score)
print(y_true)
# %%
th = 9988
FN = ((y_score >= th) == False) & y_true
idx = np.where(FN == 1)[0]
print(idx)
# %%

# Plotting post process results
idx = 59
if cfg:
    _, ssim_img             = ssim_for_batch(x, reconstructs[0])
else:
    _, ssim_img             = ssim_for_batch(x, reconstructs)
ano_maps, ood_score     = post_process_ssim(x, ssim_img)
plotting(x, ssim_img, np.expand_dims(ano_maps, axis=1), idx=idx)
print(np.sum(ood_score[idx]))
# %%
sobel_filt = (sobel(x[idx,1].numpy()) > .005).astype(int)
plt.imshow(sobel_filt)
# %%
