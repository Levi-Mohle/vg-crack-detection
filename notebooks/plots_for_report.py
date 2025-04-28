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
from skimage.filters import sobel
import skimage.morphology as morhpology
from tqdm import tqdm
from PIL import Image
from cv2 import inpaint, INPAINT_NS

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from src.models.components.utils.evaluation import *
from notebooks.utils.synthetic_cracks import *
from notebooks.utils.dataset import HDF5PatchesDatasetReconstructs

# Choose if run from local machine or SURF cloud
local = True
# %% Load the data

if local:
    data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto"
    MPEG_path   = r"C:\Users\lmohle\Documents\2_Coding\data\Datasets\MPEG400"
else:
    MPEG_path   = r"/data/storage_crack_detection/datasets/MPEG400"

lightning_data = IMPASTO_DataModule(data_dir           = data_dir,
                                    batch_size         = 80,
                                    variant            = "512x512_local",
                                    crack              = "realAB"
                                    )

lightning_data.setup()
loader = lightning_data.test_dataloader()
# %% Retrieve images from dataloader

for i, (rgb, height, id) in enumerate(loader):
     if i==1:
        break
# %% Plot noise levels
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

# %% Plot cracks with lifted edges
# print(torch.where(id == 1)[0])
# idx = 9

extent = [0,4,0,4]
fs = 14

# um conversion
height2 = (height.to(torch.float32)) * 25*10**(-4)
cracks = [rgb, height2]
T = ["RGB", "Height"]
fig, axes = plt.subplots(2,1, figsize=(15,10))
for i, ax in enumerate(axes.flatten()):
    # ax.set_title(f"{T[i]}", fontsize=fs)
    im = ax.imshow(cracks[i][idx].permute(1,2,0), extent=extent)
    ax.set_ylabel("Y [mm]", fontsize=fs)
    ax.set_xlabel("X [mm]", fontsize=fs)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
    ax.tick_params(axis='both', which='major', labelsize=fs)
    # ax.axis("off")
axes[0].set_title("Original sample", fontsize=25)
# divider = make_axes_locatable(axes[1])
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = plt.colorbar(im, cax=cax)
# cbar.set_label("Height [$\mu$m]", fontsize=fs+2)
# cbar.ax.tick_params(labelsize=fs)
fig.tight_layout()
# %% Load reconstructs

if local:
    # data_dir = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-03-03_cDDPM_0.8_realAB_reconstructs.h5"
    data_dir = r"C:\Users\lmohle\Documents\2_Coding\data\output\2025-02-11_Reconstructs\2025-02-11_real_reconstructs.h5"
else:
    pass

cfg = False
reconstruct_dataset = HDF5PatchesDatasetReconstructs(data_dir,
                                                     cfg= cfg,
                                                     rgb_transform=revert_normalize_rgb(),
                                                     height_transform= revert_normalize_height())

dataloader = DataLoader(reconstruct_dataset, batch_size=80, shuffle=False)

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
extent  = [0,4,0,4]
fs      = 14
cracks  = [rgb, height]
T       = ["RGB", "Height"]

_, ssim_img = ssim_for_batch(x, reconstructs)

ssim_filt   = np.zeros_like(ssim_img)
ano_maps    = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))
sobel_filt  = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))
for idx in range(ssim_img.shape[0]):
    sobel_filt[idx] = sobel(x[idx,1].cpu().numpy())
    sobel_filt[idx] = (sobel_filt[idx] > .02).astype(int)
    for i in range(ssim_img.shape[1]):
        ssim_filt[idx, i] = (ssim_img[idx,i] > np.percentile(ssim_img[idx,i], q=95)).astype(int)
        ssim_filt[idx,i] = morhpology.binary_erosion(ssim_filt[idx,i])

    ano_maps[idx] = (
                    (ssim_filt[idx,0]   == 1) & 
                    (ssim_filt[idx,1]   == 1) 
                    # (sobel_filt[idx]    == 1)
                    ).astype(int)
    
    # Opening (Erosion + Dilation) to remove noise + connect shapes
    ano_maps[idx] = morhpology.opening(ano_maps[idx])

idx     = 9
fig, axes = plt.subplots(2,1, figsize=(12,8))
for i, ax in enumerate(axes.flatten()):
    # ax.set_title(f"{T[i]}", fontsize=fs)
    ax.imshow(ssim_img[idx][i], extent=extent)
    ax.set_ylabel("Y [mm]", fontsize=fs)
    ax.set_xlabel("X [mm]", fontsize=fs)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
    ax.tick_params(axis='both', which='both', labelsize=fs)
    # ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
fig.tight_layout()

fig, axes = plt.subplots(1,1, figsize=(5,5))
for ax in [axes]:
    # ax.set_title(f"{T[i]}", fontsize=fs)
    ax.imshow(sobel_filt[idx], extent=extent)
    ax.set_ylabel("Y [mm]", fontsize=fs)
    ax.set_xlabel("X [mm]", fontsize=fs)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
    # ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', labelsize=fs)
fig.tight_layout()

fig, axes = plt.subplots(1,1, figsize=(8,8))
for ax in [axes]:
    # ax.set_title(f"{T[i]}", fontsize=fs)
    ax.imshow(ano_maps[idx], extent=extent)
    # ax.set_ylabel("Y [mm]", fontsize=fs)
    # ax.set_xlabel("X [mm]", fontsize=fs)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([0,1,2,3,4])
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
fig.tight_layout()

# %% Visualize synthetic cracks
# Get binary shape masks
cat_name    = 'brick'
img_dirs    = [os.path.join(MPEG_path, "MPEG400-Original", f) \
                   for f in os.listdir(os.path.join(MPEG_path, "MPEG400-Original")) \
                   if cat_name in f]
gt_dirs     = [os.path.join(MPEG_path, "MPEG400-GT", "png", f) \
                for f in os.listdir(os.path.join(MPEG_path, "MPEG400-GT", "png")) \
                if cat_name in f]
for (img_dir, gt_dir) in zip(img_dirs, gt_dirs):
    # Open, invert and transform to grayscale
    orig_img = np.array(PIL.ImageOps.invert(Image.open(img_dir)).convert('L'))
    orig_img = (orig_img > 100).astype(np.uint8)

    gt_img = np.array(Image.open(gt_dir).convert('L'))
    gt_img = (gt_img > 100).astype(np.uint8)
    
    # Thickening the skeletal image to improve shape extraction
    gt_img = morphology.binary_dilation(gt_img) 

    # Subtracting skeletal image from original
    diff = ((orig_img - gt_img)>0).astype(np.uint8)
    break

plt.figure()
plt.imshow(orig_img)
plt.axis("off")

plt.figure()
plt.imshow(gt_img)
plt.axis("off")

plt.figure()
plt.imshow(diff)
plt.axis("off")

# masks       = get_shapes(MPEG_path, cat_name, plot=True)
# %% Intro GIF

# color_dir   = r"C:\Users\lmohle\Documents\2_Coding\data\input\Harms Big Impasto-20250218-1643\80_ColorImage__X699.2811_Y66.7457_Z30.6423771381378.bmp"
# height_dir  = r"C:\Users\lmohle\Documents\2_Coding\data\input\Harms Big Impasto-20250218-1643\80_HeightImage__X699.2811_Y66.7457_Z30.6423771381378.bmp"
color_dir   = r"C:\Users\lmohle\Documents\2_Coding\data\input\Harms Almond Blossom-20250107-1604\40_ColorImage__X298.8397_Y149.1174_Z40.6661909818649.bmp"
height_dir  = r"C:\Users\lmohle\Documents\2_Coding\data\input\Harms Almond Blossom-20250107-1604\40_HeightImage__X298.8397_Y149.1174_Z40.6661909818649.bmp"
dirs = [color_dir, height_dir]

fs =14

fig, axes = plt.subplots(1,2, figsize=(10,15), width_ratios=[1,1.08])
extent = [0,24,0,24]
for i, ax in enumerate(axes.flatten()):
    img = Image.open(dirs[i])
    if i ==1:
        img = np.array(img)
        img = (img.astype(np.float32)) * 25*10**(-4)
        int99 = np.mean(img) - 3 * np.std(img)
        mask = (img <= int99).astype('uint8')
        img = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
    im = ax.imshow(img, extent=extent)
    ax.set_ylabel("Y [mm]")
    ax.set_xlabel("X [mm]")
    ax.set_yticks([0,5,10,15,20])
    ax.set_xticks([0,5,10,15,20])
    # ax.axis("off")
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Height [$\mu$m]", fontsize=fs+2)
cbar.ax.tick_params(labelsize=fs)
fig.tight_layout()


# %%