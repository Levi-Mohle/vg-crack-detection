# %% Load libraries + pretrained VAE
import torch
from diffusers.models import AutoencoderKL
import numpy as np
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from notebooks.utils.dataset import create_h5f_enc, append_h5f_enc
import notebooks.utils.latent_space as latent_space

 
local = False
data_dir = os.path.join(wd, "data", "impasto")
img_dir  = os.path.join(wd, "notebooks", "images")
if local:
    model_dir = r"C:\Users\lmohle\Documents\2_Coding\data\Trained_Models\AutoEncoderKL"
    device="cpu"
else:
    model_dir = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"
    device = "cuda"

vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True).to(device)

# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = data_dir,
                                    variant="512x512",
                                    crack="realAB",
                                    batch_size = 18,
                                    transform = DiffuserTransform()
                                    )
lightning_data.setup()

train_loader    = lightning_data.train_dataloader()
val_loader      = lightning_data.val_dataloader()
test_loader     = lightning_data.test_dataloader()

def undo_norm(x):
    x = (x + 1.) / 2.
    x = x.clamp(0., 1.)
    return x

# # %% Check error / SSIM before and after encoding-decoding

# # Initialize SSIM settings
# ssim = SSIM(gaussian_kernel=False,
#             data_range=1,
#             kernel_size=5).to(device)

# # Create empty lists to store SSIM scores
# ssim_RGB    = []
# ssim_HEIGHT = []

# # Loop to encode & decode images and comparing result with SSIM 
# for rgb, height, _ in (pbar := tqdm(test_loader)):
#     recon_rgb, recon_height = latent_space.encode_decode(vae, rgb, height, device)

#     ssim_RGB.append(ssim(rgb, recon_rgb))
#     ssim_HEIGHT.append(ssim(height, recon_height))

#     mean_rgb    = sum(ssim_RGB) / len(ssim_RGB)
#     mean_height = sum(ssim_HEIGHT) / len(ssim_HEIGHT)
    
#     pbar.set_description(f"{mean_rgb:.2f}, {mean_height:.2f}")

# # Compute standard deviation
# std_rgb = torch.std(torch.tensor(ssim_RGB),axis=0)
# std_height = torch.std(torch.tensor(ssim_HEIGHT),axis=0)

# # Print mean + std
# print(f"The mean SSIM for RGB image: {mean_rgb:.5f}, std: {std_rgb:.5f}")
# print(f"The mean SSIM for height images: {mean_height:.5f}, std: {std_height:.5f}")

# %% Encode - Decode data + plot results

# Encode and decode the data
for rgb, height, _ in test_loader:
    recon_rgb, recon_height = latent_space.encode_decode(vae, rgb, height, device)
    
    recon_rgb    = undo_norm(recon_rgb)
    recon_height = undo_norm(recon_height)
    break
 
# Undo normalization on input samples
rgb2 = undo_norm(rgb)
height2 = undo_norm(height)

for i in range(rgb.shape[0]):
    
    # RGB
    fig, axes = plt.subplots(1, 2, figsize=(12,8))
    axes[0].imshow(rgb2[i].permute(1,2,0))
    # axes[0].set_title(f"Original mini patch", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(recon_rgb[i].permute(1,2,0))
    # axes[1].set_title(f"Reconstructed", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt_dir = os.path.join(img_dir, f"test_rgb_{i}")
    fig.savefig(plt_dir)
    plt.close()
    
    # HEIGT
    fig, axes = plt.subplots(1, 2, figsize=(12,8))
    axes[0].imshow(height2[i,0])
    # axes[0].set_title(f"Original mini patch", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(recon_height[i,0])
    # axes[1].set_title(f"Reconstructed", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt_dir = os.path.join(img_dir, f"test_height_{i}")
    fig.savefig(plt_dir)
    plt.close()

# %% Save encoded dataset as h5 file

# output_filename_full_h5 = r"/data/storage_crack_detection/ml-crack-detection-van-gogh/data/impasto/2025-01-07_Enc_Real_Crack512x512_test.h5"
# for rgb, height, _ in tqdm(test_loader):

#     enc_rgb, enc_height = encode(vae, rgb, height)

#     if not os.path.exists(output_filename_full_h5):
#         # Creating new h5 file
#         create_h5f_enc(output_filename_full_h5, enc_rgb, enc_height)
#     else:
#         # Appending h5 file
#         append_h5f_enc(output_filename_full_h5, enc_rgb, enc_height)