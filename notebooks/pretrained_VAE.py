# %% Load libraries + pretrained VAE
import torch
from diffusers.models import AutoencoderKL
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from notebooks.preprocess_latent_space.dataset import create_h5f_enc, append_h5f_enc

device = "cuda" 
model_dir = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"

vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True).to(device)

# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto",
                                    variant="Enc_512x512",
                                    crack="synthetic",
                                    batch_size = 4,
                                    # rgb_transform = diffuser_normalize(),
                                    # height_transform = diffuser_normalize_height_idv()
                                    )
lightning_data.setup()

train_loader = lightning_data.train_dataloader()
val_loader = lightning_data.val_dataloader()
test_loader = lightning_data.test_dataloader()

img_dir = "/data/storage_crack_detection/lightning-hydra-template/notebooks/images"
# %% Encode - Decode data

def encode_decode(vae, rgb, height, device="cpu"):
    """
    Encodes and subsequently decodes rgb and height images with given pre-trained vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        rgb (Tensor) : Tensor containing rgb images [N,3,h,w]
        height (Tensor): Tensor containing height images [N,1,h,w]

    Returns:
        recon_rgb (Tensor) : Tensor containing recontructed rgb images [N,3,h,w]
        recon_height (Tensor): Tensor containing reconstructed height images [N,1,h,w]
    """
    vae.to(device)
    with torch.no_grad():
        # Encode
        enc_rgb, enc_height = encode(vae, rgb, height, device)  
        # Decode
        recon_rgb, recon_height = decode(vae, enc_rgb, enc_height, device)
    
    return recon_rgb.cpu(), recon_height.cpu()

def encode(vae, rgb, height, device="cpu"):
    """
    Encodes rgb and height images with given pre-trained vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        rgb (Tensor) : Tensor containing rgb images [N,3,h,w]
        height (Tensor): Tensor containing height images [N,1,h,w]

    Returns:
        enc_rgb (Tensor) : Tensor containing encoded rgb images [N,4,h/8,w/8]
        enc_height (Tensor): Tensor containing encoded height images [N,4,h/8,w/8]
    """
    # Duplicate height channel to fit the vae
    height = torch.cat((height,height,height), dim=1)
    
    vae.to(device)

    # Encode
    with torch.no_grad():
        enc_rgb     = vae.encode(rgb.to(device)).latent_dist.sample().mul_(0.18215)
        enc_height  = vae.encode(height.to(device)).latent_dist.sample().mul_(0.18215)

    return enc_rgb, enc_height

def decode(vae, enc_rgb, enc_height, device="cpu"):
    """
    Decodes rgb and height images with given pre-trained vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        enc_rgb (Tensor) : Tensor containing encoded rgb images [N,4,h/8,w/8]
        enc_height (Tensor): Tensor containing encoded height images [N,4,h/8,w/8]

    Returns:
        rgb (Tensor) : Tensor containing rgb images [N,3,h,w]
        height (Tensor): Tensor containing height images [N,1,h,w]
    """
    vae.to(device)
    
    # Decode
    recon_rgb      = vae.decode(enc_rgb.to(device)/0.18215).sample
    recon_height   = vae.decode(enc_height.to(device)/0.18215).sample[:,0].unsqueeze(1)

    return recon_rgb, recon_height

def undo_norm(x):
    x = (x + 1.) / 2.
    x = x.clamp(0., 1.)
    return x
# %% Plot results rgb

# for rgb, height, _ in train_loader:
#     recon_rgb, recon_height = encode_decode(vae, rgb, height, device)
    
#     recon_rgb = undo_norm(recon_rgb)
#     recon_height = undo_norm(recon_height)
#     break

# rgb2 = undo_norm(rgb)

# i = 3
# fig, axes = plt.subplots(1, 2, figsize=(12,8))
# axes[0].imshow(rgb2[i].permute(1,2,0))
# axes[0].set_title(f"Original mini patch", fontsize=16)
# axes[0].axis('off')

# axes[1].imshow(recon_rgb[i].permute(1,2,0))
# axes[1].set_title(f"Reconstructed", fontsize=16)
# axes[1].axis('off')

# plt_dir = os.path.join(img_dir, "test_rgb")
# fig.savefig(plt_dir)
# plt.close()

# # %% Plot results height

# height2 = undo_norm(height)

# i, j = 3, 2
# fig, axes = plt.subplots(1, 2, figsize=(12,8))
# axes[0].imshow(height2[i,0])
# axes[0].set_title(f"Original mini patch", fontsize=16)
# axes[0].axis('off')

# axes[1].imshow(recon_height[i,0])
# axes[1].set_title(f"Reconstructed", fontsize=16)
# axes[1].axis('off')

# plt_dir = os.path.join(img_dir, "test_height")
# fig.savefig(plt_dir)
# plt.close()

# ONLY DECODING

# for rgb, height, _ in train_loader:
#     recon_rgb, recon_height = decode(vae, rgb.float(), height.float(), device)
    
#     recon_rgb = undo_norm(recon_rgb)
#     recon_height = undo_norm(recon_height)
#     break

# i = 3
# fig, axes = plt.subplots(1, 1, figsize=(12,8))

# axes[0].imshow(recon_rgb[i].permute(1,2,0))
# axes[0].set_title(f"Reconstructed", fontsize=16)
# axes[0].axis('off')

# plt_dir = os.path.join(img_dir, "test_rgb")
# fig.savefig(plt_dir)
# plt.close()

# # %% Plot results height

# i, j = 3, 2
# fig, axes = plt.subplots(1, 1, figsize=(12,8))

# axes[0].imshow(recon_height[i,0])
# axes[0].set_title(f"Reconstructed", fontsize=16)
# axes[0].axis('off')

# plt_dir = os.path.join(img_dir, "test_height")
# fig.savefig(plt_dir)
# plt.close()


# %% Save encoded dataset as h5 file

# output_filename_full_h5 = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto/2025-01-07_Enc_Real_Crack512x512_test.h5"
# for rgb, height, _ in tqdm(test_loader):

#     enc_rgb, enc_height = encode(vae, rgb, height)

#     if not os.path.exists(output_filename_full_h5):
#         # Creating new h5 file
#         create_h5f_enc(output_filename_full_h5, enc_rgb, enc_height)
#     else:
#         # Appending h5 file
#         append_h5f_enc(output_filename_full_h5, enc_rgb, enc_height)
# %% Check error / SSIM before and after encoding-decoding

# # Initialize SSIM settings
# ssim = SSIM(gaussian_kernel=False,
#             data_range=1,
#             kernel_size=11).to(device)

# # Create empty lists to store SSIM scores
# ssim_RGB    = []
# ssim_HEIGHT = []

# # Loop to encode & decode images and comparing result with SSIM 
# for rgb, height, _ in (pbar := tqdm(test_loader)):
#     recon_rgb, recon_height = encode_decode(vae, rgb, height)

#     ssim_RGB.append(ssim(rgb.to(device), recon_rgb))
#     ssim_HEIGHT.append(ssim(height.to(device), recon_height))

#     mean_rgb    = sum(ssim_RGB) / len(ssim_RGB)
#     mean_height = sum(ssim_HEIGHT) / len(ssim_HEIGHT)
    
#     pbar.set_description(f"{mean_rgb:.2f}, {mean_height:.2f}")

# # Print mean
# print(f"The mean SSIM for RGB image: {mean_rgb:.5f}")
# print(f"The mean SSIM for height images: {mean_height:.5f}")