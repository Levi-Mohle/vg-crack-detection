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
                                    variant="512x512",
                                    crack="real",
                                    batch_size = 16,
                                    rgb_transform = diffuser_normalize(),
                                    height_transform = diffuser_normalize_height_idv()
                                    )
lightning_data.setup()

train_loader = lightning_data.train_dataloader()
val_loader = lightning_data.val_dataloader()
test_loader = lightning_data.test_dataloader()

# for i, (rgb, height, id) in enumerate(test_loader):
#     break
# %% Encode - Decode data

def encode_decode(vae, rgb, height):
    height = torch.cat((height,height,height), dim=1)

    with torch.no_grad():
        enc_rgb     = vae.encode(rgb).latent_dist.sample().mul_(0.18215)
        recon_rgb   = vae.decode(enc_rgb/0.18215).sample
        enc_height     = vae.encode(height).latent_dist.sample().mul_(0.18215)
        recon_height   = vae.decode(enc_height/0.18215).sample[:,0].unsqueeze(1)
    
    return recon_rgb, recon_height

def encode(vae, rgb, height):
    height = torch.cat((height,height,height), dim=1)

    with torch.no_grad():
        enc_rgb     = vae.encode(rgb.to(device)).latent_dist.sample().mul_(0.18215)
        enc_height  = vae.encode(height.to(device)).latent_dist.sample().mul_(0.18215)

    return enc_rgb.cpu(), enc_height.cpu()

# reconstructed_rgb = encode_decode(vae, rgb)

# height_stacked          = torch.cat((height,height,height), dim=1)
# reconstructed_height    = encode_decode(vae, height_stacked)
# %% Plot results rgb

# rgb2 = (rgb + 1.) / 2.
# rgb2 = rgb2.clamp(0., 1.)

# i = 3
# fig, axes = plt.subplots(1, 2, figsize=(12,8))
# axes[0].imshow(rgb2[i].permute(1,2,0))
# axes[0].set_title(f"Original mini patch", fontsize=16)
# axes[0].axis('off')

# axes[1].imshow(reconstructed_rgb[i].permute(1,2,0))
# axes[1].set_title(f"Reconstructed", fontsize=16)
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()

# %% Plot results height

# height2 = (height + 1.) / 2.
# height2 = height2.clamp(0., 1.)

# i, j = 3, 2
# fig, axes = plt.subplots(1, 2, figsize=(12,8))
# axes[0].imshow(height2[i,0])
# axes[0].set_title(f"Original mini patch", fontsize=16)
# axes[0].axis('off')

# axes[1].imshow(reconstructed_height[i,j])
# axes[1].set_title(f"Reconstructed", fontsize=16)
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()


# %% Save encoded dataset as h5 file

output_filename_full_h5 = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto/2025-01-07_Enc_Real_Crack512x512_test.h5"
for rgb, height, _ in tqdm(test_loader):

    enc_rgb, enc_height = encode(vae, rgb, height)

    if not os.path.exists(output_filename_full_h5):
        # Creating new h5 file
        create_h5f_enc(output_filename_full_h5, enc_rgb, enc_height)
    else:
        # Appending h5 file
        append_h5f_enc(output_filename_full_h5, enc_rgb, enc_height)
# %% Check error / SSIM before and after encoding-decoding

ssim = SSIM(gaussian_kernel=False,
            data_range=1,
            kernel_size=5)

ssim_RGB    = []
ssim_HEIGHT = []
for rgb, height, _ in tqdm(train_loader):
    recon_rgb, recon_height = encode_decode(vae, rgb, height)

    ssim_RGB.append(ssim(rgb, recon_rgb))
    ssim_HEIGHT.append(ssim(height, recon_height))