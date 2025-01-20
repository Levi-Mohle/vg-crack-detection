# %% Load libraries + pretrained VAE
import torch
from diffusers.models import AutoencoderKL
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
from notebooks.preprocess_latent_space.dataset import create_h5f_enc, append_h5f_enc


model_dir = r"C:\Users\lmohle\Documents\2_Coding\data\Trained_Models\AutoEncoderKL"

vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True)

# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"C:\Users\lmohle\Documents\2_Coding\data\input\Training_data\512x512",
                                    variant="512x512",
                                    batch_size = 16,
                                    rgb_transform = diffuser_normalize(),
                                    height_transform = diffuser_normalize_height_idv()
                                    )
lightning_data.setup()
test_loader = lightning_data.test_dataloader()

for i, (rgb, height, id) in enumerate(test_loader):
    break
# %% Encode - Decode data

def encode_decode(vae, rgb):
    with torch.no_grad():
        latent = vae.encode(rgb).latent_dist.sample().mul_(0.18215)
        reconstructed = vae.decode(latent/0.18215).sample

    reconstructed = (reconstructed + 1.) / 2.
    reconstructed = reconstructed.clamp(0., 1.)
    return reconstructed

def encode(vae, rgb, height):
    height = torch.cat((height,height,height), dim=1)

    with torch.no_grad():
        enc_rgb     = vae.encode(rgb).latent_dist.sample().mul_(0.18215)
        enc_height  = vae.encode(height).latent_dist.sample().mul_(0.18215)

    return enc_rgb, enc_height

# reconstructed_rgb = encode_decode(vae, rgb)

# height_stacked          = torch.cat((height,height,height), dim=1)
# reconstructed_height    = encode_decode(vae, height_stacked)
# %% Plot results rgb

rgb2 = (rgb + 1.) / 2.
rgb2 = rgb2.clamp(0., 1.)

i = 3
fig, axes = plt.subplots(1, 2, figsize=(12,8))
axes[0].imshow(rgb2[i].permute(1,2,0))
axes[0].set_title(f"Original mini patch", fontsize=16)
axes[0].axis('off')

axes[1].imshow(reconstructed_rgb[i].permute(1,2,0))
axes[1].set_title(f"Reconstructed", fontsize=16)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# %% Plot results height

height2 = (height + 1.) / 2.
height2 = height2.clamp(0., 1.)

i, j = 3, 2
fig, axes = plt.subplots(1, 2, figsize=(12,8))
axes[0].imshow(height2[i,0])
axes[0].set_title(f"Original mini patch", fontsize=16)
axes[0].axis('off')

axes[1].imshow(reconstructed_height[i,j])
axes[1].set_title(f"Reconstructed", fontsize=16)
axes[1].axis('off')

plt.tight_layout()
plt.show()


# %% Save encoded dataset as h5 file

output_filename_full_h5 = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto\2024-11-26_Enc_Crack512x512_test.h5"
for rgb, height, id in test_loader:

    enc_rgb, enc_height = encode(vae, rgb, height)

    if not os.path.exists(output_filename_full_h5):
        # Creating new h5 file
        create_h5f_enc(output_filename_full_h5, enc_rgb, enc_height, id)
    else:
        # Appending h5 file
        append_h5f_enc(output_filename_full_h5, enc_rgb, enc_height, id)

# %%
