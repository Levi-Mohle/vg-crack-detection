# %% Imports

import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.components.transforms import *
from notebooks.preprocess_latent_space.dataset import HDF5PatchesDatasetCustom
from notebooks.preprocess_latent_space.cracks import *
from notebooks.preprocess_latent_space.latent_space import *

# %% Load the data & model
data_dir = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto"
# data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto"
# IMPASTO_train_dir = "2024-11-26_512x512_train.h5"
IMPASTO_train_dir = "2024-11-26_512x512_val.h5"
# IMPASTO_train_dir = "2025-02-18_Real_Cracks512x512_test.h5"
data_train = HDF5PatchesDatasetCustom(hdf5_file_path = os.path.join(data_dir, IMPASTO_train_dir))

dataloader_train = DataLoader(  dataset    = data_train,
                                batch_size = 16,
                                shuffle    =False,
                            )

device      = "cpu" 
model_dir  = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"
# model_dir   = r"C:\Users\lmohle\Documents\2_Coding\data\Trained_Models\AutoEncoderKL"

# with torch.no_grad():
#     vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True).to(device)

# %% Adding cracks + encoding

# Get binary shape masks
MPEG_path   = r"/data/storage_crack_detection/datasets/MPEG400"
# MPEG_path   = r"C:\Users\lmohle\Documents\2_Coding\data\Datasets\MPEG400"
cat_name    = 'brick'
masks       = get_shapes(MPEG_path, cat_name, plot=False)

# %%
output_dir = data_dir
output_filename = r"2024-11-26_mix_512x512_val.h5"
output_filename_full_h5 = os.path.join(output_dir, output_filename)
# For adding p percentage of cracks + encoding
# add_enc_synthetic_cracks_to_h5(dataloader   = dataloader_train, 
    #                            masks        = masks, 
    #                            p            = 1, 
    #                            filename     = output_filename_full_h5, 
    #                            vae          = vae,
    #                            add_cracks   = False,
    #                            segmentation = False, 
    #                            device       = device
    #                            )

# For only encoding dataset
# encode_and_add2h5(dataloader = dataloader_train, 
#                   filename   = output_filename_full_h5, 
#                   vae        = vae,
#                   device     = device
#                  )

# # For only encoding AND augmenting dataset
# encode_and_augment2h5(dataloader = dataloader_train, 
#                       filename   = output_filename_full_h5, 
#                       vae        = vae,
#                       device     = device
#                      )

# For adding p percentage of cracks
add_synthetic_cracks_to_h5( dataloader   = dataloader_train, 
                            masks        = masks, 
                            p            = 1, 
                            filename     = output_filename_full_h5, 
                            add_cracks   = True,
                            segmentation = False, 
                            )

### TEMPORARY ###
# for i, (rgb, height, id) in tqdm(enumerate(dataloader_train)):
#     height_cracks, rgb_cracks, _ = Create_cracks_with_lifted_edges(height, 
#                                                                 rgb, 
#                                                                 masks=masks, 
#                                                                 decay_rate=2)
#     break

# bs, _, img_h, img_w= [4, 1, 512, 512]
# # Pick a mask
# batch_masks = random.sample(masks, bs)
# batch_masks = torch.tensor(np.array(batch_masks))
# # Pick random transformation for the shape
# transform = transforms.Compose([
#                 transforms.RandomRotation(degrees=[-180,180]),
#                 transforms.RandomResize(min_size=300, max_size=450)
#             ])
# transformed_masks = transform(batch_masks)

# _, mask_h, mask_w = transformed_masks.shape

# x_start = np.random.randint(0, img_h - mask_h, size=bs)
# y_start = np.random.randint(0, img_w - mask_w, size=bs)

# segmentation_mask = torch.zeros((bs, 1, img_h, img_w))

# for i, (x, y) in enumerate(zip(x_start, y_start)):
#     segmentation_mask[i, 0, x:x+mask_h,y:y+mask_w] = transformed_masks[i]

# concat  = torch.concat([segmentation_mask, segmentation_mask, segmentation_mask], dim=1)
# z       = encode_(vae, concat, device=device)
# xhat    = decode_(vae, z, device=device)

# %%

# xhat2 = (xhat + 1)/2
# xhat3 = (xhat2 > 0.8).to(torch.uint8)

# fig, axes = plt.subplots(2,2)
# for i, ax in enumerate(axes.flatten()):
#     # ax.imshow(rgb_cracks[i].permute(1,2,0))
#     ax.imshow(segmentation_mask[i,0])
#     ax.axis("off")

# fig, axes = plt.subplots(2,2)
# for i, ax in enumerate(axes.flatten()):
#     # ax.imshow(rgb_cracks[i].permute(1,2,0))
#     ax.imshow(xhat3[i,0].detach())
#     ax.axis("off")

# %%

