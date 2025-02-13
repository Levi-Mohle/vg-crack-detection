# %% Imports

import os
import sys
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL.ImageOps 
from torch.utils.data import DataLoader
import torch
from skimage import morphology, measure
from skimage.transform import resize, rotate
from skimage.color import rgb2hsv, hsv2rgb 
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.components.transforms import *
from notebooks.preprocess_latent_space.dataset import append_h5f_enc, create_h5f_enc, HDF5PatchesDatasetCustom
from notebooks.pretrained_VAE import encode
from notebooks.preprocess_latent_space.cracks import *

# %% Load the data & model
data_dir = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto"
IMPASTO_train_dir = "2025-01-07_Real_Cracks512x512_test.h5"
data_train = HDF5PatchesDatasetCustom(hdf5_file_path = os.path.join(data_dir, IMPASTO_train_dir))

dataloader_train = DataLoader(  dataset=data_train,
                                batch_size= 1,
                                shuffle=False,
                            )

device     = "cuda" 
model_dir  = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"
add_cracks = False

with torch.no_grad():
    vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True).to(device)

# %% Adding cracks + encoding

# Get binary shape masks
MPEG_path   = r"/data/storage_crack_detection/datasets/MPEG400"
cat_name    = 'brick'
masks       = get_shapes(MPEG_path, cat_name, plot=False)

output_dir = data_dir
output_filename = r"2025-01-07_Enc_Real_Crack512x512_test.h5"
output_filename_full_h5 = os.path.join(output_dir, output_filename)
for i, (rgb, height, id) in tqdm(enumerate(dataloader_train)):

    # id = None # Uncomment if you want to ignore original labels
    # Add, transform and encode synthetic cracks
    if (i % 2 == 0) & add_cracks:
        print("adding cracks")
        height_cracks, rgb_cracks = add_cracks_with_lifted_edges_V2(height, rgb, 
                                                                    masks=masks, 
                                                                    decay_rate=2)
        rgb_cracks      = normalize_rgb(rgb_cracks)
        height_cracks   = rescale_diffuser_height_idv(height_cracks)
        enc_rgb_cracks, enc_height_cracks   = encode(vae, rgb_cracks, height_cracks)
    else:
        enc_rgb_cracks    = None
        enc_height_cracks = None
    
    # Transform and encode normal samples
    rgb                 = normalize_rgb(rgb)
    height              = rescale_diffuser_height_idv(height)
    enc_rgb, enc_height = encode(vae, rgb, height)
    
    if not os.path.exists(output_filename_full_h5):
        # Creating new h5 file
        create_h5f_enc(output_filename_full_h5, 
                       rgb          = enc_rgb,
                       rgb_cracks   = enc_rgb_cracks,
                       height       = enc_height,
                       height_cracks= enc_height_cracks,
                       target       = id
                       )
    else:
        # Appending h5 file
        append_h5f_enc(output_filename_full_h5, 
                       rgb          = enc_rgb,
                       rgb_cracks   = enc_rgb_cracks,
                       height       = enc_height,
                       height_cracks= enc_height_cracks,
                       target       = id
                       )

