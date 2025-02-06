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

# %% Functions
def crop_mask(bin_mask, margin=10):
    """
    Crops only the relevant part (+margin) from a binary mask 
    and rescales to 1 size.

    Args:
        bin_mask (np.array) : 2D binary mask 
        margin (int) : pixels to take as margin around the shape in mask

    Returns:
        cropped_mask (np.array) : cropped 2D binary mask
        (x_min, y_min, x_max, y_max) (tuple) : outer coordinates with margin of shape
        
    """
    # Get rows and columns where mask>0
    rows, cols = np.where(bin_mask>0)

    # Determin outer coordinates of shape in mask
    x_min, x_max = rows.min(), rows.max()
    y_min, y_max = cols.min(), cols.max()

    # Add margins
    x_min = max(0, x_min - margin)
    x_max = min(bin_mask.shape[0], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(bin_mask.shape[1], y_max + margin)

    # Crop
    cropped_mask = bin_mask[x_min:x_max, y_min:y_max]

    # Normalize to 1 size
    cropped_mask = (resize(cropped_mask, (300,300))>0).astype(np.uint8)

    return cropped_mask, (x_min, y_min, x_max, y_max)

def get_shapes(MPEG_path, cat_name, plot=False):
    """
    Extracts shapes from the MPEG400 dataset combined with GT skeletal data from
    https://github.com/cong-yang/skeview . By using subtracting skeletal data from orignal
    images, a fragmented image is created. Skimage extracts the individual shapes and are saved
    into a list. 

    Args:
        MPEG_path (str) : File location to folder containing MPEG400-GT and MPEG400-Original
        cat_name (str) : Category name from which you want to extract shapes (i.e. bat, bell, brick etc.)
        plot (bool) : Boolean to turn on/off plots of the shapes

    Returns:
        all_masks (list) : list of 2D binary masks (np.array) containing extracted shapes 
        
    """
    # Get original and skeletal image directions based on chosen category
    img_dirs    = [os.path.join(MPEG_path, "MPEG400-Original", f) \
                   for f in os.listdir(os.path.join(MPEG_path, "MPEG400-Original")) \
                   if cat_name in f]
    gt_dirs     = [os.path.join(MPEG_path, "MPEG400-GT", "png", f) \
                   for f in os.listdir(os.path.join(MPEG_path, "MPEG400-GT", "png")) \
                    if cat_name in f]
    

    all_masks = []
    for (img_dir, gt_dir) in zip(img_dirs, gt_dirs):
        # Open, invert and transform to grayscale
        orig_img = np.array(PIL.ImageOps.invert(Image.open(img_dir)).convert('L'))
        orig_img = (orig_img > 0).astype(np.uint8)

        # Open and transform to grayscale
        gt_img = np.array(Image.open(gt_dir).convert('L'))
        gt_img = (gt_img > 0).astype(np.uint8)
        # Thickening the skeletal image to improve shape extraction
        gt_img = morphology.binary_dilation(gt_img) 

        # Subtracting skeletal image from original
        diff = ((orig_img - gt_img)>0).astype(np.uint8)

        # Extract shapes
        labeled_sub_shapes = measure.label(diff, connectivity=1)

        # Putting shapes into list
        sub_shape_masks = [labeled_sub_shapes == i for i in range(1, labeled_sub_shapes.max() + 1)]

        # Only pick sufficiently large shapes (but not too big)
        area_sum = [np.sum(sub_shape_masks[i]) for i in range(0, labeled_sub_shapes.max())]
        area_mask = list((np.array(area_sum) > 1000) & (np.array(area_sum) < 4000))
        large_masks = [b.astype(np.uint8) for a, b in zip(area_mask, sub_shape_masks) if a]

        # Crop the mask and rescale to same size
        cropped_masks = []
        for j in range(len(large_masks)):
            cropped_mask, _ = crop_mask(large_masks[j])
            cropped_masks.append(cropped_mask)

        # Add to the list
        all_masks.append(cropped_masks)

        # Plot shapes
        if plot:
            plt.figure()
            fig, axes = plt.subplots(3, 4, figsize=(12,8))
            for _, (ax, mask) in enumerate(zip(axes.flatten(), cropped_masks)):
                ax.imshow(mask)
                ax.axis("off")
            plt.show()
    
    # Flatten the list
    all_masks = sum(all_masks, [])
    
    return all_masks

def get_grad_mask(mask, flap_height, decay_rate, seed=None):
    """
    Applies an exponential gradient to the given mask in a random direction

    Args:
        mask (np.array) : 2D binary mask containing a shape
        flap_height (int) : maximum height value to be added to the existing height map
        decay_rate (float) : Rate of decay for gradient on flap
        seed (int) : fixed seed

    Returns:
        grad_mask (np.array) : 2D mask containing the shape with exponential gradient
        
    """
    # Get mask shape
    mask_h, mask_w = mask.shape

    # If defined, fix the seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Pick random gradient angle
    angle_grad = random.uniform(0, 2*np.pi)
    x = np.linspace(0, 1, mask_w)
    y = np.linspace(0, 1, mask_h)
    xv, yv = np.meshgrid(x,y)

    # Rotate gradient
    lin_grad = np.cos(angle_grad) * xv + np.sin(angle_grad) * yv

    # Normalize to range [0,1]
    lin_grad -= lin_grad.min()
    lin_grad /= lin_grad.max()

    # Apply exponentional 
    exp_grad = np.exp(decay_rate * lin_grad)

    # Normalize to range [0,flap_height]
    exp_grad -= exp_grad.min()
    exp_grad /= exp_grad.max()
    exp_grad *= flap_height
    
    # Apply gradient to mask
    grad_mask = mask * exp_grad
    grad_mask -= grad_mask.min()

    return grad_mask

def add_cracks_with_lifted_edges_V2(height, rgb, masks, flap_height= None, decay_rate=2, seed=None):
    """
    Adds a crack with lifted edge on top of a mini-patch, by adding a shape from the mask list
    to the height map, overlayed with a gradient of exponentially increasing values. On the rgb image the edge pixels 
    of the shape are darkened where the height difference of the lifted area exceeds a threshold 

    Args:
        height (torch.Tensor): height mini patch containing (1,1,height,width)
        rgb (torch.Tensor): rgb mini patch (1,3,height,width)
        masks (list) : list containing binary masks of flap shapes
        flap_height (int) : maximum height value to be added to the existing height map
        decay_rate (float) : Rate of decay for gradient on flap
        seed (int) : fixed seed

    Returns:
        cracked_rgb (torch.Tensor): rgb mini patch containing cracks (1,3,height,width)
        cracked_height (torch.Tensor): height mini patch containing cracks (1,1,height,width)
        
    """
    # Get shape
    _, _, img_h, img_w= height.shape

    # Operations not possible on uint16, so conversion is required
    height = height.to(torch.float)
    
    # If not defined, take average height * factor to add (maximum)
    if flap_height is None:
        flap_height = (height.mean() - height.min()).item() * 3

    # If defined, fix the seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Pick a mask
    mask = random.choice(masks)
    # Pick random transformation for the shape
    angle = random.choice([-90,180,90,0])
    x, y = random.sample([300,350,400,450], 2)
    mask = rotate(mask, angle)
    mask = (resize(mask, (x, y))>0).astype(np.uint8)

    # Get mask shape
    mask_h, mask_w = mask.shape

    # Generate random top-left cornder coordinates for mask
    x_start = random.randint(0, img_h - mask_h)
    y_start = random.randint(0, img_w - mask_w)

    # Apply exponential gradient to mask
    grad_mask = get_grad_mask(mask, flap_height, decay_rate, seed)

    # Apply gradient mask to height map
    cracked_height = height.clone()
    cracked_height[0,0, x_start:x_start+mask_h,y_start:y_start+mask_w] += torch.tensor(grad_mask)

    # Change RGB to black pixels if difference > threshold
    np_mask = mask.astype(np.uint8)
    edge_mask = morphology.binary_dilation(\
        morphology.binary_dilation(\
        morphology.binary_dilation(\
        (cv.Canny(image=np_mask, threshold1=0, threshold2=1) > 0).astype(np.uint8)))) 
    mask2 = grad_mask * edge_mask >= grad_mask.max() * 0.7

    # Apply rgb mask
    rgb_cracked = rgb.clone().numpy()
    hsv = rgb2hsv(rgb_cracked[0], channel_axis=0) # Conversion to hsv to decrease value channel
    hsv[:, x_start:x_start+mask_h,y_start:y_start+mask_w][2, mask2] *= 0.2
    rgb_cracked = torch.tensor(hsv2rgb(hsv, channel_axis=0)).unsqueeze(0) * 255

    return cracked_height.to(torch.uint16), rgb_cracked.to(torch.uint8)

# %% Load the data & model
data_dir = r"/data/storage_crack_detection/lightning-hydra-template/data/impasto"
IMPASTO_train_dir = "2024-11-26_512x512_val.h5"
data_train = HDF5PatchesDatasetCustom(hdf5_file_path = os.path.join(data_dir, IMPASTO_train_dir))

dataloader_train = DataLoader(
                                dataset=data_train,
                                batch_size= 1,
                                shuffle=True,
                            )

device = "cuda" 
model_dir = r"/data/storage_crack_detection/Pretrained_models/AutoEncoderKL"

vae =  AutoencoderKL.from_pretrained(model_dir, local_files_only=True).to(device)

# %% Adding cracks + encoding

# Get binary shape masks
MPEG_path   = r"/data/storage_crack_detection/datasets/MPEG400"
cat_name    = 'brick'
masks       = get_shapes(MPEG_path, cat_name, plot=False)

output_dir = data_dir
output_filename = r"2024-11-26_Enc_synthetic_mix_512x512_val.h5"
output_filename_full_h5 = os.path.join(output_dir, output_filename)
for i, (rgb, height, id) in tqdm(enumerate(dataloader_train)):
    
    # Add, transform and encode synthetic cracks
    if i % 2 == 0:
        height_cracks, rgb_cracks = add_cracks_with_lifted_edges_V2(height, rgb, 
                                                                    masks=masks, 
                                                                    decay_rate=2)
        rgb_cracks      = normalize_rgb(rgb_cracks)
        height_cracks   = rescale_diffuser_height_idv(height_cracks)
        enc_rgb_cracks, enc_height_cracks   = encode(vae, rgb_cracks, height_cracks)
    else:
        enc_rgb_cracks = None
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
                       )
    else:
        # Appending h5 file
        append_h5f_enc(output_filename_full_h5, 
                       rgb          = enc_rgb,
                       rgb_cracks   = enc_rgb_cracks,
                       height       = enc_height,
                       height_cracks= enc_height_cracks,
                       )

