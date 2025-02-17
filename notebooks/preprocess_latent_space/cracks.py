import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL.ImageOps 
import torch
from skimage import morphology, measure
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb 
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF

from src.data.components.transforms import *
from notebooks.preprocess_latent_space.latent_space import encode_2ch
from notebooks.preprocess_latent_space.dataset import append_h5f_enc, create_h5f_enc

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

def get_grad_mask(masks, flap_height, decay_rate, seed=None):
    """
    Applies an exponential gradient to the given mask in a random direction

    Args:
        masks (np.array) : 2D binary mask containing a shape
        flap_height (int) : maximum height value to be added to the existing height map
        decay_rate (float) : Rate of decay for gradient on flap
        seed (int) : fixed seed

    Returns:
        grad_mask (np.array) : 2D mask containing the shape with exponential gradient
        
    """
    # Get mask shape
    bs, mask_h, mask_w = masks.shape

    # If defined, fix the seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Create gradients in different random directions
    x = np.linspace(0, 1, mask_w)
    y = np.linspace(0, 1, mask_h)
    xv, yv = np.meshgrid(x,y)

    exp_grads = np.zeros((bs, mask_h, mask_w))
    for i in range(bs):
        # Pick random gradient angle
        angle_grad = random.uniform(0, 2*np.pi)
        
        # Rotate gradient
        lin_grad = np.cos(angle_grad) * xv + np.sin(angle_grad) * yv

        # Normalize to range [0,1]
        lin_grad -= lin_grad.min()
        lin_grad /= lin_grad.max()

        # Apply exponentional 
        exp_grad = np.exp(decay_rate * lin_grad)

        # Normalize to range [0,flap_height]
        exp_grad        -= exp_grad.min()
        exp_grad        /= exp_grad.max()
        exp_grads[i]    = exp_grad * flap_height[i].item()
    
    exp_grads = torch.tensor(exp_grads)
    # Apply gradient to mask
    grad_mask = masks * exp_grads

    # Find second smallest value + subtract for smooth gradient
    reshape_mask    = grad_mask.view(bs,-1)
    sort_mask, _    = torch.sort(reshape_mask, dim=1)
    min_vals        = sort_mask[:, 1]

    grad_mask -= min_vals.view(bs,1,1)

    return grad_mask

def Create_cracks_with_lifted_edges(height, rgb, masks, flap_height= None, decay_rate=2, seed=None
                                    ,get_seg_map=False):
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
    bs, _, img_h, img_w= height.shape

    # Operations not possible on uint16, so conversion is required
    height = height.to(torch.float)
    
    # If not defined, take average height * factor to add (maximum)
    if flap_height is None:
        flap_height = (height.mean(dim=(2,3)) - height.amin(dim=(2,3))) * 3

    # If defined, fix the seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Pick a mask
    batch_masks = random.sample(masks, bs)
    batch_masks = torch.tensor(np.array(batch_masks))
    # Pick random transformation for the shape
    transform = transforms.Compose([
                    transforms.RandomRotation(degrees=[-180,180]),
                    transforms.RandomResize(min_size=300, max_size=450)
                ])
    transformed_masks = transform(batch_masks)
    # Get mask shape
    _, mask_h, mask_w = transformed_masks.shape

    # Generate random top-left corner coordinates for mask
    x_start = np.random.randint(0, img_h - mask_h, size=bs)
    y_start = np.random.randint(0, img_w - mask_w, size=bs)

    # Apply exponential gradient to mask
    grad_masks = get_grad_mask(transformed_masks, flap_height, decay_rate, seed)

    # Clone and randomly flip images as data augmentation
    cracked_height  = height.clone()
    rgb_cracked     = rgb.clone()
    if random.random() < 0.5:
        cracked_height  = TF.hflip(cracked_height)
        rgb_cracked     = TF.hflip(rgb_cracked)

    if random.random() < 0.5:
        cracked_height  = TF.vflip(cracked_height)
        rgb_cracked     = TF.vflip(rgb_cracked)
    
    # Apply gradient mask to height map
    segmentation_masks = torch.zeros((bs, 1, img_h, img_w))
    for i, (x, y) in enumerate(zip(x_start, y_start)):
        cracked_height[i,0, x:x+mask_h,y:y+mask_w] += grad_masks[i]
        segmentation_masks[i, 0, x:x+mask_h,y:y+mask_w] = transformed_masks[i]
    

    # Change RGB to black pixels if difference > threshold
    np_masks     = (grad_masks > 0).numpy().astype(np.uint8)
    edge_masks   = np.zeros_like(np_masks)
    for i in range(bs):
        edge_masks[i] = morphology.binary_dilation(\
                        morphology.binary_dilation(\
                            morphology.binary_dilation(\
                            (cv.Canny(image=np_masks[i], threshold1=0, threshold2=1) > 0).astype(np.uint8)))) 
    edge_masks = torch.tensor(edge_masks)
    mask2 = grad_masks * edge_masks >= torch.amax(grad_masks, dim=(1,2), keepdim=True) * 0.7

    # Apply rgb mask
    hsv = rgb2hsv(rgb_cracked.numpy(), channel_axis=1) # Conversion to hsv to decrease value channel
    for i, (x, y) in enumerate(zip(x_start, y_start)):
        hsv[i, :, x:x+mask_h,y:y+mask_w][2, mask2[i].numpy()]*= 0.2
    rgb_cracked = torch.tensor(hsv2rgb(hsv, channel_axis=1)) * 255
 
    return cracked_height.to(torch.uint16), rgb_cracked.to(torch.uint8), segmentation_masks

def add_synthetic_cracks_to_h5(dataloader, masks, p, filename, vae, add_cracks=True, device="cpu"):
    """
    Adds p percentage of synthetic cracks to the dataset provided with the dataloader. New data gets immediately
    encoded useing a vae. Saves new dataset as h5 file as given filename

    Args:
        dataloader : dataloader containing the original dataset
        masks (np.ndarray) : 2D arrays containing shapes for the cracks
        p (float) : perctage of cracks that should be added to the dataset
        filename (str) : output filename of the h5 file
        add_cracks (bool) : boolean value to turn on/off adding any cracks
        vae (AutoEncoderKL): pre-trained vae

    Returns:
        
    """
    for i, (rgb, height, id) in enumerate(tqdm(dataloader)):

        id = None # Uncomment if you want to ignore original labels

        # Add, transform and encode synthetic cracks
        every_n_samples = int(1/p)
        if (i % every_n_samples == 0) & add_cracks:
            height_cracks, rgb_cracks, _ = Create_cracks_with_lifted_edges(height, rgb, 
                                                                            masks=masks, 
                                                                            decay_rate=2)
            rgb_cracks      = normalize_rgb(rgb_cracks)
            height_cracks   = rescale_diffuser_height_idv(height_cracks)

            rgb_cracks, height_cracks   = encode_2ch(vae, rgb_cracks, height_cracks, device=device)
        else:
            rgb_cracks    = None
            height_cracks = None
        
        # Transform and encode normal samples
        rgb                 = normalize_rgb(rgb)
        height              = rescale_diffuser_height_idv(height)
        rgb, height         = encode_2ch(vae, rgb, height, device=device)
        
        if not os.path.exists(filename):
            # Creating new h5 file
            create_h5f_enc(filename, 
                        rgb          = rgb,
                        rgb_cracks   = rgb_cracks,
                        height       = height,
                        height_cracks= height_cracks,
                        target       = id
                        )
        else:
            # Appending h5 file
            append_h5f_enc(filename, 
                        rgb          = rgb,
                        rgb_cracks   = rgb_cracks,
                        height       = height,
                        height_cracks= height_cracks,
                        target       = id
                        )