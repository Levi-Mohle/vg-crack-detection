"""
Routines to create a preprocessed dataset consisting of mini-patches from a keyence file

    Source Name : preprocess.py
    Contents    : 
    Date        : 2025

 """

import numpy as np
from torch.utils.data import Dataset
import cv2
import h5py
import keyence
import log
import torch

############################################################
# Support functions
############################################################

def snake_matrix(start, size):
    """
    Creates a matrix of size patch_size, which has incremental values starting from start,
    going in a snake-like structure, starting left upper corner to the right.

    Args:
        start (int): value of the upperleft corner of the matrix
        size ([int, int]): size of the matrix

    Returns:
        end (int): last value of the snake-like matrix
        snake (np.array): the snake-like matrix itself
    """
    # Define matrix
    snake = np.zeros((size[0], size[1]))

    for x in range(size[0]):
        if x%2==0:
            for y in range(size[1]):
                snake[x,y] = start
                start += 1
        else:
            for y in reversed(range(size[1])):
                snake[x,y] = start
                start += 1
    end = start
    return end, snake

def index_libary(nr_patches, patch_size):
    """ Creates a indices matrix which represent main patch indices on first axis and
        mini-patch indices on second and third axis

    Args:
        nr_patches (int): Total number of main patches in h5 file
        patch_size ([int,int]): division of mini-patches in 1 main patch

    Returns:
        index library (np.array) for all mini-patches of 1 painting
    """
    # Create zero array
    indices = np.zeros((nr_patches, patch_size[0], patch_size[1]))

    start = 0
    for n in range(nr_patches):
        start, indices[n,:,:] = snake_matrix(start, patch_size)

    return indices.astype('uint16')

def assign_indices(lst):
    """ Function to convert main patch indices to a range starting from 0.
        [3,3,5,7] becomes [0,0,1,2]
    Args:
        lst ([int]): list of main patch indices
    Returns:
        lst ([int]): list of converted indices
    """
    unique_values = sorted(set(lst))
    value_to_index = {value: idx for idx, value in enumerate(unique_values)}

    return [value_to_index[value] for value in lst]

def encode_data(vae, data):
    with torch.no_grad():
        latent = vae.encode(data).latent_dist.sample().mul_(0.18215)

    return latent

############################################################
# Classes
############################################################

class HDF5CreatePatchesDataset(Dataset):
    """ A class which creates mini-patches as a Dataset object 
        from height and rgb images directly from a Keyence h5 file

    """
    def __init__(self, input_filename, idx_mini_range, patch_size
                 , inpaint=False, transform=None, latent=False,
                 down_scale_factor = 1):
        """
        Args:
            input_filename (str): name of the input file including full path and extension
            mini_idx_range ([int]) : list of integers indices of mini-patch indices 
                                    from original painting
            patch_size ([int,int]) : The new grid you want to divide the original patches in
            inpaint (bool): Pre-processing height images by inpainting 0 values
            down_scale_factor (int) : factor by which you downscale original painting patches
        """

        self.input_filename = input_filename
        self.idx_mini_range = idx_mini_range
        self.patch_size = patch_size
        self.transform = transform
        self.latent = latent
        self.down_scale_factor = down_scale_factor

        # Open the input Keyence data file for reading, and read some required parameters from it
        self.kr = keyence.Read(self.input_filename, verbose=False)

        # Convert mini-patch indices to main patch indices
        indices = index_libary(self.kr.num_images(), self.patch_size)
        self.idx_mini_range.sort()
        self.idx_range = []
        self.mini_patch_indices = []
        for idx in self.idx_mini_range:
            main, row, col = np.where(indices == idx)
            self.idx_range.append(main[0])
            self.mini_patch_indices.append([row[0], col[0]])

        # Prevent opening same main patch twice
        self.idx_range_unique = list(set(self.idx_range))
        self.idx_range_unique.sort()

        self.num_rows                = int(self.kr.num_rows()/self.down_scale_factor)
        self.num_cols                = int(self.kr.num_cols()/self.down_scale_factor)

        if self.num_rows // self.patch_size[0] == 0:
            log.error(f"Original dimension of {self.num_rows} not divisible by {self.patch_size[0]}")
        elif self.num_cols // self.patch_size[1] == 0:
            log.error(f"Original dimension of {self.num_cols} not divisible by {self.patch_size[1]}")

        self.chunk_size = [self.num_rows // self.patch_size[0],
                           self.num_cols // self.patch_size[1]]

        self.total_chunks = len(self.idx_mini_range)

        # Create empty lists to fill from h5 file
        self.height = []
        self.rgb = []
        # Loop across all image patches.
        for idx_r in self.idx_range_unique:
            log.info(f"Reading image and position data for index {idx_r}")

            height_data = cv2.resize(self.kr.height_image(idx_r),
                                        (self.num_rows, self.num_cols),
                                        interpolation=cv2.INTER_NEAREST)

            # Fill in 0 values by inpaint
            if inpaint:
                # Calculate the lowerbound of the 99.7% confidence interval
                int99 = np.mean(height_data) - 3 * np.std(height_data)
                mask = (height_data <= int99).astype('uint8')
                height_data = cv2.inpaint(height_data, mask, 10, cv2.INPAINT_NS)

            rgb_data = cv2.resize(self.kr.rgb_image(idx_r),
                                    (self.num_rows, self.num_cols),
                                    interpolation=cv2.INTER_CUBIC)
            # position_data = self.kr.position_m(idx_r)
            self.height.append(height_data)
            self.rgb.append(rgb_data)

        # Concatenate all rgb and height images separate
        self.height = np.expand_dims(np.stack(self.height), axis=1) # size [num_rows, num_cols, 1]
        self.rgb = np.stack(self.rgb).transpose(0,3,1,2) # size [num_rows, num_cols, 3]

        # Concatenate rgb and height images together
        # self.all = np.concatenate((self.rgb, np.expand_dims(self.height, axis=-1)), axis=-1) # size [num_rows, num_cols, 4]

        # Close the Keyence file for reading
        self.kr.close()

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx: int) -> tuple[any, any]:

        # Calculate which image and which chunk within the image
        img_idx = assign_indices(self.idx_range)[idx]
        row_idx = self.mini_patch_indices[idx][0]
        col_idx = self.mini_patch_indices[idx][1]

        row_start = row_idx * self.chunk_size[0]
        col_start = col_idx * self.chunk_size[1]

        height_chunks = self.height[img_idx, :, row_start:row_start + self.chunk_size[0], col_start: col_start+self.chunk_size[1]]
        rgb_chunks = self.rgb[img_idx, :, row_start:row_start + self.chunk_size[0], col_start: col_start+self.chunk_size[1]]

        # all_chunks = self.all[img_idx, row_start:row_start + self.chunk_size[0], col_start: col_start+self.chunk_size[1], :]
        minipatch_nr = self.idx_mini_range[idx]

        if self.transform is not None:
            all_chunks = self.transform(all_chunks)

        return rgb_chunks, height_chunks, minipatch_nr # all_chunks