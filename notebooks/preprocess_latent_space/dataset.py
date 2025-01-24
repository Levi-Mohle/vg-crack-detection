"""
Routines to create a dataset consisting of mini-patches from a keyence file

    Source Name : dataset.py
    Contents    : classes and functions to create dataset object from keyence file
    Date        : 2024

 """

import numpy as np
from torch.utils.data import Dataset
import cv2
import h5py
import os
# from vg.fileio import keyence
# from vg.utils import log

# --------------------------------------------------------------------------------------------------
# SUPPORT FUNCTIONS
# --------------------------------------------------------------------------------------------------

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

def get_inner_minipatch_index(main_size, mini_size):
    
    _, indices = snake_matrix(0, main_size)

    inner_patch_index = indices[1:main_size[0] -1, 1:main_size[1] -1].flatten().astype('int').tolist()

    library = index_libary(main_size[0]*main_size[1], mini_size)

    inner_mini_patches = library[inner_patch_index,:,:].flatten().tolist()

    return inner_mini_patches

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

def save_dataset_to_hdf5(filename, dataset):
    """ Function to save a dataset instance back into an h5 file.
    Args:
        filename (str): name of the output file including full path and extension
        dataset (dataset object): dataset object created by HDF5CreatePatchesDataset
    """
    # Create h5 file and start writing
    with h5py.File(filename, 'w') as f:
        num_samples = len(dataset)
        rgb_shape = dataset[0][0].shape
        height_shape = dataset[0][1].shape

        # Create dataset for RGB images
        rgb_dset = f.create_dataset(
            'meas_capture/rgb', shape=(num_samples, *rgb_shape), dtype='uint8'
        )

        # Create dataset for height maps
        height_dset = f.create_dataset(
            'meas_capture/height', shape=(num_samples, *height_shape), dtype='uint16'
        )

        # Create dataset for mini-patch ids
        ood_dset = f.create_dataset(
            'extra/OOD', shape=(num_samples), dtype='uint8'
        )

        # Fill respective datasets
        for idx, (rgb, height, target) in enumerate(dataset):
            rgb_dset[idx] = rgb
            height_dset[idx] = height
            ood_dset[idx] = target

    f.close()

def split_h5_file(input_file, n_splits):
    """
    Splits data in existing h5 file with specific structure into n evenly divided
    smaller h5 file. Smaller h5 file get saved into a folder in the input file 
    directory

    This function splits h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)

    Args:
        input_file (str) : full name + directory of h5 file which you want to split
        n_splits (int) : Number of smaller h5 files to split the original file in.

    Returns:
        
    """

    # Get the directory of the input file
    input_dir = os.path.dirname(input_file)

    output_prefix = os.path.basename(input_file)[:-3]
    # Create a folder to store the split files
    split_folder = os.path.join(input_dir, f"{output_prefix}_splits")
    os.makedirs(split_folder, exist_ok=True)
    print(f"Split files will be stored in: {split_folder}")

    with h5py.File(input_file, 'r') as f:
        # Access dataset in the file
        height  = f["meas_capture/height"]
        rgb     = f["meas_capture/rgb"]
        OOD     = f["extra/OOD"]

        total_items = height.shape[0]
        assert rgb.shape[0] == total_items and OOD.shape[0] == total_items, \
            "All datasets must ahve the same number of items along the first dimension."
        
        # Calculate chunk size
        chunk_size = total_items // n_splits
        remainder = total_items % n_splits

        start = 0
        for i in range(n_splits):
            end = start + chunk_size + (1 if i < remainder else 0)

            # Create a new split file
            output_file = os.path.join(split_folder, f"{output_prefix}_part_{i}.h5")

            with h5py.File(output_file, 'w') as split_f:
                # Create the same structure in the split file
                meas_capture_grp = split_f.create_group("meas_capture")
                extra_grp = split_f.create_group("extra")

                # Add split datasets
                meas_capture_grp.create_dataset("height", data=height[start:end])
                meas_capture_grp.create_dataset("rgb", data=rgb[start:end])
                extra_grp.create_dataset("OOD", data=OOD[start:end])

            print(f"Created {output_file} with {end - start} entries.")
            start = end

def combine_h5_files(input_folder, output_file):
    """
    Combines data from multiple h5 file with specific structure into 1 bigger h5 file. 
    The combined h5 file gets saved in the directory specified in output_file

    This function combines h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)

    Args:
        input_folder (str) : full name + directory of folder which contains smaller h5 files
        output_file (int) : full name + directory of combined h5 file

    Returns:
        
    """
    # Get the list of smaller .h5 files
    h5_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".h5")]
    h5_files.sort()
    
    if not h5_files:
        raise ValueError("No .h5 files found in the specified folder")
    
    with h5py.File(output_file, 'w') as combined_f:
        # Create lists to be filled later
        combined_height = []
        combined_rgb = []
        combined_OOD = []

        # Loop over all h5 files and fill lists
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                combined_height.append(f["meas_capture/height"][:])
                combined_rgb.append(f["meas_capture/rgb"][:])
                combined_OOD.append(f["extra/OOD"][:])

        # Create 1 array from list
        height  = np.concatenate(combined_height, axis=0)
        rgb     = np.concatenate(combined_rgb, axis=0)
        OOD     = np.concatenate(combined_OOD, axis=0)

        # Create the same structure in the split file
        meas_capture_grp    = combined_f.create_group("meas_capture")
        extra_grp           = combined_f.create_group("extra")

        # Add split datasets
        meas_capture_grp.create_dataset("height", data=height)
        meas_capture_grp.create_dataset("rgb", data=rgb)
        extra_grp.create_dataset("OOD", data=OOD)

    print(f"Combined file created: {output_file}")

def create_h5f_enc(output_filename_full_h5, rgb, height, id):    
    """
    Create and save h5 file to store crack and normal tiny patches in

    This function creates h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)

    Args:
        output_filename_full_h5 (str): filename + location of h5 file you want to create and save
        rgb_cracks (torch.Tensor): rgb tiny patches containing cracks (N,3,height,width)
        height_cracks (torch.Tensor): height tiny patches containing cracks (N,1,height,width)
        rgb_normal (torch.Tensor): rgb tiny patches containing normal samples (N,3,height,width)
        height_normal (torch.Tensor): height tiny patches containing normal samples (N,1,height,width)

    Returns:
        
    """
    
    with h5py.File(output_filename_full_h5, 'w') as h5f:
        h5f.create_dataset('meas_capture/height',
                            data = height,
                            maxshape = (None, 4, 64, 64),
                            dtype='float')
        h5f.create_dataset('meas_capture/rgb',
                            data = rgb,
                            maxshape = (None, 4, 64, 64),
                            dtype='float')
        h5f.create_dataset('extra/OOD',
                           data = id,
                           maxshape= (None,),
                           dtype= 'float')
        # Close the Keyence file for reading and the Keyence file for writing
        h5f.close()   

def append_h5f_enc(output_filename_full_h5, rgb, height, id):
    """
    Open and append a h5 file to store crack and normal tiny patches in

    This function opens h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)
            
    Args:
        output_filename_full_h5 (str): filename + location of h5 file you want to open and append
        rgb_cracks (torch.Tensor): rgb tiny patches containing cracks (N,3,height,width)
        height_cracks (torch.Tensor): height tiny patches containing cracks (N,1,height,width)
        rgb_normal (torch.Tensor): rgb tiny patches containing normal samples (N,3,height,width)
        height_normal (torch.Tensor): height tiny patches containing normal samples (N,1,height,width)
        tiny_size (np.ndarray): tiny patch size in pixels (heigh, width)

    Returns:
    """
    with h5py.File(output_filename_full_h5, 'a') as hdf5:
        rgbs     = hdf5['meas_capture/rgb']
        heights  = hdf5['meas_capture/height']
        OODs     = hdf5['extra/OOD']

        original_size = rgbs.shape[0]

        rgbs.resize(original_size + rgb.shape[0], axis=0)
        heights.resize(original_size + height.shape[0], axis=0)
        OODs.resize(original_size + id.shape[0], axis=0)

        rgbs[original_size:]     = rgb
        heights[original_size:]  = height

        OODs[original_size:]     = id

        # Close the Keyence file for reading and the Keyence file for writing
        hdf5.close()

# --------------------------------------------------------------------------------------------------
# CLASSES
# --------------------------------------------------------------------------------------------------

class HDF5CreatePatchesDataset(Dataset):
    """ A class which creates mini-patches as a Dataset object 
        from height and rgb images directly from a Keyence h5 file

    """
    def __init__(self, input_filename, idx_mini_range, patch_size
                 , inpaint=False, transform=None, down_scale_factor = 1):
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

class HDF5CreatePatchesDatasetFULL(Dataset):
    """ A class which creates mini-patches as a Dataset object 
        from height and rgb images directly from a Keyence h5 file

    """
    def __init__(self, input_filename, mp_list , mp_grid, tp_grid, 
                 inpaint=False, down_scale_factor = 1):
        """
        Args:
            input_filename (str): name of the input file including full path and extension
            mini_idx_range ([int]) : list of integers indices of mini-patch indices 
                                    from original painting
            patch_size ([int,int]) : The new grid you want to divide the original patches in
            inpaint (bool): Pre-processing height images by inpainting 0 values
            down_scale_factor (int) : factor by which you downscale original painting patches
        """

        self.input_filename     = input_filename
        self.mp_list            = mp_list
        self.mp_grid            = mp_grid
        self.tp_grid            = tp_grid
        self.down_scale_factor  = down_scale_factor

        # Open the input Keyence data file for reading, and read some required parameters from it
        self.kr = keyence.Read(self.input_filename, verbose=False)
        self.num_images = self.kr.num_images()
        self.total_tp = len(mp_list) * self.tp_grid[0] * self.tp_grid[1]
        self.tp_per_mp          = self.tp_grid[0] * self.tp_grid[1]

        # Convert mini-patch indices to main patch indices
        indices = index_libary(self.kr.num_images(), self.mp_grid)
        self.mp_list.sort()
        self.idx_range = []
        self.mini_patch_indices = []
        self.tiny_patch_indices = [[i, j] for i in range(self.tp_grid[0]) for j in range(self.tp_grid[1])]
        for idx in self.mp_list:
            main, row, col = np.where(indices == idx)
            self.idx_range.append(main[0])
            self.mini_patch_indices.append([row[0], col[0]])
       
        # Prevent opening same main patch twice
        self.idx_range_unique = list(set(self.idx_range))
        self.idx_range_unique.sort()

        self.img_list = [i for i in self.idx_range_unique for _ in range(self.tp_per_mp)]

        self.num_rows                = int(self.kr.num_rows()/self.down_scale_factor)
        self.num_cols                = int(self.kr.num_cols()/self.down_scale_factor)

        if self.num_rows // self.mp_grid[0] == 0:
            log.error(f"Original dimension of {self.num_rows} not divisible by {self.mp_grid[0]}")
        elif self.num_cols // self.mp_grid[1] == 0:
            log.error(f"Original dimension of {self.num_cols} not divisible by {self.mp_grid[1]}")

        self.mp_size = [self.num_rows // self.mp_grid[0],
                           self.num_cols // self.mp_grid[1]]
        
        self.tp_size = [self.num_rows // (self.mp_grid[0]*self.tp_grid[0]),
                           self.num_cols // (self.mp_grid[1]*self.tp_grid[1])]

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
        return self.total_tp

    def __getitem__(self, idx: int) -> tuple[any, any]:

        # Calculate which image and which chunk within the image
        
        mp_idx  = idx // self.tp_per_mp
        tp_idx  = idx % self.tp_per_mp
        img_idx = assign_indices(self.idx_range)[mp_idx]
        
        row_start_mp = self.mini_patch_indices[img_idx][0] * self.mp_size[0] 
        col_start_mp = self.mini_patch_indices[img_idx][1] * self.mp_size[1]

        row_start_tp = row_start_mp + self.tiny_patch_indices[tp_idx][0] * self.tp_size[0]
        col_start_tp = col_start_mp + self.tiny_patch_indices[tp_idx][1] * self.tp_size[1]

        height_chunks = self.height[img_idx, :, row_start_tp:row_start_tp + self.tp_size[0], col_start_tp: col_start_tp+self.tp_size[1]]
        rgb_chunks = self.rgb[img_idx, :, row_start_tp:row_start_tp + self.tp_size[0], col_start_tp: col_start_tp+self.tp_size[1]]

        # All samples are normal
        target = 0

        # if self.transform is not None:
        #     all_chunks = self.transform(all_chunks)

        return rgb_chunks, height_chunks, target # all_chunks
    
class HDF5PatchesDataset(Dataset):
    """ A class that reads h5 files created by HDF5CreatePatchesDataset into a Dataset object
    
    """
    def __init__(self, hdf5_file_path, transform=None):
        """
        Args:
            hdf5_file (str): name of the input file including full path and extension
            transform (torchvision.transform): transformations to be applied on input
        """

        # Open h5 file
        self.hdf5_file_path = hdf5_file_path
        self.h5_file = h5py.File(self.hdf5_file_path, 'r')

        self.transform = transform

        # Assign data from h5 file
        self.rgb = self.h5_file['meas_capture/rgb']
        self.height = self.h5_file['meas_capture/height']
        self.ids = self.h5_file['extra/OOD']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rgb = self.rgb[idx][:]
        height = self.height[idx][:]
        id_number = self.ids[idx]

        if self.transform:
            rgb = self.transform(rgb)
            height = self.transform(height)

        # Target label (only zeros because all are ID), for classification purposes
        # TODO for test dataset create proper assignment of target labels
        target = np.zeros(self.__len__())[idx]

        return rgb, height, id_number, target

class HDF5PatchesDatasetCustom(Dataset):
    """ A class that reads h5 files created by build_crackset.py into a Dataset object.
        This function can only read h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)
    
    """
    def __init__(self, hdf5_file_path, rgb_transform=None, height_transform=None):
        """
        Args:
            hdf5_file (str): name of the input file including full path and extension
            transform (torchvision.transform): transformations to be applied on input
        """

        # Open h5 file
        self.hdf5_file_path = hdf5_file_path
        self.h5_file = h5py.File(self.hdf5_file_path, 'r')

        self.rgb_transform      = rgb_transform
        self.height_transform   = height_transform

        # Assign data from h5 file
        self.rgb = self.h5_file['meas_capture/rgb']
        self.height = self.h5_file['meas_capture/height']
        self.target = self.h5_file['extra/OOD']

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        rgb     = self.rgb[idx][:]
        height  = self.height[idx][:]
        target  = self.target[idx]

        if self.rgb_transform:
            rgb     = self.rgb_transform(rgb.transpose(1,2,0))
            
        if self.height_transform:
            height  = self.height_transform(height.transpose(1,2,0))

        return rgb, height, target
    
class HDF5PatchesDatasetReconstructs(Dataset):
    """ A class that reads h5 files created by build_crackset.py into a Dataset object.
        This function can only read h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)
    
    """
    def __init__(self, hdf5_file_path, rgb_transform=None, height_transform=None):
        """
        Args:
            hdf5_file (str): name of the input file including full path and extension
            transform (torchvision.transform): transformations to be applied on input
        """

        # Open h5 file
        self.hdf5_file_path = hdf5_file_path
        self.h5_file = h5py.File(self.hdf5_file_path, 'r')

        self.rgb_transform      = rgb_transform
        self.height_transform   = height_transform

        # Assign data from h5 file
        self.rgb        = self.h5_file['meas_capture/rgb']
        self.height     = self.h5_file['meas_capture/height']
        self.r_rgb      = self.h5_file['reconstructed/rgb']
        self.r_height   = self.h5_file['reconstructed/height']
        self.target     = self.h5_file['extra/OOD']

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        rgb         = self.rgb[idx][:]
        height      = self.height[idx][:]
        r_rgb       = self.r_rgb[idx][:]
        r_height    = self.r_height[idx][:]
        target      = self.target[idx]

        if self.rgb_transform:
            rgb     = self.rgb_transform(rgb.transpose(1,2,0))
            r_rgb   = self.rgb_transform(r_rgb.transpose(1,2,0))
            
        if self.height_transform:
            height      = self.height_transform(height.transpose(1,2,0))
            r_height    = self.height_transform(r_height.transpose(1,2,0))

        return rgb, height, r_rgb, r_height, target
    
