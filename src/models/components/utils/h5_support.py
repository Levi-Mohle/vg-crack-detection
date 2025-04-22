import os
import h5py

def save_reconstructions_to_h5(output_file_name, batch, cfg=False):
    if not os.path.exists(output_file_name):
        # Creating new h5 file
        create_h5f_reconstruct(output_file_name, batch, cfg)
    else:
        # Appending h5 file
        append_h5f_reconstruct(output_file_name, batch, cfg)
    return 0

def create_h5f_reconstruct(output_filename_full_h5, batch, cfg):    
    """
    Create and save h5 file to store crack and normal tiny patches in

    This function creates h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_0
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_1
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
                            data = batch[0][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
        h5f.create_dataset('meas_capture/rgb',
                            data = batch[0][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
        if cfg:
            # Reconstruction with label 0 ("non-cracks")
            h5f.create_dataset('reconstructed_0/height',
                            data = batch[1][0][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
            h5f.create_dataset('reconstructed_0/rgb',
                            data = batch[1][0][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
            # Reconstruction with label 1 ("cracks")
            h5f.create_dataset('reconstructed_1/height',
                            data = batch[1][1][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
            h5f.create_dataset('reconstructed_1/rgb',
                            data = batch[1][1][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
        else:
            h5f.create_dataset('reconstructed_0/height',
                            data = batch[1][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
            h5f.create_dataset('reconstructed_0/rgb',
                            data = batch[1][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
            
        h5f.create_dataset('extra/OOD',
                           data = batch[2],
                           maxshape= (None,),
                           dtype= 'uint8')
        # Close the Keyence file for reading and the Keyence file for writing
        h5f.close()   

def append_h5f_reconstruct(output_filename_full_h5, batch, cfg=False):
    """
    Open and append a h5 file to store crack and normal tiny patches in

    This function opens h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_0
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_1
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

    Returns:
    """
    with h5py.File(output_filename_full_h5, 'a') as hdf5:
        rgbs        = hdf5['meas_capture/rgb']
        heights     = hdf5['meas_capture/height']
        r0_rgbs      = hdf5['reconstructed_0/rgb']
        r0_heights   = hdf5['reconstructed_0/height']
        OODs        = hdf5['extra/OOD']

        original_size = rgbs.shape[0]

        if cfg:
            r1_rgbs      = hdf5['reconstructed_1/rgb']
            r1_heights   = hdf5['reconstructed_1/height']

            r0_rgb      = batch[1][0][:,:3]
            r0_height   = batch[1][0][:,3:]
            r1_rgb      = batch[1][1][:,:3]
            r1_height   = batch[1][1][:,3:]

            r1_rgbs.resize(original_size + r1_rgb.shape[0], axis=0)
            r1_heights.resize(original_size + r1_height.shape[0], axis=0)

            r1_rgbs[original_size:]      = r1_rgb
            r1_heights[original_size:]   = r1_height
        else:
            r0_rgb      = batch[1][:,:3]
            r0_height   = batch[1][:,3:]

        rgb         = batch[0][:,:3]
        height      = batch[0][:,3:]
        id          = batch[2]
 
        rgbs.resize(original_size + rgb.shape[0], axis=0)
        heights.resize(original_size + height.shape[0], axis=0)
        r0_rgbs.resize(original_size + r0_rgb.shape[0], axis=0)
        r0_heights.resize(original_size + r0_height.shape[0], axis=0)
        OODs.resize(original_size + id.shape[0], axis=0)

        rgbs[original_size:]        = rgb
        heights[original_size:]     = height
        r0_rgbs[original_size:]      = r0_rgb
        r0_heights[original_size:]   = r0_height

        OODs[original_size:]     = id

        # Close the Keyence file for reading and the Keyence file for writing
        hdf5.close()