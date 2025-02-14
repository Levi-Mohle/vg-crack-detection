from torch.utils.data import Dataset
import h5py


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
    def __init__(self, hdf5_file_path, transform=None):
        """
        Args:
            hdf5_file_path (str): name of the input file including full path and extension
            transform (torchvision.transform): transformations to be applied on input
        """

        # Open h5 file
        self.hdf5_file_path = hdf5_file_path
        self.h5_file = h5py.File(self.hdf5_file_path, 'r')

        # self.rgb_transform      = rgb_transform
        # self.height_transform   = height_transform
        self.transform = transform

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

        if self.transform:
            rgb, height = self.transform(rgb, height)

        return rgb, height, target