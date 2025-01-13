"""
File io routines for the Keyence height sensor

    Source Name : keyence.py
    Contents    : routines for accessing Keyence sensor data files
    Date        : 2023

    In brief, these are the groups and datasets inside a Keyence h5 file:

    keyence_file.h5
    ├───file_info
    │   ├────date_time               (string)
    │   ├────device_temperature_C    (32-bit float scalar)
    │   ├────factor_height_to_meters (32-bit float scalar)
    │   ├────file_version            (string)
    │   ├────image_height_m          (32-bit float scalar, height of patch image in m)
    │   ├────image_width_m           (32-bit float scalar, width of patch image in m)
    │   ├────num_cols                (32-bit integer scalar, number of patch image columns)
    │   ├────num_images              (32-bit integer scalar)
    │   ├────num_rows                (32-bit integer scalar, number of patch image rows)
    │   └────operator_name           (string)
    └───meas_capture
        ├────height_um               (16-bit uint array size num_images x num_rows x num_cols)
        ├────position_xyz_ m         (32-bit float array size num_images x 3)
        └────rgb                     (16-bit uint array size num_images x num_rows x num_cols x 3)
 """

import sys
from pathlib import Path
import h5py
import numpy as np
from time import gmtime, strftime
from vg.utils import log
#from vg.utils import polynomial

CURRENT_FILE_VERSION = "2.1.0"


LEGACY_METERS_PER_KEYENCE_HEIGHT_INT = 0.000000250000011874363


class Read:
    """ A class which opens a Keyence h5 file and has member functions to read elements from it.

    Attributes:
        h5f (h5py.File): handle to hdf5 file which is opened for reading upon class initialization
        verbose (boolean): to print or not to print intermediate messages for debugging purposes
    """

    def __init__(self, in_file, verbose=False, legacy=False):
        """
        Args:
            in_file (str): name of the input file including full path and extension
            verbose (boolean): when True, verbose text is printed for debugging, default is False
            legacy (bool): When True, older hdf5 files of version 1.0.0 can be 
                           imported by which missing elements are given default 
                           values. For example, file_info>image_height_m is not
                           in v1.0.0 hdf5 files. By default an error would be
                           thrown ("Error while reading..."), but with legacy
                           True, the float 0.024 meters is given.
        """

        try:
            if verbose:
                log.info('Opening Keyence file for reading...')
            self.h5f = h5py.File(in_file, "r")
        except FileNotFoundError:
            log.error(f'Cannot open Keyence file for reading: {in_file}')
            sys.exit(1)
        else:
            self.verbose = verbose

        self.is_legacy = False
        if legacy and self.file_version == '1.0.0':
            self.is_legacy = True
        elif legacy:
            log.error("While setting legacy"
                      f" - file is v{self.file_version}, but legacy only works for v1.0.0.")
            sys.exit()

    def num_rows(self):
        """Read the number of image rows from the opened Keyence data file.
        Returns:
            num_rows (int32): scalar value
        """

        if self.verbose:
            log.info("Reading the number of image rows from the Keyence file ...")
        num_rows = np.array(self.h5f["file_info/num_rows"])[0]
        if num_rows.dtype != np.int32:
            log.error(f'While reading with function num_rows() in file {__file__}'
                      ' - expected num_rows to be of type int32.')
            sys.exit()
        return num_rows    

    @property
    def file_version(self):
        """Read the version number from the opened Keyence data file.
        Returns:
            file_version (str): string in semantic versioning form "X.Y.Z"
        """
        if self.verbose:
            log.info("Reading the version number from the Keyence file ...")
        file_version = str(self.h5f["file_info/file_version"].asstr()[...])
        return file_version

    def num_cols(self):
        """Read the number of image columns from the opened Keyence data file.
        Returns:
            num_cols (int32): scalar value
        """

        if self.verbose:
            log.info("Reading the number of image columns from the Keyence file ...")
        num_cols = np.array(self.h5f["file_info/num_cols"])[0]
        if num_cols.dtype != np.int32:
            log.error(f'While reading with function num_cols() in file {__file__}'
                      ' - expected num_cols to be of type int32.')
            sys.exit()
        return num_cols

    def num_images(self):
        """Read the number of images from the opened Keyence data file.
        Returns:
            num_images (int32): scalar value
        """

        if self.verbose:
            log.info("Reading the number of images in the Keyence file ...")
        num_images = np.array(self.h5f["file_info/num_images"])[0]
        if num_images.dtype != np.int32:
            log.error(f'While reading with function num_images() in file {__file__}'
                      ' - expected num_images to be of type int32.')
            sys.exit()
        return num_images

    def height_image(self, im_indices):
        """Read height values from the opened Keyence data file.
        Args:
            im_indices (scalar or list): image indices to be retrieved
        Returns:
            height_data (uint16): 2D numpy array with height image if im_indices is a scalar
                                  3D numpy array with N height images if length(im_indices) == N
        """

        if self.verbose:
            log.info("Reading Keyence height data...")
        num_images = self.num_images()
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
            sys.exit()
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
            sys.exit()

        height_data = None

        if self.is_legacy:
            height_data = np.array(self.h5f["meas_capture/height_um"][im_indices,:,:]).squeeze()
        else:
            try:
                height_data = np.array(self.h5f["meas_capture/height"][im_indices,:,:]).squeeze()
            except KeyError:
                log.error(f'While reading with function height_image() in file {__file__}'
                          ' - height not in file.')
                sys.exit()

        # if height_data.dtype != np.uint16:  # Prevent clipping
        #     log.error(f'While reading with function height_image() in file {__file__}'
        #               ' - expected height data to be of type uint16.')
        #     sys.exit()

        return height_data

    def nan_mask(self, im_indices):
        """Read NaN mask values from the opened Keyence data file. A NaN mask value is 1 when there
        is a NaN in the height image. A NaN mask value is 0 otherwise.

        Args:
            im_indices (scalar or list): image indices to be retrieved
        Returns:
            nan_mask_data (uint8): 2D numpy array with NaN mask image if im_indices is a scalar
                                   3D numpy array with N NaN mask images if length(im_indices) == N
        """

        if self.verbose:
            log.info("Reading Keyence NaN mask data...")
        num_images = self.num_images()
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
            sys.exit()
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
            sys.exit()

        nan_mask_data = None

        if self.is_legacy:
            log.error(f'There is no legacy for NaN mask.')
            sys.exit()
        else:
            try:
                nan_mask_data = np.array(self.h5f["meas_capture/nan_mask"][im_indices,:,:]).squeeze()
            except KeyError:
                log.error(f'While reading with function nan_mask() in file {__file__}'
                          ' - NaN mask not in file.')
                sys.exit()

        if nan_mask_data.dtype != np.uint8:
            log.error(f'While reading with function nan_mask() in file {__file__}'
                      ' - expected NaN mask data to be of type uint8.')
            sys.exit()

        return nan_mask_data

    def factor_height_to_meters(self):
        """Reads the scaling factor to convert height image values with arbitrary unit to height
           image values in meters.

           Legacy returns np.float32(0.000000250000011874363).
        Returns:
            factor_height_to_meters (float32): scalar value
        """

        factor_height_to_meters = None

        if self.is_legacy:
            factor_height_to_meters = (
                np.float32(LEGACY_METERS_PER_KEYENCE_HEIGHT_INT)
            )
        elif self.file_version == '2.0.0':
            log.warning("Factor_height_to_meters is likely not yet filled in "
                        "correctly in HDF5 file ... taking the value 2.5E-7 for now.")
            # factor_height_to_meters = np.array(self.h5f["file_info/factor_height_to_meters"])[0]
            # factor_height_to_meters = factor_height_to_meters / np.float32(1000) # correct mm to m
            factor_height_to_meters = np.float32(2.5E-7)
        else:
            if self.verbose:
                log.info("Reading factor_height_to_meters from the Keyence file ...")
            try:
                factor_height_to_meters = \
                    np.array(self.h5f["file_info/factor_height_to_meters"])[0]
            except ValueError:
                log.error(f'While reading with factor_height_to_meters() in file {__file__}.'
                          ' - file_info/factor_height_to_meters not in file.')
                sys.exit()

        if factor_height_to_meters.dtype != np.float32:
            log.error(f'While reading with factor_height_to_meters() in file {__file__}'
                      ' - expected factor_height_to_meters to be of type float32.')
            sys.exit()

        return factor_height_to_meters

    def rgb_image(self, im_indices):
        """Read RGB data from the Keyence input file.
        Args:
            im_indices (scalar or list): image indices to be retrieved
        Returns:
            rgb_data (uint8): numpy array with RGB pixel values
        """

        if self.verbose:
            log.info("Reading Keyence rgb data...")
        num_images = self.num_images()
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
            sys.exit()
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
            sys.exit()
        r_data = np.array(self.h5f["meas_capture/rgb"][im_indices,:,:,0])
        g_data = np.array(self.h5f["meas_capture/rgb"][im_indices,:,:,1])
        b_data = np.array(self.h5f["meas_capture/rgb"][im_indices,:,:,2])
        rgb_data = np.stack([r_data, g_data, b_data], axis=-1)
        if rgb_data.dtype != np.uint8:
            log.error(f'While reading with function rgb_image() in file {__file__}'
                      ' - expected RGB data to be of type uint8.')
            sys.exit()

        return rgb_data

    def position_m(self, im_indices):
        """Read (x,y,z) CAS stage position data from the Keyence input file.

        Legacy returns (x,y), but no z.

        Returns:
            position_data (float32): numpy array with (x,y) coordinates in m
        """

        if self.verbose:
            log.info("Reading CAS stage position data...")
        num_images = self.num_images()
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')

        position_data = None

        if self.is_legacy:
            position_data = np.array(self.h5f["meas_capture/position_xy_mm"][im_indices,:])
            position_data = position_data / 1000.0
        else:
            try:
                position_data = np.array(self.h5f["meas_capture/position_xyz_m"][im_indices,:])
            except KeyError:
                log.error(f'While reading with function position_m() in file {__file__}'
                          ' - meas_capture/position_xyz_m not in file.')
                sys.exit()

        if self.file_version == '2.0.0':
            position_data = position_data / 1000.0

        if position_data.dtype != np.float32:
            log.error(f'While reading with function position_m() in file {__file__}'
                      ' - expected position data to be of type float32.')
            sys.exit()

        # to test the not-connected patch case, uncomment the next lines
        #position_data[40,0] = 0.4
        #position_data[40,1] = 0.2
        #position_data[70,0] = 0.3
        #position_data[70,1] = 0.1

        return position_data

    def image_height_m(self):
        """Read height of a Keyence image in m (all images in h5 file have same height).

        Legacy returns 0.024.

        Returns:
            image_height_m (float32): scalar value with image height in m
        """

        if self.verbose:
            log.info("Reading the Keyence image height in m ...")

        image_height_m = None

        if self.is_legacy:
            image_height_m = np.float32(0.024)
        else:
            try:
                image_height_m = np.array(self.h5f["file_info/image_height_m"])[0]
            except KeyError:
                log.error(f'While reading with function image_height_m() in file {__file__}'
                          ' - image_height_m not in file.')
                sys.exit()

        if image_height_m.dtype != np.float32:
            log.error(f'While reading with function image_height_m() in file {__file__}'
                      ' - expected image_height_m to be of type float32.')
            sys.exit()

        return image_height_m

    def image_width_m(self):
        """Read width of a Keyence image in m (all images in h5 file have same width).
        
        Legacy returns 0.024.
        
        Returns:
            image_width_m (float32): scalar value with image width in m
        """

        if self.verbose:
            log.info("Reading the Keyence image width in m ...")

        image_width_m = None

        if self.is_legacy:
            image_width_m = np.float32(0.024)
        else:
            try:
                image_width_m = np.array(self.h5f["file_info/image_width_m"])[0]
            except KeyError:
                log.error(f'While reading with function image_width_m() in file {__file__}'
                          ' - image_width_m not in file.')
                sys.exit()

        if image_width_m.dtype != np.float32:
            log.error(f'While reading with function image_width_m() in file {__file__}'
                      ' - expected image_width_m to be of type float32.')
            sys.exit()

        return image_width_m

    def close(self):
        """Close the Keyence data file for reading.
        """
        if self.verbose:
            log.info("Closing Keyence file...")
        self.h5f.close()


class Write:
    """ A class which opens a Keyence h5 file and has member functions to write elements to it.

    Attributes:
        h5f (h5py.File): handle to hdf5 file which is opened for writing upon class initialization
    """

    def __init__(self, out_file, image_height_m, image_width_m, num_rows, num_cols, num_images,
                 operator_name, verbose=False, create_if_does_not_exist=False):
        """
        Args:
            out_file (str): name of the output file including full path and extension
            verbose (boolean): when True, verbose text is printed for debugging, default is False
        """

        self.verbose = verbose

        try:
            if verbose:
                log.info(f'Opening Keyence file for writing: {out_file}')
            self.h5f = h5py.File(out_file, "w")
        except FileNotFoundError:
            if not create_if_does_not_exist:
                log.error(
                    'Keyence file does not exist. '
                    'Set flag create_if_does_not_exist=True to attempt to '
                    'create the file and its parent directory.')
                sys.exit(1)
            if verbose:
                log.info('Keyence file does not exist.')
            out_directory = Path(out_file).parent
            if not out_directory.exists():
                if verbose:
                    log.info(
                        'Parent directory does not exist for file. '
                        f'Creating directory {out_directory}.')
                out_directory.mkdir(parents=True)
            try:
                if verbose:
                    log.info('Creating file and opening for writing.')
                self.h5f = h5py.File(out_file, "w-")
            except FileNotFoundError:
                log.error(
                    'Cannot create or open Keyence file for writing: '
                    f'{out_file}.')
                sys.exit(1)

        self.image_height_m = image_height_m
        self.image_width_m = image_width_m
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_images = num_images

        if verbose:
            log.info('Creating hdf5 group file_info')

        self.h5f.create_group("file_info")

        date_time = strftime("%d/%m/%Y, %H:%M:%S", gmtime())
        dt = h5py.string_dtype(encoding='utf-8')
        dset = self.h5f.create_dataset('file_info/date_time', (1,), dtype=dt)
        dset[0] = date_time

        dset = self.h5f.create_dataset('file_info/device_temperature_C', (1,), dtype='float32')
        dset[0] = np.float32(0.0)

        dset = self.h5f.create_dataset('file_info/factor_height_to_meters', (1,), dtype='float32')
        dset[0] = np.float32(2.5e-7)

        dt = h5py.string_dtype(encoding='utf-8')
        dset = self.h5f.create_dataset('file_info/file_version', (1,), dtype=dt)
        dset[0] = CURRENT_FILE_VERSION

        dset = self.h5f.create_dataset('file_info/image_height_m', (1,), dtype='float32')
        dset[0] = image_height_m

        dset = self.h5f.create_dataset('file_info/image_width_m', (1,), dtype='float32')
        dset[0] = image_width_m

        dset = self.h5f.create_dataset('file_info/num_rows', (1,), dtype='int32')
        dset[0] = num_rows

        dset = self.h5f.create_dataset('file_info/num_cols', (1,), dtype='int32')
        dset[0] = num_cols

        dset = self.h5f.create_dataset('file_info/num_images', (1,), dtype='int32')
        dset[0] = num_images

        dt = h5py.string_dtype(encoding='utf-8')
        dset = self.h5f.create_dataset('file_info/operator_name', (1,), dtype=dt)
        dset[0] = operator_name

        if verbose:
            log.info('Creating hdf5 group meas_capture')

        self.h5f.create_group("meas_capture")

        self.height_image_dset = \
            self.h5f.create_dataset('meas_capture/height',
                                    (num_images,num_rows,num_cols,1),
                                    dtype='float32')  # Prevent clipping
        self.nan_mask_dset = \
            self.h5f.create_dataset('meas_capture/nan_mask',
                                    (num_images,num_rows,num_cols,1),
                                    dtype='uint8')
        self.rgb_image_dset = \
            self.h5f.create_dataset('meas_capture/rgb',
                                    (num_images,num_rows,num_cols,3),
                                    dtype='uint8')
        self.position_xyz_dset = \
            self.h5f.create_dataset('meas_capture/position_xyz_m',
                                    (num_images,3),
                                    dtype='float32')

    def close(self):
        """Close the Keyence data file for writing.
        """
        if self.verbose:
            log.info("Closing Keyence file...")
        self.h5f.close()

    def position_m(self, position_data, im_indices):
        """Write (x,y,z) CAS stage position data to the Keyence output file.
        """

        if self.verbose:
            log.info("Writing CAS stage position data...")
        num_images = self.num_images
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
        if position_data.dtype != np.float32:
            log.error(f'While writing with function position_m() in file {__file__}'
                      ' - expected position data to be of type float32.')
            sys.exit()

        self.position_xyz_dset[im_indices] = position_data

    def height_image(self, height_data, im_indices):
        """Write height values to the opened Keyence data file.
        Args:
            height_data ():
            im_indices (scalar or list): image indices to be written
        """

        if self.verbose:
            log.info("Writing Keyence height data...")
        num_images = self.num_images
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
            sys.exit()
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
            sys.exit()
        # if height_data.dtype != np.uint16:  # Prevent clipping
        #     log.error(f'While writing with function height_image() in file {__file__}'
        #               ' - expected height data to be of type uint16.')
        #     sys.exit()

        self.height_image_dset[im_indices,:,:,0] = height_data

    def nan_mask(self, nan_mask_data, im_indices):
        """Write NaN mask values to the opened Keyence data file.
        Args:
            nan_mask_data ():
            im_indices (scalar or list): image indices to be written
        """

        if self.verbose:
            log.info("Writing Keyence NaN mask data...")
        num_images = self.num_images
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
            sys.exit()
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
            sys.exit()
        if nan_mask_data.dtype != np.uint8:
            log.error(f'While writing with function nan_mask() in file {__file__}'
                      ' - expected NaN mask data to be of type uint8.')
            sys.exit()

        self.nan_mask_dset[im_indices,:,:,0] = nan_mask_data

    def rgb_image(self, rgb_data, im_indices):
        """Write RGB values to the opened Keyence data file.
        Args:
            rgb_data ():
            im_indices (scalar or list): image indices to be written
        """

        if self.verbose:
            log.info("Writing Keyence RGB data...")
        num_images = self.num_images
        if np.max(im_indices) >= num_images:
            log.error(f'Image index too large in file {__file__}')
            sys.exit()
        if np.min(im_indices) < 0:
            log.error(f'Image index is negative (not allowed) in file {__file__}')
            sys.exit()
        if rgb_data.dtype != np.uint8:
            log.error(f'While writing with function rgb_image() in file {__file__}'
                      ' - expected rgb data to be of type uint8.')
            sys.exit()

        self.rgb_image_dset[im_indices,:,:,:] = rgb_data
