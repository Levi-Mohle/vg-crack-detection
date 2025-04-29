"""
Script for combining split HDF5 files

    Source Name : report_plotting.py
    Contents    : Function for combining HDF5 files
    Date        : 2025

 """

from utils.dataset import combine_h5_files

input_folder    = r"/data/storage_rtx2080/repos/lightning-hydra-template/data/impasto/512x512/splits"
output_file     = r"/data/storage_rtx2080/repos/lightning-hydra-template/data/impasto/2024-11-26_512x512_train.h5"

# Combine h5 files
combine_h5_files(input_folder, output_file)