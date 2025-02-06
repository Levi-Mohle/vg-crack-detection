from preprocess_latent_space.dataset import combine_h5_files

input_folder = r"/data/storage_rtx2080/repos/lightning-hydra-template/data/impasto/512x512/splits"
output_file = r"/data/storage_rtx2080/repos/lightning-hydra-template/data/impasto/2024-11-26_512x512_train.h5"

combine_h5_files(input_folder, output_file)