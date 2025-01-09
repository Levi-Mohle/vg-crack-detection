import sys
from pathlib import Path

def impasto_dataset_variant(variant):
    if variant == "32x32":
        IMPASTO_train_dir   = "Normal32x32_train.h5"
        IMPASTO_val_dir     = "Normal32x32_val.h5"
        IMPASTO_test_dir    = "Crack32x32_test.h5"
    elif variant == "AE512x512":
        IMPASTO_train_dir   = "AE512x512_train.h5"
        IMPASTO_val_dir     = "AE512x512_val.h5"
        IMPASTO_test_dir    = "AE512x512_test.h5"
    return IMPASTO_train_dir, IMPASTO_val_dir, IMPASTO_test_dir