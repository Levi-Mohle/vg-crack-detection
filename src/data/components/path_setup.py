def impasto_dataset_variant(variant, crack):
    # For 32x32 dataset
    if variant == "32x32":
        IMPASTO_train_dir   = "Normal32x32_train.h5"
        IMPASTO_val_dir     = "Normal32x32_val.h5"
        IMPASTO_test_dir    = "Crack32x32_test.h5"
    # For running 512x512 dataset on local machine 
    elif variant == "512x512_local":
        IMPASTO_train_dir   = "2024-11-26_Enc_synthetic_mix_512x512_train.h5"
        IMPASTO_val_dir     = "2024-11-26_Enc_synthetic_mix_512x512_train.h5"
        IMPASTO_test_dir    = "AE512x512_test.h5"
    # For 512x512 dataset
    elif variant == "512x512":
        IMPASTO_train_dir   = "2024-11-26_512x512_train.h5"
        IMPASTO_val_dir     = "2024-11-26_512x512_val.h5"
        if crack == "realAB":
            IMPASTO_test_dir    = "2025-01-07_Real_Cracks_512x512_test.h5"
        elif crack == "realBI":
            IMPASTO_test_dir    = "2025-02-18_Real_Cracks_512x512_test.h5"
        else:
            IMPASTO_test_dir    = "2024-11-26_Cracks_512x512_test.h5"
    # For 512x512 dataset with synthetic cracks
    elif variant == "mix_512x512":
        IMPASTO_train_dir   = "2024-11-26_mix_512x512_train.h5"
        IMPASTO_val_dir     = "2024-11-26_mix_512x512_val.h5"
        if crack == "realAB":
            IMPASTO_test_dir    = "2025-01-07_Real_Cracks_512x512_test.h5"
        elif crack == "realBI":
            IMPASTO_test_dir    = "2025-02-18_Real_Cracks_512x512_test.h5"
        else:
            IMPASTO_test_dir    = "2024-11-26_Cracks_512x512_test.h5"
    # For Encoded 512x512 dataset (effective size after encoding 8x64x64)
    elif variant == "Enc_512x512":
        IMPASTO_train_dir   = "2024-11-26_Enc_aug_512x512_train.h5"
        IMPASTO_val_dir     = "2024-11-26_Enc_512x512_val.h5"
        if crack == "realAB":
            IMPASTO_test_dir    = "2025-01-07_Enc_Real_Cracks_512x512_test.h5"
        elif crack == "realBI":
            IMPASTO_test_dir    = "2025-02-18_Enc_Real_Cracks_512x512_test.h5"
        else:
            IMPASTO_test_dir    = "2024-11-26_Enc_Cracks_512x512_test.h5"
    # For Encoded 512x512 dataset with synthetic cracks (effective size after encoding 8x64x64)
    elif variant == "Enc_mix_512x512":
        IMPASTO_train_dir   = "2024-11-26_Enc_synthetic_mix_512x512_train.h5"
        IMPASTO_val_dir     = "2024-11-26_Enc_synthetic_mix_512x512_val.h5"
        if crack == "realAB":
            IMPASTO_test_dir    = "2025-01-07_Enc_Real_Cracks_512x512_test.h5"
        elif crack == "realBI":
            IMPASTO_test_dir    = "2025-02-18_Enc_Real_Cracks_512x512_test.h5"
        else:
            IMPASTO_test_dir    = "2024-11-26_Enc_Cracks_512x512_test.h5"
    # For Encoded 512x512 dataset with synthetic cracks and segmentation masks (effective size after encoding 8x64x64)
    elif variant == "Enc_mix_seg_512x512":
        IMPASTO_train_dir   = "2024-11-26_Enc_synthetic_mix_seg_512x512_train.h5"
        IMPASTO_val_dir     = "2024-11-26_Enc_synthetic_mix_seg_512x512_val.h5"
        IMPASTO_test_dir    = "2024-11-26_Enc_synthetic_mix_seg_512x512_val.h5"
    
    return IMPASTO_train_dir, IMPASTO_val_dir, IMPASTO_test_dir