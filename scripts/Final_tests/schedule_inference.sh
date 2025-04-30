#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.FM_param.reconstruct=.4 model.FM_param.plot=false model.FM_param.ood=true data.batch_size=16 ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_16-41-10_FM_aug_new/checkpoints/epoch_068.ckpt 

##############
python src/eval.py +experiment=impasto_gc_FM2 trainer=gpu data.variant=Enc_mix_512x512 data.crack=realBI model.unet.n_classes=3 model.FM_param.reconstruct=.4  model.FM_param.plot=false data.batch_size=16 model.FM_param.ood=true ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_19-21-45_gc_FM_new/checkpoints/epoch_034.ckpt 

##############
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=false data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot=false data.batch_size=16 model.DDPM_param.ood=true ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_08-43-01_DDPM_aug_new/checkpoints/epoch_072.ckpt 

###############
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot=false data.batch_size=16 model.DDPM_param.ood=true ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_08-43-01_DDPM_aug_new/checkpoints/epoch_072.ckpt 

