#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Runs on models trained without augmented data
python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.FM_param.reconstruct=.4 model.FM_param.plot_ids=[2,5,10,12,13,15] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-13_00-23-08_FM_new/checkpoints/epoch_064.ckpt

python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.FM_param.reconstruct=.4 model.FM_param.plot_ids=[2,5,10,12,13,15] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_16-41-10_FM_aug_new/checkpoints/epoch_068.ckpt 
##############
# python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[2,5,10,12,13,15] ckpt_path= 


################
# python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[2,5,10,12,13,15] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_21-12-29_DDPM_new/checkpoints/epoch_023.ckpt 

# python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[2,5,10,12,13,15] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_08-43-01_DDPM_aug_new/checkpoints/epoch_072.ckpt 