#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.skip_steps=10,25,50,100 model.DDPM_param.condition_weight=0,1,2,4 model.DDPM_param.plot_ids=[2,5,10,12,13,15] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-03-12_08-43-01_DDPM_aug_new/checkpoints/epoch_072.ckpt --multirun