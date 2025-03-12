#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Runs on models trained without augmented data
python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.FM_param.reconstruct=.4 model.FM_param.plot_ids=[0,9] ckpt_path= 

##############
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[0,9] ckpt_path= 

################
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[0,9] ckpt_path= 