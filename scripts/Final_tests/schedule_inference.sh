#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.FM_param.reconstruct=.4 model.FM_param.plot=false model.FM_param.ood=true ckpt_path= 

##############
python src/eval.py +experiment=impasto_gc_FM2 trainer=gpu data.variant=Enc_mix_512x512 data.crack=realBI model.unet.n_classes=3 model.FM_param.reconstruct=.4  model.FM_param.plot=false model.FM_param.ood=true ckpt_path= 

##############
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[0,9] model.DDPM_param.ood=true ckpt_path= 

###############
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.4 model.DDPM_param.plot=false model.DDPM_param.ood=true ckpt_path= 

