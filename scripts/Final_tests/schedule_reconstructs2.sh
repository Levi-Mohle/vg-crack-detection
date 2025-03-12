#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Multi-runs for reconstruction levels
python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=synthetic model.FM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.FM_param.plot_ids=[0,1] ckpt_path= --multirun

python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.FM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.FM_param.plot_ids=[0,9] ckpt_path= --multirun

##############

python src/eval.py +experiment=impasto_gc_FM2 trainer=gpu data.variant=Enc_mix_512x512 data.crack=synthetic model.unet.n_classes=3 model.FM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.FM_param.plot_ids=[0,1] ckpt_path= --multirun

python src/eval.py +experiment=impasto_gc_FM2 trainer=gpu data.variant=Enc_mix_512x512 data.crack=realBI model.unet.n_classes=3 model.FM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.FM_param.plot_ids=[0,1] ckpt_path= --multirun

##############

python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 data.crack=synthetic model.DDPM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.DDPM_param.plot_ids=[0,1] ckpt_path= --multirun

python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 data.crack=realBI model.DDPM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.DDPM_param.plot_ids=[0,9] ckpt_path= --multirun

################

python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=synthetic model.DDPM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.DDPM_param.plot_ids=[0,1] ckpt_path= --multirun

python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.05,.1,.2,.3,.4,.5,.8 model.DDPM_param.plot_ids=[0,9] ckpt_path= --multirun


#################
#################
# Test for AB on 1 reconstruction level
python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realAB model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[1,9] ckpt_path= 

python src/eval.py +experiment=impasto_cddpm2 trainer=gpu data.variant=Enc_512x512 data.crack=realAB model.DDPM_param.reconstruct=.4 model.DDPM_param.plot_ids=[1,9] ckpt_path=

python src/eval.py +experiment=impasto_FM2 trainer=gpu data.variant=Enc_mix_512x512 data.crack=realAB model.FM_param.reconstruct=.4 model.FM_param.plot_ids=[1,9] ckpt_path=

python src/eval.py +experiment=impasto_gc_FM2 trainer=gpu data.variant=Enc_mix_512x512 data.crack=realAB model.unet.n_classes=3 model.FM_param.reconstruct=.4 model.FM_param.plot_ids=[1,9] ckpt_path=
