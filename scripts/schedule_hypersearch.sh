#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=synthetic model.DDPM_param.reconstruct=.35,.38,.43 model.DDPM_param.skip_steps=20,25,30,35 model.DDPM_param.condition_weight=1.5,1.75,2.25 model.DDPM_param.plot_ids=[0,1] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun

python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.35,.38,.43 model.DDPM_param.skip_steps=20,25,30,35 model.DDPM_param.condition_weight=1.5,1.75,2.25 model.DDPM_param.plot_ids=[0,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun

python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realAB model.DDPM_param.reconstruct=.35,.38,.43 model.DDPM_param.skip_steps=20,25,30,35 model.DDPM_param.condition_weight=1.5,1.75,2.25 model.DDPM_param.plot_ids=[1,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun
