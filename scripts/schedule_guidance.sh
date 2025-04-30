#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/eval.py +experiment=impasto_FM trainer=gpu data.variant=Enc_mix_512x512 data.crack=synthetic model.FM_param.reconstruct=.2,.5 model.FM_param.solver=euler,midpoint model.FM_param.plot_ids=[0,1] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-20_00-12-59_FM_OT_aug_long/checkpoints/epoch_082.ckpt --multirun

# python src/eval.py +experiment=impasto_FM trainer=gpu data.variant=Enc_mix_512x512 data.crack=realBI model.FM_param.reconstruct=.2,.5 model.FM_param.solver=euler,midpoint model.FM_param.plot_ids=[0,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-20_00-12-59_FM_OT_aug_long/checkpoints/epoch_082.ckpt --multirun

# python src/eval.py +experiment=impasto_gc_FM trainer=gpu data.variant=Enc_mix_512x512 data.crack=realBI model.unet.n_classes=3 model.FM_param.reconstruct=.2,.5 model.FM_param.solver=euler,midpoint model.FM_param.plot_ids=[0,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-17_13-03-35_cc_FM_OT_CFG_Full_dataset/checkpoints/epoch_050.ckpt --multirun



# python src/eval.py +experiment=impasto_gc_FM trainer=gpu data.variant=Enc_mix_512x512 data.crack=synthetic model.unet.n_classes=3 model.FM_param.reconstruct=.2,.5 model.FM_param.guidance_strength=.3,.5,.8 model.FM_param.plot_ids=[0,1] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-17_13-03-35_cc_FM_OT_CFG_Full_dataset/checkpoints/epoch_050.ckpt --multirun

# python src/eval.py +experiment=impasto_gc_FM trainer=gpu data.variant=Enc_mix_512x512 data.crack=realBI model.unet.n_classes=3 model.FM_param.reconstruct=.2,.5 model.FM_param.guidance_strength=.3,.5,.8 model.FM_param.plot_ids=[0,1] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-17_13-03-35_cc_FM_OT_CFG_Full_dataset/checkpoints/epoch_050.ckpt --multirun



# python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=synthetic model.DDPM_param.reconstruct=.5,.8 model.DDPM_param.condition_weight=1,2,4 model.DDPM_param.plot_ids=[0,1] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun

# python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.5,.8 model.DDPM_param.condition_weight=1,2,4 model.DDPM_param.plot_ids=[0,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun

# python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=synthetic model.DDPM_param.reconstruct=.8 model.DDPM_param.skip_steps=10,25,50,100 model.DDPM_param.plot_ids=[0,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun

python src/eval.py +experiment=impasto_ddpm trainer=gpu data.variant=Enc_512x512 model.DDPM_param.use_cond=true data.crack=realBI model.DDPM_param.reconstruct=.8 model.DDPM_param.skip_steps=10,25,50,100 model.DDPM_param.plot_ids=[0,9] ckpt_path=/data/storage_models_crack_detection/Trained_models/2025-02-26_20-44-52_DDPM_aug/checkpoints/epoch_130.ckpt --multirun
