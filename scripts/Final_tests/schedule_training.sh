#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train.py experiment=impasto_FM2 trainer=ddp trainer.devices=2 data.variant=Enc_aug_512x512 model.FM_param.plot=false model.FM_param.ood=false model.FM_param.plot_n_epoch=20 

# python src/train.py experiment=impasto_gc_FM2 trainer=ddp trainer.devices=2 data.variant=Enc_mix_512x512 model.FM_param.plot=false model.FM_param.ood=false model.unet.n_classes=3 model.FM_param.plot_n_epoch=20

# python src/train.py experiment=impasto_cddpm2 trainer=ddp trainer.devices=2 data.variant=Enc_512x512 model.DDPM_param.plot=false model.DDPM_param.ood=false model.DDPM_param.plot_n_epoch=20

python src/train.py experiment=impasto_FM2 trainer=ddp trainer.devices=2 data.variant=Enc_512x512 model.FM_param.plot=false model.FM_param.ood=false model.FM_param.plot_n_epoch=20