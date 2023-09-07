#!/usr/bin/env bash

gpus=2
data_name=change256
net_G=ChangeFormerV6 #This is the best version
split=test
vis_root=/home/brendan/LASTFormer/ChangeFormer-main/vis
project_name=2bCD_ChangeFormerV6_change256_b8_lr0.0001_adamw_train_valid_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256

checkpoints_root=/home/brendan/LASTFormer/ChangeFormer-maincheckpoints/2bCD_ChangeFormerV6_change256_b8_lr0.0001_adamw_train_valid_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/

checkpoint_name=best_ckpt.pt

img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/brendan/LASTFormer/ChangeFormer-main/eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}



