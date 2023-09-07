#!/usr/bin/env bash

gpus=2
data_name=change256
net_G=ChangeFormerV6 #This is the best version
split=test

img_size=256  
#batch_size=16
batch_size=8
lr=0.0001         
max_epochs=200
embed_dim=256

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False


vis_root=/home/brendan/LASTFormer/ChangeFormer-main/vis
checkpoint_root=ED_CD_ChangeFormerV6_change256_b8_lr0.0001_adamw_train_valid_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/
checkpoint_name=best_ckpt.pt

project_name=1EVAL_CD_ChangeFormerV6${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/brendan/LASTFormer/ChangeFormer-main/main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}


