#!/usr/bin/env bash

#Config file to run: Fully convolutional siamese networks for change detection" (Concatenation Version)
#Credit:
# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

#GPUs
gpus=2

#Set paths
checkpoint_root=/home/brendan/LASTFormer/ChangeFormer-maincheckpoints
vis_root=/home/brendan/LASTFormer/ChangeFormer-main/vis
data_name=change256 #LEVIR, DSIFNB

img_size=256                #Choices=128, 256, 512
batch_size=8               #Choices=8, 16, 32, 64
lr=0.0001         
max_epochs=200

net_G=SiamUnet_conc

lr_policy=linear
optimizer=adamw               #Choices: sgd, adam, adamw
loss=BK_Dice_CE_Loss                     #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

pretrain=/home/brendan/LASTFormer/ChangeFormer-main/SConCUNet_best_ckpt.pt


#Train and Validation splits
split=train         #trainval
split_val=valid      #test
project_name=2aCD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/brendan/LASTFormer/ChangeFormer-main/main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} 