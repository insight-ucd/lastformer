Forked from: https://github.com/wgcban/ChangeFormer

### Training Models
For training models you can explore the scripts folder for pre-made scripts. 
```
gpus=1
checkpoint_root=/path/to/checkpoint/
vis_root=/path/to/vis/dir
data_name=change256 #LEVIR, DSIFN

img_size=256 # size of the input image e.g. 256x256
batch_size=8
lr=0.01
max_epochs=200

net_G=base_transformer_pos_s4_dd8 # model  to train
embed_dim=256

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

pretrain=/home/brendan/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth

#Train and Validation splits
split=train         #trainval
split_val=valid      #test
project_name=experiment_name

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./ChangeFormer-main/main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
```

