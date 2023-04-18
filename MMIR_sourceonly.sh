#!/bin/bash

sourcedomain="D2"
targetdomain="D3"
rgbdatapath="/workspace/MM-SADA-code-master/EPICS/rgb"
flowdatapath="/workspace/MM-SADA-code-master/EPICS/flow"
rgbpretrained="/workspace/pytorch-i3d/model/model_rgb.pth"
flowpretrained="/workspace/pytorch-i3d/modes/model_flow.pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /workspace/pytorch-i3d/train_i3d.py  --train=True --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain_source"  \
                           --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
                           --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
                           --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain"\
                           --lr=0.01 --num_gpus=4 --batch_size=48 --max_step=4001  --modality="joint" --num_labels=8  \
                           --steps_before_update 1 --restore_mode="pretrain" --domain_mode="PretrainM"   \
                           --epoch=100 #--pred_synch=True    
                           
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /workspace/pytorch-i3d/train_i3d.py  --train=True --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_source"   \
                           --trained_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain_source/saved_model_"$sourcedomain"_"$targetdomain"_0.01_0.9/004000.pt" \
                           --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" --restoring=True \
                           --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain" \
                           --lr=0.001 --num_gpus=4 --batch_size=48 --max_step=8001 --modality="joint" --num_labels=8 \
                           --steps_before_update 1 --restore_mode="model" --domain_mode="PretrainM" --epoch=200 #--pred_synch=True
