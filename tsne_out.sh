sourcedomain="D2"
targetdomain="D3"
rgbdatapath="/workspace/MM-SADA-code-master/EPICS/rgb"
flowdatapath="/workspace/MM-SADA-code-master/EPICS/flow"
rgbpretrained="/workspace/pytorch-i3d/models/rgb_imagenet.pt"
flowpretrained="/workspace/pytorch-i3d/models/flow_imagenet.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspace/pytorch-i3d/TSNE.py  --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain"  \
                        --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
                        --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
                        --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain" \
                        --lr=0.0001 --num_gpus=2 --batch_size=16 --max_steps=13714  --modality="joint" --num_labels=8  \
                        --steps_before_update 1 --restore_mode="pretrain" --domain_mode="PretrainM" --pred_synch=True --trained_path="/workspace/pytorch-i3d/EPIC_D2_D3_mmsada_self_nobn/saved_model_D2_D3_0.0001_0.9/008001.pt" 