sourcedomain="D2"
targetdomain="D3"
rgbdatapath="/workspace/MM-SADA-code-master/EPICS/rgb"
flowdatapath="/workspace/MM-SADA-code-master/EPICS/flow"
rgbpretrained="/workspace/pytorch-i3d/models/rgb_imagenet.pt"
flowpretrained="/workspace/pytorch-i3d/models/flow_imagenet.pt"


                           
                           
for modelnum in {4000..5600..1000}; do
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /workspace/pytorch-i3d/train_i3d.py  --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain"  \
                        --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
                        --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
                        --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain" \
                        --lr=0.0001 --num_gpus=2 --batch_size=16 --max_steps=13714  --modality="joint" --num_labels=8  \
                        --steps_before_update 1 --restore_mode="pretrain" --domain_mode="PretrainM" --pred_synch=True --trained_path="/workspace/pytorch-i3d/EPIC_D2_D3_mmsada_pretrain6/saved_model_D2_D3_0.01_0.9/00"$modelnum".pt" 
done
#--pred_synch=True
# modelnum=12000
# CUDA_VISIBLE_DEVICES=3 python /workspace/pytorch-i3d/train_i3d.py  --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain"  \
#                            --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
#                            --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
#                            --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain"\
#                            --lr=0.00005 --num_gpus=1 --batch_size=64 --max_steps=13714  --modality="joint" --num_labels=8  \
#                            --steps_before_update 1 --restore_mode="pretrain" --domain_mode="PretrainM" --step=$modelnum\
#                            --trained_path="/workspace/pytorch-i3d/trained_model/0"$modelnum".pt" 

# modelnum=1200
# CUDA_VISIBLE_DEVICES=0 python /workspace/pytorch-i3d/train_i3d.py  --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain"  \
#                            --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
#                            --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
#                            --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain"\
#                            --lr=0.00005 --num_gpus=2 --batch_size=64 --max_steps=13714  --modality="joint" --num_labels=8  \
#                            --steps_before_update 1 --restore_mode="pretrain" --domain_mode="DANN" --step=$modelnum --trained_path="/workspace/pytorch-i3d/EPIC_D2_D3_mmsada_pretrain2/saved_model_D2_D3_1e-05_0.9/00"$modelnum".pt" &
# modelnum=1400
# CUDA_VISIBLE_DEVICES=1 python /workspace/pytorch-i3d/train_i3d.py  --results_path="EPIC_"$sourcedomain"_"$targetdomain"_mmsada_pretrain"  \
#                            --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
#                            --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
#                            --dataset="/workspace/MM-SADA-code-master/Annotations/$sourcedomain" --unseen_dataset="/workspace/MM-SADA-code-master/Annotations/$targetdomain"\
#                            --lr=0.00005 --num_gpus=2 --batch_size=64 --max_steps=13714  --modality="joint" --num_labels=8  \
#                            --steps_before_update 1 --restore_mode="pretrain" --domain_mode="DANN" --step=$modelnum --trained_path="/workspace/pytorch-i3d/EPIC_D2_D3_mmsada_pretrain2/saved_model_D2_D3_1e-05_0.9/00"$modelnum".pt" &
