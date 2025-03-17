#! /bin/bash

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# run_cmd="$environs python train_video.py --base configs/cogvideox_5b_i2v_lora.yaml configs/sft.yaml --seed $RANDOM"

run_cmd="$environs python train_video_v2v.py --base configs/cogvideox_5b_i2v_lora.yaml configs/sft.yaml --seed 3407"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"