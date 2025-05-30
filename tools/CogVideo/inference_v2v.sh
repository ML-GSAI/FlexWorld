#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video_v2v.py --base configs/cogvideox_5b_v2v.yaml configs/inference_v2v.yaml --seed $RANDOM --image2video"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"