#! /bin/bash

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video_v2v.py --base configs/cogvideox_5b_v2v.yaml configs/sft_v2v.yaml --seed 3407"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"