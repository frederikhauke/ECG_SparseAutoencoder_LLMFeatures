#!/bin/bash
# ++++++++++++ IMPORTANT +++++++++++++
# first run 
# salloc --gres=gpu:1 --mem=100G --cpus-per-task=16
# then run 
# srun bash start_xray_train_interactive.bash

cd /home/homesOnMaster/fhauke/ECG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

source /home/homesOnMaster/fhauke/miniconda3/etc/profile.d/conda.sh
conda activate ecg

export PYTHONPATH=/home/homesOnMaster/fhauke/ECG

python interpret_features.py