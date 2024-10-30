#!/bin/sh
source activate /env/dvd/
cd /code/dvd/
export PYTHONPATH=./

train_config=configs/train.yaml
dataset_config=configs/dataset.yaml
fusion_config=configs/fusion.yaml
now=$(date +"%Y-%m-%d-%H-%M-%S")

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch \
  --master_port=29501 \
  --nproc_per_node=4 \
  train.py \
  --config=${train_config} \
  --dataset_config=${dataset_config} \
  --fusion_config=${fusion_config} \
  --datetime=${now}