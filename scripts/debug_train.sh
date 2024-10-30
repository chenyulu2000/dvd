#!/bin/sh
source activate /env/dvd/
cd /code/dvd/
export PYTHONPATH=./

train_config=configs/train.yaml
dataset_config=configs/dataset.yaml
fusion_config=configs/fusion.yaml

#file_name=data/readers.py
#file_name=data/dataset.py
#file_name=models/encoders/disentanglement/image_encoder.py
#file_name=models/encoders/disentanglement/question_encoder.py
#file_name=models/encoders/disentanglement/history_encoder.py
#file_name=models/encoders/multimodal_fusion/multimodal_fusion.py
file_name=train.py

CUDA_VISIBLE_DEVICES=2 \
  python -m torch.distributed.launch \
  --master_port=29512 \
  --nproc_per_node=1 \
  ${file_name} \
  --config=${train_config} \
  --dataset_config=${dataset_config} \
  --fusion_config=${fusion_config} \
  --debug=True
