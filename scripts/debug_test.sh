#!/bin/sh
source activate /env/dvd/
cd /code/dvd/
export PYTHONPATH=./

test_config=configs/test.yaml
dataset_config=configs/dataset.yaml
fusion_config=configs/fusion.yaml

file_name=test.py

python ${file_name} \
--config=${test_config} \
--dataset_config=${dataset_config} \
--fusion_config=${fusion_config} \
--debug=True

