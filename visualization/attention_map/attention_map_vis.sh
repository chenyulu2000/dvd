#!/bin/sh
source activate /env/dvd/
cd /code/dvd/
export PYTHONPATH=./

python visualization/attention_map/attention_map_vis.py
