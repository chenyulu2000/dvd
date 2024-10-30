#!/bin/sh
source activate /env/dvd/
cd /code/dvd/
export PYTHONPATH=./

python visualization/case_analysis/case_vis.py
