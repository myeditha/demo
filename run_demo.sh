#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $1
pip3 install datasets

python demo.py
