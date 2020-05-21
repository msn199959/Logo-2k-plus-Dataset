#!/bin/bash
srun  -n 8 --gres=gpu:V100:4i  --mail-type=ALL
source activate wj 
python /drna-master/train.py
