#!/bin/sh
#JSUB -m gpu05
#JSUB -q qgpu
#JSUB -n 1
#JSUB -e /gpfs/home/huowei/STGAN/log_hw/val/error.%J
#JSUB -o /gpfs/home/huowei/STGAN/log_hw/val/output.%J
#JSUB -J last版本-val

your_model=/gpfs/home/huowei/STGAN/codes/run/exp/experiments/STGAN/model_val
dataset=val_1st

/gpfs/home/huowei/anaconda3/envs/STGAN/bin/python val.py \
  --model_path=${your_model} \
  --exp_name=STGAN_val \
  --dataset=${dataset} \
  --save_path=./results
