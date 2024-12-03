#!/bin/sh
#JSUB -m gpu05
#JSUB -q qgpu
#JSUB -n 1
#JSUB -e /gpfs/home/huowei/STGAN/log_hw/error.%J
#JSUB -o /gpfs/home/huowei/STGAN/log_hw/output.%J
#JSUB -J last版本


/gpfs/home/huowei/anaconda3/envs/STGAN/bin/python train.py -opt yamls/STGAN.yml

