#!/bin/bash

echo "MIMIC dataset"
SEED=2028

CUDA_VISIBLE_DEVICES=3 python /opt/workspace/4.MIMIC/OnlyClinicalSeq_diabetes/TotalMain.py \
--cuda_num=3 --seed=$SEED --model_num=0 &
CUDA_VISIBLE_DEVICES=3 python /opt/workspace/4.MIMIC/OnlyClinicalSeq_diabetes/TotalMain.py \
--cuda_num=3 --seed=$SEED --model_num=1 &
CUDA_VISIBLE_DEVICES=3 python /opt/workspace/4.MIMIC/OnlyClinicalSeq_diabetes/TotalMain.py \
--cuda_num=3 --seed=$SEED --model_num=2 &

echo "MIMIC dataset End"