#!/usr/bin/env bash

JOB_NAME=jobs/lepton_$1
BASE_BATCH_SIZE=32

echo $JOB_NAME

python -m work.train \
    --data-dir data/index/mix \
    --job-dir $JOB_NAME \
    --batch-size $BASE_BATCH_SIZE \
    --model unet \
    --loader basic \
    --model-path jobs/lepton_mix_v2/checkpoints/checkpoint.17-1.62.h5 \
    --epochs 50 -l
