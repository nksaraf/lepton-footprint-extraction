#!/usr/bin/env bash

JOB_DIR=jobs/lepton_$1
MODEL_NAME=models/lepton_$1.h5

python -i -m work.predict \
    --data-dir data/index/hyd \
    --job-dir $JOB_DIR \
    --model unet \
    --model-path $MODEL_NAME -l \
    --loader basic \
    --batch-size 3 \
    --seed $2 \
    --data-pre /Users/nikhilsaraf/Documents