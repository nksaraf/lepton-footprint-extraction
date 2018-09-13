#!/usr/bin/env bash

JOB_DIR=jobs/lepton_$1
MODEL_NAME=models/lepton_$1.h5

python -m work.main \
    --job-dir $JOB_DIR \
    --model-path $MODEL_NAME -l \
    -f /Users/nikhilsaraf/Documents/Nanded.tif