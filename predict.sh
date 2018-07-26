#!/usr/bin/env bash

python2 -i -m work.predict \
    --data-dir data/dev \
    --job-dir working \
    --model unet \
    --model-path data/model.h5 -l -d --epochs 1 \
    --batch-size-train 3 \
    --seed 1001