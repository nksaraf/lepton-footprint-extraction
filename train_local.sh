#!/usr/bin/env bash

((LEPTON_MODEL=$LEPTON_MODEL+1))
JOB_NAME=jobs/lepton_$LEPTON_MODEL
MAIN_TRAINER_MODULE="work.train"
BASE_BATCH_SIZE=32
GPUS=1
((BATCH_SIZE=$BASE_BATCH_SIZE*$GPUS))

#TRAINER_PACKAGE_PATH="./work"
#now=$(date +"%Y%m%d_%H%M%S")
#JOB_NAME="lepton_$now"
#MAIN_TRAINER_MODULE="work.train"
#JOB_DIR="gs://lepton/jobs/$JOB_NAME"
#PACKAGE_STAGING_PATH="gs://lepton"
#DATA_DIR="gs://lepton/data"

python2 -i -m work.train \
    --data-dir data/csv \
    --job-dir $JOB_NAME \
    --batch-size-train $BASE_BATCH_SIZE \
    --batch-size-val $BASE_BATCH_SIZE \
    --model unet \
    --epochs 5 -l
