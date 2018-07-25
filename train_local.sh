#!/usr/bin/env bash

#TRAINER_PACKAGE_PATH="./trainer"
#now=$(date +"%Y%m%d_%H%M%S")
#JOB_NAME="lepton_$now"
#MAIN_TRAINER_MODULE="trainer.train"
#JOB_DIR="gs://lepton/jobs/$JOB_NAME"
#PACKAGE_STAGING_PATH="gs://lepton"
#DATA_DIR="gs://lepton/data"

#gcloud ml-engine jobs submit training $JOB_NAME  \
#    --package-path $TRAINER_PACKAGE_PATH \
#    --module-name $MAIN_TRAINER_MODULE \
#    --staging-bucket $PACKAGE_STAGING_PATH \
#    --job-dir $JOB_DIR \
#    --config config.yaml \
#    --stream-logs --\
#    --data-dir $DATA_DIR \
#    --batch-size-train 1 \
#    --batch-size-val 1 \
#    --epochs 1 \
#    -d

python2 -i -m trainer.train \
    --data-dir data/bang \
    --job-dir working \
    --batch-size-train 10 \
    --batch-size-val 10 \
    --model unet \
    --model-path data/model.h5 \
    --epochs 1 --gpus 1 -l -d
