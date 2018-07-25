#!/usr/bin/env bash

TRAINER_PACKAGE_PATH="./work"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="lepton_$now"
MAIN_TRAINER_MODULE="work.train"
JOB_DIR="gs://lepton/jobs/$JOB_NAME"
PACKAGE_STAGING_PATH="gs://lepton"
DATA_DIR="gs://lepton/data/csv"
BASE_BATCH_SIZE=32
GPUS=1
((BATCH_SIZE=$BASE_BATCH_SIZE*$GPUS))

#gcloud ml-engine jobs submit training $JOB_NAME  \
#    --package-path $TRAINER_PACKAGE_PATH \
#    --module-name $MAIN_TRAINER_MODULE \
#    --staging-bucket $PACKAGE_STAGING_PATH \
#    --job-dir $JOB_DIR \
#    --config config.yaml \
#    --stream-logs --\
#    --data-dir  $DATA_DIR \
#    --batch-size-train $BATCH_SIZE \
#    --batch-size-val $BATCH_SIZE \
#    --epochs 100 \
#    --gpus $GPUS \
#    --base.py unet


gcloud ml-engine jobs submit training $JOB_NAME  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR \
    --config config.yaml \
    --stream-logs --\
    --data-dir  $DATA_DIR \
    --batch-size-train $BATCH_SIZE \
    --batch-size-val $BATCH_SIZE \
    --epochs 10 \
    --gpus $GPUS \
    --model unet


# gcloud ml-engine local train --package-path work \
# 	--module-name work.train --\
# 	--train-image-dir $TRAIN_IMAGE_DIR \
# 	--train-mask-dir $TRAIN_IMAGE_DIR \
# 	--val-image-dir $VAL_IMAGE_DIR \
# 	--val-mask-dir $VAL_MASK_DIR \
# 	--job-dir $JOB_DIR \
#     --batch-size 10 \
#     --first_layer_channels 64 \
#     --depth 3 \
#     --dropout 0.2 \
#     -lr 1e-4 \
#     --epochs 10\
#     --train-steps 230 \
#     --val-steps 80 \
#     --checkpoint_epochs 10