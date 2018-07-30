from __future__ import print_function

import argparse
import os
import logging

import trainer.model as model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ProgbarLogger
from tensorflow.python.lib.io import file_io

logging.basicConfig(format='%(levelname)s:  %(message)s', level=logging.DEBUG)

SEED = 100
MODEL_PATH = 'lepton.h5'

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())

def train(
        job_dir,
        train_image_dir, 
        train_mask_dir, 
        val_image_dir, 
        val_mask_dir, 
        batch_size,
        first_layer_channels,
        depth,
        dropout,
        batch_norm,
        learning_rate,
        epochs,
        train_steps,
        val_steps,
        checkpoint_epochs
        ):

    train_generator = model.generator(train_image_dir, train_mask_dir, target_size=(256, 256), seed=SEED, batch_size=batch_size)
    val_generator = model.generator(val_image_dir, val_mask_dir, target_size=(256, 256), seed=SEED, batch_size=batch_size)

    os.makedirs(job_dir)

    lepton = model.build_model(start_ch=first_layer_channels, depth=depth, dropout=dropout, batchnorm=batch_norm)
    model.compile_model(lepton, 
        optimizer=Adam, 
        learning_rate=learning_rate, 
        loss='binary_crossentropy', 
        metrics=[model.jaccard_coef]
    )

    checkpoint_path = 'checkpoint.{epoch:02d}.h5'
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, period=checkpoint_epochs)
    tensor_board = TensorBoard(os.path.join(job_dir, 'logs'), write_graph=True)
    prog_bar = ProgbarLogger(count_mode='steps')

    callbacks = [model_checkpoint, tensor_board, prog_bar]

    history = lepton.fit_generator(
        train_generator, 
        steps_per_epoch=train_steps, 
        epochs=epochs, 
        verbose=1,
        callbacks=callbacks, 
        validation_data=val_generator,
        validation_steps=val_steps
    )

    if job_dir.startswith("gs://"):
        lepton.save(MODEL_PATH)
        copy_file_to_gcs(job_dir, MODEL_PATH)
    else:
        lepton.save(os.path.join(job_dir, MODEL_PATH))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(lepton, os.path.join(job_dir, 'export'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, type=str, help='Folder to save model in')
    parser.add_argument('--train-image-dir', required=True, type=str, help='Folder containing images folder for training')
    parser.add_argument('--train-mask-dir', required=True, type=str, help='Folder containing masks folder for training')
    parser.add_argument('--val-image-dir', required=True, type=str, help='Folder containing images folder for validation')
    parser.add_argument('--val-mask-dir', required=True, type=str, help='Folder containing masks folder for validation')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training and validation')
    parser.add_argument('--first_layer_channels', default=32, type=int, help='Batch size for training and validation')
    parser.add_argument('--depth', default=32, type=int, help='Batch size for training and validation')
    parser.add_argument('--dropout', default=0.2, type=float, help='Batch size for training and validation')
    parser.add_argument('-bnorm', '--batch-norm', default=False, action='store_true', help='Set flag to include batch normalization')
    parser.add_argument('-lr', '--learning-rate', default=1e-5, type=float, help='Learning rate of optimizer')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs for training')
    parser.add_argument('--train-steps', required=True, type=int, help='Times a batch sampled from the training data per epoch')
    parser.add_argument('--val-steps', required=True, type=int, help='Times a batch sampled from the validation data per epoch')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Interval of epochs for checkpoints')

    parse_args, unknown = parser.parse_known_args()
    train(**parse_args.__dict__)


