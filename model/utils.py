import os
import random
import threading

import cv2
import joblib
import numpy as np
import pandas as pd
from imageio import imread, imwrite
from keras.callbacks import Callback
from keras.utils import Sequence
from tensorflow.python.lib.io import file_io

from model.connections import Transformer

INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'area': cv2.INTER_AREA,
}


class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, model, job_dir, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.job_dir = job_dir
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.model_to_save = model
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    pass
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            save_model_weights(self.model_to_save, self.job_dir, filepath)
                        else:
                            save_model(self.model_to_save, self.job_dir, filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    save_model_weights(self.model_to_save, self.job_dir, filepath)
                else:
                    save_model(self.model_to_save, self.job_dir, filepath)


def save_model_weights(model, job_dir, filepath):
    if job_dir.startswith('gs://'):
        model.save_weights(filepath, overwrite=True)
        copy_file_to_gcs(job_dir, filepath)
    else:
        model.save_weights(os.path.join(job_dir, filepath), overwrite=True)


def save_model(model, job_dir, filepath):
    if job_dir.startswith('gs://'):
        model.save(filepath, overwrite=True)
        copy_file_to_gcs(job_dir, filepath)
    else:
        model.save(os.path.join(job_dir, filepath), overwrite=True)


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batch(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self, *args, **kwargs):
        raise NotImplementedError

    def _get_batch(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class CSVLoader(Transformer):
    __out__ = ('x', 'y')

    def __init__(self, name, x_column, y_column, prefix=''):
        super(CSVLoader, self).__init__(name)
        self.x_column = x_column
        self.y_column = y_column
        self.prefix = prefix

    def __transform__(self, filename, train_mode):
        self.log("Loading data from {filename}".format(filename=filename))
        with open(filename, 'r') as fp:
            data = pd.read_csv(fp)
        x = data[self.x_column].map(lambda a: os.path.join(self.prefix, a))
        if train_mode:
            y = data[self.y_column].map(lambda a: os.path.join(self.prefix, a))
        else:
            y = None
        return {'x': x,
                'y': y}

class CSVLoaderXYZ(Transformer):
    __out__ = ('x', 'y', 'z')

    def __init__(self, name, x_column, y_column, prefix=''):
        super(CSVLoaderXYZ, self).__init__(name)
        self.x_column = x_column
        self.y_column = y_column
        self.prefix = prefix

    def __transform__(self, filename, train_mode):
        self.log("Loading data from {filename}".format(filename=filename))
        with open(filename, 'r') as fp:
            data = pd.read_csv(fp)
        x = data[self.x_column].map(lambda a: os.path.join(self.prefix, a))
        if train_mode:
            y = data[self.y_column].map(lambda a: os.path.join(self.prefix, a))
        else:
            y = None

        z = data[self.x_column]
        return {'x': x,
                'y': y,
                'z': z }


def save_image(path, x, scale=True):
    """Saves an image stored as a Numpy array to a path or file object.

    # Arguments
        path: Path or file object.
        image: Numpy array.
        scale: Whether to rescale image values to be within `[0, 255]`.
    """
    if scale:
        # x = x + max(-np.min(x), 0)
        # x_max = np.max(x)
        # if x_max != 0:
        #     x /= x_max
        x *= 255
    x = x.astype(np.uint8)

    if len(x.shape) > 2:
        if x.shape[2] == 1:
            x = x.reshape((x.shape[0], x.shape[1]))

    with open(path, mode='wb') as fp:
        imwrite(fp, x)


def load_image(path, color_mode='rgb'):
    """Loads an image into numpy array.

    # Arguments
        path: Path to image file.
        color_mode: Boolean, whether to load the image as gray or rgb.
            
    # Returns
        A numpy array representing image.

    # Raises
        ValueError: if image or color_mode is not supported
    """
    assert color_mode in ['rgb', 'gray'], 'Invalid color_mode (should be either rgb or gray), got {}'.format(color_mode)
    with open(path, mode='rb') as fp:
        image = imread(fp)

    if image is None:
        raise ValueError('Unable to load {}'.format(path))

    return image


def load_joblib(filepath):
    with open(filepath, 'r') as fp:
        return joblib.load(fp)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_to_csv(file_path, data, columns, **kwargs):
    df = pd.DataFrame(data=data, columns=columns)
    df = df.dropna()
    with open(file_path, 'w') as fp:
        df.to_csv(fp, **kwargs)


def copy_file_to_gcs(job_dir, file_path):
    if job_dir.startswith('gs://'):
        with file_io.FileIO(file_path, mode='rb') as input_f:
            with file_io.FileIO(os.path.join(job_dir, file_path), mode='wb') as output_f:
                output_f.write(input_f.read())
