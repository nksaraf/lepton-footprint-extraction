from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import argparse
import os

from tensorflow.python.lib.io import file_io

from model.connections import Wire
from model.base import UNetModel, ResnetUNetModel
from model.trainer import GeneratorTrainer
from model.config import create_config
from model.loaders import ImageLoaderAugmented, ImageLoaderInference, ImageLoaderNormalizedAugmented
from model.utils import CSVLoader

trainers = {
    'unet': UNetModel,
    'unet_resnet': ResnetUNetModel
}

loaders = {
    'basic': ImageLoaderAugmented,
    'normalized': ImageLoaderNormalizedAugmented
}


def train(config, args):
    try:
        os.makedirs(os.path.join(config.job_dir, 'checkpoints'))
    except:
        pass

    plug = Wire(train_path=os.path.join(args.data_dir, 'train.csv'),
                val_path=os.path.join(args.data_dir, 'val.csv'),
                train_mode=True)

    train_loader = (plug(filename='train_path') 
        | CSVLoader(name='xy_train', **config.xy_splitter)
        | ImageLoaderAugmented('train_loader', True, **config.loader))

    val_loader = (plug(filename='val_path') 
        | CSVLoader(name='xy_valid', **config.xy_splitter)
        | ImageLoaderInference('val_loader', True, **config.loader))

    loader = train_loader(datagen='generator') + val_loader(validation_datagen='generator')

    unet = trainers[config.model_name](**config.model)
    trainer_model = GeneratorTrainer(config.model_name, unet)

    # if a model path was given, load the weights from that path before training,
    # allows training on old models
    if len(args.model_path) > 0:
        trainer_model.setup(path=args.model_path, load_weights=True, gpus=args.gpus)
    else:
        trainer_model.setup(gpus=args.gpus)

    trained_model = loader | trainer_model
    trained_model.model.save(config.job_dir, 'trained_model.h5')
    return loader, trained_model


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--job-dir', required=True, type=str, help='Working folder')
    config_parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    config_parser.add_argument('--epochs', required=True, type=int, help='Number of epochs for training')
    config_parser.add_argument('--model', default='unet', type=str, help='Model to train (unet/unet_resnet)')
    config_parser.add_argument('--seed', '-s', default=6581, type=int, help='Seed')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str, help='Folder to get data files csv from')
    parser.add_argument('--gpus', default=1, type=int, help='Number of gpus for training')
    parser.add_argument('--model-path', default='', type=str, help="Model path")
    parser.add_argument('--loader', default='basic', type=str, help="Image data loader to use")
    parser.add_argument('--data-pre', required=False, type=str, default='', help="Data files prefix")
    
    config_args, unknown = config_parser.parse_known_args()
    config = create_config(**config_args.__dict__)

    args , _ = parser.parse_known_args(unknown)
    result = train(config, args)
