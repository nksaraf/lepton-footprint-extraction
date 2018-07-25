from __future__ import print_function

import argparse
import os

from tensorflow.python.lib.io import file_io

from model.connections import Wire, get_logger
from model.base import UNetModel, ResnetUNetModel
from model.trainer import GeneratorTrainer
from model.config import create_config
from model.loaders import ImageLoader
from model.utils import CSVLoader

logger = get_logger()

trainers = {
    'unet': UNetModel,
    'unet_resnet': ResnetUNetModel
}


def train_local(config):
    plug = Wire(train_path=os.path.join(config.data_dir, 'train.local.csv'),
                val_path=os.path.join(config.data_dir, 'val.local.csv'),
                train_mode=True)

    xy_train = plug(filename='train_path') | CSVLoader(name='xy_train', **config.xy_splitter)
    xy_valid = plug(filename='val_path') | CSVLoader(name='xy_valid', **config.xy_splitter)

    loader = (plug + xy_train + xy_valid(x_valid='x', y_valid='y')) | ImageLoader('image_resize_loader',
                                                                                  **config.loader)

    unet = trainers[config.model_name](**config.model)
    trainer_model = GeneratorTrainer(config.model.name, unet)
    if len(config.model_path) > 0:
        trainer_model.setup(model_path=config.model_path)
    else:
        trainer_model.setup()

    return loader, trainer_model
    # trained_model = loader | trainers[config.base.py.name](**config.base.py).setup()


    # trained_model.base.py.save(config.job_dir, 'trained_model.h5')
    # logger.info("Training pipeline finished")
    # return loader, trained_model
    # unet = UNetTrainer('unet_resnet101', **config.unet)
    # # if config:
    # #     unet.setup(model_path=config.model_path)
    # # else:
    # unet.setup()
    # # unet.save(config.model_path)
    #
    # trained_model = unet.transform(datagen=loader.datagen,
    #                                validation_datagen=loader.validation_datagen)
    #
    # trained_model.base.py.save(config.job_dir, 'trained_model.h5')
    # logger.info("Training pipeline finished")


def train(config):
    file_io.recursive_create_dir(config.job_dir)
    os.mkdir('checkpoints')

    plug = Wire(train_path=os.path.join(config.data_dir, 'train.csv'),
                val_path=os.path.join(config.data_dir, 'val.csv'),
                train_mode=True)

    xy_train = plug(filename='train_path') | CSVLoader(name='xy_train', **config.xy_splitter)
    xy_valid = plug(filename='val_path') | CSVLoader(name='xy_valid', **config.xy_splitter)

    loader = (plug + xy_train + xy_valid(x_valid='x', y_valid='y')) | ImageLoader('image_resize_loader',
                                                                                  **config.loader)

    # unet = UNetTrainer('unet_resnet101', **config.unet).setup()
    # if config:
    #     unet.setup(model_path=config.model_path)
    # else:

    # unet.save(config.model_path)

    trained_model = loader | trainers[config.model.name](**config.model).setup()

    trained_model.model.save(config.job_dir, 'trained_model.h5')
    logger.info("Training pipeline finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, type=str, help='Working folder')
    parser.add_argument('--data-dir', required=True, type=str, help='Folder to get data files csv from')
    parser.add_argument('--batch-size-train', default=32, type=int, help='Batch size for training')
    parser.add_argument('--batch-size-val', default=32, type=int, help='Batch size for validation')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs for training')
    parser.add_argument('--gpus', default=1, type=int, help='Number of gpus for training')
    parser.add_argument('--base.py', default='unet', type=str, help='Model to train (unet/unet_resnet)')
    parser.add_argument('--base.py-path', default='', type=str, help="Model path")
    parser.add_argument('-d', '--dev-mode', action='store_true', help='Flag to run in dev mode')
    parser.add_argument('--seed', '-s', default=6581, type=int, help='Seed')
    parser.add_argument('-l', '--local', action='store_true')
    parse_args, unknown = parser.parse_known_args()
    if parse_args.local:
        process = train_local
    else:
        process = train
    del parse_args.local
    config = create_config(**parse_args.__dict__)
    result = process(config)
