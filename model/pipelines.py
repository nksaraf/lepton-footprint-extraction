from model.loaders import ImageDataLoaderResize
from model.utils import CSVLoader
from model.base import UNetTrainer
import os


def train_pipeline(plug, config, logger):
    logger.info("Training pipeline...")
    xy_train = CSVLoader(name='xy_train', **config.xy_splitter).transform(filename=plug.train_filepath,
                                                                          train_mode=plug.train_mode)
    xy_valid = CSVLoader(name='xy_valid', **config.xy_splitter).transform(filename=plug.val_filepath,
                                                                          train_mode=plug.train_mode)

    loader = ImageDataLoaderResize('image_resize_loader', **config.loader).transform(x=xy_train.x,
                                                                                     y=xy_train.y,
                                                                                     train_mode=plug.train_mode,
                                                                                     x_valid=xy_valid.x,
                                                                                     y_valid=xy_valid.y)

    unet = UNetTrainer('unet_resnet101', **config.unet)
    if config.env.load_model:
        unet.setup(model_path=config.env.model_path)
    else:
        unet.setup()
        unet.save(config.env.model_path)

    trained_model = unet.transform(datagen=loader.datagen,
                                   validation_datagen=loader.validation_datagen)

    trained_model.model.save(os.path.join(config.working_dir, 'trained_model.h5'))
    logger.info("Training pipeline finished")

    return loader, trained_model


def predict_pipeline(plug, config):
    xy_train = CSVLoader(name='xy_train', **config.xy_splitter).transform(filename=plug.train_filepath,
                                                                          train_mode=plug.train_mode)
    xy_valid = CSVLoader(name='xy_valid', **config.xy_splitter).transform(filename=plug.val_filepath,
                                                                          train_mode=plug.train_mode)

    loader = ImageDataLoaderResize('image_resize_loader', **config.loader).transform(x=xy_train.x,
                                                                                     y=xy_train.y,
                                                                                     train_mode=plug.train_mode,
                                                                                     x_valid=xy_valid.x,
                                                                                     y_valid=xy_valid.y)

    unet = UNetTrainer('unet_resnet101', **config.unet)
    unet.log("Loading trained model for {name}")
    unet.setup(model_path=os.path.join(config.working_dir, 'model.h5'))
    return unet, loader
