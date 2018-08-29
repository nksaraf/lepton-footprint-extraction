from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import argparse
import os

from model.predictor import Predictor, Evaluator
from postprocessing.utils import PolygonViewer
from model.connections import Wire
from model.base import UNetModel, ResnetUNetModel
from model.config import create_config
from model.loaders import ImageLoaderBasicTest, ImageLoaderNormalizedTest
from model.utils import CSVLoader
from postprocessing.polygons import Polygonizer

predictors = {
    'unet': UNetModel,
    'unet_resnet': ResnetUNetModel
}

loaders = {
    'basic': ImageLoaderBasicTest,
    'normalized': ImageLoaderNormalizedTest
}


def predict(config, args):
    plug = Wire(train_path=os.path.join(args.data_dir, 'train.local.csv'),
                val_path=os.path.join(args.data_dir, 'val.local.csv'),
                train_mode=True)

    xy_train = plug(filename='train_path') | CSVLoader(name='xy_train', **config.xy_splitter)
    xy_valid = plug(filename='val_path') | CSVLoader(name='xy_valid', **config.xy_splitter)

    loader = (plug + xy_train + xy_valid(x_valid='x', y_valid='y')) | loaders[args.loader]('image_loader', **config.loader)
    unet = predictors[config.model_name](**config.model)
    predictor = Predictor('unet_predictor', unet, need_setup=True)
    if len(args.model_path) > 0:
        predictor.setup(path=args.model_path, load_weights=True)
    else:
        raise Exception("No model path provided")

    evaluator = Evaluator('evaluator', predictor.predictor, need_setup=False)

    viewer = PolygonViewer('viewer', save=args.save, job_dir=config.job_dir)
    polygonizer = Polygonizer('polygons')

    while True:
        x, y = loader.validation_datagen[0].next()
        batch = Wire(x=x, y=y)
        prediction = batch | predictor | polygonizer
        evaluation = batch | evaluator
        grouth_truth = batch(predictions='y') | polygonizer
        batch(images='x') + prediction(predictions='polygons') + grouth_truth(truths='polygons') + evaluation | viewer
        if raw_input('More? (y/n): ') not in ["Y", 'y']:
            break


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--job-dir', required=True, type=str, help='Working folder')
    config_parser.add_argument('--batch-size-train', default=32, type=int, help='Batch size for training')
    config_parser.add_argument('--batch-size-val', default=32, type=int, help='Batch size for validation')
    config_parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
    config_parser.add_argument('--model', default='unet', type=str, help='Model to train (unet/unet_resnet)')
    config_parser.add_argument('--seed', '-s', default=6581, type=int, help='Seed')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str, help='Folder to get data files csv from')
    parser.add_argument('--gpus', default=1, type=int, help='Number of gpus for training')
    parser.add_argument('--model-path', default='', type=str, help="Model path")
    parser.add_argument('--loader', default='basic', type=str, help="Image data loader to use")
    parser.add_argument('--save', action='store_true', default=False)
    
    config_args, unknown = config_parser.parse_known_args()
    config = create_config(**config_args.__dict__)

    args , _ = parser.parse_known_args(unknown)
    result = predict(config, args)
