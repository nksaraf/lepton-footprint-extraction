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
from model.loaders import ImageLoaderInference, ImageLoaderNormalizedInference
from model.utils import CSVLoaderXYZ
from postprocessing.polygons import Polygonizer
from postprocessing.shapefile import ShapefileCreator, get_transform

predictors = {
    'unet': UNetModel,
    'unet_resnet': ResnetUNetModel
}

loaders = {
    'basic': ImageLoaderInference,
    'normalized': ImageLoaderNormalizedInference
}

def predict(config, args):
    plug = Wire(filename=os.path.join(args.data_dir, 'val.csv'), train_mode=True)

    loader = (plug 
        | CSVLoaderXYZ(name='xyz', prefix=args.data_pre, **config.xy_splitter) 
        | ImageLoaderInference('image_loader', True, **config.loader))
    
    unet = predictors[config.model_name](**config.model)
    predictor = Predictor('unet_predictor', unet, need_setup=True)
    predictor.setup(path=args.model_path, load_weights=True)

    evaluator = Evaluator('evaluator', predictor.predictor, need_setup=False)
    viewer = PolygonViewer('viewer', save=args.save, job_dir=config.job_dir)
    polygonizer = Polygonizer('polygons')
    generator, steps = loader.generator

    while True:
        x, y, z = generator.next()
        batch = Wire(x=x, y=y)
        prediction = batch | predictor | polygonizer
        creator = ShapefileCreator('shapefile')
        for i in range(len(x)):
            shp_path = os.path.join('/Users/nikhilsaraf/Documents', os.path.basename(z[i]).replace('image', 'pred').replace('.jpg', '.shp'))
            creator.transform(polygons=prediction.polygons[i], filename=shp_path, transform=get_transform(os.path.join(args.data_pre, z[i])))
        # return prediction.polygons[2]



        # evaluation = batch | evaluator
        # ground_truth = batch(predictions='y') | polygonizer
        # batch(images='x') + prediction(predictions='polygons') + ground_truth(truths='polygons') + evaluation | viewer
        # if raw_input('More? (y/n): ') not in ["Y", 'y']:
        #     break


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--job-dir', required=True, type=str, help='Working folder')
    config_parser.add_argument('--batch-size', default=32, type=int, help='Batch size for inference')
    config_parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
    config_parser.add_argument('--model', default='unet', type=str, help='Model to train (unet/unet_resnet)')
    config_parser.add_argument('--seed', '-s', default=6581, type=int, help='Seed')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str, help='Folder to get data files csv from')
    parser.add_argument('--gpus', default=1, type=int, help='Number of gpus for training')
    parser.add_argument('--model-path', required=True, type=str, help="Model path")
    parser.add_argument('--loader', default='basic', type=str, help="Image data loader to use")
    parser.add_argument('--data-pre', required=False, type=str, default='', help="Data files prefix")
    parser.add_argument('--save', action='store_true', default=False)
    
    config_args, unknown = config_parser.parse_known_args()
    config = create_config(**config_args.__dict__)

    args , _ = parser.parse_known_args(unknown)
    result = predict(config, args)
