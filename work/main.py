from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import argparse
import os

from model.predictor import Predictor
from model.connections import Wire
from model.base import UNetModel
from model.config import create_config
from model.loaders import ImageLoaderInference
from postprocessing.polygons import Polygonizer
from postprocessing.shapefile import ShapefileCreator, make_transform

import rasterio
import numpy as np

"""
The primary prediction pipeline. Includes:
	- fix image if the dimensions are not perfect for tiling by adding extra blank pixels on the edges
	- tiling arbitrarily large image into 256x256 sized images
	- predicting masks for all those images using the model
	- combine predicted masks and stitch together to original dimensions
	- extract polygons from large mask
	- save to shapefile
"""

def tile_image(image, tile_h, tile_w):
    """
    Return an array of shape (n, tile_h, tile_w, channels) which
    preserves the number of channels from image, and splits the image into
    subblocks of dimensions (tile_h, tile_w, channels).
    """
    h, w, c = image.shape
    return (image.reshape(h//tile_h, tile_h, -1, tile_w, c)
            .swapaxes(1,2)
            .reshape(-1, tile_h, tile_w, c))


def untile_image(image, h, w, c):
    """
    shape of image => (number_of_tiles, tile_h, tile_w, channels)
    Return an array of shape (h, w, c) that stitches the tiles together in the original order
    """
    n, tile_h, tile_w, c = image.shape
    return (image.reshape(h//tile_h, -1, tile_h, tile_w, c)
               .swapaxes(1,2)
               .reshape(h, w, c))

def open_image(file_path):
	"""Open a geocoded image, return the RGB image as numpy array, projection system used,
	and transformation matrix.
	""" 
    with rasterio.open(file_path) as jpg:
        transform = jpg.transform
        crs = jpg.crs
        assert jpg.count == 3
        r = jpg.read(1)
        g = jpg.read(2)
        b = jpg.read(3)
        image = np.stack([r, g, b], axis=-1)

    return image, transform, crs

def adjust_image(image):
	"""Adjust image to be divisible into tiles of 256x256, but adding blank pixels to the
	right and bottom edges of the image as necessary.
	"""
    h, w, _ = image.shape
    h_add = 256 - ( h % 256 )
    w_add = 256 - ( w % 256 )

    new_h = h if h_add == 256 else (h + h_add)
    new_w = w if w_add == 256 else (w + w_add)

    adjusted_image = np.zeros(( new_h, new_w, 3 ), dtype=image.dtype)
    adjusted_image[:h, :w] = image
    return adjusted_image

def predict(config, args):
    image, transform, crs = open_image(args.file_path)
    adjusted_image = adjust_image(image)
    tiles = tile_image(adjusted_image, 256, 256)

    unet = UNetModel(**config.model)
    predictor = Predictor('unet_predictor', unet, need_setup=True)
    predictor.setup(path=args.model_path, load_weights=True)

    predictions = Wire(x=tiles, batch_size=5) | predictor

    remade_image = untile_image(predictions.predictions, adjusted_image.shape[0], adjusted_image.shape[1], 1)
    polygons = Wire(predictions=remade_image) | Polygonizer('polygons')
    (Wire(filename=args.file_path[:args.file_path.index('.')] + '.shp', transform=make_transform(transform)) + polygons 
        | ShapefileCreator('shapefile', crs=crs.to_dict()))


if __name__ == '__main__':
	# arguments used to form the config for prediction operation
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--job-dir', required=True, type=str, help='Working folder')
    config_parser.add_argument('--batch-size', default=32, type=int, help='Batch size for inference')
    config_parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
    config_parser.add_argument('--model', default='unet', type=str, help='Model to train (unet/unet_resnet)')
    config_parser.add_argument('--seed', '-s', default=0, type=int, help='Seed')

	# arguments for the ``predict`` function
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, type=str, help="Model path")
    parser.add_argument('-f', '--file-path', required=True, type=str)

    config_args, unknown = config_parser.parse_known_args()
    config = create_config(**config_args.__dict__)
    
    args , _ = parser.parse_known_args(unknown)
    result = predict(config, args)
