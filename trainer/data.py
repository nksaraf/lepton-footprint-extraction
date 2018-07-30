from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import cv2
import sys
import logging
import tqdm

OUTPUT_DIR = 'output'
OUTPUT_SUFFIX = '_output'
IMAGE_SIZE = 256
WHITE_LIST_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

logging.basicConfig(format='%(levelname)s:  %(message)s', level=logging.INFO)

def save_img(path, x, scale=True):
    """Saves an image stored as a Numpy array to a path or file object.

    # Arguments
        path: Path or file object.
        image: Numpy array.
        scale: Whether to rescale image values to be within `[0, 255]`.
    """
    if len(x.shape) > 2:
        if x.shape[2] == 1: # grayscale
            x = x.reshape((x.shape[0], x.shape[1]))
        else: # RGB -> BGR
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    cv2.imwrite(path, x)

def load_img(path, grayscale=False, target_size=None, interpolation='nearest'):
    """Loads an image into numpy array.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", "bicubic",
            "lanczos", "area"
            
    # Returns
        A numpy array representing image.

    # Raises
        ValueError: if interpolation method is not supported.
    """
    image = cv2.imread(path, -1)
    if image is None:
            raise ValueError('Unable to load {}'.format(path))

    if image.dtype == np.uint16:
        imin, imax = np.min(image), np.max(image)
        image -= imin
        imf = np.array(image,'float32')
        imf *= 255./(imax-imin)
        image = np.asarray(np.round(imf), 'uint8')

    print(image)
    print(image.shape)

    if grayscale:
        image = image.reshape((image.shape[0], image.shape[1], 1))
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(interpolation, ", ".join(INTERPOLATION_METHODS.keys())))
            resample = INTERPOLATION_METHODS[interpolation]
            img = cv2.resize(image, width_height_tuple, resample)
    return image

def _iter_valid_files(directory, white_list_formats, get_output):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        get_output: whether to return output filenames

    # Yields
        Tuple of (root, filename) with extension in `white_list_formats` and has valid output file.
    """
    for root, subdirs, files in os.walk(directory, topdown=True):
        try:
            output_dir_idx = subdirs.index(OUTPUT_DIR)
            del subdirs[output_dir_idx]
        except ValueError:
            if get_output:
                logging.info('No output(masks) folder found in ' + root)

        for file in sorted(files):
            fname, ext = os.path.splitext(file)
            if ext in white_list_formats:
                output_filename = '{}{}{}'.format(fname, OUTPUT_SUFFIX, ext)
                if get_output:
                    output_path_same = os.path.join(root, output_filename)
                    output_path_inside = os.path.join(os.path.join(root, OUTPUT_DIR), output_filename)
                    if os.path.exists(output_path_same):
                        yield os.path.join(root, file), output_path_same
                    elif os.path.exists(output_path_inside):
                        yield os.path.join(root, file), output_path_inside
                    else:
                        logging.info('No corresponding output found for {} in {}'.format(file, root))
                else:
                    yield os.path.join(root, file)

def _list_valid_filenames_in_directory(directory, white_list_formats, split=[], get_output=False):
    """Lists paths of files in `directory` with extensions in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory
        get_output: whether to return output filenames

    # Returns
        filenames: 
    """
    if split:
        files = list(_iter_valid_files(directory, white_list_formats, get_output))
        num_files = len(files)
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = files[start: stop]
    else:
        valid_files = list(_iter_valid_files(directory, white_list_formats, get_output))
    return valid_files

def _fix_image_size(image, tile_size=256, step_size=256):
    height, width = image.shape[0], image.shape[1]
    logging.info('Image has shape ({}, {})'.format(height, width))

    rem_width = width % tile_size
    rem_height = height % tile_size

    if rem_width == 0:
        border_right = 0
    else:
        border_right = tile_size - rem_width

    if rem_height == 0:
        border_bottom = 0
    else:
        border_bottom = tile_size - rem_height

    logging.info('Adding border of pixel width {} - right, {} - bottom'.format(border_right, border_bottom))
    if border_bottom > 0 or border_right > 0:
        new_image = cv2.copyMakeBorder(image, (0, border_bottom, 0, border_right), cv2.BORDER_REFLECT_101)
        return new_image
    else:
        return image

def _get_tiles(img_array, tile_size):
    rows = int(img_array.shape[0]/tile_size)
    cols = int(img_array.shape[1]/tile_size)
    for i in range(rows):
        for j in range(cols):
            x, y = i*tile_size, j*tile_size
            tile = img_array[x:x+tile_size, y:y+tile_size]
            yield i, j, tile

def make_tiles(path, output_path, prefix='', suffix='', file_format='jpeg', tile_size=256, skip_filename=False):
    image = load_img(path)
    logging.info('Loaded image {} with shape {}'.format(path, image.shape))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logging.info('Created directory {}'.format(output_path))

    image = _fix_image_size(image)
    rows = int(image.shape[0]/tile_size)
    cols = int(image.shape[1]/tile_size)
    number_of_tiles = rows * cols

    filename, ext = os.path.splitext(os.path.basename(path))

    logging.info('Splitting {} into {} tiles'.format(filename, number_of_tiles))

    if not skip_filename:
        prefix = '{}_{}'.format(prefix, filename)
    if len(suffix) > 0:
        suffix = '_{}'.format(suffix)

    for i, j, tile in tqdm.tqdm(_get_tiles(image, tile_size), total=number_of_tiles):
        save_img(os.path.join(output_path, '{}_{}_{}{}.{}'.format(prefix, i, j, suffix, file_format)), tile, scale=False)

if __name__ == '__main__':
    pass
    # valid_files = _list_valid_filenames_in_directory('raw_data/bangalore', white_list_formats, get_output=True)
    # print(len(valid_files))
    # print(valid_files[0])
    # make_tiles('raw_data/bang_input.jpeg', 'raw_data/bangalore', prefix='bangalore', file_format='jpeg', skip_filename=True)

    # make_tiles('raw_data/bang_output.jpeg', 'raw_data/bangalore/output', prefix='bangalore', suffix='output', file_format='jpeg', skip_filename=True)