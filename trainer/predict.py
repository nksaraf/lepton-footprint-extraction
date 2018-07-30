from __future__ import print_function

import argparse
import os
import logging
import random
from collections import defaultdict

import trainer.model as model
import trainer.util as util

import numpy as np
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.lib.io import file_io
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.ops
import rasterio
import rasterio.features
import pandas as pd

SEED = 100
MODEL_PATH = 'lepton.h5'
THRESHOLD = 0.5
MIN_POLYGON_AREA = 0.

def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    poly_list = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df

def mask_for_polygons(polygons):
    img_mask = np.zeros((256, 256), np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def mask_to_polygons(mask, epsilon=10., min_area=MIN_POLYGON_AREA):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1).astype(np.uint8) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            try:
                poly = Polygon(
                    shell=cnt[:, 0, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= min_area])
                all_polygons.append(poly)
            except:
                pass
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def post_process(masks):
    return masks.reshape(masks.shape[:-1])


def predict( job_dir, image_dir, mask_dir):

    lepton = load_model(os.path.join(job_dir, MODEL_PATH), custom_objects={'jaccard_coef': model.jaccard_coef})
    test_generator = model.generator(image_dir, mask_dir, target_size=(256, 256), seed=random.randint(0, 100000), batch_size=1)

    for i in range(1):
        image, mask = next(test_generator)
        predicted_mask = lepton.predict(image, verbose=1)
        print(image.shape)
        print(predicted_mask.shape)
        predicted_mask = post_process(predicted_mask)
        mask = post_process(mask)
        print(predicted_mask.shape)
        # predicted_mask[predicted_mask < THRESHOLD] = 0.
        print(predicted_mask)
        predicted_mask = (predicted_mask > THRESHOLD).astype(np.uint8)
        print(predicted_mask)
        polygons = mask_to_poly(predicted_mask[0])
        print(polygons)
        # pred_mask = mask_for_polygons(polygons)
        # pred_masks = np.zeros(predicted_mask.shape)
        # for i in range(1):
        #     polygons = mask_to_polygons(predicted_mask[i])
        #     print(polygons)
        #     pred_mask = mask_for_polygons(polygons)
        #     pred_masks[i] = pred_mask

        # print(pred_mask.shape)
        # pred_binary_mask = np.zeros(predicted_mask.shape)
        # threshold = 0.3
        # pred_binary_mask[predicted_mask >= threshold] = 1.
        util.display_images([image[0], predicted_mask[0], mask[0]], 1, 3)
        # util.three_by_three(image, pred_masks, mask, False)
        # util.display_images([image[0], predicted_mask[0], mask[0]], 1, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, type=str, help='Folder to save model in')
    parser.add_argument('--image-dir', required=True, type=str, help='Folder containing images folder for inference')
    parser.add_argument('--mask-dir', required=True, type=str, help='Folder containing masks folder for inference')
    #parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training and validation')

    parse_args, unknown = parser.parse_known_args()
    predict(**parse_args.__dict__)


