from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from model.connections import Transformer

MIN_POLYGON_AREA = 1.


class Polygonizer(Transformer):
    __out__ = ('polygons', )
    
    def __init__(self, name, epsilon=1.5, min_area=MIN_POLYGON_AREA):
        super(Polygonizer, self).__init__(name)
        self.epsilon = epsilon
        self.min_area = min_area

    def polygonize(self, mask):
        mask[mask < 0.5] = 0
        mask[mask > 0] = 1

        # first, find contours with cv2: it's much faster than shapely
        image, contours, hierarchy = cv2.findContours(
            ((mask == 1).astype(np.uint8) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        # create approximate contours to have reasonable submission size
        approx_contours = [cv2.approxPolyDP(cnt, self.epsilon, True)
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
            if idx not in child_contours and cv2.contourArea(cnt) >= self.min_area:
                assert cnt.shape[1] == 1
                try:
                    poly = Polygon(
                        shell=cnt[:, 0, :],
                        holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                               if cv2.contourArea(c) >= self.min_area])
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

    def __transform__(self, predictions):
        if len(predictions.shape) == 4:
            polygons = []
            for i in range(predictions.shape[0]):
                polygons.append(self.polygonize(predictions[i]))
        else:
            polygons = self.polygonize(predictions)
        return {'polygons': polygons}


