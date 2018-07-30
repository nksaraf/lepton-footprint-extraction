# from __future__ import absolute_import
#
# import argparse
# import json
# import logging
# import os
#
# import apache_beam as beam
# import cv2
# import imageio
# import joblib
# import numpy as np
# from apache_beam import pvalue
# from apache_beam.io.filesystems import FileSystems
# from pycocotools import mask as cocomask
# from scipy import ndimage as ndi
# from scipy.ndimage.morphology import distance_transform_edt
#
# logger = logging.getLogger()
#
#
# def create_mask_one_image(image_name, working_dir, coco, erode, dilate, border_width, small_annotations_size):
#     image_size = (300, 300)
#     mask_overlayed = np.zeros(image_size).astype('uint8')
#     distances = np.zeros(image_size)
#     annotations = coco["annotation"]
#
#     if erode < 0 or dilate < 0:
#         raise ValueError('erode and dilate cannot be negative')
#
#     if erode == 0:
#         mask, distances = overlay_masks_from_annotations(annotations=annotations,
#                                                          image_size=image_size,
#                                                          distances=distances)
#     elif dilate == 0:
#         mask, _ = overlay_masks_from_annotations(annotations=annotations,
#                                                  image_size=image_size)
#         mask_eroded, distances = overlay_eroded_masks_from_annotations(annotations=annotations,
#                                                                        image_size=image_size,
#                                                                        erode=erode,
#                                                                        distances=distances,
#                                                                        small_annotations_size=small_annotations_size)
#         mask = add_dropped_objects(mask, mask_eroded)
#     else:
#         mask, distances = overlay_eroded__dilated_masks_from_annotations(annotations=annotations,
#                                                                          image_size=image_size,
#                                                                          erode=erode,
#                                                                          dilate=dilate,
#                                                                          distances=distances,
#                                                                          small_annotations_size=small_annotations_size)
#     mask_overlayed = np.where(mask, 1, mask_overlayed)
#
#     sizes = get_size_matrix(mask_overlayed)
#     distances, second_nearest_distances = clean_distances(distances)
#
#     if border_width > 0:
#         borders = (second_nearest_distances < border_width) & (~mask_overlayed)
#         borders_class_id = mask_overlayed.max() + 1
#         mask_overlayed = np.where(borders, borders_class_id, mask_overlayed)
#
#     mask_overlayed = np.where(mask_overlayed > 0, 255, 0)
#
#     target_filepath = os.path.join(working_dir, "masks", image_name + ".png")
#     target_filepath_dist = os.path.join(working_dir, "distances", image_name)
#     target_filepath_sizes = os.path.join(working_dir, "sizes", image_name)
#     image_filepath = os.path.join(working_dir, "images", image_name + ".jpg")
#
#     return {
#         "csv": '{},{},{}'.format(image_name, image_filepath, target_filepath),
#         "masks": (target_filepath, mask_overlayed.astype(np.uint8)),
#         "distances": (target_filepath_dist, distances),
#         "sizes": (target_filepath_sizes, sizes)
#     }
#
#
# def overlay_masks_from_annotations(annotations, image_size, distances=None):
#     mask = np.zeros(image_size)
#     for ann in annotations:
#         rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
#         m = cocomask.decode(rle)
#         m = m.reshape(image_size)
#         if is_on_border(m, 2):
#             continue
#         if distances is not None:
#             distances = update_distances(distances, m)
#         mask += m
#     return np.where(mask > 0, 1, 0).astype('uint8'), distances
#
#
# def overlay_eroded_masks_from_annotations(annotations, image_size, erode, distances, small_annotations_size):
#     mask = np.zeros(image_size)
#     for ann in annotations:
#         rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
#         m = cocomask.decode(rle)
#         m = m.reshape(image_size)
#         if is_on_border(m, 2):
#             continue
#         m_eroded = get_simple_eroded_mask(m, erode, small_annotations_size)
#         if distances is not None:
#             distances = update_distances(distances, m_eroded)
#         mask += m_eroded
#     return np.where(mask > 0, 1, 0).astype('uint8'), distances
#
#
# def overlay_eroded__dilated_masks_from_annotations(annotations, image_size, erode, dilate, distances,
#                                                    small_annotations_size):
#     mask = np.zeros(image_size)
#     for ann in annotations:
#         rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
#         m = cocomask.decode(rle)
#         m = m.reshape(image_size)
#         if is_on_border(m, 2):
#             continue
#         m_ = get_simple_eroded_dilated_mask(m, erode, dilate, small_annotations_size)
#         if distances is not None:
#             distances = update_distances(distances, m_)
#         mask += m_
#     return np.where(mask > 0, 1, 0).astype('uint8'), distances
#
#
# def update_distances(dist, mask):
#     if dist.sum() == 0:
#         distances = distance_transform_edt(1 - mask)
#     else:
#         distances = np.dstack([dist, distance_transform_edt(1 - mask)])
#     return distances
#
#
# def clean_distances(distances):
#     if len(distances.shape) < 3:
#         distances = np.dstack([distances, distances])
#     else:
#         distances.sort(axis=2)
#         distances = distances[:, :, :2]
#     second_nearest_distances = distances[:, :, 1]
#     distances_clean = np.sum(distances, axis=2)
#     return distances_clean.astype(np.float16), second_nearest_distances
#
#
# def get_simple_eroded_mask(mask, size, small_annotations_size):
#     if mask.sum() > small_annotations_size**2:
#         mask_eroded = cv2.erode(mask, kernel=(size, size))
#     else:
#         mask_eroded = mask
#     return mask_eroded
#
#
# def get_simple_eroded_dilated_mask(mask, erode_size, dilate_size, small_annotations_size):
#     if mask.sum() > small_annotations_size**2:
#         mask_ = cv2.erode(mask, kernel=(erode_size, erode_size))
#     else:
#         mask_ = cv2.dilate(mask, kernel=(dilate_size, dilate_size))
#     return mask_
#
#
# def get_size_matrix(mask):
#     sizes = np.ones_like(mask)
#     labeled = label(mask)
#     for label_nr in range(1, labeled.max() + 1):
#         label_size = (labeled == label_nr).sum()
#         sizes = np.where(labeled == label_nr, label_size, sizes)
#     return sizes
#
#
# def is_on_border(mask, border_width):
#     return not np.any(mask[border_width:-border_width, border_width:-border_width])
#
#
# def label(mask):
#     labeled, nr_true = ndi.label(mask)
#     return labeled
#
#
# def add_dropped_objects(original, processed):
#     reconstructed = processed.copy()
#     labeled = label(original)
#     for i in range(1, labeled.max() + 1):
#         if not np.any(np.where((labeled == i) & processed)):
#             reconstructed += (labeled == i)
#     return reconstructed.astype('uint8')
#
#
# # Dataflow DoFn create mask for given image_id
# def create_masks(element,
#                  working_dir,
#                  erode=0,
#                  dilate=0,
#                  border_width=0,
#                  small_annotations_size=14):
#     with FileSystems.open(FileSystems.join(working_dir, "annotations", '{}.json'.format(element))) as fp:
#         coco = json.load(fp)
#
#     result = create_mask_one_image(element, working_dir, coco, erode, dilate,
#                                    border_width, small_annotations_size)
#
#     yield result['csv']
#     yield pvalue.TaggedOutput('masks', result['masks'])
#     yield pvalue.TaggedOutput('distances', result['distances'])
#     yield pvalue.TaggedOutput('sizes', result['sizes'])
#
#
# # Dataflow DoFn write image to path
# def write_images(element):
#     path, image = element
#     logger.info("Writing image to {}...".format(path))
#     with FileSystems.create(path) as fp:
#         imageio.imwrite(fp, image, format=path.split('.')[-1])
#     return True
#
#
# # Dataflow DoFn joblib dump to path
# def joblib_dump(element):
#     path, data = element
#     logger.info("Writing to {}...".format(path))
#     with FileSystems.create(path) as fp:
#         joblib.dump(data, fp)
#     return True
#
#
# def run(argv=None):
#     # Create and set your PipelineOptions.
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-w', '--working')
#     parser.add_argument('-d', '--dev-mode', action='store_true')
#     args, pipeline_args = parser.parse_known_args(argv)
#
#     # Create the Pipeline with the specified options.
#     with beam.Pipeline(argv=pipeline_args) as pipe:
#         logger.info("Building pipeline...")
#         index_file = FileSystems.join(args.working, 'index.txt')
#
#         results = (pipe
#                    | beam.io.ReadFromText(index_file)
#                    | 'CreatingMasks' >> beam.FlatMap(create_masks, working_dir=args.working)
#                    .with_outputs('masks', 'distances', 'sizes', main='csv'))
#
#         csv, masks, distances, sizes = results
#
#         for sub_dir in ['masks', 'distances', 'sizes']:
#             try:
#                 FileSystems.mkdirs(FileSystems.join(args.working, sub_dir))
#             except:
#                 pass
#
#         csv | 'WritingCSV' >> beam.io.WriteToText(os.path.join(args.working, 'mapping_challenge3'),
#                                                   '.csv',
#                                                   num_shards=0,
#                                                   shard_name_template='',
#                                                   header='image_id,file_path_image,file_path_mask',
#                                                   append_trailing_newlines=True)
#
#         masks | 'SavingMasks' >> beam.Map(write_images)
#         distances | 'SavingDistances' >> beam.Map(joblib_dump)
#         sizes | 'SavingSizes' >> beam.Map(joblib_dump)
