from __future__ import print_function

from itertools import izip
import logging

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import print_summary
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import trainer.unet as unet
import trainer.cloud as cloud

SMOOTH = 1e-12
PROJECT = "lepton-maps-207611"
BUCKET = "bangalore_data"

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return K.mean(jac)

def gs_generator(image_dir, mask_dir, target_size, seed, batch_size=32):
    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_last',
        rescale=1./255
    )

    flow_from_dir_args = dict(
        target_size=target_size,
        class_mode=None,
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
        classes=['data']
    )

    data_generator = ImageDataGenerator(**data_gen_args)
    image_generator = cloud.flow_from_google_storage( data_generator, PROJECT, BUCKET, image_dir, color_mode='rgb', **flow_from_dir_args )
    mask_generator = cloud.flow_from_google_storage( data_generator, PROJECT, BUCKET, mask_dir, color_mode='grayscale', **flow_from_dir_args )
    return izip(image_generator, mask_generator)

def generator(image_dir, mask_dir, target_size, seed, batch_size=32):
    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_last',
        rescale=1./255
    )

    flow_from_dir_args = dict(
        target_size=target_size,
        class_mode=None,
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
        classes=['data']
    )

    data_generator = ImageDataGenerator(**data_gen_args)
    image_generator = data_generator.flow_from_directory( image_dir, color_mode='rgb', **flow_from_dir_args )
    mask_generator = data_generator.flow_from_directory( mask_dir, color_mode='grayscale', **flow_from_dir_args )
    return izip(image_generator, mask_generator)

def build_model(start_ch=16, depth=3, dropout=0.5, batchnorm=False):
    logging.info('-'*30)
    logging.info("Building model...")

    model = unet.UNet(img_shape=(256,256,3), out_ch=1, start_ch=start_ch, depth=depth, activation='relu', 
         dropout=dropout, batchnorm=batchnorm, maxpool=True, upconv=True, residual=False)

    logging.info("Built.")
    logging.info('-'*30)
    print_summary(model, print_fn=logging.debug)

    return model

def compile_model(model, optimizer, learning_rate=1e-3, loss='binary_crossentropy', metrics=[jaccard_coef]):
    logging.info('-'*30)
    logging.info("Compiling model...")

    model.compile(optimizer=optimizer(lr=learning_rate), loss=loss, metrics=metrics + ['accuracy'])

    logging.info("Compiled.")
    logging.info('-'*30)

def to_savedmodel(model, export_path):
    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'image': model.inputs[0]},
                                    outputs={'mask': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )

    builder.save()