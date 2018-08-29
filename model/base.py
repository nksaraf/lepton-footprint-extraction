import os
from abc import ABCMeta, abstractmethod
from functools import partial, update_wrapper

from keras.callbacks import EarlyStopping, ProgbarLogger, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf

from model.loss import dice_coef, get_weights, jaccard_index, jaccard_index_loss, mixed_iou_cross_entropy_loss, \
    weighted_cross_entropy
from model.resnet101 import Scale
from model.unet import UNetResNet101, BaseUNet
from model.utils import save_model, save_model_weights


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, architecture_config, training_config, callbacks_config):
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config
        self.model = None
        self.training_model = None
        self.callbacks = None
        self.custom_objects = None

    def setup(self, path=None, load_weights=False, gpus=1):
        if path is not None:
            if load_weights:
                self.model = self.compile_model(gpus=gpus, **self.architecture_config)
                self.training_model.load_weights(path)
            else:
                self.model = load_model(path, custom_objects=self.custom_objects)
        else:
            self.model = self.compile_model(gpus=gpus, **self.architecture_config)

    def compile_model(self, gpus, model_params, optimizer_params, compiler_params, loss_params):
        optimizer = self.build_optimizer(**optimizer_params)
        loss = self.build_loss(**loss_params)
        if gpus <= 1:
            model = self.build_model(**model_params)
            self.training_model = model
            self.training_model.compile(optimizer=optimizer, loss=loss, **compiler_params)
        else:
            with tf.device("/cpu:0"):
                model = self.build_model(**model_params)
            model.compile(optimizer=optimizer, loss=loss, **compiler_params)
            self.training_model = multi_gpu_model(model, gpus=gpus)
            self.training_model.compile(optimizer=optimizer, loss=loss, **compiler_params)
        return model

    @abstractmethod
    def create_callbacks(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_model(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_loss(self, **kwargs):
        raise NotImplementedError

    def save_weights(self, job_dir, filepath):
        save_model_weights(self.model, job_dir, filepath)

    def load_weights(self, job_dir, filepath):
        self.model.load_weights(os.path.join(job_dir, filepath))

    def save(self, job_dir, filepath):
        save_model(self.model, job_dir, filepath)

    def load(self, job_dir, filepath):
        self.model = load_model(os.path.join(job_dir, filepath), custom_objects=self.custom_objects)
        return self


class AbstractUNetModel(Model):
    __metaclass__ = ABCMeta

    def create_callbacks(self, **callbacks_config):
        self.callbacks = []
        callbacks_index = dict(progbar_logger=ProgbarLogger,
                               plateau_lr_scheduler=ReduceLROnPlateau, early_stopping=EarlyStopping,
                               tensor_board=TensorBoard)
        for name, callback in callbacks_index.iteritems():
            if name in callbacks_config:
                self.callbacks.append(callback(**callbacks_config[name]))

    def build_optimizer(self, lr, decay):
        return Adam(lr=lr, epsilon=10e-8, decay=decay)

    def build_loss(self, loss_weights, bce, iou):
        weights_function = partial(get_weights, **bce)
        update_wrapper(weights_function, get_weights)
        weighted_loss = partial(weighted_cross_entropy, weights_function=weights_function)
        update_wrapper(weighted_loss, weighted_cross_entropy)
        iou_loss = partial(jaccard_index_loss, **iou)
        update_wrapper(iou_loss, jaccard_index_loss)
        loss = partial(mixed_iou_cross_entropy_loss,
                       iou_loss_func=iou_loss,
                       iou_weight=loss_weights['iou_mask'],
                       cross_entropy_weight=loss_weights['bce_mask'],
                       cross_entropy_loss_func=weighted_loss)
        update_wrapper(loss, mixed_iou_cross_entropy_loss)
        return loss


class UNetModel(AbstractUNetModel):
    __metaclass__ = ABCMeta

    def __init__(self, architecture_config, training_config, callbacks_config):
        super(AbstractUNetModel, self).__init__(architecture_config, training_config, callbacks_config)
        self.custom_objects = {
            'mixed_iou_cross_entropy_loss': self.build_loss(**self.architecture_config['loss_params']),
            'jaccard_index': jaccard_index,
            'dice_coef': dice_coef
        }

    def build_model(self, **model_params):
        unet = BaseUNet()
        return unet.build_model(**model_params)


class ResnetUNetModel(AbstractUNetModel):
    __metaclass__ = ABCMeta

    def __init__(self, architecture_config, training_config, callbacks_config):
        super(AbstractUNetModel, self).__init__(architecture_config, training_config, callbacks_config)
        self.custom_objects = {
            'mixed_iou_cross_entropy_loss': self.build_loss(**self.architecture_config['loss_params']),
            'Scale': Scale,
            'jaccard_index': jaccard_index,
            'dice_coef': dice_coef
        }

    def build_model(self, **model_params):
        unet = UNetResNet101()
        return unet.build_model(**model_params)