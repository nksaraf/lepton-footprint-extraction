import os
from abc import ABCMeta, abstractmethod
from functools import partial, update_wrapper

from keras.callbacks import EarlyStopping, ProgbarLogger, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from connections import Transformer
from model.loss import dice_coef, get_weights, jaccard_index, jaccard_index_loss, mixed_iou_cross_entropy_loss, \
    weighted_cross_entropy
from model.resnet101 import Scale
from model.unet import UNetResNet101
from model.utils import ModelCheckpoint, save_model, save_model_weights


class Model(Transformer):
    __metaclass__ = ABCMeta

    def __init__(self, name, architecture_config, training_config, callbacks_config):
        super(Model, self).__init__(name, need_setup=True)
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config
        self.model = None
        self.parallel_model = None
        self.callbacks = None
        self.custom_objects = None

    def __setup__(self, model_path=None):
        self.log("Setting up model...")
        if model_path is not None:
            self.model = load_model(model_path, custom_objects=self.custom_objects)
        else:
            self.model = self._compile_model(**self.architecture_config)

    def _compile_model(self, model_params, optimizer_params, compiler_params, loss_params):
        model = self._build_model(**model_params)
        optimizer = self._build_optimizer(**optimizer_params)
        loss = self._build_loss(**loss_params)
        self.log("Compiling model...")
        try:
            self.parallel_model = multi_gpu_model(model)
        except:
            self.parallel_model = model

        self.parallel_model.compile(optimizer=optimizer, loss=loss, **compiler_params)
        self.log("Done building model.")
        return model

    @abstractmethod
    def _create_callbacks(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _build_model(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _build_optimizer(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _build_loss(self, **kwargs):
        raise NotImplementedError

    def save_weights(self, job_dir, filepath):
        save_model_weights(self.model, job_dir, filepath)

    def load_weights(self, job_dir, filepath):
        self.model.load_weights(os.path.join(job_dir, filepath))

    def save(self, job_dir, filepath):
        self.log('Saving model at {}'.format(filepath))
        save_model(self.model, job_dir, filepath)

    def load(self, job_dir, filepath):
        self.log("Loading model from file {}".format(filepath))
        self.model = load_model(os.path.join(job_dir, filepath), custom_objects=self.custom_objects)
        return self


class AbstractXYTrainer(Model):
    __metaclass__ = ABCMeta
    __out__ = ('model',)

    def __transform__(self, x, y, validation_data):
        self.log("Training model using x-y data...")
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.parallel_model.fit(x, y,
                       validation_data=validation_data,
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        self.log("Trained model")
        return {"model": self}


class AbstractGeneratorTrainer(Model):
    __metaclass__ = ABCMeta
    __out__ = ('model',)

    def __transform__(self, datagen, validation_datagen):
        self.log("Training model with generator...")
        self._create_callbacks(**self.callbacks_config)
        train_flow, train_steps = datagen
        valid_flow, valid_steps = validation_datagen
        self.parallel_model.fit_generator(train_flow,
                                 steps_per_epoch=train_steps,
                                 validation_data=valid_flow,
                                 validation_steps=valid_steps,
                                 callbacks=self.callbacks,
                                 verbose=1,
                                 **self.training_config)
        self.log("Trained model")
        return {"model": self}


class UNetTrainer(AbstractGeneratorTrainer):
    def __init__(self, name, architecture_config, training_config, callbacks_config):
        super(UNetTrainer, self).__init__(name, architecture_config, training_config, callbacks_config)
        self.custom_objects = {
            'mixed_iou_cross_entropy_loss': self._build_loss(**self.architecture_config['loss_params']),
            'Scale': Scale,
            'jaccard_index': jaccard_index,
            'dice_coef': dice_coef
        }

    def __setup__(self, model_path=None):
        Model.__setup__(self, model_path)

    def _create_callbacks(self, **callbacks_config):
        self.callbacks = []
        callbacks_index = dict(model_checkpoint=ModelCheckpoint, progbar_logger=ProgbarLogger,
                               plateau_lr_scheduler=ReduceLROnPlateau, early_stopping=EarlyStopping,
                               tensor_board=TensorBoard)
        for name, callback in callbacks_index.iteritems():
            if name in callbacks_config:
                self.callbacks.append(callback(**callbacks_config[name]))

    def _build_model(self, **model_params):
        unet = UNetResNet101()
        return unet.build_model(**model_params)

    def _build_optimizer(self, lr, decay):
        return Adam(lr=lr, epsilon=10e-8, decay=decay)

    def _build_loss(self, loss_weights, bce, iou):
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
