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
    """An abstract Keras-backed machine learning model that can be used for training or prediction. 
    It or its weights can be loaded from and saved to disk. It can also be configured to run on
    multiple gpus. 

    The model needs to be compiled before training or prediction. The four functions that need to 
    be overrided allow the compilation of the model to be very customized in terms of the 
    parameters accepted. The model is initialized with 3 config dictionaries that are used during
    compilation and training. Their purpose is as follows:

    architecture_config: configuration for compilation. must contain the following four subdictionaries
        model_params: parameters for ``build_model``
        optimizer_params: parameters for ``build_optimizer``
        loss_params: parameters for ``build_loss``
        compiler_params: other parameters for the Keras compiler apart from ``optimizer`` and ``loss``
    callbacks_config: The subdictionaries included determine which callbacks will be included during
                      the training. The custom ModelCheckpoint callback is always added to the list
                      of callbacks. Used as parameter for ``create_callbacks``
    training_config: other parameters passed on to the Keras functions for training (``keras.models.Model.fit``)
                     apart form ``validation_data`` (received as argument) and ``callbacks`` (uses attribute)

    # Attributes:
        architecture_config: Described above
        training_config: Described above
        callbacks_config: Described above
        model: A Keras model instance or ``None`` if not yet built
        training_model: A Keras model instance, same as ``model``, except that on multi-GPU devices, 
            it optimizes the training process to run faster. This model is used during training, while
            the other is always used for saving and loading (their weights are always equal)
        custom_objects: A dictionary of objects that are part of the Keras model but are custom-defined. 
            This is needed during loading such a model from file.
    """ 
    __metaclass__ = ABCMeta

    def __init__(self, architecture_config, training_config, callbacks_config):
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config
        self.model = None
        self.training_model = None
        self.custom_objects = {}

    def setup(self, path=None, load_weights=False, gpus=1):
        """Setup (compile) the model to be used for training or prediction. Load the model or just 
        the weights from file if required. Also setup the model to be used on Multi-GPU devices. Doesn't
        return anything, assigns ``self.model`` and ``self.training_model``. Uses
        the ``architecture_config`` dictionary while compiling model.

        # Args:
            path: Filepath to saved model/weights, or ``None`` if should not load from file. 
                When loading weights, the model weights contained in the filepath given should 
                have the same exact build as the one being loaded to. If loading model from file, ensure that
                the necessary ``custom_objects`` are given in the attributes.
            load_weights: True if need to load only weights from file, False for complete model
            gpus: [int] Number of gpus to setup model for
        """
        if path is not None:
            if load_weights:
                self.model = self.compile_model(gpus=gpus, **self.architecture_config)
                self.training_model.load_weights(path)
            else:
                self.model = load_model(path, custom_objects=self.custom_objects)
        else:
            self.model = self.compile_model(gpus=gpus, **self.architecture_config)

    def compile_model(self, gpus, model_params, optimizer_params, loss_params, compiler_params):
        """Build and compile a ``Keras.models.Model`` based on the configuration provided.

        # Args:
            gpus: Number of gpus available. If >1, the model needs to be setup for a multi-gpu configuration.
            model_params: dict of parameters to the ``build_model`` function
            optimzer_params: dict of parameters to the ``build_optimizer`` function
            loss_params: dict of parameters to the ``build_loss`` function
            compiler_params: dict of other parameters for the ``Keras.models.Model.compile`` function apart
                from ``optimizer`` and ``loss``

        # Returns:
            A compiled ``Keras.models.Model``. The ``self.training_model`` is also assigned to the 
            same model, if gpus <= 1, or a multi-gpu model (see ``keras.utils.multi_gpu_model``) if 
            gpus > 1.
        """
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
        """Create a list of callbacks to be used during the training/prediction of the model.
        Uses ``callbacks_config`` to create the list.

        # Returns:
            A list of ``keras.callbacks.Callback``
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self, **kwargs):
        """Build the architecture of the model.

        # Returns:
            A ``keras.models.Model``
        """
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, **kwargs):
        """Build an optimizer for the model (used to compile the model).

        # Returns:
            A ``keras.optimizers.Optimizer``
        """
        raise NotImplementedError

    @abstractmethod
    def build_loss(self, **kwargs):
        """Build a loss function that received predicted output and expected output and computes
        a numerical loss value. It is used to compile the model

        # Returns:
            A function with signature: (target, output) -> loss (float)
        """
        raise NotImplementedError

    def save_weights(self, job_dir, filepath):
        """Save model weights to ``{job_dir}/{filepath}``"""
        save_model_weights(self.model, job_dir, filepath)

    def load_weights(self, job_dir, filepath):
        """Load model weights from ``{job_dir}/{filepath}``"""
        self.model.load_weights(os.path.join(job_dir, filepath))

    def save(self, job_dir, filepath):
        """Save Keras model to `{job_dir}/{filepath}``"""
        save_model(self.model, job_dir, filepath)

    def load(self, job_dir, filepath):
        """Load Keras model from `{job_dir}/{filepath}`` and assign to ``self.model``. Needs
        ``self.custom_objects`` to be properly setup.
        """
        self.model = load_model(os.path.join(job_dir, filepath), custom_objects=self.custom_objects)
        return self


class AbstractUNetModel(Model):
    """An abstract UNet architecture based ``Model``. The optimizer and loss functions are setup 
    specifically for UNet models used for semantic segmentation problems. The model architecture is
    not included and needs to be provided. 
    """
    __metaclass__ = ABCMeta

    def create_callbacks(self, **callbacks_config):
        """See base class."""
        self.callbacks = []
        callbacks_index = dict(progbar_logger=ProgbarLogger,
                               plateau_lr_scheduler=ReduceLROnPlateau, early_stopping=EarlyStopping,
                               tensor_board=TensorBoard)
        for name, callback in callbacks_index.iteritems():
            if name in callbacks_config:
                self.callbacks.append(callback(**callbacks_config[name]))

    def build_optimizer(self, lr, decay):
        """See base class. Returns a Adam Optimizer with the given parameters"""
        return Adam(lr=lr, epsilon=10e-8, decay=decay)

    def build_loss(self, loss_weights, bce, iou):
        """See base class. Returns a mixed loss function which uses the discrete IoU loss function 
        (approximated by the jaccard_index) and the cross entropy loss function. These two loss functions
        are weighted according to the ``loss_weights`` parameter
            iou_mask -> jaccard index weight
            bce_mask -> cross entropy weight

        # Args:
            loss_weights: A dictionary containing the weights of the two loss functions:
                iou_mask -> jaccard index weight
                bce_mask -> cross entropy weight
            bce: A dictionary containing parameters to the cross entropy loss function
            iou: A dictionary containing parameters to the jaccard index loss function
        """
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
    """A UNet model that does not use a pre-trained encoder. Refer to https://arxiv.org/abs/1505.04597
    to understand the architecture of the model. The model is highly configurable. See ``BaseUNet`` to 
    see how to configure the model.
    """
    def __init__(self, architecture_config, training_config, callbacks_config):
        super(AbstractUNetModel, self).__init__(architecture_config, training_config, callbacks_config)
        self.custom_objects = {
            'mixed_iou_cross_entropy_loss': self.build_loss(**self.architecture_config['loss_params']),
            'jaccard_index': jaccard_index,
            'dice_coef': dice_coef
        }

    def build_model(self, **model_params):
        """See base class. Uses ``BaseUnet`` as its model architecture."""
        unet = BaseUNet()
        return unet.build_model(**model_params)


class ResnetUNetModel(AbstractUNetModel):
    """A UNet model that does uses the Resnet101 image classification model as its pre-trained encoder.
    The decoder is the same as before. The model can be used with or without the pre-trained weights for
    the encoder.
    """
    def __init__(self, architecture_config, training_config, callbacks_config):
        super(AbstractUNetModel, self).__init__(architecture_config, training_config, callbacks_config)
        self.custom_objects = {
            'mixed_iou_cross_entropy_loss': self.build_loss(**self.architecture_config['loss_params']),
            'Scale': Scale,
            'jaccard_index': jaccard_index,
            'dice_coef': dice_coef
        }

    def build_model(self, **model_params):
        """See base class. Uses ``UNetResNet101`` as its model architecture."""
        unet = UNetResNet101()
        return unet.build_model(**model_params)