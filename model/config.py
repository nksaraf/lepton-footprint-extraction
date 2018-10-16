"""Creates configuration dictionaries for building models, training and prediction. Also
contains constants to be used in the project.
"""

import os

from model.connections import Wire
from model.loss import dice_coef, jaccard_index

SIZE_COLUMNS = ['height', 'width']
SEED = 6581
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'
Y_COLUMNS_SCORING = ['ImageId']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

image_h, image_w = (256, 256)

def unet_resnet_config(job_dir, epochs):
    return Wire({
        'architecture_config': {
            'model_params': {
                'input_shape': (image_h, image_w),
                'dropout': 0.1,
                'in_channels': 3,
                'out_channels': 1,
                'l2_reg': 0.0001,
                'is_deconv': True,
                'resnet_pretrained': True,
                'num_filters': 32,
                'resnet_weights_path': 'data/resnet101_weights.h5'
            },
            'optimizer_params': {
                'lr': 0.001,
                'decay': 0.0,
            },
            'compiler_params': {
                'metrics': ['binary_accuracy', jaccard_index, dice_coef]
            },
            'loss_params': {
                'loss_weights': {
                    'bce_mask': 1.0,
                    'iou_mask': 1.0,
                },
                'bce': {
                    'w0': 50,
                    'sigma': 10,
                    'imsize': (image_h, image_w)
                },
                'iou': {
                    'smooth': 1.,
                    'log': True
                },
            }
        },
        'training_config': {
            'epochs': epochs,
        },
        'callbacks_config': {
            'model_checkpoint': {
                'job_dir': job_dir,
                'filepath': os.path.join('checkpoints',
                                         'checkpoint.{epoch:02d}-{val_loss:.2f}.h5'),
                'period': 1,
                'save_best_only': True,
                'verbose': 1,
                'save_weights_only': True
            },
            'plateau_lr_scheduler': {
                'factor': 0.3,
                'patience': 30
            },
            'progbar_logger': {
                'count_mode': 'steps',
            },
            'early_stopping': {
                'patience': 30,
            },
            'tensor_board': {
                'log_dir': os.path.join(job_dir, 'logs')
            }
        },
    })

def base_unet_config(job_dir, epochs):
    return Wire({
        'architecture_config': {
            'model_params': {
                'input_shape': (image_h, image_w),
                'dropout': 0.1,
                'in_channels': 3,
                'out_channels': 1,
                'l2_reg': 0.0001,
                'is_deconv': True,
                'num_filters': 32,
                'depth': 4,
                'batch_norm': True
            },
            'optimizer_params': {
                'lr': 0.001,
                'decay': 0.0,
            },
            'compiler_params': {
                'metrics': ['binary_accuracy', jaccard_index, dice_coef]
            },
            'loss_params': {
                'loss_weights': {
                    'bce_mask': 1.0,
                    'iou_mask': 1.0,
                },
                'bce': {
                    'w0': 50,
                    'sigma': 10,
                    'imsize': (image_h, image_w)
                },
                'iou': {
                    'smooth': 1.,
                    'log': True
                },
            }
        },
        'training_config': {
            'epochs': epochs,
        },
        'callbacks_config': {
            'model_checkpoint': {
                'job_dir': job_dir,
                'filepath': os.path.join('checkpoints',
                                         'checkpoint.{epoch:02d}-{val_loss:.2f}.h5'),
                'period': 1,
                'save_best_only': True,
                'verbose': 1,
                'save_weights_only': True
            },
            'plateau_lr_scheduler': {
                'factor': 0.3,
                'patience': 30
            },
            'progbar_logger': {
                'count_mode': 'steps',
            },
            'early_stopping': {
                'patience': 30,
            },
            'tensor_board': {
                'log_dir': os.path.join(job_dir, 'logs')
            }
        },
    })

model_configs = {
    'unet_resnet': unet_resnet_config,
    'unet': base_unet_config
}

def model_config(model, job_dir, epochs):
    return model_configs[model](job_dir, epochs)

def create_config(job_dir,
                  batch_size,
                  epochs,
                  model,
                  seed):
    return Wire({
        'seed': seed,
        'job_dir': job_dir,
        'xy_splitter': {
            'x_column': X_COLUMN,
            'y_column': Y_COLUMN,
        },
        'mean': MEAN,
        'std': STD,
        'reader_single': {
            'x_columns': X_COLUMN,
            'y_columns': Y_COLUMN,
        },
        'loader': {
            'dataset_params': {
                'h': image_h,
                'w': image_w
            },
            'loader_params': {
                'batch_size': batch_size,
                'shuffle': False,
                'seed': seed
            },
        },
        'model_name': model,
        'model': model_config(model, job_dir, epochs)
    })
