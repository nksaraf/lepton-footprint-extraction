import os

from connections import Wire
from model.loss import dice_coef, jaccard_index

# CONFIG_FILEPATH = 'pipeline.yaml'
#
# params = load_params(CONFIG_FILEPATH)

# PARAMS = params
SIZE_COLUMNS = ['height', 'width']
SEED = 6581
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'
Y_COLUMNS_SCORING = ['ImageId']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

image_h, image_w = (256, 256)


# GLOBAL_CONFIG = {
#     'exp_root': params.working_dir,
#     'num_workers': params.num_workers,
#     'num_classes': 2,
#     'image_shape': (params.image_h, params.image_w),
#     'batch_size_train': params.batch_size_train,
#     'batch_size_inference': params.batch_size_inference,
#     'loader_mode': params.loader_mode,
#     'stream_mode': params.stream_mode
# }

def create_config(job_dir,
                  data_dir,
                  batch_size_train,
                  batch_size_val,
                  epochs,
                  dev_mode):

    return Wire({
        'dev_mode': dev_mode,
        'data_dir': data_dir,
        'seed': 6581,
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
                'w': image_w,
                'distances': False,
            },
            'loader_params': {
                'training': {
                    'batch_size': batch_size_train,
                    'shuffle': True,
                    'seed': SEED
                },
                'inference': {
                    'batch_size': batch_size_val,
                    'shuffle': False,
                    'seed': SEED
                },
            },
        },
        'unet': {
            'architecture_config': {
                'model_params': {
                    'input_shape': (image_h, image_w),
                    'dropout': 0.1,
                    'in_channels': 3,
                    'out_channels': 1,
                    'l2_reg': 0.0001,
                    'is_deconv': True,
                    'resnet_pretrained': False,
                    'num_filters': 32,
                    'resnet_weights_path': os.path.join(data_dir, 'resnet101_weights.h5')
                },
                'optimizer_params': {
                    'lr': 0.0001,
                    'decay': 0.1,
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
                    'filepath': os.path.join('checkpoints', 'checkpoint.{epoch:02d}-{val_loss:.2f}.h5'),
                    'period': 1,
                    'save_best_only': True,
                    'verbose': 1,
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
        },
        # 'postprocessor': {
        #     'mask_dilation': {
        #         'dilate_selem_size': params.dilate_selem_size
        #     },
        #     'mask_erosion': {
        #         'erode_selem_size': params.erode_selem_size
        #     },
        #     'prediction_crop': {
        #         'h_crop': params.crop_image_h,
        #         'w_crop': params.crop_image_w
        #     },
        #     'scoring_model': params.scoring_model,
        #     'lightGBM': {
        #         'model_params': {
        #             'learning_rate': params.lgbm__learning_rate,
        #             'boosting_type': 'gbdt',
        #             'objective': 'regression',
        #             'metric': 'regression_l2',
        #             'sub_feature': 1.0,
        #             'num_leaves': params.lgbm__num_leaves,
        #             'min_data': params.lgbm__min_data,
        #             'max_depth': params.lgbm__max_depth
        #         },
        #         'training_params': {
        #             'number_boosting_rounds': params.lgbm__number_of_trees,
        #             'early_stopping_rounds': params.lgbm__early_stopping
        #         },
        #         'train_size': params.lgbm__train_size,
        #         'target': params.lgbm__target
        #     },
        #     'random_forest': {
        #         'train_size': params.lgbm__train_size,
        #         'target': params.lgbm__target
        #     },
        #     'nms': {
        #         'iou_threshold': params.nms__iou_threshold,
        #         'num_threads': params.num_threads
        #     },
        # }
    })
