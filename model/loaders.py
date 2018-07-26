import os

import numpy as np
from attrdict import AttrDict

import model.transforms as transforms
from config import MEAN, STD
from model.connections import Transformer
from model.transforms import Augmenter, fast_seq, fix_shape_after_transform, fix_shape_before_transform, image_seq
from model.utils import Iterator, load_image, load_joblib


class ImageDataGenerator(Iterator):
    def __init__(self, x, image_transform, image_augment_with_target,
                 mask_transform, image_augment, iter_params):
        Iterator.__init__(self, len(x),
                          batch_size=iter_params.batch_size,
                          shuffle=iter_params.shuffle,
                          seed=iter_params.seed)
        self.x = x
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batch(index_array)

    def _get(self, index):
        image = load_image(self.x[index], color_mode='rgb')
        if self.image_augment is not None:
            image = self.image_augment(image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, None

    def _get_batch(self, index_array):
        x, y = self._get(index_array[0])
        batch_x = np.zeros(shape=(len(index_array),) + x.shape)
        batch_x[0] = x
        if y is not None:
            batch_y = np.zeros(shape=(len(index_array),) + y.shape)
            batch_y[0] = y
        else:
            batch_y = None

        for i in range(1, len(index_array)):
            x, y = self._get(index_array[i])
            batch_x[i] = x
            if y is not None:
                batch_y[i] = y

        if batch_y is None:
            return batch_x

        return batch_x, batch_y


class ImageDataGeneratorXY(ImageDataGenerator):
    def __init__(self, x, y,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment, iter_params):

        ImageDataGenerator.__init__(self, x,
                                    image_transform, image_augment_with_target,
                                    mask_transform, image_augment, iter_params)
        self.y = y

    def _get(self, index):
        image = load_image(self.x[index], color_mode='rgb')
        mask = load_image(self.y[index], color_mode='gray')
        mask = fix_shape_before_transform(mask)
        if self.image_augment_with_target is not None:
            image, mask = self.image_augment_with_target(image, mask)
        if self.image_augment is not None:
            image = self.image_augment(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        if self.image_transform is not None:
            image = self.image_transform(image)
        mask = fix_shape_after_transform(mask)
        return image, mask


class ImageDataGeneratorXYDistances(ImageDataGeneratorXY):
    def _get(self, index):
        image = load_image(self.x[index], color_mode='rgb')
        mask_filepath = self.y[index]
        mask = load_image(mask_filepath, color_mode='gray')
        distance_filepath = mask_filepath.replace("/masks/", "/distances/")
        distance_filepath = os.path.splitext(distance_filepath)[0]
        size_filepath = distance_filepath.replace("/distances/", "/sizes/")
        dist = load_joblib(distance_filepath)
        dist = dist.astype(np.uint16)
        sizes = load_joblib(size_filepath).astype(np.uint16)
        sizes = np.sqrt(sizes).astype(np.uint16)

        if self.image_augment_with_target is not None:
            image, mask, dist, sizes = self.image_augment_with_target(image, mask, dist, sizes)
        if self.image_augment is not None:
            image = self.image_augment(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            dist = self.mask_transform(dist)
            sizes = self.mask_transform(sizes)
            mask = np.concatenate((mask, dist, sizes))
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, mask


class AbstractImageLoader(Transformer):
    __out__ = ('datagen', 'validation_datagen')

    def __init__(self, name,
                 loader_params,
                 dataset_params,
                 image_transform=None,
                 mask_transform=None,
                 image_augment_with_target_train=None,
                 image_augment_with_target_inference=None,
                 image_augment_train=None,
                 image_augment_inference=None):
        super(AbstractImageLoader, self).__init__(name=name)
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_augment_with_target_train = image_augment_with_target_train
        self.image_augment_with_target_inference = image_augment_with_target_inference
        self.image_augment_train = image_augment_train
        self.image_augment_inference = image_augment_inference

    def __transform__(self, x, y=None, x_valid=None, y_valid=None, train_mode=True):
        if train_mode and y is not None:
            gen, steps = self._get_datagen(x, y, True, self.dataset_params.distances, self.loader_params.training)
        else:
            gen, steps = self._get_datagen(x, None, False, False, self.loader_params.inference)

        if x_valid is not None and y_valid is not None:
            val_gen, val_steps = self._get_datagen(x_valid, y_valid, False,
                                                   self.dataset_params.distances, self.loader_params.inference)
        else:
            val_gen = None
            val_steps = None
        return {'datagen': (gen, steps),
                'validation_datagen': (val_gen, val_steps)}

    def _get_datagen(self, x, y, train_mode, distances, loader_params):
        if train_mode or y is not None:
            generator = ImageDataGeneratorXYDistances if distances else ImageDataGeneratorXY
            datagen = generator(x, y,
                                image_augment=self.image_augment_train,
                                image_augment_with_target=self.image_augment_with_target_train,
                                mask_transform=self.mask_transform,
                                image_transform=self.image_transform,
                                iter_params=loader_params
                                )
        else:
            datagen = ImageDataGenerator(x,
                                         image_augment=self.image_augment_inference,
                                         image_augment_with_target=self.image_augment_with_target_inference,
                                         mask_transform=self.mask_transform,
                                         image_transform=self.image_transform,
                                         iter_params=loader_params)
        steps = len(datagen)
        return datagen, steps


def normalize_resize(h, w, mean=0., std=1.):
    return transforms.Compose([transforms.Resize((h, w)),
                               transforms.Normalize(mean=mean, std=std)])


class ImageLoader(AbstractImageLoader):
    def __init__(self, name, loader_params, dataset_params):
        super(ImageLoader, self).__init__(name,
                                          loader_params,
                                          dataset_params,
                                          image_transform=normalize_resize(dataset_params['h'], dataset_params['w'], MEAN,
                                                                           STD),
                                          mask_transform=normalize_resize(dataset_params['h'], dataset_params['w']),
                                          image_augment_with_target_train=Augmenter(fast_seq),
                                          image_augment_train=Augmenter(image_seq))


class ImageLoaderTest(AbstractImageLoader):
    def __init__(self, name, loader_params, dataset_params):
        super(ImageLoaderTest, self).__init__(name,
                                              loader_params,
                                              dataset_params,
                                              image_transform=normalize_resize(dataset_params['h'], dataset_params['w'], MEAN,
                                                                               STD),
                                              mask_transform=normalize_resize(dataset_params['h'], dataset_params['w']))
