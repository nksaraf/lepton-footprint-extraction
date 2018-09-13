import os
from collections import namedtuple

import numpy as np
from attrdict import AttrDict

import model.transforms as transforms
from config import MEAN, STD
from model.connections import Transformer
from model.transforms import Augmenter, fast_seq, fix_shape_after_transform, fix_shape_before_transform, image_seq
from model.utils import Iterator, load_image, load_joblib

def_iter_params = {
    'batch_size': 1,
    'shuffle': False,
    'seed': 0
}

ImageModifier = namedtuple('ImageModifier',
                           ['image_transform', 'mask_transform', 'image_augment', 'image_augment_with_mask'])
ImageModifier.__new__.__defaults__ = (None, None, None, None)


class BatchGenerator(Iterator):

    def __init__(self, x, y=None, z=None, iter_params=def_iter_params):
        Iterator.__init__(self,
                          len(x),
                          batch_size=iter_params.batch_size,
                          shuffle=iter_params.shuffle,
                          seed=iter_params.seed)
        self.x = x
        self.y = y
        self.z = z

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batch(index_array)

    def _load(self, index):
        x = self.x[index] if self.x is not None else None
        y = self.y[index] if self.y is not None else None
        z = self.z[index] if self.z is not None else None
        return x, y, z

    def _get(self, index):
        return self._load(index)

    def _get_batch(self, index_array):
        x, y, z = self._get(index_array[0])
        size = len(index_array)
        batch = {}
        for i in ['x', 'y', 'z']:
            a = locals()[i]
            if a is not None:
                if isinstance(a, np.ndarray):
                    batch[i] = np.zeros(
                        shape=(size, ) + a.shape, dtype=a.dtype)
                    batch[i][0] = a
                else:
                    batch[i] = [a] * size
            else:
                batch[i] = None

        for i in range(1, size):
            x, y, z = self._get(index_array[i])
            for k in ['x', 'y', 'z']:
                a = locals()[k]
                if a is not None:
                    batch[k][i] = a

        batches = tuple()
        for i in ['x', 'y', 'z']:
            if batch[i] is not None:
                batches += (batch[i], )

        return batches


class ImageGenerator(BatchGenerator):

    def __init__(self, x,
                 y=None,
                 z=None,
                 iter_params=def_iter_params,
                 modifier=ImageModifier()):
        super(ImageGenerator, self).__init__(x, y, z, iter_params)
        self.modifier = modifier

    def _get(self, index):
        image, mask, meta = self._load(index)

        image = fix_shape_before_transform(image)
        if mask is not None:
            mask = fix_shape_before_transform(mask)
            if self.modifier.image_augment_with_mask is not None:
                image, mask = self.modifier.image_augment_with_mask(
                    image, mask)
            if self.modifier.mask_transform is not None:
                mask = self.modifier.mask_transform(mask)
            mask = fix_shape_after_transform(mask)

        if self.modifier.image_augment is not None:
            image = self.modifier.image_augment(image)
        if self.modifier.image_transform is not None:
            image = self.modifier.image_transform(image)
        image = fix_shape_after_transform(image)

        return image, mask, meta


class ImageFromPathGenerator(ImageGenerator):

    def _load(self, index):
        x = load_image(self.x[index], color_mode='rgb')
        y = load_image(
            self.y[index], color_mode='gray') if self.y is not None else None
        z = self.z[index] if self.z is not None else None
        return x, y, z


class ImageLoader(Transformer):
    __out__ = ('generator', )

    def __init__(self, name, path_mode, loader_params, modifier):
        super(ImageLoader, self).__init__(name)
        self.loader_params = AttrDict(loader_params)
        self.generator = ImageFromPathGenerator if path_mode else ImageGenerator
        self.modifier = modifier

    def __transform__(self, x, y=None, z=None):
        gen = self.generator(x, y, z, modifier=self.modifier,
                             iter_params=self.loader_params)
        steps = len(gen)
        return {'generator': (gen, steps)}


def normalize_resize(h, w, mean=0., std=1.):
    return transforms.Compose([transforms.Resize((h, w)),
                               transforms.Normalize(mean=mean, std=std)])


class ImageLoaderNormalizedAugmented(ImageLoader):

    def __init__(self, name, path_mode, loader_params, dataset_params):
        super(ImageLoaderNormalized, self).__init__(name,
                                                    path_mode,
                                                    loader_params,
                                                    modifier=ImageModifier(
                                                        image_transform=normalize_resize(dataset_params['h'], dataset_params['w'], MEAN,
                                                                                         STD),
                                                        mask_transform=normalize_resize(
                                                            dataset_params['h'], dataset_params['w']),
                                                        image_augment_with_mask=Augmenter(
                                                            fast_seq),
                                                        image_augment=Augmenter(image_seq))
                                                    )


class ImageLoaderAugmented(ImageLoader):

    def __init__(self, name, path_mode, loader_params, dataset_params):
        super(ImageLoaderBasic, self).__init__(name,
                                               path_mode,
                                               loader_params,
                                               modifier=ImageModifier(
                                                   image_transform=transforms.Resize(
                                                       (dataset_params['h'], dataset_params['w'])),
                                                   mask_transform=normalize_resize(
                                                       dataset_params['h'], dataset_params['w']),
                                                   image_augment_with_mask=Augmenter(
                                                       fast_seq),
                                                   image_augment_train=Augmenter(image_seq))
                                               )


class ImageLoaderNormalizedInference(ImageLoader):

    def __init__(self, name, path_mode, loader_params, dataset_params):
        super(ImageLoaderNormalizedInference, self).__init__(name, path_mode,
                                                             loader_params,
                                                             modifier=ImageModifier(
                                                                 image_transform=normalize_resize(dataset_params['h'], dataset_params['w'], MEAN,
                                                                                                  STD),
                                                                 mask_transform=normalize_resize(dataset_params['h'], dataset_params['w']))
                                                             )


class ImageLoaderInference(ImageLoader):

    def __init__(self, name, path_mode, loader_params, dataset_params):
        super(ImageLoaderInference, self).__init__(name,
                                                   path_mode,
                                                   loader_params,
                                                   modifier=ImageModifier(
                                                       image_transform=transforms.Resize(
                                                           (dataset_params['h'], dataset_params['w'])),
                                                       mask_transform=normalize_resize(dataset_params['h'], dataset_params['w']))
                                                   )
