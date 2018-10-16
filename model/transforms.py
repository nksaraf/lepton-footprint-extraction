from __future__ import division

import math
import numbers
import random
import types
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from imgaug import augmenters as iaa

""" Utility classes and functions to apply common transformations to images in bulk.
Also includes support for augmentation during training process."""


fast_seq = iaa.SomeOf((2, 4),
                      [iaa.Fliplr(0.75),
                       iaa.Flipud(0.75),
                       iaa.OneOf([iaa.Affine(rotate=0), iaa.Affine(rotate=90),
                                  iaa.Affine(rotate=180), iaa.Affine(rotate=270)]),
                       ], random_order=True)

image_seq = iaa.SomeOf((1, 2),
                       [iaa.Add((-25, 25)),
                        iaa.AddToHueAndSaturation((-25, 25)),
                        iaa.ContrastNormalization((0.9, 1.1))],
                       random_order=True)


INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'area': cv2.INTER_AREA,
}


class ImageTransform(object):
    __metaclass__ = ABCMeta

    def _setup(self):
        pass

    @abstractmethod
    def transform(self, img):
        return img

    def __call__(self, *images):
        self._setup()
        transformed = [self.transform(image) for image in images]
        if len(transformed) == 1:
            return transformed[0]
        else:
            return transformed


class Augmenter(ImageTransform):
    def __init__(self, augmenters):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        self.augmenters = augmenters
        self.seq_det = None

    def _setup(self):
        seq = iaa.Sequential(self.augmenters)
        seq.reseed()
        self.seq_det = seq.to_deterministic()

    def transform(self, img):
        return self.seq_det.augment_image(img)


class Compose(ImageTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` Transforms): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize(ImageTransform):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def transform(self, img):
        """
        Args:
            img (image): numpy array of dim (H, W, C).

        Returns:
            Tensor: Normalized image.
        """
        if isinstance(self.mean, list):
            assert len(img.shape) == 3 and img.shape[2] == len(self.mean), "Invalid casting"
        return ((img / 255.).astype(np.float32) - self.mean) / self.std


class Denormalize(ImageTransform):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def transform(self, img):
        """
        Args:
            img (image): numpy array of dim (H, W, C).

        Returns:
            Tensor: Normalized image.
        """
        if isinstance(self.mean, list):
            assert len(img.shape) == 3 and img.shape[2] == len(self.mean), "Invalid casting"
        return (((img * self.std) + self.mean) * 255).astype(np.uint8)


class Resize(ImageTransform):
    """Rescale the input image as numpy array to the given size.

    Args:
        size (height, width)
        interpolation (str, optional): Desired interpolation. Default is
            ``nearest``
    """

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        assert interpolation in INTERPOLATION_METHODS.keys(), "Invalid interpolation method"
        self.interpolation = interpolation

    def transform(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        return cv2.resize(img, self.size, interpolation=INTERPOLATION_METHODS[self.interpolation])


class CenterCrop(ImageTransform):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def transform(self, img):
        """
        Args:
            img (np.array): Image to be cropped.

        Returns:
            img: Cropped image.
        """
        h, w = img.shape[:2]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1 + th, x1:x1 + tw]


class Pad(ImageTransform):
    """Pad the given PIL.Image on all sides with the given "pad" value.

    Args:
        padding (int or sequence): Padding on each border. If a sequence of
            length 4, it is used to pad left, top, right and bottom borders respectively.
        fill: Pixel fill value. Default is 0.
    """

    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def transform(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """
        left, top, right, bottom = self.padding
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, self.fill)


class Lambda(ImageTransform):
    """Apply a user-defined lambda as a transform.

    Args:
        func (function): Lambda/function to be used for transform.
    """

    def __init__(self, func):
        assert isinstance(func, types.LambdaType)
        self.func = func

    def transform(self, img):
        return self.func(img)


class RandomCrop(ImageTransform):
    """Crop the given numpy image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def transform(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            left, top, right, bottom = self.padding
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        h, w, _ = img.shape
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[y1:y1 + th, x1:x1 + tw]


class RandomHorizontalFlip(ImageTransform):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def transform(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return cv2.flip(img, 0)
        return img


class RandomSizedCrop(ImageTransform):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: 'bilinear'
    """

    def __init__(self, size, interpolation='bilinear'):
        assert interpolation in INTERPOLATION_METHODS.keys(), "Invalid interpolation method"
        self.size = size
        self.interpolation = interpolation

    def transform(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img[y1: y1 + h, x1: x1 + w]
                assert (img.size == (h, w))
                return cv2.resize(img, (self.size, self.size), INTERPOLATION_METHODS[self.interpolation])

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class Flip(ImageTransform):
    """
        Flips image in given direction
        0 - horizontal, 1
    """
    HORIZONTAL = 0
    VERTICAL = 1
    BOTH = -1

    def __init__(self, direction):
        assert direction in [Flip.HORIZONTAL, Flip.VERTICAL, Flip.BOTH]
        self.direction = direction

    def transform(self, img):
        cv2.flip(img, self.direction)


def fix_shape_before_transform(img):
    if len(img.shape) == 2 or img.shape[2] == 3:
        return img
    elif img.shape[2] == 1:
        return img.reshape(img.shape[:2])
    else:
        return img


def fix_shape_after_transform(img):
    if len(img.shape) == 2:
        return img.reshape((img.shape[0], img.shape[1], 1))
    else:
        return img
