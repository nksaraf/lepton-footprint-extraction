import multiprocessing.pool
from functools import partial
from keras.preprocessing.image import Iterator
import warnings
import numpy as np
import keras.backend as K
import keras
from google.cloud import storage
import os

# rewrite of flow_from_directory
# https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py


def flow_from_google_storage(imageDataGen, project, bucket, directory,
                             target_size=(256, 256), color_mode='rgb',
                             classes=None, class_mode='categorical',
                             batch_size=32, shuffle=True, seed=None,
                             save_to_dir=None,
                             save_prefix='',
                             save_format='png',
                             follow_links=False,
                             subset=None,
                             interpolation='nearest'):
    """Takes the path to a directory, and generates batches of augmented/normalized data.
    # Arguments
            directory: path to the target directory.
             It should contain one subdirectory per class.
             Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator.
            See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) for more details.
            target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
             The dimensions to which all images found will be resized.
            color_mode: one of "grayscale", "rbg". Default: "rgb".
             Whether the images will be converted to have 1 or 3 color channels.
            classes: optional list of class subdirectories (e.g. `['dogs', 'cats']`). Default: None.
             If not provided, the list of classes will be automatically
             inferred from the subdirectory names/structure under `directory`,
             where each subdirectory will be treated as a different class
             (and the order of the classes, which will map to the label indices, will be alphanumeric).
             The dictionary containing the mapping from class names to class
             indices can be obtained via the attribute `class_indices`.
            class_mode: one of "categorical", "binary", "sparse", "input" or None. Default: "categorical".
             Determines the type of label arrays that are returned: "categorical" will be 2D one-hot encoded labels,
             "binary" will be 1D binary labels, "sparse" will be 1D integer labels, "input" will be images identical
             to input images (mainly used to work with autoencoders).
             If None, no labels are returned (the generator will only yield batches of image data, which is useful to use
             `model.predict_generator()`, `model.evaluate_generator()`, etc.).
              Please note that in case of class_mode None,
               the data still needs to reside in a subdirectory of `directory` for it to work correctly.
            batch_size: size of the batches of data (default: 32).
            shuffle: whether to shuffle the data (default: True)
            seed: optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None). This allows you to optionally specify a directory to which to save
             the augmented pictures being generated (useful for visualizing what you are doing).
            save_prefix: str. Prefix to use for filenames of saved pictures (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: whether to follow symlinks inside class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
             `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to resample the image if the
             target size is different from that of the loaded image.
             Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
             If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
             supported. If PIL version 3.4.0 or newer is installed, `"box"` and
             `"hamming"` are also supported. By default, `"nearest"` is used.
    # Returns
        A DirectoryIterator yielding tuples of `(x, y)` where `x` is a numpy array containing a batch
        of images with shape `(batch_size, *target_size, channels)` and `y` is a numpy array of corresponding labels.
    """
    return GoogleStorageIterator(project, bucket,
                                 directory, imageDataGen,
                                 target_size=target_size, color_mode=color_mode,
                                 classes=classes, class_mode=class_mode,
                                 data_format=imageDataGen.data_format,
                                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                                 save_to_dir=save_to_dir,
                                 save_prefix=save_prefix,
                                 save_format=save_format,
                                 follow_links=follow_links,
                                 subset=subset,
                                 interpolation=interpolation)


class GoogleStorageIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, project, bucket, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

        # init gs
        self.storage_client = storage.Client(project)
        self.bucket = self.storage_client.get_bucket(bucket)
        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            labels_folder_iter = self.bucket.list_blobs(delimiter="/", prefix=self.directory)
            list(labels_folder_iter)  # populate labels_folder_iter
            classes = [p[len(self.directory):-1] for p in sorted(labels_folder_iter.prefixes)]

        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(self._count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links,
                                   split=split)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(self.directory, subdir) for subdir in classes)))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))
        print(self.class_indices)

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(self.directory, subdir) for subdir in classes):
            results.append(pool.apply_async(self._list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats, split,
                                             self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)

        pool.close()
        pool.join()
        super(GoogleStorageIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            blob = self.bucket.get_blob(os.path.join(self.directory, fname), self.storage_client)
            img = self.load_img_from_string(blob.download_as_string(self.storage_client),
                                       grayscale=grayscale,
                                       target_size=self.target_size,
                                       interpolation=self.interpolation)

            x = keras.preprocessing.image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # TODO write save to gs
        # optionally save augmented images to disk for debugging purposes
#        if self.save_to_dir:
#            for i, j in enumerate(index_array):
#                img = keras.preprocessing.image.array_to_img(batch_x[i], self.data_format, scale=True)
#                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                  index=j,
#                                                                  hash=np.random.randint(1e7),
#                                                                  format=self.save_format)
#                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _count_valid_files_in_directory(self, directory, white_list_formats, split, follow_links):
        """Count files with extension in `white_list_formats` contained in directory.
        # Arguments
            directory: absolute path to the directory
                containing files to be counted
            white_list_formats: set of strings containing allowed extensions for
                the files to be counted.
            split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
                account a certain fraction of files in each directory.
                E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
                of images in each directory.
            follow_links: boolean.
        # Returns
            the count of files with extension in `white_list_formats` contained in
            the directory.
        """
        num_files = len(list(self._iter_valid_files(directory, white_list_formats, follow_links)))
        if split:
            start, stop = int(split[0] * num_files), int(split[1] * num_files)
        else:
            start, stop = 0, num_files
        return stop - start

    def _iter_valid_files(self, directory, white_list_formats, follow_links):
        """Count files with extension in `white_list_formats` contained in directory.
        # Arguments
            directory: absolute path to the directory
                containing files to be counted
            white_list_formats: set of strings containing allowed extensions for
                the files to be counted.
            follow_links: boolean.
        # Yields
            tuple of (root, filename) with extension in `white_list_formats`.
        """
        def _recursive_list(subpath):
            # TODO should return all file path relative to subpath walk trhough any directory it find
            if subpath[-1] != '/':
                subpath = subpath + '/'
            iter_blobs = self.bucket.list_blobs(delimiter="/", prefix=subpath)
            blobs = list(iter_blobs)
            return sorted(map(lambda blob: (subpath, blob.name[len(subpath):]), blobs), key=lambda x: x[1])

        for root, fname in _recursive_list(directory):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname

    def _list_valid_filenames_in_directory(self, directory, white_list_formats, split,
                                           class_indices, follow_links):
        """List paths of files in `subdir` with extensions in `white_list_formats`.
        # Arguments
            directory: absolute path to a directory containing the files to list.
                The directory name is used as class label and must be a key of `class_indices`.
            white_list_formats: set of strings containing allowed extensions for
                the files to be counted.
            split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
                account a certain fraction of files in each directory.
                E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
                of images in each directory.
            class_indices: dictionary mapping a class name to its index.
            follow_links: boolean.
        # Returns
            classes: a list of class indices
            filenames: the path of valid files in `directory`, relative from
                `directory`'s parent (e.g., if `directory` is "dataset/class1",
                the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
        """
        dirname = os.path.basename(directory)

        if split:
            num_files = len(list(self._iter_valid_files(directory, white_list_formats, follow_links)))
            start, stop = int(split[0] * num_files), int(split[1] * num_files)
            valid_files = list(self._iter_valid_files(directory, white_list_formats, follow_links))[start: stop]
        else:
            valid_files = self._iter_valid_files(directory, white_list_formats, follow_links)

        classes = []
        filenames = []
        for root, fname in valid_files:
            classes.append(class_indices[dirname])
            absolute_path = os.path.join(root, fname)
            relative_path = os.path.join(dirname, os.path.relpath(absolute_path, directory))

            filenames.append(relative_path)

        return classes, filenames

    def load_img_from_string(self, img_string, grayscale=False, target_size=None,
                             interpolation='nearest'):
        from PIL import Image as pil_image
        import io
        _PIL_INTERPOLATION_METHODS = {
            'nearest': pil_image.NEAREST,
            'bilinear': pil_image.BILINEAR,
            'bicubic': pil_image.BICUBIC,
        }
        """Loads an image into PIL format.
        # Arguments
            path: Path to image file
            grayscale: Boolean, whether to load the image as grayscale.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        # Returns
            A PIL Image instance.
        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(io.BytesIO(img_string))
        if grayscale:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img