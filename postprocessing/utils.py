from __future__ import print_function
from matplotlib import pyplot as plt
from model.connections import Transformer
from descartes import PolygonPatch
import numpy as np
import random
import os

no_labels = dict(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )


def plt_image(plot, image, title=None):
    plt.subplot(*plot)
    if title is not None:
        plt.title(title)
    plt.imshow(image.astype(np.uint8))
    plt.tick_params(**no_labels)


def plt_mask(plot, mask, title=None):
    plt.subplot(*plot)
    if title is not None:
        plt.title(title)
    plt.imshow(mask.reshape(mask.shape[:-1]), cmap='gist_gray')
    plt.tick_params(**no_labels)


def plt_polygons(plot, polygons, image, color='red', title=None):
    ax = plt.subplot(*plot)
    if title is not None:
        plt.title(title)
    plt.imshow(image.astype(np.uint8))
    try:
        ax.add_patch(PolygonPatch(polygons, edgecolor=color, facecolor='none', linewidth=1))
    except:
        pass
    plt.tick_params(**no_labels)


class Viewer(Transformer):
    __out__ = ('status', )

    def __init__(self, name, save=False, job_dir=''):
        super(Viewer, self).__init__(name)
        self.save = save
        if self.save:
            assert len(job_dir) > 0
            pred_dir = os.path.join(job_dir, 'predictions')
            try:
                os.makedirs(pred_dir)
            except:
                pass
            self.dir = pred_dir

    def __transform__(self, image, mask, prediction):
        plt.figure(figsize=(10, 4))
        plt_image((1, 3, 1), image, title='Input image (256x256)')
        plt_mask((1, 3, 2), mask, title='Ground truth mask')
        plt_mask((1, 3, 3), prediction, title='Predicted building mask')
        plt.tight_layout()
        if self.save:
            path = os.path.join(self.dir, 'prediction_{}.png'.format(random.randint(1000, 9999)))
            self.log('Saving to {}'.format(path))
            plt.savefig(path)
        plt.show()
        return {'status': True}


class ThreeByThreeViewer(Transformer):
    __out__ = ('status', )

    def __init__(self, name, save=False, job_dir=''):
        super(ThreeByThreeViewer, self).__init__(name)
        self.save = save
        if self.save:
            assert len(job_dir) > 0
            pred_dir = os.path.join(job_dir, 'predictions')
            try:
                os.makedirs(pred_dir)
            except:
                pass
            self.dir = pred_dir

    def __transform__(self, images, masks, predictions, evaluation):
        plt.figure(figsize=(10, 8))
        plt.suptitle('binary accuracy: {:.2f}, jaccard index: {:.2f}, dice coef: {:.2f}'.format(evaluation[1], evaluation[2], evaluation[3]))
        for i in range(3):
            plt_image((3, 3, ((i * 3) + 1)), images[i])
            plt_mask((3, 3, ((i * 3) + 2)), masks[i])
            plt_mask((3, 3, ((i * 3) + 3)), predictions[i])
        plt.tight_layout()
        if self.save:
            path = os.path.join(self.dir, 'prediction_{}.png'.format(random.randint(1000, 9999)))
            self.log('Saving to {}'.format(path))
            plt.savefig(path)
        plt.show()
        return {'status': True}


class PolygonViewer(ThreeByThreeViewer):
    def __transform__(self, images, truths, predictions, evaluation):
        plt.figure(figsize=(6, 8))
        plt.suptitle(
            'binary accuracy: {:.2f}, jaccard index: {:.2f}, dice coef: {:.2f}'.format(evaluation[1] * 100,
                                                                                       evaluation[2] * 100,
                                                                                       evaluation[3] * 100))
        for i in range(3):
            plt_image((3, 3, ((i * 3) + 1)), images[i])
            plt_polygons((3, 3, ((i * 3) + 2)), truths[i], images[i], 'blue')
            plt_polygons((3, 3, ((i * 3) + 3)), predictions[i], images[i], 'red')
        plt.tight_layout()
        if self.save:
            path = os.path.join(self.dir, 'prediction_{}.png'.format(random.randint(1000, 9999)))
            self.log('Saving to {}'.format(path))
            plt.savefig(path)
        plt.show()
        return {'status': True}
