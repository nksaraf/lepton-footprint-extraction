from matplotlib import pyplot as plt
from model.connections import Transformer

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

# class Viewer(Transformer):
#     def __init__(self, name, rows, columns):
#         super(Viewer, self).__init__(name)
#         pass
#
#     def __transform__(self, images):
#         plt.figure(figsize=(10, 8))
#         for i in range(3):
#             plt.subplot(3, 3, ((i * 3) + 1))
#             plt.imshow(images[i])
#             if i == 0:
#                 plt.title('Input image (256x256)')
#             plt.tick_params(**no_labels)
#             plt.subplot(3, 3, ((i * 3) + 2))
#             plt.imshow(predictions[i], cmap='gist_gray')
#             if i == 0:
#                 plt.title('Predicted building mask')
#             plt.tick_params(**no_labels)
#             plt.subplot(3, 3, ((i * 3) + 3))
#             plt.imshow(masks[i], cmap='gist_gray')
#             if i == 0:
#                 plt.title('Ground truth building mask')
#             plt.tick_params(**no_labels)
#         plt.tight_layout()
#         if save:
#             plt.savefig('data/predictions/pred_mask_{}.jpeg'.format(random.randint(1000000, 9999999)))
#         plt.show()


class ThreeByThreeViewer(Transformer):
    __out__ = ('output', )

    def __transform__(self, images, masks, predictions):
        plt.figure(figsize=(10, 8))
        for i in range(3):
            plt.subplot(3, 3, ((i * 3) + 1))
            plt.imshow(images[i])
            if i == 0:
                plt.title('Input image (256x256)')
            plt.tick_params(**no_labels)
            plt.subplot(3, 3, ((i * 3) + 2))
            plt.imshow(predictions[i].reshape(predictions[i].shape[:-1]), cmap='gist_gray')
            if i == 0:
                plt.title('Predicted building mask')
            plt.tick_params(**no_labels)
            plt.subplot(3, 3, ((i * 3) + 3))
            plt.imshow(masks[i].reshape(masks[i].shape[:-1]), cmap='gist_gray')
            if i == 0:
                plt.title('Ground truth building mask')
            plt.tick_params(**no_labels)
        plt.tight_layout()
        plt.show()
        return {'output': None}
        # if save:
        #     plt.savefig('data/predictions/pred_mask_{}.jpeg'.format(random.randint(1000000, 9999999)))
        # plt.show()

