import numpy as np
from matplotlib import pyplot as plt
import logging
import random
# from segment import takeGradientMag, getDisplayGradient

# IMAGES_TRAIN_DIR = 'data/train/images'
# MASKS_TRAIN_DIR = 'data/train/masks'
# IMAGES_VAL_DIR = 'data/val/images'
# MASKS_VAL_DIR = 'data/val/masks'
# IMAGES_TEST_DIR = 'data/test/images'
# MASKS_TEST_DIR = 'data/test/masks'

PROJECT_NAME = 'lepton-building-extraction'

def init_logger():
    logger = logging.getLogger(PROJECT_NAME)
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)
    ch_va.setFormatter(fmt=message_format)
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger(PROJECT_NAME)

def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed

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

def display_images(images, rows, columns):
    fig=plt.figure()
    for i in range(1, columns*rows +1):
        img = images[i-1]
        if img.shape[-1] == 1:
            img = np.reshape(img, (img.shape[:-1]))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='gist_gray')
        else:
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
    plt.show()

def display_image_mask(image, mask):
    display_images([image, mask], 1, 2)

def edge_detection(image, mask):
    # gradientImage = takeGradientMag(batch_image[0])
    # display = getDisplayGradient(gradientImage)
    image = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image,50, 255)
    print(edges.shape)
    print(edges)
    display_images([image, edges, mask], 1, 3)

def masked_display(image, mask):
    fig = plt.figure()
    plt.imshow((image * 255).astype(np.uint8), interpolation='nearest')
    plt.imshow(mask, 'binary', interpolation='none', alpha=0.4)
    plt.show()

def three_by_three(images, predictions, masks, save=False):
    # predictions = predictions.reshape(predictions.shape[:-1])
    plt.figure(figsize=(10, 8))
    for i in range(3):
        plt.subplot(3, 3, ((i * 3) + 1))
        plt.imshow(images[i])
        if i == 0:
            plt.title('Input image (256x256)')
        plt.tick_params(**no_labels)
        plt.subplot(3, 3, ((i * 3) + 2))
        plt.imshow(predictions[i], cmap='gist_gray')
        if i == 0:
            plt.title('Predicted building mask')
        plt.tick_params(**no_labels)
        plt.subplot(3, 3, ((i * 3) + 3))
        plt.imshow(masks[i], cmap='gist_gray')
        if i == 0:
            plt.title('Ground truth building mask')
        plt.tick_params(**no_labels)
    plt.tight_layout()
    if save:
        plt.savefig('data/predictions/pred_mask_{}.jpeg'.format(random.randint(1000000, 9999999)))
    plt.show()

def save_image_mask(image, prediction, mask):
    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    plt.xticks([], [])
    fig.add_subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gist_gray')
    plt.xticks([], [])
    fig.add_subplot(1, 3, 3)
    plt.imshow(mask, cmap='gist_gray')
    plt.xticks([], [])
    plt.show()
