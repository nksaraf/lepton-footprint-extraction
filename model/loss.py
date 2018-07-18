from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np


def weighted_cross_entropy(target, output, weights_function=None):
    """Calculate weighted Cross Entropy loss for multiple classes.

    This function calculates torch.nn.CrossEntropyLoss(), but each pixel loss is weighted.
    Target for weights is defined as a part of target, in target[:, 1:, :, :].
    If weights_function is not None weights are calculated by applying this function on target[:, 1:, :, :].
    If weights_function is None weights are taken from target[:, 1, :, :].

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x (1 + K) x H x W). Where K is number of different weights.
        weights_function (function, optional): Function applied on target for weights.

    Returns:
        torch.Tensor: Loss value.

    """
    if target.shape[3] == 1:
        weights = K.ones_like(target)
    elif target.shape[3] == 2:
        weights = target[:, :, :, 1]
    else:
        weights = weights_function(target[:, :, :, 1:])
    target = target[:, :, :, 0]
    loss_per_pixel = K.binary_crossentropy(K.flatten(target), K.flatten(output))
    return K.mean(loss_per_pixel * K.flatten(weights))


def get_weights(target, w0, sigma, imsize):
    """
    w1 is temporarily torch.ones - it should handle class imbalance for the whole dataset
    """
    C = K.sqrt(imsize[0] * imsize[1]) / 2.
    distances = target[:, :, :, 0]
    sizes = target[:, :, :, 1]

    w1 = np.ones(distances.shape)  # TODO: fix it to handle class imbalance
    size_weights = _get_size_weights(sizes, C)
    distance_weights = _get_distance_weights(distances, w1, w0, sigma)
    weights = distance_weights * size_weights
    return weights


def _get_distance_weights(d, w1, w0, sigma):
    weights = w1 + w0 * K.exp(-(d ** 2.) / (sigma ** 2.))
    weights[d == 0] = 1
    return weights


def _get_size_weights(sizes, C):
    sizes_ = sizes.copy()
    sizes_[sizes == 0] = 1
    size_weights = C / sizes_
    size_weights[sizes_ == 1] = 1
    return size_weights


def mixed_iou_cross_entropy_loss(output, target, iou_weight=0.5, iou_loss_func=None,
                                  cross_entropy_weight=0.5, cross_entropy_loss_func=None):
    """Calculate mixed Dice and Cross Entropy Loss.

    Args:
        output: Model output of shape (N x H x W x 1).
        target:
            Target of shape (N x H x W x (1 + K).
            Where K is number of different weights for Cross Entropy.
        iou_weight (float, optional): Weight of Dice loss. Defaults to 0.5.
        iou_loss_func (function, optional): Dice loss function. If None multiclass_dice_loss() is being used.
        cross_entropy_weight (float, optional): Weight of Cross Entropy loss. Defaults to 0.5.
        cross_entropy_loss_func (function, optional):
            Cross Entropy loss function.
            If None torch.nn.CrossEntropyLoss() is being used.
        iou_smooth (float, optional): Smoothing factor for Dice loss. Defaults to 0.

    Returns:
        float: Loss value.

    """
    output = K.cast(output, dtype='float32')
    iou_target = K.cast(target[:, :, :, 0], dtype='float32')
    cross_entropy_target = K.cast(target, dtype='float32')
    if cross_entropy_loss_func is None:
        cross_entropy_loss_func = binary_crossentropy
        cross_entropy_target = iou_target
    if iou_loss_func is None:
        iou_loss_func = jaccard_index_loss

    iou_loss = iou_weight * iou_loss_func(iou_target, output)
    cross_entropy_loss = cross_entropy_weight * cross_entropy_loss_func(cross_entropy_target, output)
    return iou_loss + cross_entropy_loss


def dice_coef(y_true, y_pred, smooth=1.):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return ((2. * intersection) + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)


def dice_coef_loss(y_true, y_pred, smooth=1.):
    return 1 - dice_coef(y_true, y_pred, smooth)


def jaccard_index(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jaccard_index_loss(y_true, y_pred, smooth=1., log=True):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is useful for unbalanced data sets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disappearing
    gradient.

    """
    if log:
        return -K.log(jaccard_index(y_true, y_pred, smooth))
    else:
        return (1. - jaccard_index(y_true, y_pred, smooth)) * smooth