import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.utils.generic_utils import get_custom_objects

from seg_metrics import iou_score, f_score

SMOOTH = 1.

__all__ = [
    'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
    'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
]


# ============================== Jaccard Losses ==============================

def jaccard_loss(gt, pr):
    r"""Jaccard loss function for imbalanced datasets:

    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Jaccard loss in range [0, 1]

    """
    return 1 - iou_score(gt, pr)


def bce_jaccard_loss(gt, pr):
    r"""Sum of binary crossentropy and jaccard losses:

    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + jaccard_loss(A, B)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch (only for jaccard loss)

    Returns:
        loss

    """
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + jaccard_loss(gt, pr)
    return loss


def cce_jaccard_loss(gt, pr):
    r"""Sum of categorical crossentropy and jaccard losses:

    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + jaccard_loss(A, B)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        loss

    """
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return 1 * cce + jaccard_loss(gt, pr)


# Update custom objects
get_custom_objects().update({
    'jaccard_loss': jaccard_loss,
    'bce_jaccard_loss': bce_jaccard_loss,
    'cce_jaccard_loss': cce_jaccard_loss,
})


# ============================== Dice Losses ================================

def dice_loss(gt, pr):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - f_score(gt, pr)


def bce_dice_loss(gt, pr):
    r"""Sum of binary crossentropy and dice losses:

    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + dice_loss(A, B)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss

    """
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + dice_loss(gt, pr)
    return loss


def cce_dice_loss(gt, pr):
    r"""Sum of categorical crossentropy and dice losses:

    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + dice_loss(A, B)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss

    """
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return 1 * cce + dice_loss(gt, pr)


# Update custom objects
get_custom_objects().update({
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'cce_dice_loss': cce_dice_loss,
})
