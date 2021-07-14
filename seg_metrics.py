import keras.backend as K


__all__ = [
    'iou_score', 'jaccard_score', 'f1_score', 'f2_score', 'dice_score',
    'get_f_score', 'get_iou_score', 'get_jaccard_score',
]

SMOOTH = 1.


# ============================ Jaccard/IoU score ============================


def iou_score(gt, pr):
    class_weights=1.; smooth=1.; per_image=True; threshold=None
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round

    Returns:
        IoU/Jaccard score in range [0, 1]

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    """
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    if threshold is not None:
        pr = K.greater(pr, threshold)
        pr = K.cast(pr, K.floatx())

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


def IOU(gt, pr):
    """Change default parameters of IoU/Jaccard score

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round

    Returns:
        ``callable``: IoU/Jaccard score
    """
    return iou_score(gt, pr)


# ============================== F/Dice - score ==============================

def f_score(gt, pr):
    class_weights=1; beta=1; smooth=1.; per_image=True; threshold=None
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round

    Returns:
        F-score in range [0, 1]

    """
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    if threshold is not None:
        pr = K.greater(pr, threshold)
        pr = K.cast(pr, K.floatx())

    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


def Dice(gt, pr):
    """Change default parameters of F-score score

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        beta: f-score coefficient
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round

    Returns:
        ``callable``: F-score
    """
    return f_score(gt, pr)
