import pdb
import numpy as np
import tensorflow as tf
from keras import backend as K

# this package is for metrics and related loss fucntions

# all these functions are for two classes even if there is a third class
# it will be ignored as it is background we do not care about it


class Class_weighting:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.dimension = K.int_shape(y_pred)

    def general_weighting(self):
        for i in range(self.dimension[-1]):
            desired_class = K.sum(self.y_true[:,:,:,i], axis=(1,2))
            all = K.sum(self.y_true, axis=(1,2,3))
            weight_factor = (all-desired_class+ K.epsilon())/(all+ K.epsilon())

            setattr(self, "weight"+str(i), weight_factor)
            setattr(self, "y_t"+str(i), self.y_true[:,:,:,i])
            setattr(self, "y_p"+str(i), self.y_pred[:,:,:,i])



def general_dice_weighted(y_true, y_pred):
    Weights = Class_weighting(y_true, y_pred)
    Weights.general_weighting()

    for i in range(Weights.dimension[-1]):
        w = getattr(Weights, "weight"+str(i))
        y1 = getattr(Weights, "y_t"+str(i))
        y2 = getattr(Weights, "y_p"+str(i))

        Sum = K.sum(y1 * y2, axis=(1,2))
        Sum_true = K.sum(y1, axis=(1,2))
        Sum_pred = K.sum(y2, axis=(1,2))

        if i == 0:
            Nominator = w*K.sum(Sum)
            Denominator = w*K.sum(Sum_true) + w*K.sum(Sum_pred)

        else:
            Nominator += w*K.sum(Sum)
            Denominator += w*K.sum(Sum_true)

    Nominator += K.epsilon()
    Denominator += K.epsilon()

    return tf.keras.backend.mean((Nominator/Denominator))

def general_loss_dice_weighted(y_true, y_pred):
    return ( 1-general_dice_weighted(y_true, y_pred) )



def dice(y_true, y_pred, smooth=K.epsilon()):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]
    intersection = K.sum(y_true * y_pred)
    # return (2. * intersection + smooth) / (K.sum(K.square(y_t),-1) + K.sum(K.square(y_p),-1) + smooth)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def loss_dice(y_true, y_pred):
    return 1-dice(y_true, y_pred)



def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)



def weighting_no_background(y_true, y_pred):
    Coef = K.int_shape(y_pred)[1]/5
    Coef = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2]/Coef
    Coef = 1/Coef

    Weights = []
    Y1 = []
    Y2 = []
    if K.int_shape(y_pred)[-1]==2:
        y1 = y_true[:,:,:,-1]
        y2 = y_pred[:,:,:,-1]
        y_true_class1_w = K.sum(y1, axis=(1,2))
        weight = (y_true_class1_w + 1)/(y_true_class1_w + 1)

        Y1=y1
        Y2=y2
        Weights=weight

    else:
        for Class in range(K.int_shape(y_pred)[-1]-1):
            Class += 1
            y1 = y_true[:,:,:,Class]
            y2 = y_pred[:,:,:,Class]

            y_true_class1_w = K.sum(y1, axis=(1,2))
            y_true_others = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2] - y_true_class1_w

            weights = (y_true_others)/(K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2])

            weights = K.cast(weights,'float32')
            Condition_inverse = K.greater(weights, Coef)
            weights = K.cast(Condition_inverse,'float32')*weights
            Condition = K.equal(weights, 0)
            weight = weights+K.cast(Condition,'float32')

            Y1.append(y1)
            Y2.append(y2)
            Weights.append(weight)
    return (Y1, Y2, Weights)

def dice_weighted(y_true, y_pred):
    Y1, Y2, Weights = weighting_no_background(y_true, y_pred)

    Sum_weights = 0
    if K.int_shape(y_pred)[-1]==2:
        Sum = K.sum(Y1 * Y2, axis=(1,2))
        Sum_true = K.sum(Y1, axis=(1,2))
        Sum_pred = K.sum(Y2, axis=(1,2))

        Nominator = 2*( K.sum(Sum) )
        Denominator = ( K.sum(Sum_true)+K.sum(Sum_pred)+K.epsilon() )
        DICE = tf.keras.backend.mean((Nominator/Denominator))
        Sum_weights = 1
    else:
        for N, (y1, y2, weight) in enumerate(zip(Y1, Y2, Weights)):
            Sum = K.sum(y1 * y2, axis=(1,2))
            Sum_true = K.sum(y1, axis=(1,2))
            Sum_pred = K.sum(y2, axis=(1,2))
            Sum_weights += weight

            Nominator = weight*2*( K.sum(Sum) )
            Denominator = ( K.sum(Sum_true) + K.sum(Sum_pred)+K.epsilon() )

            if N==0:
                DICE = tf.keras.backend.mean((Nominator/Denominator))
            else:
                DICE += tf.keras.backend.mean((Nominator/Denominator))
    return DICE/Sum_weights

def loss_dice_weighted(y_true, y_pred):
    return 1-dice_weighted(y_true, y_pred)



def weighting_traditional(y_true, y_pred):
    Coef = K.int_shape(y_pred)[1]/5
    Coef = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2]/Coef
    Coef = 1/Coef

    Weights = []
    Y1 = []
    Y2 = []
    for Class in range(K.int_shape(y_pred)[-1]):
        y1 = y_true[:,:,:,Class]
        y2 = y_pred[:,:,:,Class]

        y_true_class1_w = K.sum(y1, axis=(1,2))
        y_true_others = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2] - y_true_class1_w

        weights = ( y_true_others )/( K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2] )

        weight = weights
        # weights = K.cast(weights,'float32')
        # Condition_inverse = K.greater(weights, Coef)
        # weights = K.cast(Condition_inverse,'float32')*weights
        # Condition = K.equal(weights, 0)
        # weight = weights+K.cast(Condition,'float32')

        Y1.append(y1)
        Y2.append(y2)
        Weights.append(weight)
    return (Y1, Y2, Weights)

def dice_weighted_traditional(y_true, y_pred):
    Y1, Y2, Weights = weighting_traditional(y_true, y_pred)

    Sum_weights = 0
    for N, (y1, y2, weight) in enumerate(zip(Y1, Y2, Weights)):
        Sum = K.sum(y1 * y2, axis=(1,2))
        Sum_true = K.sum(y1, axis=(1,2))
        Sum_pred = K.sum(y2, axis=(1,2))
        Sum_weights += weight

        Nominator = weight*2*( K.sum(Sum) )
        Denominator = ( K.sum(Sum_true) + K.sum(Sum_pred)+K.epsilon() )
        result = tf.keras.backend.mean((Nominator/Denominator))

        if N==0:
            DICE = result
        else:
            DICE += result
    return DICE/Sum_weights

def loss_dice_weighted_traditional(y_true, y_pred):
    return 1-dice_weighted_traditional(y_true, y_pred)



def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot



def generalised_dice(y_true, y_pred):
    ground_truth = y_true
    prediction = y_pred

    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])


    ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                     reduction_axes=[0])
    seg_vol = tf.reduce_sum(prediction, 0)


    weights = tf.reciprocal(tf.square(ref_vol))

    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return generalised_dice_score

def generalised_dice_loss(y_true, y_pred):
    return 1-generalised_dice(y_true, y_pred)



def wasserstein_disagreement_map(prediction, ground_truth, M):
    n_classes = K.int_shape(prediction)[-1]
    ground_truth = tf.cast(ground_truth, dtype=tf.float64)
    prediction = tf.cast(prediction, dtype=tf.float64)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(prediction[:,i], ground_truth[:,j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map

def generalised_wasserstein_dice(y_true, y_pred):
    M_tree_4 = np.array([[0., 1., 1., 1.,],
                     [1., 0., 0.6, 0.5],
                     [1., 0.6, 0., 0.7],
                     [1., 0.5, 0.7, 0.]], dtype=np.float64)
    n_classes = K.int_shape(y_pred)[-1]

    ground_truth = tf.cast(tf.reshape(y_true,(-1,n_classes)), dtype=tf.int64)
    pred_proba = tf.cast(tf.reshape(y_pred,(-1,n_classes)), dtype=tf.float64)

    M = M_tree_4
    delta = wasserstein_disagreement_map(pred_proba, ground_truth, M)
    all_error = tf.reduce_sum(delta)
    one_hot = tf.cast(ground_truth, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL, dtype=tf.float32)

def generalised_wasserstein_dice_loss(y_true, y_pred):
    return 1- (generalised_wasserstein_dice(y_true, y_pred)+dice_weighted(y_true, y_pred))/2

def generalised_wasserstein_dice_loss2(y_true, y_pred):
    return 1- (generalised_wasserstein_dice(y_true, y_pred))



def get_iou( gt , pr , n_classes ):
    EPS = K.epsilon()
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum(( gt == cl )*( pr == cl ))
        union = np.sum(np.maximum( ( gt == cl ) , ( pr == cl ) ))
        iou = float(intersection)/( union + EPS )
        class_wise[ cl ] = iou
    return class_wise



def sensitivity(y_true, y_pred):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())



def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())



def weighting(y_true, y_pred):
    if K.int_shape(y_pred)[-1] > 2:
        y1 = y_true[:,:,:,1:]
        y2 = y_pred[:,:,:,1:]

        y_true_class1_w = K.sum(y1[:,:,:,0], axis=(1,2))
        y_true_class2_w = K.sum(y1[:,:,:,1], axis=(1,2))

    else:
        y1 = y_true[:,:,:,:]
        y2 = y_pred[:,:,:,:]

        y_true_class1_w = K.sum(y1, axis=(1,2))
        y_true_class2_w = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2] - y_true_class1_w

    weight1 = (y_true_class1_w+ 1)/(y_true_class2_w+y_true_class1_w+ 1)
    weight2 = (y_true_class2_w+ 1)/(y_true_class2_w+y_true_class1_w+ 1)
    return (y1, y2, weight1, weight2)

def sensitivity_weighted(y_true, y_pred):
    y_true, y_pred, weight1, weight2 = weighting(y_true, y_pred)

    true_positives = K.sum(K.round(K.clip(y_true[:,:,:,0] * y_pred[:,:,:,0], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:,:,:,0], 0, 1)))
    sensitivity = weight2 * true_positives / (possible_positives + K.epsilon())

    true_positives = K.sum(K.round(K.clip(y_true[:,:,:,1] * y_pred[:,:,:,1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:,:,:,1], 0, 1)))
    sensitivity = sensitivity+weight1 * true_positives / (possible_positives + K.epsilon())
    return K.minimum(sensitivity, K.ones(shape=1))

def specificity_weighted(y_true, y_pred):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    y_true, y_pred, weight1, weight2 = weighting(y_true, y_pred)

    true_negatives = K.sum(K.round(K.clip( (1-y_true[:,:,:,0]) * (1-y_pred[:,:,:,0]), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true[:,:,:,0], 0, 1)))
    specifcity = weight2 * true_negatives / (possible_negatives + K.epsilon())

    true_negatives = K.sum(K.round(K.clip( (1-y_true[:,:,:,1]) * (1-y_pred[:,:,:,1]), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true[:,:,:,1], 0, 1)))
    specifcity = specifcity + weight1 * true_negatives / (possible_negatives + K.epsilon())
    return K.minimum(specifcity, K.ones(shape=1))
