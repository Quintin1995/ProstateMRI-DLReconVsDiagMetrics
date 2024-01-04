import tensorflow as tf
from tensorflow.image import ssim
from tensorflow.keras import backend as K


# Structural Similarity index on images.
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(ssim(tf.squeeze(y_true, 4), tf.squeeze(y_pred, 4), 1.0))


# Peak signal to noise ratio on images.
def psnr_metric(y_true, y_pred, max_val=1.0):
    return tf.reduce_mean(tf.image.psnr(tf.squeeze(y_true, 4), tf.squeeze(y_pred, 4), max_val=max_val), axis=-1)


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(ssim(tf.squeeze(y_true, 4), tf.squeeze(y_pred, 4), 1.0))
