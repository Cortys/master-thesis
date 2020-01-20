from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class AvgVertPooling(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def get_config(self):
    base_config = super().get_config()

    return base_config

  def call(self, io):
    (X, A, n), Y = io

    Y_shape = tf.shape(Y)
    max_n = Y_shape[-2]
    n_mask = ops.vec_mask(n, max_n)

    Y = Y * n_mask

    y = tf.transpose(
      tf.math.divide_no_nan(
        tf.transpose(tf.reduce_sum(Y, 1)),
        tf.cast(n, tf.float32)))

    return y
