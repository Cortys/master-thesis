from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class AvgEdgePooling(keras.layers.Layer):
  def __init__(
    self, squeeze_output=False):
    super().__init__()
    self.squeeze_output = squeeze_output

  def get_config(self):
    base_config = super().get_config()
    base_config["squeeze_output"] = self.squeeze_output

    return base_config

  def call(self, io):
    (X, A, n), Y = io

    Y_sum = tf.reduce_sum(Y, axis=(1, 2))
    n_2 = tf.cast(tf.expand_dims(n * n, axis=-1), tf.float32)

    y = tf.math.divide_no_nan(Y_sum, n_2)

    if self.squeeze_output:
      y = tf.squeeze(y, axis=-1)

    return y
