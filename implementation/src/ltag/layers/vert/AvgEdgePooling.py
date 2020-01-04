from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class AvgEdgePooling(keras.layers.Layer):
  def __init__(
    self, squeeze_output=False):
    super(AvgEdgePooling, self).__init__()
    self.squeeze_output = squeeze_output

  def get_config(self):
    base_config = super(AvgEdgePooling, self).get_config()
    base_config["squeeze_output"] = self.squeeze_output

    return base_config

  def call(self, io):
    (X, A, n), Y = io

    y = tf.transpose(
      tf.math.divide_no_nan(
        tf.transpose(tf.reduce_sum(Y, axis=(1, 2))),
        tf.cast(n * n, tf.float32)))

    if self.squeeze_output:
      y = tf.squeeze(y, axis=-1)

    return y
