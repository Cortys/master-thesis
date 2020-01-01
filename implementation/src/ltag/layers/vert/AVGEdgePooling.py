from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class AVGEdgePooling(keras.layers.Layer):
  def __init__(
    self, sparse=False, squeeze_output=False):
    super(AVGEdgePooling, self).__init__()
    self.sparse = sparse
    self.squeeze_output = squeeze_output

  def get_config(self):
    base_config = super(AVGEdgePooling, self).get_config()
    base_config["sparse"] = self.sparse
    base_config["squeeze_output"] = self.squeeze_output

    return base_config

  def call(self, io):
    (X, A, n), Y = io

    if self.sparse:
      Y = tf.sparse.to_dense(Y)

    y = tf.transpose(
      tf.math.divide_no_nan(
        tf.transpose(tf.reduce_sum(Y, axis=(1, 2, 3))),
        tf.cast(n * n, tf.float32)))

    if self.squeeze_output:
      y = tf.squeeze(y, axis=-1)

    return y
