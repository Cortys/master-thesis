from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np

class SortPooling(keras.layers.Layer):
  def __init__(
    self, k_pool=1):
    super().__init__()
    self.k_pool = k_pool

  def get_config(self):
    base_config = super().get_config()
    base_config['k_pool'] = self.k_pool

    return base_config

  def build(self, input_shape):
    self.W = self.add_weight(
      "W", shape=[self.k_pool],
      trainable=True, initializer=tf.initializers.Ones)

  def call(self, io):
    inputs, Y = io
    e_map = inputs[-2]
    v_count = inputs[-1]

    Y = tf.squeeze(Y, axis=-1)
    N = tf.shape(v_count)[0]
    row_splits = tf.ragged.segment_ids_to_row_splits(e_map, N)
    Y_r = tf.RaggedTensor.from_row_splits(Y, row_splits)
    Y_d = Y_r.to_tensor(default_value=-np.inf)
    k_pool = tf.math.minimum(self.k_pool, tf.shape(Y_d)[-1])

    Y_sort, _ = tf.math.top_k(Y_d, k_pool)
    Y_top = Y_sort[:, 0:k_pool]
    W = tf.nn.softmax(self.W[0:k_pool])
    Y_w = tf.where(tf.math.is_inf(Y_top), 0.0, Y_top) * W
    y = tf.reduce_sum(Y_w, -1)

    return y
