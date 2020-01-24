from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class AvgPooling(keras.layers.Layer):
  def __init__(
    self, normalize_pool=True, vert_only_pool=False):
    super().__init__()
    self.normalize_pool = normalize_pool
    self.vert_only_pool = vert_only_pool

  def get_config(self):
    base_config = super().get_config()
    base_config["normalize_pool"] = self.normalize_pool
    base_config["vert_only_pool"] = self.vert_only_pool

    return base_config

  def call(self, io):
    inputs, Y = io
    X_in = inputs[0]
    e_map = inputs[-2]
    v_count = inputs[-1]

    if self.vert_only_pool:
      vert_mask = X_in[:, 0] == 1
      Y = tf.boolean_mask(Y, vert_mask)
      e_map = tf.boolean_mask(e_map, vert_mask)

    N = tf.shape(v_count)[0]

    if self.normalize_pool:
      y = tf.math.unsorted_segment_mean(Y, e_map, num_segments=N)
    else:
      y = tf.math.unsorted_segment_sum(Y, e_map, num_segments=N)

    # y = tf.math.divide_no_nan(
    #   y,
    #   tf.expand_dims(tf.cast(v_count, tf.float32), axis=1))

    return y
