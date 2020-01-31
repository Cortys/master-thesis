from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class SagPooling(keras.layers.Layer):
  def __init__(
    self, att_threshold=0.5,
    softmax_pool=True,
    vert_only_pool=False):
    super().__init__()
    self.softmax_pool = softmax_pool
    self.vert_only_pool = vert_only_pool
    self.att_threshold = att_threshold

  def get_config(self):
    base_config = super().get_config()
    base_config["softmax_pool"] = self.softmax_pool
    base_config["vert_only_pool"] = self.vert_only_pool
    base_config["att_threshold"] = self.att_threshold

    return base_config

  def call(self, io):
    inputs, (Y, Y_att) = io
    X_in = inputs[0]
    e_map = inputs[-2]
    v_count = inputs[-1]

    mask = None

    if self.softmax_pool:
      Y_att = tf.exp(Y_att)
      Y *= Y_att
    else:
      mask = Y_att[:, 0] >= self.att_threshold
      Y *= Y_att

    if self.vert_only_pool:
      vert_mask = X_in[:, 0] == 1
      if self.softmax_pool:
        mask = vert_mask
      else:
        mask &= vert_mask

    if mask is not None:
      Y = tf.boolean_mask(Y, mask)
      e_map = tf.boolean_mask(e_map, mask)

    N = tf.shape(v_count)[0]

    if not self.softmax_pool:
      y = tf.math.unsorted_segment_mean(Y, e_map, num_segments=N)
    else:
      y = tf.math.unsorted_segment_sum(Y, e_map, num_segments=N)
      y_att = tf.math.unsorted_segment_sum(Y_att, e_map, num_segments=N)
      y = y / y_att

    # y = tf.math.divide_no_nan(
    #   y,
    #   tf.expand_dims(tf.cast(v_count, tf.float32), axis=1))

    return y
