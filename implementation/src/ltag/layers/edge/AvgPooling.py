from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class AvgPooling(keras.layers.Layer):
  def __init__(
    self, squeeze_output=False):
    super(AvgPooling, self).__init__()
    self.squeeze_output = squeeze_output

  def get_config(self):
    base_config = super(AvgPooling, self).get_config()
    base_config["squeeze_output"] = self.squeeze_output

    return base_config

  def call(self, io):
    (X, ref_a, ref_b, e_map, v_count), Y = io

    N = tf.shape(v_count)[0]

    y = tf.math.unsorted_segment_mean(Y, e_map, num_segments=N)

    # y = tf.math.divide_no_nan(
    #   y,
    #   tf.expand_dims(tf.cast(v_count, tf.float32), axis=1))

    if self.squeeze_output:
      y = tf.squeeze(y, axis=-1)

    return y
