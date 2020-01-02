from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class AVGPooling(keras.layers.Layer):
  def __init__(
    self, squeeze_output=False):
    super(AVGPooling, self).__init__()
    self.squeeze_output = squeeze_output

  def get_config(self):
    base_config = super(AVGPooling, self).get_config()
    base_config["squeeze_output"] = self.squeeze_output

    return base_config

  def call(self, io):
    (X, ref_a, ref_b, e_map, v_count), Y = io

    N = tf.shape(v_count)[0]

    y = tf.math.unsorted_segment_mean(Y, e_map, num_segments=N)

    if self.squeeze_output:
      y = tf.squeeze(y, axis=-1)

    return y
