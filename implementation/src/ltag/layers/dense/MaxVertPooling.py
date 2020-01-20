from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class MaxVertPooling(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def get_config(self):
    base_config = super().get_config()
    base_config["squeeze_output"] = self.squeeze_output

    return base_config

  def call(self, io):
    (X, A, n), Y = io

    y = tf.math.reduce_max(Y, 1)

    return y
