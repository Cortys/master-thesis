from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

class EFGCNLayer(keras.layers.Layer):
  def __init__(self, num_outputs):
    super(EFGCNLayer, self).__init__()
    self.num_outputs = num_outputs

  def get_config(self):
    base_config = super(EFGCNLayer, self).get_config()
    base_config['num_outputs'] = self.num_outputs

    return base_config

  def build(self, input_shape):
    X_shape, A_shape = input_shape
    self.W = self.add_weight(
      "W", shape=[X_shape[2], self.num_outputs],
      trainable=True, initializer='uniform')
    self.bias = self.add_weight(
      "bias", shape=[self.num_outputs],
      trainable=True, initializer='zero')

  def call(self, input):
    X, A = input

    XW = tf.linalg.matmul(X, self.W)

    X_out = tf.nn.relu(XW + self.bias)

    return X_out, A
