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
    base_config["num_outputs"] = self.num_outputs

    return base_config

  def build(self, input_shape):
    X_shape, A_shape = input_shape
    vert_dim = X_shape[-1]

    self.W = self.add_weight(
      "W", shape=[vert_dim, self.num_outputs],
      trainable=True, initializer="uniform")
    self.bias = self.add_weight(
      "bias", shape=[self.num_outputs],
      trainable=True, initializer="zero")

  def call(self, input):
    X, A = input

    X_dim = tf.shape(X)
    Id = tf.eye(X_dim[-2], batch_shape=X_dim[:-2])
    AI = A + Id
    diags = tf.reduce_sum(AI, axis=-1)
    diags_norm = tf.pow(diags, -0.5)
    D_norm = tf.linalg.diag(diags_norm)
    A_norm = tf.linalg.matmul(D_norm, tf.linalg.matmul(AI, D_norm))

    XW = tf.linalg.matmul(X, self.W)
    XWA = tf.linalg.matmul(A_norm, XW)

    X_out = tf.nn.leaky_relu(XWA + self.bias)

    return X_out, A
