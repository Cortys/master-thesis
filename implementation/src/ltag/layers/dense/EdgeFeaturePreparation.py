from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class EdgeFeaturePreparation(keras.layers.Layer):
  def __init__(
    self, sparse=False, bias=False, neighborhood_mask=-1):
    super().__init__()
    self.sparse = sparse
    self.bias = bias
    self.neighborhood_mask = neighborhood_mask

  def get_config(self):
    base_config = super().get_config()
    base_config["sparse"] = self.sparse
    base_config["bias"] = self.bias
    base_config["neighborhood_mask"] = self.neighborhood_mask

    return base_config

  def call(self, input):
    X, A, n = input

    if self.sparse:
      X = tf.sparse.to_dense(X)
      A = tf.sparse.to_dense(A)

    X_t = tf.linalg.matrix_transpose(X)
    X_d = tf.linalg.diag(X_t)
    X_e = tf.transpose(X_d, perm=(0, 2, 3, 1))

    A_e = tf.expand_dims(A, axis=-1)

    AX_e = tf.concat([X_e, A_e], axis=-1)

    if self.sparse:
      AX_e = tf.sparse.from_dense(AX_e)

    X_shape = tf.shape(X)

    if self.neighborhood_mask == -1:
      max_n = X_shape[-2]
      mask = ops.matrix_mask(n, max_n)
    else:
      mask = ops.neighborhood_mask(
        AX_e, self.neighborhood_mask)

    return AX_e, mask, n
