from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class EF2GCNLayer(keras.layers.Layer):
  def __init__(
    self, out_dim, in_dim=None,
    act="relu", sparse=False, masked_bias=False):
    super(EF2GCNLayer, self).__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.act = keras.activations.get(act)
    self.sparse = sparse
    self.masked_bias = masked_bias

  def get_config(self):
    base_config = super(EF2GCNLayer, self).get_config()
    base_config["out_dim"] = self.out_dim
    base_config["in_dim"] = self.in_dim
    base_config["act"] = keras.activations.serialize(self.act)
    base_config["sparse"] = self.sparse
    base_config["masked_bias"] = self.masked_bias

    return base_config

  def build(self, input_shape):
    print("ef2gcn", input_shape)
    if self.masked_bias:
      X_shape, mask_shape, n_shape = input_shape
    else:
      X_shape, n_shape = input_shape

    edge_dim = X_shape[-1]

    if edge_dim is None:
      edge_dim = self.in_dim

    self.W = self.add_weight(
      "W", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_prop = self.add_weight(
      "W_prop", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.masked_bias:
      self.W_bias = self.add_weight(
        "W_bias", shape=[self.out_dim],
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    if self.masked_bias:
      X, mask, n = input
    else:
      X, n = input

    if self.sparse:
      X = tf.sparse.to_dense(X)

    X_prop = ops.aggregate_edge_features(X, tf.multiply)

    XW = tf.linalg.matmul(X, self.W)
    XW_prop = tf.linalg.matmul(X_prop, self.W_prop)

    if self.masked_bias:
      XW_comb = tf.nn.bias_add(XW + XW_prop, self.W_bias) * mask
    else:
      XW_comb = XW + XW_prop

    X_out = self.act(XW_comb)

    if self.sparse:
      X_out = tf.sparse.from_dense(X_out)

    if self.masked_bias:
      return X_out, mask, n
    else:
      return X_out, n
