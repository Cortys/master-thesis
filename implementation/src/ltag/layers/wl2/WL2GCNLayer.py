from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class WL2GCNLayer(keras.layers.Layer):
  def __init__(
    self, out_dim, in_dim=None,
    act="relu", bias=True):
    super().__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.act = keras.activations.get(act)
    self.bias = bias

  def get_config(self):
    base_config = super().get_config()
    base_config["out_dim"] = self.out_dim
    base_config["in_dim"] = self.in_dim
    base_config["act"] = keras.activations.serialize(self.act)
    base_config["bias"] = self.bias

    return base_config

  def build(self, input_shape):
    X_shape = input_shape[0]

    edge_dim = X_shape[-1]

    if edge_dim is None:
      edge_dim = self.in_dim

    self.W = self.add_weight(
      "W", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_prop = self.add_weight(
      "W_prop", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.bias:
      self.W_bias = self.add_weight(
        "W_bias", shape=[self.out_dim],
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X, ref_a, ref_b, e_map, v_count = input

    X_prop = ops.aggregate_edge_features_using_refs(
      X, ref_a, ref_b, tf.multiply)

    XW = tf.linalg.matmul(X, self.W)
    XW_prop = tf.linalg.matmul(X_prop, self.W_prop)

    if self.bias:
      XW_comb = tf.nn.bias_add(XW + XW_prop, self.W_bias)
    else:
      XW_comb = XW + XW_prop

    X_out = self.act(XW_comb)

    return X_out, ref_a, ref_b, e_map, v_count
