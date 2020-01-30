from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops
from ltag.layers.DenseLayer import DenseLayer

class CWL2GCNLayer(keras.layers.Layer):
  def __init__(
    self, out_dim, in_dim=None,
    act="relu", bias=True,
    local_act="relu", intersperse_dense=False):
    super().__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.act = keras.activations.get(act)
    self.local_act = keras.activations.get(local_act)
    self.bias = bias
    self.intersperse_dense = intersperse_dense

  def get_config(self):
    base_config = super().get_config()
    base_config["out_dim"] = self.out_dim
    base_config["in_dim"] = self.in_dim
    base_config["act"] = keras.activations.serialize(self.act)
    base_config["local_act"] = keras.activations.serialize(self.local_act)
    base_config["bias"] = self.bias
    base_config["intersperse_dense"] = self.intersperse_dense

    return base_config

  def build(self, input_shape):
    X_shape = input_shape[0]

    edge_dim = X_shape[-1]

    if edge_dim is None:
      edge_dim = self.in_dim

    self.W = self.add_weight(
      "W", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_back = self.add_weight(
      "W_back", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_prop = self.add_weight(
      "W_prop", shape=[edge_dim, self.out_dim],
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.bias:
      self.b = self.add_weight(
        "b", shape=[self.out_dim],
        trainable=True, initializer=tf.initializers.Zeros)
      self.b_prop = self.add_weight(
        "b_prop", shape=[self.out_dim],
        trainable=True, initializer=tf.initializers.Zeros)

    if self.intersperse_dense:
      self.dense_layer = DenseLayer(
        in_dim=self.out_dim, out_dim=self.out_dim,
        act=self.act, bias=self.bias)

  def call(self, input):
    X, ref_a, ref_b, backref, e_map, v_count = input

    XW = tf.linalg.matmul(X, self.W)
    XW_back = tf.linalg.matmul(X, self.W_back)
    XW_prop = tf.linalg.matmul(X, self.W_prop)

    def local_hash(X_a, X_b):
      S = X_a + X_b

      if self.bias:
        S = tf.nn.bias_add(S, self.b_prop)

      return self.local_act(S)

    X_conv = XW_back * ops.wl2_convolution_compact.python_function(
      XW_prop, ref_a, ref_b, backref, local_hash)

    if self.bias:
      XW_comb = tf.nn.bias_add(XW + X_conv, self.b)
    else:
      XW_comb = XW + X_conv

    X_out = self.act(XW_comb)

    if self.intersperse_dense:
      X_out = self.dense_layer(X_out)

    return X_out, ref_a, ref_b, backref, e_map, v_count
