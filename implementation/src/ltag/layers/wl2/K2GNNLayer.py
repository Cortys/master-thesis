from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers as regs

import ltag.ops as ops
from ltag.layers.DenseLayer import DenseLayer

class K2GNNLayer(keras.layers.Layer):
  def __init__(
    self, out_dim, in_dim=None,
    act="relu", bias=True,
    W_regularizer=None, W_prop_regularizer=None,
    b_regularizer=None, intersperse_dense=False):
    super().__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.act = keras.activations.get(act)
    self.bias = bias
    self.intersperse_dense = intersperse_dense
    self.W_regularizer = regs.get(W_regularizer)
    self.W_prop_regularizer = regs.get(W_prop_regularizer)
    self.b_regularizer = regs.get(b_regularizer)

  def get_config(self):
    base_config = super().get_config()
    base_config["out_dim"] = self.out_dim
    base_config["in_dim"] = self.in_dim
    base_config["act"] = keras.activations.serialize(self.act)
    base_config["bias"] = self.bias
    base_config["intersperse_dense"] = self.intersperse_dense
    base_config["W_regularizer"] = regs.serialize(self.W_regularizer)
    base_config["W_prop_regularizer"] = regs.serialize(self.W_prop_regularizer)
    base_config["b_regularizer"] = regs.serialize(self.b_regularizer)

    return base_config

  def build(self, input_shape):
    X_shape = input_shape[0]

    edge_dim = X_shape[-1]

    if edge_dim is None:
      edge_dim = self.in_dim

    self.W = self.add_weight(
      "W", shape=[edge_dim, self.out_dim],
      regularizer=self.W_regularizer,
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_prop = self.add_weight(
      "W_prop", shape=[edge_dim, self.out_dim],
      regularizer=self.W_prop_regularizer,
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.bias:
      self.b = self.add_weight(
        "b", shape=[self.out_dim],
        regularizer=self.b_regularizer,
        trainable=True, initializer=tf.initializers.Zeros)

    if self.intersperse_dense:
      self.dense_layer = DenseLayer(
        in_dim=self.out_dim, out_dim=self.out_dim,
        W_regularizer=self.W_regularizer,
        b_regularizer=self.b_regularizer,
        act=self.act, bias=self.bias)

  def call(self, input):
    X, ref_a, backref, e_map, v_count = input

    XW = tf.linalg.matmul(X, self.W)
    XW_prop = tf.linalg.matmul(X, self.W_prop)

    X_conv = ops.wl1_convolution_compact.python_function(
      XW_prop, ref_a, backref)

    if self.bias:
      XW_comb = tf.nn.bias_add(XW + X_conv, self.b)
    else:
      XW_comb = XW + X_conv

    X_out = self.act(XW_comb)

    if self.intersperse_dense:
      X_out = self.dense_layer(X_out)

    return X_out, ref_a, backref, e_map, v_count