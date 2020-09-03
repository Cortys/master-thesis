from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers as regs

import ltag.ops as ops
from ltag.layers.DenseLayer import DenseLayer

class GINLayer(keras.layers.Layer):
  def __init__(
    self, out_dim, in_dim=None, hidden_dim=None,
    act="relu", bias=True,
    W_regularizer=None,
    b_regularizer=None):
    super().__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.act = keras.activations.get(act)
    self.bias = bias
    self.W_regularizer = regs.get(W_regularizer)
    self.b_regularizer = regs.get(b_regularizer)

  def get_config(self):
    base_config = super().get_config()
    base_config["out_dim"] = self.out_dim
    base_config["in_dim"] = self.in_dim
    base_config["hidden_dim"] = self.hidden_dim
    base_config["act"] = keras.activations.serialize(self.act)
    base_config["bias"] = self.bias

    return base_config

  def build(self, input_shape):
    X_shape = input_shape[0]

    vert_dim = X_shape[-1]

    if vert_dim is None:
      vert_dim = self.in_dim

    hidden_dim = self.hidden_dim or max(vert_dim, self.out_dim)

    self.W_hidden = self.add_weight(
      "W_hidden", shape=[vert_dim, hidden_dim],
      regularizer=self.W_regularizer,
      trainable=True, initializer=tf.initializers.GlorotUniform)

    self.W_out = self.add_weight(
      "W_out", shape=[hidden_dim, self.out_dim],
      regularizer=self.W_regularizer,
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.bias:
      self.b_hidden = self.add_weight(
        "b_hidden", shape=[hidden_dim],
        regularizer=self.b_regularizer,
        trainable=True, initializer=tf.initializers.Zeros)
      self.b_out = self.add_weight(
        "b_out", shape=[self.out_dim],
        regularizer=self.b_regularizer,
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X, ref_a, ref_b, v_map, v_count = input

    X_agg = ops.gin_convolution_compact.python_function(
      X, ref_a, ref_b)

    X_hid = tf.matmul(X_agg, self.W_hidden)
    if self.bias:
      X_hid = tf.nn.bias_add(X_hid, self.b_hidden)
    X_hid = self.act(X_hid)

    X_out = tf.matmul(X_hid, self.W_out)
    if self.bias:
      X_out = tf.nn.bias_add(X_out, self.b_out)
    X_out = self.act(X_out)

    return X_out, ref_a, ref_b, v_map, v_count
