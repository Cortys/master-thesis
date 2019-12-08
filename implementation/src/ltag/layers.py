from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.ops as ops

class EFGCNLayer(keras.layers.Layer):
  def __init__(
    self, num_outputs,
    act="relu",
    k_dim=0, k_act="sigmoid",
    normalize_adj=True):
    super(EFGCNLayer, self).__init__()
    self.num_outputs = num_outputs
    self.act = keras.activations.get(act)
    self.k_dim = k_dim
    self.k_act = keras.activations.get(k_act)
    self.normalize_adj = normalize_adj

  def get_config(self):
    base_config = super(EFGCNLayer, self).get_config()
    base_config["num_outputs"] = self.num_outputs
    base_config["act"] = keras.activations.serialize(self.act)
    base_config["k_dim"] = self.k_dim
    base_config["k_act"] = keras.activations.serialize(self.k_act)
    base_config["normalize_adj"] = self.normalize_adj

    return base_config

  def build(self, input_shape):
    X_shape, A_shape = input_shape

    vert_dim = X_shape[-1]

    self.W = self.add_weight(
      "W", shape=[vert_dim, self.num_outputs],
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_bias = self.add_weight(
      "W_bias", shape=[self.num_outputs],
      trainable=True, initializer=tf.initializers.Zeros)

    if self.k_dim > 0:
      self.K = self.add_weight(
        "K", shape=[vert_dim, self.k_dim],
        trainable=True, initializer=tf.initializers.Ones)
      self.K_bias = self.add_weight(
        "K_bias", shape=[self.k_dim],
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X, A = input

    X_dim = tf.shape(X)
    Id = tf.eye(X_dim[-2], batch_shape=X_dim[:-2])

    if self.k_dim > 0:
      XK = tf.nn.bias_add(tf.linalg.matmul(X, self.K), self.K_bias)
      A_filter = tf.nn.sigmoid(tf.matmul(XK, XK, adjoint_b=True))
      AI = A * A_filter + Id
    else:
      AI = A + Id

    if self.normalize_adj:
      AI = ops.normalize_mat(AI)

    XW = tf.nn.bias_add(tf.linalg.matmul(X, self.W), self.W_bias)
    XWA = tf.linalg.matmul(AI, XW)

    X_out = self.act(XWA)


    XW = tf.linalg.matmul(X, self.W)
    XWA = tf.linalg.matmul(A_norm, XW)

    X_out = tf.nn.tanh(XWA + self.bias)

    return X_out, A
