from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
import funcy as fy
from tensorflow import keras

from ltag.layers import EFGCNLayer, EF2GCNLayer

def gcn(
  layer, layer_dims=[],
  input_transformer=None,
  **kwargs):

  X = keras.Input(shape=(None, layer_dims[0]), name="X")
  A = keras.Input(shape=(None, None), name="A")
  n = keras.Input(shape=(), dtype=tf.int32, name="n")
  inputs = (X, A, n)

  h = input_transformer(inputs) if input_transformer else inputs

  for f_dim in layer_dims[1:]:
    h = layer(f_dim, **kwargs)(h)

  Y = h[0]

  return inputs, Y


def as_model(name, io_fn, *args, **kwargs):
  def Model(*args2, **kwargs2):
    inputs, output = io_fn(*args, *args2, **kwargs, **kwargs2)

    return keras.Model(inputs=inputs, outputs=output, name=name)

  return Model


def edge_featured_inputs(inputs):
  X, A, n = inputs

  X_t = tf.linalg.matrix_transpose(X)
  X_d = tf.linalg.diag(X_t)
  X_e = tf.transpose(X_d, perm=(0, 2, 3, 1))

  A_e = tf.expand_dims(A, axis=-1)

  AX_e = tf.concat([A_e, X_e], axis=-1)

  return AX_e, n


def avg_verts(io):
  (X, A, n), Y = io

  Y_shape = tf.shape(Y)
  max_n = Y_shape[-2]
  n_mask = tf.broadcast_to(
    tf.expand_dims(tf.sequence_mask(n, maxlen=max_n), 2), Y_shape)

  Y = tf.where(n_mask, Y, tf.zeros_like(Y))

  y = tf.transpose(
    tf.math.divide_no_nan(
      tf.transpose(tf.reduce_sum(Y, 1)),
      tf.cast(n, tf.float32)))

  return (X, A, n), y


def avg_edges(io):
  (X, A, n), Y = io

  Y_shape = tf.shape(Y)
  max_n = Y_shape[-2]
  n_mask = tf.cast(tf.sequence_mask(n, maxlen=max_n), tf.int32)
  m_1 = tf.expand_dims(n_mask, 2)
  m_2 = tf.expand_dims(n_mask, 1)
  m = tf.broadcast_to(
    tf.cast(tf.expand_dims(tf.linalg.matmul(m_1, m_2), -1), tf.float32),
    Y_shape)

  Y = Y * m

  y = tf.transpose(
    tf.math.divide_no_nan(
      tf.transpose(tf.reduce_sum(Y, axis=(1, 2, 3))),
      tf.cast(n * n, tf.float32)))

  return (X, A, n), y


# Vertex models:

EFGCN = as_model("EFGCN", gcn, EFGCNLayer)

EF2GCN = as_model(
  "EF2GCN", gcn, EF2GCNLayer, input_transformer=edge_featured_inputs)


# Averaging models:

AVG_EFGCN = as_model("AVG_EFGCN", fy.rcompose(gcn, avg_verts), EFGCNLayer)

AVG_EF2GCN = as_model(
  "AVG_EF2GCN", fy.rcompose(gcn, avg_edges),
  EF2GCNLayer,
  input_transformer=edge_featured_inputs)
