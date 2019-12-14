from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.chaining.pipeline as cp
import ltag.chaining.model as cm
from ltag.layers import EFGCNLayer, EF2GCNLayer

@cm.model_inputs
def gcn_inputs(layer_dims):
  X = keras.Input(shape=(None, layer_dims[0]), name="X")
  A = keras.Input(shape=(None, None), name="A")
  n = keras.Input(shape=(), dtype=tf.int32, name="n")

  return (X, A, n)

@cm.model_step
def with_layers(inputs, layer, layer_dims, **kwargs):
  layer = cp.tolerant(layer)
  h = inputs

  for f_dim in layer_dims[1:]:
    h = layer(f_dim, **kwargs)(h)

  return h

@cm.model_step
def with_edge_featured_propagation(inputs):
  X, A, n = inputs

  X_t = tf.linalg.matrix_transpose(X)
  X_d = tf.linalg.diag(X_t)
  X_e = tf.transpose(X_d, perm=(0, 2, 3, 1))

  A_e = tf.expand_dims(A, axis=-1)

  AX_e = tf.concat([A_e, X_e], axis=-1)

  return AX_e, n

@cm.model_step
def select_features(inputs):
  return inputs[0]

@cp.pipeline_step
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

@cp.pipeline_step
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

def gcn_model(name, steps):
  return cm.create_model(name, [gcn_inputs, *steps])


# Non-aggregating GCNs:

EFGCN = gcn_model("EFGCN", [with_layers(EFGCNLayer), select_features])
EF2GCN = gcn_model("EF2GCN", [
  with_edge_featured_propagation,
  with_layers(EF2GCNLayer),
  select_features])

# Averaging GCNs:

AVG_EFGCN = EFGCN.extend("AVG_EFGCN", [avg_verts])
AVG_EF2GCN = EF2GCN.extend("EF2GCN", [avg_edges])
