from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.chaining.pipeline as cp
import ltag.chaining.model as cm
import ltag.ops as ops

from ltag.layers import (
  EdgeFeaturePreparation,
  EFGCNLayer, EF2GCNLayer,
  AVGEdgePooling
)

@cm.model_inputs
def gnn_inputs(layer_dims, sparse=False):
  X = keras.Input(shape=(None, layer_dims[0]), name="X", sparse=sparse)
  A = keras.Input(shape=(None, None), name="A", sparse=sparse)
  n = keras.Input(shape=(), dtype=tf.int32, name="n")

  return X, A, n

@cm.model_step
def select_features(inputs):
  return inputs[0]

@cp.pipeline_step
def avg_verts(io):
  (X, A, n), Y = io

  Y_shape = tf.shape(Y)
  max_n = Y_shape[-2]
  n_mask = ops.vec_mask.python_function(n, max_n)

  Y = Y * n_mask

  y = tf.transpose(
    tf.math.divide_no_nan(
      tf.transpose(tf.reduce_sum(Y, 1)),
      tf.cast(n, tf.float32)))

  return (X, A, n), y

def gnn_model(name, steps):
  return cm.create_model(name, [gnn_inputs, *steps])


# Non-aggregating GCNs:

EFGCN = gnn_model("EFGCN", [
  cm.with_layers(EFGCNLayer),
  select_features])

EF2GCN = gnn_model("EF2GCN", [
  cm.with_layer(EdgeFeaturePreparation),
  cm.with_layers(EF2GCNLayer),
  select_features])

# Averaging GCNs:

AVG_EFGCN = EFGCN.extend("AVG_EFGCN", [avg_verts])
AVG_EF2GCN = EF2GCN.extend("AVG_EF2GCN", [
  cm.with_layer(AVGEdgePooling, with_inputs=True)])
