from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

import ltag.chaining.pipeline as cp
import ltag.chaining.model as cm
import ltag.ops as ops

import ltag.layers.vert as vl

@cm.model_inputs
def gnn_vert_inputs(layer_dims, sparse=False):
  X = keras.Input(shape=(None, layer_dims[0]), name="X", sparse=sparse)
  A = keras.Input(shape=(None, None), name="A", sparse=sparse)
  n = keras.Input(shape=(), dtype=tf.int32, name="n")

  return X, A, n

@cm.model_inputs
def gnn_edge_inputs(layer_dims):
  X = keras.Input(shape=(None, layer_dims[0]), name="X")
  A = keras.Input(shape=(None, None), name="A")
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

def gnn_model(name, steps, edge_inputs=False):
  return cm.create_model(name, [
    gnn_edge_inputs if edge_inputs else gnn_vert_inputs,
    *steps])


# Non-aggregating GCNs:

VertEFGCN = gnn_model("VertEFGCN", [
  cm.with_layers(vl.EFGCNLayer),
  select_features])

VertEF2GCN = gnn_model("VertEF2GCN", [
  cm.with_layer(vl.EdgeFeaturePreparation),
  cm.with_layers(vl.EF2GCNLayer),
  select_features])

# Averaging GCNs:

AVG_VertEFGCN = VertEFGCN.extend("AVG_VertEFGCN", [avg_verts])
AVG_VertEF2GCN = VertEF2GCN.extend("AVG_VertEF2GCN", [
  cm.with_layer(vl.AVGEdgePooling, with_inputs=True)])
