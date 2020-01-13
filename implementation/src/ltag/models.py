from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import funcy as fy

import ltag.chaining.model as cm
import ltag.layers.dense as ld
import ltag.layers.wl2 as lwl2

@cm.model_inputs
def gnn_dense_inputs(layer_dims, sparse=False):
  X = keras.Input(shape=(None, layer_dims[0]), name="X", sparse=sparse)
  A = keras.Input(shape=(None, None), name="A", sparse=sparse)
  n = keras.Input(shape=(), dtype=tf.int32, name="n")

  return X, A, n

@cm.model_inputs
def gnn_wl2_inputs():
  X = keras.Input(shape=(None,), dtype=tf.float32, name="X")
  ref_a = keras.Input(shape=(None,), dtype=tf.int32, name="ref_a")
  ref_b = keras.Input(shape=(None,), dtype=tf.int32, name="ref_b")
  e_map = keras.Input(shape=(), dtype=tf.int32, name="e_map")
  v_count = keras.Input(shape=(), dtype=tf.int32, name="v_count")

  return X, ref_a, ref_b, e_map, v_count


input_types = {
  "dense": gnn_dense_inputs,
  "wl2": gnn_wl2_inputs
}

@cm.model_step
def select_features(inputs):
  return inputs[0]

def gnn_model(name, steps, input_type="dense"):
  m = cm.create_model(name, [
    input_types[input_type],
    *steps])

  def add_attrs(m):
    m.input_type = input_type
    m.extend = fy.rcompose(m.extend, add_attrs)
    return m

  return add_attrs(m)


# Non-aggregating GNNs:

DenseGCN = gnn_model("DenseGCN", [
  cm.with_layers(ld.GCNLayer),
  select_features])

DenseWL2GCN = gnn_model("DenseWL2GCN", [
  cm.with_layer(ld.EdgeFeaturePreparation),
  cm.with_layers(ld.WL2GCNLayer),
  select_features])

WL2GCN = gnn_model("WL2GCN", [
  cm.with_layers(lwl2.WL2GCNLayer),
  select_features],
  input_type="wl2")

# Averaging GNNs:

AvgDenseGCN = DenseGCN.extend("AvgDenseGCN", [
  cm.with_layer(ld.AvgVertPooling, with_inputs=True)])
AvgDenseWL2GCN = DenseWL2GCN.extend("AvgDenseWL2GCN", [
  cm.with_layer(ld.AvgEdgePooling, with_inputs=True)])

AvgWL2GCN = WL2GCN.extend("AvgWL2GCN", [
  cm.with_layer(lwl2.AvgPooling, with_inputs=True)])

# Max GNNs:
MaxDenseGCN = DenseGCN.extend("MaxDenseGCN", [
  cm.with_layer(ld.MaxVertPooling, with_inputs=True)])

# SortPool GNNs:
SortWL2GCN = WL2GCN.extend("SortWL2GCN", [
  cm.with_layer(lwl2.SortPooling, with_inputs=True)])
