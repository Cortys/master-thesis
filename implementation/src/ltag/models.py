from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import funcy as fy

import ltag.chaining.model as cm
import ltag.layers.vert as vl
import ltag.layers.edge as ve

@cm.model_inputs
def gnn_vert_inputs(layer_dims, sparse=False):
  X = keras.Input(shape=(None, layer_dims[0]), name="X", sparse=sparse)
  A = keras.Input(shape=(None, None), name="A", sparse=sparse)
  n = keras.Input(shape=(), dtype=tf.int32, name="n")

  return X, A, n

@cm.model_inputs
def gnn_edge2_inputs():
  X = keras.Input(shape=(None,), dtype=tf.float32, name="X")
  ref_a = keras.Input(shape=(None,), dtype=tf.int32, name="ref_a")
  ref_b = keras.Input(shape=(None,), dtype=tf.int32, name="ref_b")
  e_map = keras.Input(shape=(), dtype=tf.int32, name="e_map")
  v_count = keras.Input(shape=(), dtype=tf.int32, name="v_count")

  return X, ref_a, ref_b, e_map, v_count


input_types = {
  "vert": gnn_vert_inputs,
  "edge2": gnn_edge2_inputs
}

@cm.model_step
def select_features(inputs):
  return inputs[0]

def gnn_model(name, steps, input_type="vert"):
  m = cm.create_model(name, [
    input_types[input_type],
    *steps])

  def add_attrs(m):
    m.input_type = input_type
    return m

  add_attrs(m)
  m.extend = fy.rcompose(m.extend, add_attrs)

  return m


# Non-aggregating GNNs:

VertGCN = gnn_model("VertGCN", [
  cm.with_layers(vl.GCNLayer),
  select_features])

VertWL2GCN = gnn_model("VertWL2GCN", [
  cm.with_layer(vl.EdgeFeaturePreparation),
  cm.with_layers(vl.WL2GCNLayer),
  select_features])

EdgeWL2GCN = gnn_model("EdgeWL2GCN", [
  cm.with_layers(ve.WL2GCNLayer),
  select_features],
  input_type="edge2")

# Averaging GNNs:

AVG_VertGCN = VertGCN.extend("AVG_VertGCN", [
  cm.with_layer(vl.AVGVertPooling, with_inputs=True)])
AVG_VertWL2GCN = VertWL2GCN.extend("AVG_VertWL2GCN", [
  cm.with_layer(vl.AVGEdgePooling, with_inputs=True)])

AVG_EdgeWL2GCN = EdgeWL2GCN.extend("AVG_EdgeWL2GCN", [
  cm.with_layer(ve.AVGPooling, with_inputs=True)])
