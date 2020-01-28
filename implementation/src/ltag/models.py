from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import funcy as fy

import ltag.chaining.model as cm
import ltag.layers.dense as ld
import ltag.layers.wl2 as lwl2

@cm.model_inputs
def gnn_dense_inputs(layer_dims=None, in_dim=None, sparse=False):
  in_dim = in_dim or layer_dims[0]
  X = keras.Input(shape=(None, in_dim), name="X", sparse=sparse)
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

@cm.model_inputs
def gnn_wl2c_inputs():
  X = keras.Input(shape=(None,), dtype=tf.float32, name="X")
  ref_a = keras.Input(shape=(), dtype=tf.int32, name="ref_a")
  ref_b = keras.Input(shape=(), dtype=tf.int32, name="ref_b")
  backref = keras.Input(shape=(), dtype=tf.int32, name="backref")
  e_map = keras.Input(shape=(), dtype=tf.int32, name="e_map")
  v_count = keras.Input(shape=(), dtype=tf.int32, name="v_count")

  return X, ref_a, ref_b, backref, e_map, v_count


input_types = {
  "dense": gnn_dense_inputs,
  "wl2": gnn_wl2_inputs,
  "wl2c": gnn_wl2c_inputs
}

@cm.model_step
def select_features(inputs):
  return inputs[0]

@cm.model_step
def squeeze_output(X, squeeze_output=False):
  return tf.squeeze(X, -1)

def gnn_model(name, steps, input_type="dense"):
  m = cm.create_model(name, [
    input_types[input_type],
    *steps,
    squeeze_output],
    extend_at=-1)  # squeeze always comes last

  def add_attrs(m):
    m.input_type = input_type
    m.extend = fy.rcompose(m.extend, add_attrs)
    return m

  return add_attrs(m)

def DenseLayer(in_dim, out_dim, act="linear", bias=False, **kwargs):
  kwargs = fy.project(kwargs, [
    "kernel_initializer", "bias_initializer",
    "kernel_regularizer", "bias_regularizer", "activity_regularizer",
    "kernel_constraint", "bias_constraint"])

  return keras.layers.Dense(
    out_dim, input_shape=(in_dim,),
    activation=act, use_bias=bias,
    **kwargs)

def with_fc(model):
  "Add a fully connected NN to the given model type."
  return model.extend(model.name + "_FC", [
    cm.with_layers(DenseLayer, prefix="fc")])


# Non-aggregating GNNs:

DenseGCN = gnn_model("DenseGCN", [
  cm.with_layers(ld.GCNLayer, prefix="conv"),
  select_features])

DenseWL2GCN = gnn_model("DenseWL2GCN", [
  cm.with_layer(ld.EdgeFeaturePreparation),
  cm.with_layers(ld.WL2GCNLayer, prefix="conv"),
  select_features])

WL2GCN = gnn_model("WL2GCN", [
  cm.with_layers(lwl2.WL2GCNLayer, prefix="conv"),
  select_features],
  input_type="wl2c")

CWL2GCN = gnn_model("CWL2GCN", [
  cm.with_layers(lwl2.CWL2GCNLayer, prefix="conv"),
  select_features],
  input_type="wl2c")

# Averaging GNNs:

AvgDenseGCN = DenseGCN.extend("AvgDenseGCN", [
  cm.with_layer(ld.AvgVertPooling, with_inputs=True)])
AvgDenseWL2GCN = DenseWL2GCN.extend("AvgDenseWL2GCN", [
  cm.with_layer(ld.AvgEdgePooling, with_inputs=True)])

AvgWL2GCN = WL2GCN.extend("AvgWL2GCN", [
  cm.with_layer(lwl2.AvgPooling, with_inputs=True)])
AvgCWL2GCN = CWL2GCN.extend("AvgWCL2GCN", [
  cm.with_layer(lwl2.AvgPooling, with_inputs=True)])

# Max GNNs:
MaxDenseGCN = DenseGCN.extend("MaxDenseGCN", [
  cm.with_layer(ld.MaxVertPooling, with_inputs=True)])

# SortPool GNNs:
SortWL2GCN = WL2GCN.extend("SortWL2GCN", [
  cm.with_layer(lwl2.SortPooling, with_inputs=True)])
