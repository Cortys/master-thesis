from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn as sk

import ltag.models as models
import ltag.datasets.synthetic as synthetic
import ltag.evaluation.datasets as eval_ds
import ltag.evaluation.evaluate as evaluate
import ltag.evaluate_datasets as eval_main

import grakel as gk

# -%% codecell

def mutag_experient():
  model_class = models.AvgWL2GCN
  dsm = eval_ds.Mutag_8()

  # model_class = models.with_fc(model_class)
  model = model_class(
    act="sigmoid", squeeze_output=True,
    conv_layer_dims=[dsm.dim_wl2_features(), 64, 64, 64, 1],
    fc_layer_dims=[64, 64, 1],
    bias=True)

  opt = keras.optimizers.Adam(0.0001)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds_raw = dsm.get_all(
    output_type=model_class.input_type)
  ds = ds_raw.shuffle(1200)

  evaluate.train(
    model, ds, verbose=1, epochs=1000,
    label=f"{dsm.name}_{model.name}")
  print(model.evaluate(ds))

def proteins_experient():
  model_class = models.AvgWL2GCN
  dsm = eval_ds.Proteins_6()

  model = model_class(
    act="sigmoid", squeeze_output=False,
    conv_layer_dims=[dsm.dim_wl2_features(), 64, 64, 64, 1],
    vert_only_pool=False,
    bias=True)

  opt = keras.optimizers.Adam(0.0001)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds, ds_val = dsm.get_train_fold(
    1, output_type=model_class.input_type)

  evaluate.train(
    model, ds, ds_val, verbose=1,
    label=f"{dsm.name}_{model.name}")

def dd_experient():
  model_class = models.AvgWL2GCN
  dsm = eval_ds.DD_2()

  model = model_class(
    act="sigmoid", squeeze_output=True,
    layer_dims=[dsm.dim_wl2_features(), 64, 64, 64, 1],
    bias=True)

  opt = keras.optimizers.Adam(0.00001)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds_raw = dsm.get_all(
    output_type=model_class.input_type)
  ds = ds_raw

  evaluate.train(
    model, ds, verbose=1,
    label=f"{dsm.name}_{model.name}")
  print(model.evaluate(ds))

def nci1_experient():
  model_class = models.AvgCWL2GCN
  dsm = eval_ds.NCI1_8()

  model = model_class(
    act="sigmoid", local_act="relu",
    conv_layer_dims=[dsm.dim_wl2_features(), 64, 64, 64, 1],
    # conv_layer_args=[None, None, None, {
    #   "act": "sigmoid"
    # }],
    squeeze_output=True,
    bias=True)

  opt = keras.optimizers.Adam(0.0001)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  train, val = dsm.get_train_fold(0, output_type=model_class.input_type)
  test = dsm.get_test_fold(0, output_type=model_class.input_type)

  evaluate.train(
    model, train, test, verbose=1, epochs=500,
    label=f"{dsm.name}_{model.name}")
  # print(model.evaluate(test))

def reddit_experient():
  model_class = models.AvgWL2GCN
  dsm = eval_ds.RedditBinary_1()

  model = model_class(
    act="sigmoid", squeeze_output=True,
    layer_dims=[dsm.dim_wl2_features(), 64, 64, 64, 1],
    vert_only_pool=True,
    bias=True)

  opt = keras.optimizers.Adam(0.00001)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds_raw = dsm.get_all(
    output_type=model_class.input_type)
  ds = ds_raw

  evaluate.train(
    model, ds, verbose=1, epochs=1000,
    label=f"{dsm.name}_{model.name}")
  print(model.evaluate(ds))

def wl2_power_experiment():
  model_class = models.AvgWL2GCN
  # model_class = models.with_fc(model_class)
  dsm = synthetic.threesix_dataset()(
    wl2_neighborhood=1)  # ok(3, 2, 1)

  if model_class.input_type == "wl2c":
    in_dim = dsm.dim_wl2_features()
  else:
    in_dim = dsm.dim_node_features

  if in_dim == 0:
    in_dim = 1

  opt = keras.optimizers.Adam(0.1)

  model = model_class(
    act="sigmoid", squeeze_output=True,
    layer_dims=[in_dim, 4, 1],
    fc_layer_dims=[1, 2, 1],
    local_hash="multiply",
    neighborhood_mask=1,  # ok(3, 2), nok(-1, 1)
    bias=False)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds = dsm.get_all(
    output_type=model_class.input_type,
    shuffle=True)

  evaluate.train(
    model, ds, verbose=1,
    epochs=100, patience=200,
    label=f"{dsm.name}_{model.name}")

  print(
    list(dsm.get_all(output_type="dense"))[0][1].numpy(),
    model.predict(dsm.get_all(output_type=model_class.input_type)))


nci1_experient()
#
# list(dsm.get_all(output_type="grakel")[0])
#
# k = gk.GraphKernel(kernel=[
#   {"name": "weisfeiler_lehman", "n_iter": 5},
#   {"name": "subtree_wl"}],
#   normalize=True)
#
# k.fit_transform(*dsm.get_all(output_type="grakel"))
# k.transform(list(dsm.get_all(output_type="grakel")[0])[:1])

# eval_main.run(verbose=1)
# eval_main.quick_run(verbose=1)
# eval_main.resume(
#   "2020-01-15_14-26-15_NCI1_AvgWL2GCN_quick",
#   verbose=1)
# eval_main.summarize("2020-01-15_14-31-53_NCI1_AvgWL2GCN")

# tf.config.list_physical_devices('GPU')

# %% codecell

# import tensorflow as tf
# import networkx as nx
# import ltag.ops as ops
# import ltag.datasets.synthetic as synthetic
# import ltag.datasets.utils as utils
#
# ds = synthetic.threesix_dataset()(
#   wl2_neighborhood=1, wl2_indices=True)
#
# i = 1
# # WL1:
#
# g1 = ds.dataset[0][i]
#
# (X, A, n), _ = list(utils.to_dense_ds([g1], [1]).batch(1))[0]
#
# X_t = tf.linalg.matrix_transpose(X)
# X_d = tf.linalg.diag(X_t)
# X_e = tf.transpose(X_d, perm=(0, 2, 3, 1))
#
# A_e = tf.expand_dims(A, axis=-1)
#
# AX_e = tf.concat([X_e, A_e], axis=-1)
#
# AX_e
#
# mask = ops.neighborhood_mask(AX_e, 1)
#
# # WL2:
#
# g2 = utils.make_wl2_batch([ds.wl2_dataset[0][i]], True)
#
# x2 = tf.constant(g2[0])
# a2 = tf.constant(g2[1])
# b2 = tf.constant(g2[2])
# idx2 = tf.constant(g2[5], dtype=tf.int64)
#
# x2, idx2
#
# AX_e, x2[:, :-1]
#
# x2, a2, b2, idx2
#
# px2 = ops.aggregate_edge_features_using_refs(x2, a2, b2, tf.add)[:, :2]
#
# px2
#
# s1 = tf.SparseTensor(indices=idx2, values=px2[:, 0], dense_shape=[6, 6])
# s2 = tf.SparseTensor(indices=idx2, values=px2[:, 1], dense_shape=[6, 6])
# s1 = tf.sparse.expand_dims(s1, -1)
# s2 = tf.sparse.expand_dims(s2, -1)
# s = tf.cast(tf.sparse.to_dense(tf.sparse.concat(-1, [s1, s2])), tf.float32)
#
# ops.aggregate_edge_features(AX_e, tf.add)[0] * mask, s
