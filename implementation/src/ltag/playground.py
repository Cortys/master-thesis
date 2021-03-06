from __future__ import absolute_import, division, print_function,\
  unicode_literals

import warnings
# Ignore future warnings caused by grakel:
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn as sk
import grakel as gk
import funcy as fy
import json

import ltag.models.gnn as gnn_models
import ltag.models.kernel as kernel_models
import ltag.datasets.synthetic.datasets as synthetic
import ltag.evaluation.datasets as eval_ds
import ltag.evaluation.evaluate as evaluate
import ltag.evaluate_datasets as eval_main
from ltag.utils import cart, cart_merge, entry_duplicator, NumpyEncoder

# -%% codecell

def mutag_experient():
  model_class = gnn_models.AvgWL2GCN
  dsm = eval_ds.Mutag_8()

  # model_class = gnn_models.with_fc(model_class)
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
  model_class = gnn_models.AvgWL2GCN
  dsm = eval_ds.Proteins_6()

  model = model_class(
    act="sigmoid", squeeze_output=True,
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
  model_class = gnn_models.AvgWL2GCN
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
  model_class = gnn_models.SagCWL2GCN
  dsm = eval_ds.NCI1_8()

  model = model_class(
    act="sigmoid", local_act="relu",
    conv_layer_dims=[dsm.dim_wl2_features(), 64, 64, 64, 1],
    # conv_stack_tf="keep_input",
    att_conv_layer_dims=[dsm.dim_wl2_features(), 1],
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
    model, train, test, verbose=1, epochs=1000,
    label=f"{dsm.name}_{model.name}")
  # print(model.evaluate(test))

def reddit_experient():
  model_class = gnn_models.SagCWL2GCN
  dsm = eval_ds.RedditBinary_1()

  model = model_class(
    act="sigmoid", local_act="sigmoid", squeeze_output=True,
    conv_layer_dims=[dsm.dim_wl2_features(), 40, 40, 40, 1],
    att_conv_layer_dims=[dsm.dim_wl2_features(), 1],
    conv_stack_tf="keep_input",
    bias=True)

  opt = keras.optimizers.Adam(0.0007)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  train, val = dsm.get_train_fold(0, output_type=model_class.input_type)
  test = dsm.get_test_fold(0, output_type=model_class.input_type)

  evaluate.train(
    model, train, test, verbose=2, epochs=1000,
    label=f"{dsm.name}_{model.name}")
  print(model.evaluate(test))

def imdb_experient():
  model_class = gnn_models.SagCWL2GCN
  dsm = eval_ds.IMDB_8()

  model = model_class(
    act="sigmoid", local_act="sigmoid", squeeze_output=True,
    conv_layer_dims=[dsm.dim_wl2_features(), 40, 40, 40, 1],
    att_conv_layer_dims=[dsm.dim_wl2_features(), 1],
    conv_stack_tf="keep_input",
    bias=True)

  opt = keras.optimizers.Adam(0.0007)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  train, val = dsm.get_train_fold(0, output_type=model_class.input_type)
  test = dsm.get_test_fold(0, output_type=model_class.input_type)

  evaluate.train(
    model, train, test, verbose=2, epochs=1000,
    label=f"{dsm.name}_{model.name}")
  print(model.evaluate(test))

def wl2_power_experiment():
  # model_class = gnn_models.AvgGIN
  model_class = gnn_models.AvgCWL2GCN
  model_class = gnn_models.with_fc(model_class)
  dsm = synthetic.threesix_dataset(stored=True)(
    wl2_neighborhood=1)  # ok(3, 2, 1)

  if model_class.input_type == "dense":
    in_dim = dsm.dim_dense_features()
  elif model_class.input_type == "wl1":
    in_dim = dsm.dim_wl1_features()
  else:
    in_dim = dsm.dim_wl2_features()

  if in_dim == 0:
    in_dim = 1

  opt = keras.optimizers.Adam(0.1)

  model = model_class(
    act="sigmoid", squeeze_output=True,
    layer_dims=[in_dim, 4, 1],
    fc_layer_dims=[1, 10, 1],
    neighborhood_mask=1,  # ok(3, 2), nok(-1, 1)
    bias=False, no_local_hash=True)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds = dsm.get_all(
    output_type=model_class.input_type,
    shuffle=True)

  evaluate.train(
    model, ds, verbose=1,
    epochs=200, patience=200,
    label=f"{dsm.name}_{model.name}")

  print(
    list(dsm.get_all(output_type="dense"))[0][1].numpy(),
    model.predict(dsm.get_all(output_type=model_class.input_type)))

def synthetic_experiment2():
  model_class = gnn_models.AvgCWL2GCN
  dsm = synthetic.balanced_triangle_classification_dataset(stored=True)(
    with_holdout=False,
    wl2_neighborhood=1,
    wl2_batch_size=dict(batch_graph_count=228))

  if model_class.input_type == "dense":
    in_dim = dsm.dim_dense_features()
  else:
    in_dim = dsm.dim_wl2_features()

  if in_dim == 0:
    in_dim = 1

  opt = keras.optimizers.Adam(0.0005)

  model = model_class(
    act="sigmoid", local_act="relu",
    squeeze_output=True,
    layer_dims=[in_dim, 32, 32, 32, 1],
    att_conv_layer_dims=[in_dim, 32, 32, 32, 1],
    bias=True, no_local_hash=True)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  i = 5
  ds = dsm.get_train_fold(
    i, output_type=model_class.input_type)
  ds_test = dsm.get_test_fold(
    i, output_type=model_class.input_type)

  evaluate.train(
    model, ds, ds_test, verbose=2,
    epochs=5000, patience=2000,
    label=f"{dsm.name}_{model.name}")

def kernel_experiment():
  model_class = kernel_models.WL_sp
  model = model_class(C=0.001)
  dsm = synthetic.balanced_triangle_classification_dataset(stored=True)(
    with_holdout=False)

  for i in range(10):
    ds = dsm.get_train_fold(
      i, output_type=model_class.input_type)
    ds_test = dsm.get_test_fold(
      i, output_type=model_class.input_type)
    #ds = dsm.get_all(output_type=model_class.input_type)
    print(i)
    print(evaluate.train(model, ds, ds_test, label=f"{dsm.name}_{model.name}").history)

def ds_stats():
  datasets = [
    synthetic.balanced_triangle_classification_dataset(stored=True)(),
    eval_ds.NCI1_8(), eval_ds.Proteins_5(), eval_ds.DD_2(),
    eval_ds.RedditBinary_1(), eval_ds.IMDB_8()
    ]
  s = {}

  for ds in datasets:
    print(f"DS stats for {ds.name}:")
    stats = ds.stats()
    print(stats)
    s[ds.name] = stats
    print("-------------------------")

  with open("/data/stats.json", "w") as f:
    json.dump(s, f, cls=NumpyEncoder, indent="\t")
  print("Done.")

wl2_power_experiment()
# synthetic_experiment2()
# imdb_experient()

# synthetic.balanced_triangle_classification_dataset(stored=True)().draw(152, label_colors=True)
# synthetic.balanced_triangle_classification_dataset(stored=True)().draw(153, label_colors=True)
# synthetic.balanced_triangle_classification_dataset(stored=True)().export_dot(152)
# synthetic.balanced_triangle_classification_dataset(stored=True)().export_dot(153)
# ds_stats()
# p = eval_ds.Proteins_5()
# d = eval_ds.DD_2()
# nci = eval_ds.NCI1_8()
# imdb = eval_ds.IMDB_8()
# reddit = eval_ds.RedditBinary_1()

# nci.draw(30)
# nci.draw(33)
# nci.draw(204)
# nci.draw(1300)
# nci.draw(4101)
#
# imdb.draw(107)
# imdb.draw(999)
# imdb.export_dot(999)
#
# reddit.draw(2)
# reddit.draw(100)
# reddit.export_dot(100)
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
# g2 = utils.make_wl_batch([ds.wl2_dataset[0][i]], True)
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
