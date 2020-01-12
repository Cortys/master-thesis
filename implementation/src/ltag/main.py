from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

import ltag.models as models
import ltag.datasets.synthetic as synthetic
import ltag.datasets.disk.tu.datasets as tu

def binary_classifier(model_class, learning_rate=0.001, **params):
  model = model_class(act="sigmoid", squeeze_output=True, **params)

  opt = keras.optimizers.Adam(learning_rate)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  return model

def logistic_regression(
  model_class, act="tanh", learning_rate=0.001, **params):
  model = model_class(act=act, squeeze_output=True, **params)

  opt = keras.optimizers.Adam(learning_rate)

  model.compile(
    optimizer=opt,
    loss="mse",
    metrics=["mae"])

  return model


log_dir = "../logs"
model_class = models.SortWL2GCN

# ds_raw = synthetic.triangle_dataset()(wl2_neighborhood=3).get_all(
#   model_class.input_type, shuffle=True)
ds_manager = tu.Proteins(
  wl2_neighborhood=6,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 10000,
    "upper_batch_edge_count": 30000,
    "batch_graph_count": 20
  })

def model_factory(ds):
  in_dim = ds.element_spec[0][0].shape[-1]

  return binary_classifier(
    model_class, layer_dims=[in_dim, 8, 8, 8, 1],
    bias=True, k_pool=512)


model_factory.input_type = model_class.input_type


def train(model, train_ds, val_ds=None, label=None):
  label = "_" + label if label is not None else ""

  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}{label}/",
    histogram_freq=1,
    write_images=False)
  es = tf.keras.callbacks.EarlyStopping(
    monitor="loss" if val_ds is None else "val_loss",
    patience=50,
    min_delta=0.0001,
    restore_best_weights=True)

  model.fit(
    train_ds, validation_data=val_ds,
    epochs=300, callbacks=[tb, es])


def evaluate(model_factory, ds_manager, repeat=1):
  outer_k = ds_manager.outer_k

  ds_type = model_factory.input_type

  for k in range(outer_k):
    test_ds = ds_manager.get_test_fold(k, output_type=ds_type)
    train_ds, val_ds = ds_manager.get_train_fold(k, output_type=ds_type)

    for i in range(repeat):
      print("Iteration", i, "of split", k, "...")
      model = model_factory(train_ds)
      train(model, train_ds, val_ds, f"{train_ds.name}_{model.name}")
      results = model.evaluate(test_ds)
      print("\nEvaluation", k, i, "|", model.metrics_names, "=", results)


# -%% codecell
evaluate(model_factory, ds_manager)
