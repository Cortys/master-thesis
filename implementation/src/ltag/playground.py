from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

import ltag.models as models
import ltag.datasets.disk.tu.datasets as tu
import ltag.datasets.synthetic as synthetic
import ltag.evaluation.datasets as eval_ds
import ltag.evaluation.evaluate as evaluate

import ltag.evaluate_datasets as eval_main


def proteins_experient():
  model_class = models.AvgWL2GCN
  dsm = eval_ds.Proteins_6()

  model = model_class(
    act="sigmoid", squeeze_output=True,
    layer_dims=[dsm.dim_wl2_features(), 32, 32, 1],
    bias=True)

  opt = keras.optimizers.Adam(0.001)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  ds = dsm.get_all(
    output_type=model_class.input_type)

  evaluate.train(
    model, ds, verbose=1,
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
  ds = ds_raw.shuffle(1200)

  evaluate.train(
    model, ds, verbose=1,
    label=f"{dsm.name}_{model.name}")
  print(model.evaluate(ds))


dd_experient()

# eval_main.run(verbose=1)
# eval_main.quick_run(verbose=1)
# eval_main.resume(
#   "2020-01-15_14-26-15_NCI1_AvgWL2GCN_quick",
#   verbose=1)
# eval_main.summarize("2020-01-15_14-31-53_NCI1_AvgWL2GCN")
