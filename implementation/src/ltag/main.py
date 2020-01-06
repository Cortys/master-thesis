from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

import ltag.datasets.synthetic as synthetic
import ltag.datasets.disk as disk
import ltag.models as models

log_dir = "../logs"
modelClass = models.AvgEdgeWL2GCN

modelClass

# ds_raw = synthetic.twothree_dataset(
#   output_type=modelClass.input_type,
#   shuffle=True, neighborhood=2)
# ds_raw = disk.load_classification_dataset(
#   "nci1", output_type=modelClass.input_type,
#   shuffle=True, neighborhood=8)
ds_raw = disk.load_sparse_classification_dataset("proteins")

ds_name = ds_raw.name

# import ltag.datasets.utils as dsutils
# dsutils.draw_from_ds(synthetic.triangle_dataset(), 2)

# edge2 encoded datasets are by definition pre-batched:
ds = (
  ds_raw if modelClass.input_type == "edge2"
  else ds_raw.shuffle(1000, reshuffle_each_iteration=False).batch(50))
ds.prefetch(20)

ds.element_spec
# list(ds_raw)[0]

in_dim = ds.element_spec[0][0].shape[-1]
squeeze_output = len(ds.element_spec[1].shape) == 1

# -%% codecell

model = modelClass(
  layer_dims=[in_dim, 32, 32, 1],
  act="sigmoid", squeeze_output=squeeze_output,
  bias=True)

opt = keras.optimizers.Adam(0.005)


model.compile(
  optimizer=opt,
  loss="binary_crossentropy",
  metrics=["accuracy"])
# model.compile(
#   optimizer=opt,
#   loss="mse",
#   metrics=["mae"])

model.get_weights()

def train(label=None):
  label = "_" + label if label is not None else ""

  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}{label}/",
    histogram_freq=1,
    write_images=False)
  es = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=50,
    min_delta=0.0001,
    restore_best_weights=True)

  model.fit(ds, epochs=500, callbacks=[tb])


train(f"{ds_name}_{model.name}")

# -%% codecell
np.round(np.stack([model.predict(ds), np.array([
  y
  for ys in map(lambda x: x[1].numpy(), list(ds))
  for y in ys])], axis=1), decimals=3)
