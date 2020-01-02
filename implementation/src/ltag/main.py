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
modelClass = models.AVG_EdgeWL2GCN

ds_raw = synthetic.triangle_dataset(
  output_type=modelClass.input_type, shuffle=False)
# ds_raw = disk.load_classification_dataset("mutag")

# ds_utils.draw_from_ds(ds_raw, 1)

# edge2 encoded datasets are by definition pre-batched:
ds = ds_raw if modelClass.input_type == "edge2" else ds_raw.batch(50)

ds.element_spec

list(ds)
in_dim = ds.element_spec[0][0].shape[-1]
in_dim

# -%% codecell

model = modelClass(
  layer_dims=[in_dim, 8, 4, 1],
  act="tanh", squeeze_output=True, bias=True)

opt = keras.optimizers.Adam(0.005)

model.compile(
  optimizer=opt, loss="mse", metrics=["mae"])

model.get_weights()

def train(label=None):
  label = "_" + label if label is not None else ""

  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}{label}/",
    histogram_freq=1,
    write_images=False)

  model.fit(ds, epochs=200, callbacks=[tb_callback])


train("triangle_wl2")

# -%% codecell
model.predict(ds), np.array([
  y
  for ys in map(lambda x: x[1].numpy(), list(ds))
  for y in ys])
