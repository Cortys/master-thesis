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

ds_raw = synthetic.triangle_dataset(sparse=False)
# ds_raw = disk.load_classification_dataset("mutag")

# ds_utils.draw_from_ds(ds_raw, 1)

ds = ds_raw.batch(50)
list(range(1, 10))
ds.element_spec

list(ds)

in_dim = ds.element_spec[0][0].shape[-1]

# -%% codecell

model = models.AVG_EF2GCN(
  layer_dims=[in_dim, 4, 1],
  act="tanh", squeeze_output=False, sparse=False, masked_bias=True)

opt = keras.optimizers.Adam(0.03)

model.compile(
  optimizer=opt, loss="mse", metrics=["mae"])

model.get_weights()

def train(label=None):
  label = "_" + label if label is not None else ""

  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}{label}/",
    histogram_freq=1,
    write_images=True)

  model.fit(ds, epochs=100, callbacks=[tb_callback])


# -%% codecell
train("triangle_wl2")

model.predict(ds), np.array(list(map(lambda x: x[1].numpy(), list(ds_raw))))
