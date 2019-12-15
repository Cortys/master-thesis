from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

import ltag.datasets.synthetic as synthetic
import ltag.models as models

log_dir = "../logs"

ds_raw = synthetic.triangle_dataset()

# ds_utils.draw_from_ds(ds_raw, 1)

ds = ds_raw.batch(50)

# -%% codecell

model = models.AVG_EF2GCN(layer_dims=[2, 4, 1], act="tanh")

opt = keras.optimizers.Adam(0.01)

model.compile(optimizer=opt, loss="mse", metrics=["mae"])

model.get_weights()

def train(label=None):
  label = "_" + label if label is not None else ""

  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}{label}/",
    histogram_freq=1,
    write_images=True)

  model.fit(ds, epochs=200, callbacks=[tb_callback])


# -%% codecell
train("triangle_dataset_wl2")

model.predict(ds), np.array(list(map(lambda x: x[1].numpy(), list(ds_raw))))
