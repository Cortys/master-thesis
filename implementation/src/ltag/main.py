from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

import ltag.datasets.synthetic as synthetic
import ltag.models as models
import ltag.datasets.utils as ds_utils

log_dir = "../logs"

ds_raw = synthetic.triangle_dataset()

ds_utils.draw_from_ds(ds_raw, 0)

ds = ds_raw.batch(50)

model = models.LTA_GCN(2, 2)

model.compile(optimizer=keras.optimizers.Adam(0.1), loss="mse", metrics=["mae"])

model.get_weights()

list(ds)

def train():
  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}/",
    histogram_freq=1,
    write_images=True)

  model.fit(ds, epochs=50, callbacks=[tb_callback])


train()
model.predict(ds), np.array(list(map(lambda x: x[1].numpy(), list(ds_raw))))
