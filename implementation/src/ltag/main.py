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

ds_raw = synthetic.twothree_dataset()

ds_utils.draw_from_ds(ds_raw, 1)

ds = ds_raw.batch(50)

# -%% codecell

model = models.AVG_EFGCN([1, 4, 1], act="tanh")

model.compile(optimizer=keras.optimizers.Adam(0.1), loss="mse", metrics=["mae"])

model.get_weights()

list(ds)

def train(label=None):
  label = "_" + label if label is not None else ""

  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}{label}/",
    histogram_freq=1,
    write_images=True)

  model.fit(ds, epochs=50, callbacks=[tb_callback])


train("twothree_filtered_4")
model.predict(ds), np.array(list(map(lambda x: x[1].numpy(), list(ds_raw))))
