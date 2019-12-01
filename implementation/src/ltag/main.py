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

log_dir = "../logs"

ds = synthetic.loop_dataset(1000).batch(50)

model = models.LTA_GCN()
model.summary()

model.get_weights()

model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics=["mae"])

ds.element_spec, model.input_shape, model.output_shape

model.predict(ds)

list(ds.take(1))[0][0][0].shape
list(ds.take(1))[0][1].shape

def train():
  t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  tb_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{t}/",
    histogram_freq=1,
    write_images=True)

  model.fit(ds, epochs=30, callbacks=[tb_callback])


train()
