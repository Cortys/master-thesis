from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import ltag.datasets.synthetic as synthetic
import ltag.models as models

ds = synthetic.loop_dataset(1000).batch(30)

model = models.EFGCN()
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss="mse", metrics=["mae"])

ds.element_spec, model.input_shape, model.output_shape

model.predict(ds)[100]

list(ds.take(1))[0][0][0].shape
list(ds.take(1))[0][1].shape

hist = model.fit(ds, epochs=10)

plt.plot(hist.history["loss"])
plt.title("Loss")
plt.show()
