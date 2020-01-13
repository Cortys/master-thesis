from __future__ import absolute_import, division, print_function,\
  unicode_literals

from tensorflow import keras

import ltag.models as models

import ltag.datasets.disk.tu.datasets as tu
import ltag.datasets.synthetic as synthetic

learning_rates = [0.001]

adams = [keras.optimizers.Adam(lr) for lr in learning_rates]

Proteins = {
  "dataset": {
    "manager": tu.Proteins,
    "wl2_neighborhood": 6,
    "wl2_batch_size": {
      "fuzzy_batch_edge_count": 10000,
      "upper_batch_edge_count": 30000,
      "batch_graph_count": 20
    }
  },
  "model": {
    "class": models.SortWL2GCN,
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"]
  },
  "hyperparams": {
    "optimizer": adams,
  }
}
