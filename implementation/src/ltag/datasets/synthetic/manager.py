from __future__ import absolute_import, division, print_function,\
  unicode_literals

import pickle
import numpy as np
import funcy as fy
from pathlib import Path

import ltag.chaining.pipeline as cp
import ltag.datasets.utils as ds_utils
from ltag.datasets.manager import GraphDatasetManager
from ltag.datasets.disk.manager import TUDatasetManager
from ltag.datasets.disk.utils import (
  parse_tu_data, create_graph_from_tu_data, store_graphs_as_tu_data)

class SyntheticDatasetManager(GraphDatasetManager):
  def _load_dataset(self):
    return self._generate_dataset()

  def _generate_dataset(self):
   raise NotImplementedError

class StoredSyntheticDatasetManager(TUDatasetManager):
  data_dir = Path("../data/synthetic")

  def _download(self):
    graphs, targets = self._generate_dataset()

    store_graphs_as_tu_data(graphs, targets, self.name, self.raw_dir)

  def _generate_dataset(self):
   raise NotImplementedError

def synthetic_dataset(f):
  @fy.wraps(f)
  def w(*args, stored=False, **kwargs):
    n = f.__name__

    def generate():
      graphs, y = cp.tolerant(f)(*args, **kwargs)
      graphs_a = np.empty(len(graphs), dtype="O")
      graphs_a[:] = graphs
      y_a = np.array(y)

      print(f"Generated {n} graphs.")
      return graphs_a, y_a

    if stored:
      base = StoredSyntheticDatasetManager
    else:
      base = SyntheticDatasetManager
      graphs, y = generate()

      def generate():
        return graphs, y

    class M(base):
      name = n

      def _generate_dataset(self):
        return generate()

    if stored:
      m = M()
      graphs, y = m._load_dataset()

    if len(graphs) > 0:
      nf, ef = ds_utils.get_graph_feature_dims(graphs[0])
      t = y.shape[-1]
    else:
      nf = 0
      ef = 0
      t = 0

    cName = "".join(
      x for x in n.title()
      if not x.isspace() and x != "_")
    M.__name__ = cName + "Manager"
    M._dim_node_features = nf
    M._dim_edge_features = ef
    M._dim_target = t

    return M

  return w
