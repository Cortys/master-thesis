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
    ds = self._generate_dataset()

    store_graphs_as_tu_data(*ds, self.name, self.raw_dir)

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
      n_labels = set()
      e_labels = set()

      for g in graphs:
        for n, d in g.nodes(data=True):
          if "label" in d:
            n_labels.add(d["label"])

        for u, v, d in g.edges(data=True):
          if "label" in d:
            e_labels.add(d["label"])

      n_nl = (
        max(n_labels) if n_labels != set() else 0)
      if n_nl != 0 and min(n_labels) == 0:
        n_nl += 1

      n_el = (
        max(e_labels) if e_labels != set() else 0)
      if n_el != 0 and min(e_labels) == 0:
        n_el += 1

      if len(graphs) > 0:
        d_nf, d_ef = ds_utils.get_graph_feature_dims(graphs[0])
      else:
        d_nf = 0
        d_ef = 0

      print(f"Generated {len(graphs)} graphs.")
      return graphs_a, y_a, d_nf, d_ef, n_nl, n_el

    if stored:
      base = StoredSyntheticDatasetManager
    else:
      base = SyntheticDatasetManager
      graphs, y, d_nf, d_ef, n_nl, n_el = generate()
      t = y.shape[-1]

      def generate():
        return graphs, y, d_nf, d_ef, n_nl, n_el

    class M(base):
      name = n

      def _generate_dataset(self):
        return generate()

    if stored:
      m = M()
      graphs, y, d_nf, d_ef, n_nl, n_el = m._load_dataset()
      t = y.shape[-1]

    cName = "".join(
      x for x in n.title()
      if not x.isspace() and x != "_")
    M.__name__ = cName + "Manager"
    M._dim_node_features = d_nf
    M._dim_edge_features = d_ef
    M._num_node_labels = n_nl
    M._num_edge_labels = n_el
    M._dim_target = t

    return M

  return w
