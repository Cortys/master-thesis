from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import gc
import json
import io
import requests
import zipfile
from pathlib import Path
import pickle
import numpy as np
import networkx as nx

from ltag.utils import NumpyEncoder
from ltag.datasets.manager import GraphDatasetManager
from ltag.datasets.disk.utils import (
  parse_tu_data, create_graph_from_tu_data)

class StoredGraphDatasetManager(GraphDatasetManager):
  def _setup(self, wl2_batch_cache=True, no_wl2_load=False, **kwargs):
    self.wl2_batch_cache = wl2_batch_cache
    self._wl2_batch_dataset = None
    self.no_wl2_load = no_wl2_load
    self.root_dir = self.data_dir / self.name
    self.raw_dir = self.root_dir / "raw"
    if not self.raw_dir.exists():
      os.makedirs(self.raw_dir)
      self._download()

    self.processed_dir = self.root_dir / "processed"
    if not (self.processed_dir / f"{self.name}.pickle").exists():
      if not self.processed_dir.exists():
        os.makedirs(self.processed_dir)
      self._process()

  def _load_dataset(self):
    with open(self.processed_dir / f"{self.name}.pickle", "rb") as f:
      return pickle.load(f)

  def _load_wl2_dataset(self, neighborhood):
    wl2_dir = self.root_dir / "wl2" / f"n_{neighborhood}"

    if not (wl2_dir / f"{self.name}.pickle").exists():
      if not wl2_dir.exists():
        os.makedirs(wl2_dir)

      wl2_dataset = super()._load_wl2_dataset(neighborhood)

      with open(wl2_dir / f"{self.name}.pickle", "wb") as f:
        pickle.dump(wl2_dataset, f)

      return wl2_dataset
    else:
      with open(wl2_dir / f"{self.name}.pickle", "rb") as f:
        return pickle.load(f)

  def _load_wl2c_dataset(self, neighborhood):
    wl2_dir = self.root_dir / "wl2c" / f"n_{neighborhood}"

    if not (wl2_dir / f"{self.name}.pickle").exists():
      if not wl2_dir.exists():
        os.makedirs(wl2_dir)

      wl2_dataset = super()._load_wl2c_dataset(neighborhood)

      with open(wl2_dir / f"{self.name}.pickle", "wb") as f:
        pickle.dump(wl2_dataset, f)

      return wl2_dataset
    else:
      with open(wl2_dir / f"{self.name}.pickle", "rb") as f:
        return pickle.load(f)

  def _load_en_dataset(self):
    en_dir = self.root_dir / "en"

    if not (en_dir / f"{self.name}.pickle").exists():
      if not en_dir.exists():
        os.makedirs(en_dir)

      en_dataset = super()._load_en_dataset()

      with open(en_dir / f"{self.name}.pickle", "wb") as f:
        pickle.dump(en_dataset, f)

      return en_dataset
    else:
      with open(en_dir / f"{self.name}.pickle", "rb") as f:
        return pickle.load(f)

  def _make_splits(self):
    splits_filename = self.processed_dir / f"{self.name}_splits.json"

    if splits_filename.exists():
      return json.load(open(splits_filename, "r"))

    splits = super()._make_splits()

    with open(splits_filename, "w") as f:
      json.dump(splits[:], f, cls=NumpyEncoder)

    return splits

  def _compute_gram_matrix(self, output_fn):
    k_name = output_fn.__name__
    gram_dir = self.root_dir / "gram" / k_name
    gram_filename = gram_dir / "gram.pickle"

    if gram_filename.exists():
      with open(gram_filename, "rb") as f:
        gram = pickle.load(f)
        if gram == "OOM":
          print(f"{k_name} kernel OOM for {self.name}.")
          return None

        return gram

    if not gram_dir.exists():
      os.makedirs(gram_dir)

    try:
      gram = super()._compute_gram_matrix(output_fn)
    except MemoryError as err:
      print("Memory error during gram matrix computation:", err)
      gram = "OOM"

    with open(gram_filename, "wb") as f:
      pickle.dump(gram, f)

    if gram == "OOM":
      return None

    return gram

  def _get_wl2_batches(self, name, idxs=None):
    if not self.wl2_batch_cache:
      return super()._get_wl2_batches(name, idxs)

    graph_count = self.wl2_batch_size.get("batch_graph_count", -1)
    if graph_count == 1 and self._wl2_batch_dataset is not None:
      batches = self._wl2_batch_dataset
      if idxs is None:
        return batches
      else:
        return (batches[0][idxs], batches[1], batches[2])

    bc = f"n_{self.wl2_neighborhood}"

    if self.wl2_indices:
      bc += "_idx"

    if graph_count != 1:
      count_keys = [
        ("fuzzy_batch_edge_count", "fbec"),
        ("upper_batch_edge_count", "ubec"),
        ("batch_graph_count", "bgc")]
    else:
      count_keys = [("batch_graph_count", "bgc")]
      name = self.name

    for k, s in count_keys:
      if k in self.wl2_batch_size:
        v = self.wl2_batch_size[k]
        bc += f"_{s}_{v}"

    bn = f"{name}.pickle"

    batch_dir = self.root_dir / "wl2_batches" / bc

    if not (batch_dir / bn).exists():
      if not batch_dir.exists():
        os.makedirs(batch_dir)

      batches = super()._get_wl2_batches(
        name, idxs if graph_count != 1 else None)

      with open(batch_dir / bn, "wb") as f:
        pickle.dump(batches, f)

      if self.no_wl2_load:
        return
    elif self.no_wl2_load:  # Used for preprocessing managers.
      return
    else:
      with open(batch_dir / bn, "rb") as f:
        batches = pickle.load(f)

    if graph_count == 1:
      self._wl2_batch_dataset = batches
      self._wl2_dataset = None
      if idxs is not None:
        return (batches[0][idxs], batches[1], batches[2])

    return batches

  def prepare_wl2_batches(self, all_batch=False, normal=False, compact=True):
    if normal:
      assert self.wl2_batch_cache
      print("Normal WL2 encode...")
      if all_batch:
        print(f"Preparing full dataset batches of {self.name}...")
        self.get_all(output_type="wl2")
        gc.collect()

      for ok in range(self.outer_k or 1):
        for ik in range(self.inner_k or 1):
          print(f"Preparing train fold {ok} {ik} of {self.name}...")
          self.get_train_fold(ok, ik, output_type="wl2")
          gc.collect()

        print(f"Preparing test fold {ok} of {self.name}...")
        self.get_test_fold(ok, output_type="wl2")
        gc.collect()

      print("Prepared all normal batch folds.")

    if compact:
      print(f"Starting compact WL2 encode of {self.name}...")
      self.wl2c_dataset
      gc.collect()
      print(f"Compact WL2 encode of {self.name} completed.")

  def prepare_gram_matrices(self, kernels):
    print(f"Starting gram matrix computation of {self.name}...")

    for kernel in kernels:
      print(f"Preparing {kernel.__name__} kernel for {self.name}...")
      self.get_all(output_type=kernel)
      gc.collect()
      print(f"Prepared {kernel.__name__} kernel for {self.name}.")

    print(f"Completed gram matrix computation of {self.name}.")

  def _process(self):
    raise NotImplementedError

  def _download(self):
    raise NotImplementedError

  def export_dot(self, idx):
    dot_dir = self.root_dir / "dot"

    if not dot_dir.exists():
      os.makedirs(dot_dir)

    gs, ys = self.dataset
    g = gs[idx]
    y = ys[idx]
    name = f"g{idx}_{y}.dot"
    nx.drawing.nx_pydot.write_dot(g, dot_dir / name)


class TUDatasetManager(StoredGraphDatasetManager):
  URL = (
    "https://ls11-www.cs.tu-dortmund.de/"
    + "people/morris/graphkerneldatasets/{name}.zip")
  classification = True
  data_dir = Path("../data/tu")

  def _download(self):
    url = self.URL.format(name=self.name)
    response = requests.get(url)
    stream = io.BytesIO(response.content)
    with zipfile.ZipFile(stream) as z:
      for fname in z.namelist():
        z.extract(fname, self.raw_dir)

  def _process(self):
    graphs_data, d_nf, d_ef, n_nl, n_el = parse_tu_data(
      self.name, self.raw_dir)
    targets = graphs_data.pop("graph_labels")

    graphs, out_targets = [], []
    for i, target in enumerate(targets, 1):
      graph_data = {k: v[i] for (k, v) in graphs_data.items()}
      G = create_graph_from_tu_data(graph_data)

      if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        graphs.append(G)
        out_targets.append(target)

    graphs_a = np.empty(len(graphs), dtype="O")
    graphs_a[:] = graphs
    dataset = graphs_a, np.array(out_targets), d_nf, d_ef, n_nl, n_el

    with open(self.processed_dir / f"{self.name}.pickle", "wb") as f:
      pickle.dump(dataset, f)
