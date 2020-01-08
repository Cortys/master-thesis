import io
import os
import json
import requests
import zipfile
from pathlib import Path
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold

import ltag.datasets.disk.tu.utils as tu_utils

# Implementation adapted from https://github.com/diningphil/gnn-comparison.

data_dir = "../data/2_tu"

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

class GraphDatasetManager:
  def __init__(
    self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None, seed=42,
    holdout_test_size=0.1, use_node_degree=False, use_node_attrs=False,
    use_one=False,
    max_reductions=10, DATA_DIR=data_dir):

    self.root_dir = Path(DATA_DIR) / self.name
    self.kfold_class = kfold_class
    self.holdout_test_size = holdout_test_size
    self.use_node_degree = use_node_degree
    self.use_node_attrs = use_node_attrs
    self.use_one = use_one

    self.outer_k = outer_k
    assert (outer_k is not None and outer_k > 0) or outer_k is None

    self.inner_k = inner_k
    assert (inner_k is not None and inner_k > 0) or inner_k is None

    self.seed = seed

    self.raw_dir = self.root_dir / "raw"
    if not self.raw_dir.exists():
      os.makedirs(self.raw_dir)
      self._download()

    self.processed_dir = self.root_dir / "processed"
    if not (self.processed_dir / f"{self.name}.pickle").exists():
      if not self.processed_dir.exists():
        os.makedirs(self.processed_dir)
      self._process()

    with open(self.processed_dir / f"{self.name}.pickle", "rb") as f:
      self.dataset = pickle.load(f)

    splits_filename = self.processed_dir / f"{self.name}_splits.json"
    if not splits_filename.exists():
      self.splits = []
      self._make_splits()
    else:
      self.splits = json.load(open(splits_filename, "r"))

  @property
  def num_graphs(self):
    return len(self.dataset[0])

  @property
  def dim_target(self):
    return self._dim_target

  @property
  def dim_features(self):
    return self._dim_features

  def _process(self):
    raise NotImplementedError

  def _download(self):
    raise NotImplementedError

  def _make_splits(self):
    graphs, targets = self.dataset
    all_idxs = np.arange(len(targets))

    if self.outer_k is None:  # holdout assessment strategy
      assert self.holdout_test_size is not None

      if self.holdout_test_size == 0:
        train_o_split, test_split = all_idxs, []
      else:
        outer_split = train_test_split(
          all_idxs,
          stratify=targets,
          test_size=self.holdout_test_size)
        train_o_split, test_split = outer_split
      split = {"test": all_idxs[test_split], "model_selection": []}

      train_o_targets = targets[train_o_split]

      if self.inner_k is None:  # holdout model selection strategy
        if self.holdout_test_size == 0:
          train_i_split, val_i_split = train_o_split, []
        else:
          train_i_split, val_i_split = train_test_split(
            train_o_split,
            stratify=train_o_targets,
            test_size=self.holdout_test_size)
        split["model_selection"].append(
          {"train": train_i_split, "validation": val_i_split})

      else:  # cross validation model selection strategy
        inner_kfold = self.kfold_class(
          n_splits=self.inner_k, shuffle=True)
        for train_ik_split, val_ik_split in inner_kfold.split(
          train_o_split, train_o_targets):
          split["model_selection"].append({
            "train": train_o_split[train_ik_split],
            "validation": train_o_split[val_ik_split]
          })

      self.splits.append(split)

    else:  # cross validation assessment strategy

      outer_kfold = self.kfold_class(
        n_splits=self.outer_k, shuffle=True)

      for train_ok_split, test_ok_split in outer_kfold.split(
        X=all_idxs, y=targets):
        split = {
          "test": all_idxs[test_ok_split],
          "model_selection": []
        }

        train_ok_targets = targets[train_ok_split]

        if self.inner_k is None:  # holdout model selection strategy
          assert self.holdout_test_size is not None
          train_i_split, val_i_split = train_test_split(
            train_ok_split,
            stratify=train_ok_targets,
            test_size=self.holdout_test_size)
          split["model_selection"].append(
            {"train": train_i_split, "validation": val_i_split})

        else:  # cross validation model selection strategy
          inner_kfold = self.kfold_class(
            n_splits=self.inner_k, shuffle=True)
          for train_ik_split, val_ik_split in inner_kfold.split(
            train_ok_split, train_ok_targets):
            split["model_selection"].append({
              "train": train_ok_split[train_ik_split],
              "validation": train_ok_split[val_ik_split]
            })

        self.splits.append(split)

    filename = self.processed_dir / f"{self.name}_splits.json"
    with open(filename, "w") as f:
      json.dump(self.splits[:], f, cls=NumpyEncoder)


class TUDatasetManager(GraphDatasetManager):
  URL = (
    "https://ls11-www.cs.tu-dortmund.de/"
    + "people/morris/graphkerneldatasets/{name}.zip")
  classification = True

  def _download(self):
    url = self.URL.format(name=self.name)
    response = requests.get(url)
    stream = io.BytesIO(response.content)
    with zipfile.ZipFile(stream) as z:
      for fname in z.namelist():
        z.extract(fname, self.raw_dir)

  def _process(self):
    graphs_data, num_node_labels, num_edge_labels = tu_utils.parse_tu_data(
      self.name, self.raw_dir)
    targets = graphs_data.pop("graph_labels")

    graphs, out_targets = [], []
    for i, target in enumerate(targets, 1):
      graph_data = {k: v[i] for (k, v) in graphs_data.items()}
      G = tu_utils.create_graph_from_tu_data(
        graph_data, num_node_labels, num_edge_labels)

      if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        graphs.append(G)
        out_targets.append(target)

    dataset = (np.array(graphs), np.array(out_targets))

    with open(self.processed_dir / f"{self.name}.pickle", "wb") as f:
      pickle.dump(dataset, f)
