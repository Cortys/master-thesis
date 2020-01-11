import os
import json
import numpy as np
import pickle
import funcy as fy
from sklearn.model_selection import train_test_split, StratifiedKFold

import ltag.chaining.pipeline as cp
import ltag.datasets.utils as ds_utils

# Implementation adapted from https://github.com/diningphil/gnn-comparison.

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


class GraphDatasetManager:
  def __init__(
    self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None,
    seed=42, holdout_test_size=0.1,
    wl2_neighborhood=1, wl2_cache=True):

    self.kfold_class = kfold_class
    self.holdout_test_size = holdout_test_size
    self.wl2_neighborhood = wl2_neighborhood
    self.wl2_cache = wl2_cache

    self.outer_k = outer_k
    assert (outer_k is not None and outer_k > 0) or outer_k is None

    self.inner_k = inner_k
    assert (inner_k is not None and inner_k > 0) or inner_k is None

    self.seed = seed

    self._dataset = None
    self._wl2_dataset = None
    self._splits = None

    self._setup()

  def _setup(self):
    pass

  def _load_dataset(self):
    raise NotImplementedError

  def _load_wl2_dataset(self, neighborhood):
    graphs, targets = self.dataset

    print("encoding wl2 graphs...")

    wl2_graphs = np.array([
      ds_utils.wl2_encode(
        g, self._dim_node_features, self._dim_edge_features, neighborhood)
      for g in graphs])

    print("encoded wl2 graphs")

    return wl2_graphs, targets

  @property
  def dataset(self):
    if self._dataset is None:
        self._dataset = self._load_dataset()

    print("got dataset")

    return self._dataset

  @property
  def wl2_dataset(self):
    if self._wl2_dataset is None:
      self._wl2_dataset = self._load_wl2_dataset(self.wl2_neighborhood)

    return self._wl2_dataset

  @property
  def splits(self):
    if self._splits is None:
      self._splits = self._make_splits()

    return self._splits

  @property
  def num_graphs(self):
    return len(self.dataset[0])

  @property
  def dim_target(self):
    return self._dim_target

  @property
  def dim_node_features(self):
    return self._dim_node_features

  @property
  def dim_edge_features(self):
    return self._dim_edge_features

  @property
  def max_num_nodes(self):
    return self._max_num_nodes

  def _make_splits(self):
    graphs, targets = self.dataset
    all_idxs = np.arange(len(targets))
    splits = []

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

      splits.append(split)

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

        splits.append(split)

    return splits

  def get_all(self, output_type="dense", idxs=None, shuffle=False):
    print("getting ds...")

    if shuffle:
      if idxs is None:
        idxs = np.arange(self.num_graphs)
      else:
        idxs = np.array(idxs)

      np.random.shuffle(idxs)

    if output_type == "dense":
      graphs, targets = self.dataset

      if idxs is not None:
        graphs = graphs[idxs]
        targets = targets[idxs]

      ds = ds_utils.to_dense_ds(graphs, targets)
    elif output_type == "wl2":
      graphs, targets = self.wl2_dataset if self.wl2_cache else self.dataset

      if idxs is not None:
        graphs = graphs[idxs]
        targets = targets[idxs]

      print("creating ds...")

      ds = ds_utils.to_wl2_ds(
        graphs, targets,
        preencoded=self.wl2_cache)

    ds.name = self.name

    print("created ds")

    return ds

  def get_test_fold(
    self, outer_idx, batch_size=1, output_type="dense"):
    outer_idx = outer_idx or 0

    idxs = self.splits[outer_idx]["test"]

    return self.get_all(output_type, idxs)

  def get_train_fold(
    self, outer_idx, inner_idx=None, batch_size=1, output_type="dense"):
    outer_idx = outer_idx or 0
    inner_idx = inner_idx or 0

    idxs = self.splits[outer_idx]["model_selection"][inner_idx]

    return self.get_all(output_type, idxs)


class StoredGraphDatasetManager(GraphDatasetManager):
  def _setup(self):
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

  def _make_splits(self):
    splits_filename = self.processed_dir / f"{self.name}_splits.json"

    if splits_filename.exists():
      return json.load(open(splits_filename, "r"))

    splits = super()._make_splits()

    with open(splits_filename, "w") as f:
      json.dump(splits[:], f, cls=NumpyEncoder)

    return splits

  def _process(self):
    raise NotImplementedError

  def _download(self):
    raise NotImplementedError


class SyntheticGraphDatasetManager(GraphDatasetManager):
  pass


def synthetic_dataset(f):
  @fy.wraps(f)
  def w(*args, **kwargs):
    print("creating graphs...")
    r = cp.tolerant(f)(*args, **kwargs)

    if len(r) == 3:
      n, graphs, y = r
    else:
      n = f.__name__
      graphs, y = r

    if len(graphs) > 0:
      nf, ef = ds_utils.get_graph_feature_dims(graphs[0])
      t = y.shape[-1]
    else:
      nf = 0
      ef = 0
      t = 0

    print("created graphs")

    class M(SyntheticGraphDatasetManager):
      name = n
      _dim_node_features = nf
      _dim_edge_features = ef
      _dim_target = t

      def _load_dataset(self):
        return np.array(graphs), np.array(y)

    cName = "".join(
      x for x in n.title()
      if not x.isspace() and x != "_")
    M.__name__ = cName + "Manager"

    return M

  return w
