from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import json
import numpy as np
import pickle
import funcy as fy
from sklearn.model_selection import train_test_split, StratifiedKFold

import gc

import ltag.chaining.pipeline as cp
import ltag.datasets.utils as ds_utils
from ltag.utils import NumpyEncoder

# Implementation adapted from https://github.com/diningphil/gnn-comparison.


class GraphDatasetManager:
  def __init__(
    self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None,
    seed=42, holdout_test_size=0.1,
    dense_batch_size=50,
    wl2_neighborhood=1, wl2_cache=True,
    wl2_batch_size={}, **kwargs):

    self.kfold_class = kfold_class
    self.holdout_test_size = holdout_test_size
    self.dense_batch_size = dense_batch_size
    self.wl2_neighborhood = wl2_neighborhood
    self.wl2_cache = wl2_cache
    self.wl2_batch_size = wl2_batch_size

    self.outer_k = outer_k
    assert (outer_k is not None and outer_k > 0) or outer_k is None

    self.inner_k = inner_k
    assert (inner_k is not None and inner_k > 0) or inner_k is None

    self.seed = seed

    self._dataset = None
    self._wl2_dataset = None
    self._splits = None

    self._setup(**kwargs)

  def _setup(self, **kwargs):
    pass

  def _load_dataset(self):
    raise NotImplementedError

  def _load_wl2_dataset(self, neighborhood):
    graphs, targets = self.dataset

    print(f"Encoding {self.name} as WL2 with neighborhood {neighborhood}...")

    wl2_graphs = np.array([
      ds_utils.wl2_encode(
        g, self._dim_node_features, self._dim_edge_features, neighborhood)
      for g in graphs])

    print(f"Encoded WL2 graphs.")

    return wl2_graphs, targets

  @property
  def dataset(self):
    if self._dataset is None:
        self._dataset = self._load_dataset()

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
    if self._wl2_dataset is None:
      return len(self.dataset[0])
    else:
      return len(self._wl2_dataset[0])

  @property
  def dim_target(self):
    return self._dim_target

  @property
  def dim_node_features(self):
    return self._dim_node_features

  @property
  def dim_edge_features(self):
    return self._dim_edge_features

  @classmethod
  def dim_wl2_features(cls):
    return 3 + cls._dim_node_features + cls._dim_edge_features

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

  def _get_wl2_batches(self, name, idxs=None):
    graphs, targets = self.wl2_dataset if self.wl2_cache else self.dataset

    if idxs is not None:
      graphs = graphs[idxs]
      targets = targets[idxs]

    batches = ds_utils.to_wl2_ds(
      graphs, targets,
      dim_node_features=self._dim_node_features,
      dim_edge_features=self._dim_edge_features,
      as_list=True,
      preencoded=self.wl2_cache,
      neighborhood=self.wl2_neighborhood,
      **self.wl2_batch_size)

    return batches

  def get_all(
    self, output_type="dense", idxs=None, shuffle=False, name_suffix=""):
    ds_name = self.name + name_suffix

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
      ds = ds.batch(self.dense_batch_size)
    elif output_type == "wl2":
      batches = self._get_wl2_batches(ds_name, idxs)

      if batches is None:
        return

      ds = ds_utils.wl2_batches_to_dataset(*batches)

    ds.name = ds_name

    return ds

  def get_test_fold(
    self, outer_idx, output_type="dense"):
    outer_idx = outer_idx or 0

    idxs = self.splits[outer_idx]["test"]

    return self.get_all(output_type, idxs, name_suffix=f"_test-{outer_idx}")

  def get_train_fold(
    self, outer_idx, inner_idx=None, output_type="dense"):
    outer_idx = outer_idx or 0
    inner_idx = inner_idx or 0

    idxs = self.splits[outer_idx]["model_selection"][inner_idx]
    train_ds = self.get_all(
      output_type, idxs["train"],
      name_suffix=f"_train-{outer_idx}-{inner_idx}")
    val_ds = self.get_all(
      output_type, idxs["validation"],
      name_suffix=f"_val-{outer_idx}-{inner_idx}")

    return train_ds, val_ds


class StoredGraphDatasetManager(GraphDatasetManager):
  def _setup(self, wl2_batch_cache=True, no_wl2_load=False, **kwargs):
      self.wl2_batch_cache = wl2_batch_cache
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

  def _make_splits(self):
    splits_filename = self.processed_dir / f"{self.name}_splits.json"

    if splits_filename.exists():
      return json.load(open(splits_filename, "r"))

    splits = super()._make_splits()

    with open(splits_filename, "w") as f:
      json.dump(splits[:], f, cls=NumpyEncoder)

    return splits

  def _get_wl2_batches(self, name, idxs=None):
    if not self.wl2_batch_cache:
      return super()._get_wl2_batches(name, idxs)

    bc = f"n_{self.wl2_neighborhood}"

    for k, s in [
      ("fuzzy_batch_edge_count", "fbec"),
      ("upper_batch_edge_count", "ubec"),
      ("batch_graph_count", "bgc")]:
      if k in self.wl2_batch_size:
        v = self.wl2_batch_size[k]
        bc += f"_{s}_{v}"

    bn = f"{name}.pickle"

    batch_dir = self.root_dir / "wl2_batches" / bc

    if not (batch_dir / bn).exists():
      if not batch_dir.exists():
        os.makedirs(batch_dir)

      batches = super()._get_wl2_batches(name, idxs)

      with open(batch_dir / bn, "wb") as f:
        pickle.dump(batches, f)

      return batches
    elif not self.no_wl2_load:
      with open(batch_dir / bn, "rb") as f:
        return pickle.load(f)

  def prepare_wl2_batches(self):
    assert self.wl2_batch_cache

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

    print("Prepared all batch folds.")

  def _process(self):
    raise NotImplementedError

  def _download(self):
    raise NotImplementedError


class SyntheticGraphDatasetManager(GraphDatasetManager):
  pass


def synthetic_dataset(f):
  @fy.wraps(f)
  def w(*args, **kwargs):
    print("Creating graphs...")
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

    print(f"Created {n} graphs.")

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
