from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import funcy as fy
import networkx as nx
import tensorflow as tf
import grakel as gk
from sklearn.model_selection import train_test_split, StratifiedKFold

import ltag.chaining.pipeline as cp
import ltag.datasets.utils as ds_utils

# Implementation adapted from https://github.com/diningphil/gnn-comparison.

class GraphDatasetManager:
  def __init__(
    self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None,
    seed=42, holdout_test_size=0.1,
    dense_batch_size=50,
    wl2_neighborhood=1, wl2_cache=True,
    wl2_batch_size={},
    wl2_indices=False,
    node_one_labels=True,
    edge_one_labels=False,
    with_holdout=True,
    evaluation_args=None,
    name_suffix="",
    **kwargs):

    self.kfold_class = kfold_class
    self.holdout_test_size = holdout_test_size
    self.dense_batch_size = dense_batch_size
    self.wl2_neighborhood = wl2_neighborhood
    self.wl2_cache = wl2_cache
    self.wl2_batch_size = wl2_batch_size
    self.wl2_indices = wl2_indices
    self.node_one_labels = node_one_labels
    self.edge_one_labels = edge_one_labels
    self.with_holdout = with_holdout
    self.evaluation_args = evaluation_args
    self.name_suffix = name_suffix

    self.outer_k = outer_k
    assert (outer_k is not None and outer_k > 0) or outer_k is None

    self.inner_k = inner_k
    assert (inner_k is not None and inner_k > 0) or inner_k is None

    self.seed = seed

    self._dataset = None
    self._wl2_dataset = None
    self._wl2c_dataset = None
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
        g, self._dim_node_features, self._dim_edge_features,
        self._num_node_labels, self._num_edge_labels,
        neighborhood, with_indices=self.wl2_indices)
      for g in graphs])

    print(f"Encoded WL2 graphs.")

    return wl2_graphs, targets

  def _load_wl2c_dataset(self, neighborhood):
    graphs, targets = self.dataset

    print(f"Encoding {self.name} as WL2c with neighborhood {neighborhood}...")

    wl2_graphs = np.array([
      ds_utils.wl2_encode(
        g,
        self._dim_node_features, self._dim_edge_features,
        self._num_node_labels, self._num_edge_labels,
        neighborhood, compact=True)
      for g in graphs])

    print(f"Encoded WL2c graphs.")

    return wl2_graphs, targets

  @property
  def full_name(self):
    return self.name + self.name_suffix

  @property
  def dataset(self):
    if self._dataset is None:
      graphs, ys, d_nf, d_ef, n_nl, n_el = self._load_dataset()
      self._dataset = graphs, ys

      if (
        self._num_node_labels != n_nl or self._num_edge_labels != n_el
        or self.dim_node_features != d_nf or self._dim_edge_features != d_ef):
        raise Exception(
          "Loaded node or edge dimensions do not match with manager metadata. "
          + f"nf: {self.dim_node_features}, {d_nf} - "
          + f"ef: {self._dim_edge_features}, {d_ef} - "
          + f"nl: {self._num_node_labels}, {n_nl} - "
          + f"el: {self._num_edge_labels}, {n_el}")

      if self._num_node_labels == 0 and self.node_one_labels:
        for g in graphs:
          nx.set_node_attributes(g, 1, "label_one")

      if self._num_edge_labels == 0 and self.edge_one_labels:
        for g in graphs:
          nx.set_edge_attributes(g, 1, "label_one")

    return self._dataset

  @property
  def wl2_dataset(self):
    if self._wl2_dataset is None:
      self._wl2_dataset = self._load_wl2_dataset(self.wl2_neighborhood)

    return self._wl2_dataset

  @property
  def wl2c_dataset(self):
    if self._wl2c_dataset is None:
      self._wl2c_dataset = self._load_wl2c_dataset(self.wl2_neighborhood)

    return self._wl2c_dataset

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

  @property
  def num_node_labels(self):
    return self._num_node_labels

  @property
  def num_edge_labels(self):
    return self._num_edge_labels

  @classmethod
  def dim_dense_features(cls):
    f = cls._dim_node_features + cls._num_node_labels

    return f if f > 0 else 1

  @classmethod
  def dim_wl2_features(cls):
    return (
      3 + cls._dim_node_features + cls._dim_edge_features
      + cls._num_node_labels + cls._num_edge_labels)

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
    if self.wl2_cache:
      graphs, targets = self.wl2_dataset
    else:
      graphs, targets = self.dataset

    if idxs is not None:
      graphs = graphs[idxs]
      targets = targets[idxs]

    batches = ds_utils.to_wl2_ds(
      graphs, targets,
      dim_node_features=self._dim_node_features,
      dim_edge_features=self._dim_edge_features,
      num_node_labels=self._num_node_labels,
      num_edge_labels=self._num_edge_labels,
      with_indices=self.wl2_indices,
      as_list=True,  # no laziness due to slow wl2 batching
      preencoded=self.wl2_cache,
      neighborhood=self.wl2_neighborhood,
      **self.wl2_batch_size)

    return batches

  def _get_wl2c_batches(self, name, idxs=None):
    if self.wl2_cache:
      graphs, targets = self.wl2c_dataset
    else:
      graphs, targets = self.dataset

    if idxs is not None:
      graphs = graphs[idxs]
      targets = targets[idxs]

    batches = ds_utils.to_wl2_ds(
      graphs, targets,
      dim_node_features=self._dim_node_features,
      dim_edge_features=self._dim_edge_features,
      num_node_labels=self._num_node_labels,
      num_edge_labels=self._num_edge_labels,
      with_indices=self.wl2_indices,
      lazy=self.wl2_cache,  # wl2c preencoded graphs can be batched quickly
      compact=True,
      preencoded=self.wl2_cache,
      neighborhood=self.wl2_neighborhood,
      **self.wl2_batch_size)

    return batches

  def _compute_gram_matrix(self, output_fn):
    return output_fn(self)

  def get_all(
    self, output_type="dense", idxs=None, train_idxs=None,
    shuffle=False, name_suffix=""):
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

      ds = ds_utils.to_dense_ds(
        graphs, targets, self._dim_node_features, self._num_node_labels)
      ds = ds.batch(self.dense_batch_size)
    elif output_type == "grakel":
      graphs, targets = self.dataset

      if idxs is not None:
        graphs = graphs[idxs]
        targets = targets[idxs]

      if self._num_node_labels > 0:
        node_labels_tag = "label"
      elif self.node_one_labels:
        node_labels_tag = "label_one"
      else:
        node_labels_tag = None

      if self._num_edge_labels > 0:
        edge_labels_tag = "label"
      elif self.edge_one_labels:
        edge_labels_tag = "label_one"
      else:
        edge_labels_tag = None

      return gk.graph_from_networkx(
        graphs,
        node_labels_tag=node_labels_tag,
        edge_labels_tag=edge_labels_tag), targets
    # Custom kernel:
    elif callable(output_type):
      _, targets = self.dataset
      gram = self._compute_gram_matrix(output_type)

      if gram is None:
        return

      if idxs is not None:
        train_idxs = idxs if train_idxs is None else train_idxs
        gram, targets = gram[idxs, :][:, train_idxs], targets[idxs]

      return gram, targets
    elif output_type == "wl2":
      batches = self._get_wl2_batches(ds_name, idxs)

      if batches is None:
        return

      if isinstance(batches, tf.data.Dataset):
        batches.name = ds_name
        return batches

      ds = ds_utils.wl2_batches_to_dataset(*batches)
    elif output_type == "wl2c":
      batches = self._get_wl2c_batches(ds_name, idxs)

      if batches is None:
        return

      if isinstance(batches, tf.data.Dataset):
        batches.name = ds_name
        return batches

      ds = ds_utils.wl2_batches_to_dataset(*batches, compact=True)

    ds.name = ds_name

    return ds

  def get_test_fold(
    self, outer_idx, inner_idx=None, output_type="dense"):
    outer_idx = outer_idx or 0
    inner_idx = inner_idx or 0

    idxs = self.splits[outer_idx]["test"]
    # Training indices are required to fetch kernel gram matrices:
    if callable(output_type):
      ms = self.splits[outer_idx]["model_selection"][inner_idx]
      train_idxs = ms["train"]

      if not self.with_holdout:
        train_idxs = np.concatenate([train_idxs, ms["validation"]])
    else:
      train_idxs = None

    return self.get_all(
      output_type, idxs,
      train_idxs=train_idxs,
      name_suffix=f"_test-{outer_idx}")

  def get_train_fold(
    self, outer_idx, inner_idx=None, output_type="dense"):
    outer_idx = outer_idx or 0
    inner_idx = inner_idx or 0

    idxs = self.splits[outer_idx]["model_selection"][inner_idx]
    train_idxs = idxs["train"]
    val_idxs = idxs["validation"]

    if not self.with_holdout:
      return self.get_all(
        output_type, np.concatenate([train_idxs, val_idxs]),
        name_suffix=f"_trainval-{outer_idx}-{inner_idx}")

    train_ds = self.get_all(
      output_type, train_idxs,
      name_suffix=f"_train-{outer_idx}-{inner_idx}")
    val_ds = self.get_all(
      output_type, val_idxs, train_idxs=train_idxs,
      name_suffix=f"_val-{outer_idx}-{inner_idx}")

    return train_ds, val_ds

  def draw(self, idx, with_features=False):
    gs, ys = self.dataset
    g = gs[idx]
    y = ys[idx]
    return ds_utils.draw_graph(g, y, with_features=with_features)
