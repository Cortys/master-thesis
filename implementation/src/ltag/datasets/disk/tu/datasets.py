from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.datasets.disk.tu.manager as manager

# Implementation adapted from https://github.com/diningphil/gnn-comparison.

class NCI1(manager.TUDatasetManager):
  name = "NCI1"
  _dim_features = 37
  _dim_target = 2
  max_num_nodes = 111


class RedditBinary(manager.TUDatasetManager):
  name = "REDDIT-BINARY"
  _dim_features = 1
  _dim_target = 2
  max_num_nodes = 3782


class Reddit5K(manager.TUDatasetManager):
  name = "REDDIT-MULTI-5K"
  _dim_features = 1
  _dim_target = 5
  max_num_nodes = 3648


class Proteins(manager.TUDatasetManager):
  name = "PROTEINS_full"
  _dim_features = 3
  _dim_target = 2
  max_num_nodes = 620


class DD(manager.TUDatasetManager):
  name = "DD"
  _dim_features = 89
  _dim_target = 2
  max_num_nodes = 5748


class Enzymes(manager.TUDatasetManager):
  name = "ENZYMES"
  _dim_features = 21  # 18 attr + 3 labels
  _dim_target = 6
  max_num_nodes = 126


class IMDBBinary(manager.TUDatasetManager):
  name = "IMDB-BINARY"
  _dim_features = 1
  _dim_target = 2
  max_num_nodes = 136


class IMDBMulti(manager.TUDatasetManager):
  name = "IMDB-MULTI"
  _dim_features = 1
  _dim_target = 3
  max_num_nodes = 89


class Collab(manager.TUDatasetManager):
  name = "COLLAB"
  _dim_features = 1
  _dim_target = 3
  max_num_nodes = 492
