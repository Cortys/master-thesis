from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.datasets.disk.manager as manager

# Implementation adapted from https://github.com/diningphil/gnn-comparison.

class Mutag(manager.TUDatasetManager):
  name = "MUTAG"
  _dim_node_features = 0
  _num_node_labels = 7
  _dim_edge_features = 0
  _num_edge_labels = 4
  _dim_target = 2
  _max_num_nodes = 28


class NCI1(manager.TUDatasetManager):
  name = "NCI1"
  _dim_node_features = 0
  _num_node_labels = 37
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 2
  _max_num_nodes = 111


class RedditBinary(manager.TUDatasetManager):
  name = "REDDIT-BINARY"
  _dim_node_features = 0
  _num_node_labels = 0
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 2
  _max_num_nodes = 3782


class Reddit5K(manager.TUDatasetManager):
  name = "REDDIT-MULTI-5K"
  _dim_node_features = 0
  _num_node_labels = 0
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 5
  _max_num_nodes = 3648


class Proteins(manager.TUDatasetManager):
  name = "PROTEINS_full"
  _dim_node_features = 29
  _num_node_labels = 3
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 2
  _max_num_nodes = 620


class DD(manager.TUDatasetManager):
  name = "DD"
  _dim_node_features = 0
  _num_node_labels = 89
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 2
  _max_num_nodes = 5748

class IMDBBinary(manager.TUDatasetManager):
  name = "IMDB-BINARY"
  _dim_node_features = 0
  _num_node_labels = 0
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 2
  _max_num_nodes = 136


class IMDBMulti(manager.TUDatasetManager):
  name = "IMDB-MULTI"
  _dim_node_features = 0
  _num_node_labels = 0
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 3
  _max_num_nodes = 89


# Probably too large.
class Collab(manager.TUDatasetManager):
  name = "COLLAB"
  _dim_node_features = 0
  _num_node_labels = 0
  _dim_edge_features = 0
  _num_edge_labels = 0
  _dim_target = 3
  _max_num_nodes = 492
