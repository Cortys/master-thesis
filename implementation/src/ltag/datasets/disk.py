from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os.path as path
import pickle
import tensorflow as tf
import numpy as np
import networkx as nx
import funcy as fy

from ltag.datasets.utils import tf_dataset_generator

data_dir = "../data"

# Adapted from https://github.com/XuSShuai/GNN_tensorflow:
def convert_classification_data(data, directed=False):
  offset = 1 if data["index_from"] == 1 else 0
  graphs = data["graphs"]
  nodes_size_list = data["nodes_size_list"]
  labels = data["labels"]

  A, count = [], 0

  for index, graph in enumerate(graphs):
    A.append(np.zeros(
      [nodes_size_list[index], nodes_size_list[index]],
      dtype=np.uint8))

    for edge in graph:
      A[count][edge[0] - offset][edge[1] - offset] = 1

      if not directed:
        A[count][edge[1] - offset][edge[0] - offset] = 1

    count += 1

  Y = np.where(np.array(labels) == 1, 1, 0)
  A = np.array(A)

  X, initial_feature_channels = [], 0

  def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

  if data["vertex_tag"]:
    vertex_tag = data["vertex_tag"]
    initial_feature_channels = len(set(sum(vertex_tag, [])))

    for tag in vertex_tag:
      x = convert_to_one_hot(np.array(tag) - offset, initial_feature_channels)
      X.append(x)
  else:
    for graph in A:
      degree_total = np.sum(graph, axis=1)
      X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))

    initial_feature_channels = 1

  X = np.array(X)

  if data["feature"] is not None:
    feature = data["feature"]

    for i in range(len(X)):
      X[i] = np.concatenate([X[i], feature[i]], axis=1)

    initial_feature_channels = len(X[0][0])

  return (X, A, nodes_size_list), Y

@tf_dataset_generator
def load_classification_dataset(name):
  ds_path = path.join(data_dir, "classification", name + ".pickle")

  with open(ds_path, "rb") as file:
    (X, A, n), y = convert_classification_data(pickle.load(file))

  return name + "_classify", (X, A, n, y)
