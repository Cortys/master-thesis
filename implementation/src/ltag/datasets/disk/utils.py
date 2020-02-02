from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
from collections import defaultdict
from contextlib import ExitStack
import numpy as np
import networkx as nx

# Implementation adapted from https://github.com/diningphil/gnn-comparison.

def parse_tu_data(name, raw_dir):
  indicator_path = raw_dir / name / f'{name}_graph_indicator.txt'
  edges_path = raw_dir / name / f'{name}_A.txt'
  graph_labels_path = raw_dir / name / f'{name}_graph_labels.txt'
  node_labels_path = raw_dir / name / f'{name}_node_labels.txt'
  edge_labels_path = raw_dir / name / f'{name}_edge_labels.txt'
  node_attrs_path = raw_dir / name / f'{name}_node_attributes.txt'
  edge_attrs_path = raw_dir / name / f'{name}_edge_attributes.txt'

  unique_node_labels = set()
  unique_edge_labels = set()

  indicator, edge_indicator = [-1], [(-1, -1)]
  graph_nodes = defaultdict(list)
  graph_edges = defaultdict(list)
  node_labels = defaultdict(list)
  edge_labels = defaultdict(list)
  node_attrs = defaultdict(list)
  edge_attrs = defaultdict(list)

  with open(indicator_path, "r") as f:
    for i, line in enumerate(f.readlines(), 1):
      line = line.rstrip("\n")
      graph_id = int(line)
      indicator.append(graph_id)
      graph_nodes[graph_id].append(i)

  with open(edges_path, "r") as f:
    for i, line in enumerate(f.readlines(), 1):
      line = line.rstrip("\n")
      edge = [int(e) for e in line.split(',')]
      edge_indicator.append(edge)
      graph_id = indicator[edge[0]]
      graph_edges[graph_id].append(edge)

  if node_labels_path.exists():
    with open(node_labels_path, "r") as f:
      for i, line in enumerate(f.readlines(), 1):
        line = line.rstrip("\n")
        node_label = int(line)
        unique_node_labels.add(node_label)
        graph_id = indicator[i]
        node_labels[graph_id].append(node_label)

  if edge_labels_path.exists():
    with open(edge_labels_path, "r") as f:
      for i, line in enumerate(f.readlines(), 1):
        line = line.rstrip("\n")
        edge_label = int(line)
        unique_edge_labels.add(edge_label)
        graph_id = indicator[edge_indicator[i][0]]
        edge_labels[graph_id].append(edge_label)

  if node_attrs_path.exists():
    with open(node_attrs_path, "r") as f:
      for i, line in enumerate(f.readlines(), 1):
        line = line.rstrip("\n")
        nums = line.split(",") if line != "" else []
        node_attr = np.array([float(n) for n in nums])
        graph_id = indicator[i]
        node_attrs[graph_id].append(node_attr)

  if edge_attrs_path.exists():
    with open(edge_attrs_path, "r") as f:
      for i, line in enumerate(f.readlines(), 1):
        line = line.rstrip("\n")
        nums = line.split(",") if line != "" else []
        edge_attr = np.array([float(n) for n in nums])
        graph_id = indicator[edge_indicator[i][0]]
        edge_attrs[graph_id].append(edge_attr)

  graph_labels = []
  with open(graph_labels_path, "r") as f:
    for i, line in enumerate(f.readlines(), 1):
      line = line.rstrip("\n")
      target = int(line)
      if target == -1:
        graph_labels.append(0)
      else:
        graph_labels.append(target)

    # Shift by one to the left.
    # Apparently this is necessary for multiclass tasks.
    if min(graph_labels) == 1:
      graph_labels = [l - 1 for l in graph_labels]

  num_node_labels = (
    max(unique_node_labels) if unique_node_labels != set() else 0)
  if num_node_labels != 0 and min(unique_node_labels) == 0:
    # some datasets e.g. PROTEINS have labels with value 0
    num_node_labels += 1

  num_edge_labels = (
    max(unique_edge_labels) if unique_edge_labels != set() else 0)
  if num_edge_labels != 0 and min(unique_edge_labels) == 0:
    num_edge_labels += 1

  return {
    "graph_nodes": graph_nodes,
    "graph_edges": graph_edges,
    "graph_labels": graph_labels,
    "node_labels": node_labels,
    "node_attrs": node_attrs,
    "edge_labels": edge_labels,
    "edge_attrs": edge_attrs
  }, num_node_labels, num_edge_labels

def create_graph_from_tu_data(
  graph_data, num_node_labels, num_edge_labels):
  nodes = graph_data["graph_nodes"]
  edges = graph_data["graph_edges"]

  G = nx.Graph()
  I_n = np.eye(num_node_labels)
  I_e = np.eye(num_edge_labels)

  for i, node in enumerate(nodes):
    features = []
    label = None

    if graph_data["node_labels"] != []:
      features.extend(I_n[graph_data["node_labels"][i] - 1])
      label = graph_data["node_labels"][i]

    if graph_data["node_attrs"] != []:
      features.extend(graph_data["node_attrs"][i])

    if len(features) == 0:
      features = [1]

    if label is None:
      G.add_node(node, features=features)
    else:
      G.add_node(node, features=features, label=label)

  for i, edge in enumerate(edges):
    n1, n2 = edge
    features = []
    label = None

    if graph_data["edge_labels"] != []:
      features.extend(I_e[graph_data["edge_labels"][i] - 1])
      label = graph_data["edge_labels"][i]

    if graph_data["edge_attrs"] != []:
      features.extend(graph_data["edge_attrs"][i])

    if len(features) == 0:
      features = [1]

    G.add_edge(n1, n2, features=features)
    if label is None:
      G.add_edge(n1, n2, features=features)
    else:
      G.add_edge(n1, n2, features=features, label=label)

  return G

def store_graphs_as_tu_data(graphs, targets, name, raw_dir):
  base_dir = raw_dir / name

  indicator_path = base_dir / f'{name}_graph_indicator.txt'
  edges_path = base_dir / f'{name}_A.txt'
  graph_labels_path = base_dir / f'{name}_graph_labels.txt'
  node_attrs_path = base_dir / f'{name}_node_attributes.txt'
  edge_attrs_path = base_dir / f'{name}_edge_attributes.txt'

  if not base_dir.exists():
    os.makedirs(base_dir)

  with\
      open(indicator_path, "w") as indicator,\
      open(edges_path, "w") as edges,\
      open(graph_labels_path, "w") as graph_labels,\
      open(node_attrs_path, "w") as node_attrs,\
      open(edge_attrs_path, "w") as edge_attrs:
    v_offset = 1

    for i, (g, target) in enumerate(zip(graphs, targets), 1):
      v_lookup = {}
      graph_labels.write(f"{target}\n")

      for n, data in g.nodes(data=True):
        indicator.write(f"{i}\n")
        feat = ",".join(data.get("features", []))
        node_attrs.write(f"{feat}\n")
        v_lookup[n] = v_offset
        v_offset += 1

      for u, v, data in g.edges(data=True):
        edges.write(f"{v_lookup[u]},{v_lookup[v]}\n")
        feat = ",".join(data.get("features", []))
        edge_attrs.write(f"{feat}\n")
