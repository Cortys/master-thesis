from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import networkx as nx

from ltag.datasets.manager import synthetic_dataset

def unzip(tuples):
  return list(zip(*tuples))

# Dumbbell:
def dumbbell_graph(n, m, crossing_prob):
  g = nx.Graph()
  crosses = np.random.random_sample(m) < crossing_prob
  part_b = np.random.random_sample(m) < 0.5
  b_offset = int(n / 2)
  relative_edges = np.random.choice(b_offset, (m, 2))
  relative_edges[crosses] += np.array([0, b_offset])
  relative_edges[~crosses & part_b] += b_offset

  for i in range(m):
    g.add_edge(*(relative_edges[i]))

  return g


default_fiedler_sizes = [(i, int(i ** 1.5)) for i in range(10, 80)]

@synthetic_dataset
def fiedler_approximation_dataset(
  N, sizes=default_fiedler_sizes,
  crossing_probs=[0, 0.01, 0.02, 0.2, 0.5, 1]):
  graphs = [
    dumbbell_graph(
      *sizes[np.random.choice(len(sizes))],
      np.random.choice(crossing_probs))
    for _ in range(N)]

  y = np.array([
    [nx.linalg.algebraicconnectivity.algebraic_connectivity(
      g, normalized=True)]
    for g in graphs])

  return graphs, y


# Regular:
def regular_graph(n, d):
  return nx.random_regular_graph(d, n)


default_regular_sizes = [
  (n, d) for n in range(10, 80, 2) for d in range(1, int(n ** 0.5), 1)]

@synthetic_dataset
def regular_dataset(N, sizes=default_regular_sizes):
  graphs = [
    regular_graph(*sizes[np.random.choice(len(sizes))]) for _ in range(N)]

  adjs = np.array([nx.to_numpy_array(g) for g in graphs])
  y = np.array([
    [nx.number_connected_components(graphs[i]) / adjs[i].shape[0]]
    for i in range(N)])

  return graphs, y


# Loops:
default_loop_counts = range(1, 6)
default_loop_sizes = [3, 4, 5]
default_loop_scores = [1, -1, 0.5]

def loop_graph(loop_count, loop_sizes, loop_scores):
  g = nx.Graph()
  i = 0
  y = 0.0
  sizes_count = len(loop_sizes)

  for _ in range(loop_count):
    s_idx = np.random.choice(sizes_count)
    size, score = loop_sizes[s_idx], loop_scores[s_idx]

    nodes = range(i, i + size)
    i += size
    y += score

    nx.add_cycle(g, nodes)

  y /= loop_count

  return g, y

@synthetic_dataset
def loop_dataset(
  N, loop_counts=default_loop_counts, loop_sizes=default_loop_sizes,
  loop_scores=default_loop_scores):

  graphs, ys = unzip([
    loop_graph(np.random.choice(loop_counts), loop_sizes, loop_scores)
    for _ in range(N)])

  return graphs, np.array(ys)


# Triangles:
def triangle_graph(a=1, b=1, mix=1):
  g = nx.Graph()
  i = 0
  ab_split = a * 3
  e = np.eye(2)

  for _ in range(a + b):
    t = range(i, i + 3)

    g.add_nodes_from(
      t, features=(e[0 if i < ab_split else 1]))
    nx.add_cycle(g, t)
    i += 3

  if a <= b:
    pairing_count = a
    pairing_a = 0
    pairing_b = ab_split
  else:
    pairing_count = b
    pairing_a = ab_split
    pairing_b = 0

  for j in range(pairing_count):
    g.add_edge(pairing_a + j * 3, pairing_b + j * 3)
    g.add_edge(pairing_a + j * 3 + 1, pairing_b + j * 3 + 1)
    g.add_edge(pairing_a + j * 3 + 2, pairing_b + j * 3 + 2)

  for _ in range(mix):
    g.add_node(i, features=e[0])
    g.add_node(i + 1, features=e[0])
    g.add_node(i + 2, features=e[1])
    nx.add_cycle(g, range(i, i + 3))
    i += 3

  y = (a - b) / max(a + b, 1)

  return g, y

@synthetic_dataset
def triangle_dataset():
  # configs = [
  #   [1, 0, 0], [0, 1, 0], [2, 1, 0]]

  configs = [
    [i, j, 0]
    for i in range(0, 10)
    for j in range(0, 10)
    # for k in range(10, 20)
  ]
  configs = configs[1:]

  graphs, ys = unzip([triangle_graph(*config) for config in configs])

  return graphs, np.array(ys)

@synthetic_dataset
def twothree_dataset():
  g2 = nx.Graph()
  g2.add_edge(0, 1)

  g3 = nx.Graph()
  nx.add_cycle(g3, range(3))

  return [g2, g3], np.array([-1, 1])
