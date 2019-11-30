from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import networkx as nx

from ltag.datasets.utils import tf_dataset_generator

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

@tf_dataset_generator
def fiedler_approximation_dataset(
  N, sizes=default_fiedler_sizes,
  crossing_probs=[0, 0.01, 0.02, 0.2, 0.5, 1]):
  graphs = [
    dumbbell_graph(
      *sizes[np.random.choice(len(sizes))],
      np.random.choice(crossing_probs))
    for _ in range(N)]

  adjs = np.array([nx.to_numpy_array(g) for g in graphs])
  x = [np.ones(adjs[i].shape[0], 1) for i in range(N)]
  y = np.array([[nx.algebraic_connectivity(g)] for g in graphs])

  return x, adjs, y


# Regular:
def regular_graph(n, d):
  return nx.random_regular_graph(d, n)


default_regular_sizes = [
  (n, d) for n in range(10, 80, 2) for d in range(1, int(n ** 0.5), 1)]

@tf_dataset_generator
def regular_dataset(N, sizes=default_regular_sizes):
  graphs = [
    regular_graph(*sizes[np.random.choice(len(sizes))]) for _ in range(N)]

  adjs = np.array([nx.to_numpy_array(g) for g in graphs])
  x = [np.ones(adjs[i].shape[0], 1) for i in range(N)]
  y = np.array([[nx.number_connected_components(g)] for g in graphs])

  return x, adjs, y


# Loops:
default_loop_counts = range(1, 6)
default_loop_sizes = [3, 4, 5]
default_loop_scores = [1, -1, 0.5]

def loop_graph(loop_count, loop_sizes, loop_scores):
  g = nx.Graph()
  i = 0
  x = []
  y = 12
  sizes_count = len(loop_sizes)

  for _ in range(loop_count):
    s_idx = np.random.choice(sizes_count)
    size, score = loop_sizes[s_idx], loop_scores[s_idx]

    nodes = range(i, i + size)
    i += size
    x += [[1]] * size
    # y += score

    nx.add_cycle(g, nodes)

  adj = nx.to_numpy_array(g)

  return np.array(x), adj, [y]

@tf_dataset_generator
def loop_dataset(
  N, loop_counts=default_loop_counts, loop_sizes=default_loop_sizes,
  loop_scores=default_loop_scores):

  x, adjs, y = list(zip(*[
    loop_graph(np.random.choice(loop_counts), loop_sizes, loop_scores)
    for _ in range(N)]))

  return np.array(x), np.array(adjs), np.array(y)
