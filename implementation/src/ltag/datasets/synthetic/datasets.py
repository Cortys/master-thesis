from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import networkx as nx
import funcy as fy

from ltag.utils import cart, local_seed
from ltag.datasets.synthetic.manager import synthetic_dataset

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

  for _ in range(a + b):
    t = range(i, i + 3)

    g.add_nodes_from(
      t, label=(0 if i < ab_split else 1))
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
    g.add_node(i, label=0)
    g.add_node(i + 1, label=0)
    g.add_node(i + 2, label=1)
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
    for k in range(10, 20)
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

@synthetic_dataset
def threesix_dataset():
  g3 = nx.Graph()
  nx.add_cycle(g3, range(3))
  nx.add_cycle(g3, range(3, 6))

  g6 = nx.Graph()
  nx.add_cycle(g6, range(6))

  return [g3, g6], np.array([0, 1])

def noisy_triangle_graph(sl, sr, d, y):
  g = nx.Graph()
  y_not = 1 - y
  t_nodes = np.arange(3)
  l_nodes = np.arange(sl)
  r_nodes = np.arange(sr) + sl

  g.add_nodes_from(l_nodes, label=y)
  g.add_nodes_from(r_nodes, label=y_not)
  nx.add_cycle(g, t_nodes)

  def try_adding_non_triangle_edge(g, nodes):
    u = np.random.choice(nodes)
    v = np.random.choice(nodes)

    if len(list(nx.common_neighbors(g, u, v))) > 0:
      return

    g.add_edge(u, v)

  for _ in range(int(0.5 * sl * (sl + 1) * d)):
    try_adding_non_triangle_edge(g, l_nodes)

  for _ in range(int(0.5 * sr * (sr + 1) * d)):
    try_adding_non_triangle_edge(g, r_nodes)

  for _ in range(int(sr * sl * d)):
    ln = np.random.choice(l_nodes)
    rn = np.random.choice(r_nodes)
    g.add_edge(ln, rn)

  return g, y

@synthetic_dataset
def noisy_triangle_classification_dataset(seed=1337):
  with local_seed(seed):
    sizes = range(3, 10)
    balances = [(1, 1), (1, 3)]
    dens = [0.25, 0.5]
    classes = [0, 1]

    max_verts = 33
    repeat = 4
    configs = [
      (s * l, s * r, d, y)
      for s, (l, r), d, y in cart(sizes, balances, dens, classes)
      if s * (l + r) <= max_verts]

    graphs, ys = unzip([
      entry
      for config in configs
      for entry in fy.repeatedly(
        fy.partial(noisy_triangle_graph, *config), repeat)])

    return graphs, np.array(ys)

@synthetic_dataset
def balanced_triangle_classification_dataset(seed=1337):
  with local_seed(seed):
    sizes = range(3, 10)
    balances = [(1, 1), (1, 3), (3, 1)]
    dens = [0.25, 0.5]
    classes = [0, 1]

    max_verts = 33
    repeat = 3
    configs = [
      (s * l, s * r, d, y)
      for s, (l, r), d, y in cart(sizes, balances, dens, classes)
      if s * (l + r) <= max_verts]

    graphs, ys = unzip([
      entry
      for config in configs
      for entry in fy.repeatedly(
        fy.partial(noisy_triangle_graph, *config), repeat)])

    return graphs, np.array(ys)

def loop_neighbors(pos, size, split=False, only_next=True):
  dim = len(size)

  res = []

  for i in range(dim):
    min = 0
    max = size[i]

    if split and i == 0:
      if pos[i] < size[i] / 2:
        max = size[i] / 2
      else:
        min = size[i] / 2

    next = pos[i] + 1

    if next >= max:
      next = min

    res.append(pos[:i] + (next,) + pos[(i+1):])

    if not only_next:
      prev = pos[i] - 1
      if prev < min:
        prev = max - 1

      res.append(pos[:i] + (prev,) + pos[(i+1):])

  return res

@synthetic_dataset
def hyperloop_dataset(size=(10,)):
  if size[0] % 2 != 0:
    raise Exception("First loop dimension must be even.")

  g1 = nx.Graph()
  g2 = nx.Graph()

  nodes = {pos: i for i, pos in enumerate(cart(*[range(d) for d in size]))}

  for pos, i in nodes.items():
    n1 = loop_neighbors(pos, size, False)
    n2 = loop_neighbors(pos, size, True)

    g1.add_edges_from([(i, nodes[npos]) for npos in n1])
    g2.add_edges_from([(i, nodes[npos]) for npos in n2])

  N = 50

  return [g1] * N + [g2] * N, [0] * N + [1] * N

def hyperloop_dataset_generator(size=(10,)):
  suffix = "_" + "_".join(str(s) for s in size)

  return fy.partial(hyperloop_dataset, size, stored_suffix=suffix)


hyperloop_16 = hyperloop_dataset_generator((16,))
hyperloop_32 = hyperloop_dataset_generator((32,))
hyperloop_64 = hyperloop_dataset_generator((64,))
hyperloop_128 = hyperloop_dataset_generator((128,))
hyperloop_256 = hyperloop_dataset_generator((256,))
hyperloop_512 = hyperloop_dataset_generator((512,))
hyperloop_1024 = hyperloop_dataset_generator((1024,))
hyperloop_2048 = hyperloop_dataset_generator((2048,))
hyperloop_4096 = hyperloop_dataset_generator((4096,))
hyperloop_8192 = hyperloop_dataset_generator((8192,))
hyperloop_16384 = hyperloop_dataset_generator((16384,))

# hyperloop_256_2 = hyperloop_dataset_generator((256, 2))
# hyperloop_128_2_2 = hyperloop_dataset_generator((128, 2, 2))
# hyperloop_64_2_2_2 = hyperloop_dataset_generator((64, 2, 2, 2))

hyperloop_512_2 = hyperloop_dataset_generator((512, 2))  # deg 3
hyperloop_256_2_2 = hyperloop_dataset_generator((256, 2, 2))  # deg 4
hyperloop_128_2_2_2 = hyperloop_dataset_generator((128, 2, 2, 2))  # deg 5
hyperloop_64_2_2_2_2 = hyperloop_dataset_generator((64, 2, 2, 2, 2))  # deg 6
