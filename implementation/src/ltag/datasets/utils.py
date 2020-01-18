from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

import ltag.chaining.pipeline as cp

@cp.tolerant
def to_dense_ds(graphs, ys, ragged=False, sparse=False):
  x = [[
    data.get("features", [1])
    for _, data in g.nodes(data=True)]
    for g in graphs]

  adjs = [nx.to_numpy_array(g) for g in graphs]

  x = tf.ragged.constant(x)
  adjs = tf.ragged.constant(adjs)
  n_s = [g.order() for g in graphs]

  if not ragged:
    x = x.to_tensor()
    adjs = adjs.to_tensor()

    if sparse:
      x = tf.sparse.from_dense(x)
      adjs = tf.sparse.from_dense(adjs)

  return tf.data.Dataset.from_tensor_slices((
    (
      tf.cast(x, tf.float32),
      tf.cast(adjs, tf.float32),
      tf.constant(n_s, dtype=tf.int32)
    ),
    tf.constant(ys, dtype=tf.float32)
  ))

def eid_lookup(e_ids, i, j):
  if i > j:
    i, j = j, i

  return e_ids[(i, j)]

def wl2_encode(
  g, dim_node_features=None, dim_edge_features=None,
  neighborhood=1, with_indices=False):
  """
    Takes a graph with node and edge features and converts it
    into a edge + row/col ref list for sparse WL2 implementations.
  """
  g_p = nx.power(g, neighborhood) if neighborhood > 1 else g

  if dim_node_features is None:
    dim_node_features = 0

  if dim_edge_features is None:
    dim_edge_features = 0

  n_zero = np.zeros(dim_node_features)
  e_zero = np.zeros(dim_edge_features)

  for node, data in g.nodes(data=True):
    g_p.add_edge(node, node)
    g.add_edge(node, node, features=data.get("features", n_zero))

  x = []
  ref_a = []
  ref_b = []
  max_ref_dim = 0
  e_ids = {}

  for i, edge in enumerate(g_p.edges):
    e_ids[edge] = i

  for i, edge in enumerate(g_p.edges):
    a, b = edge
    in_g = g.has_edge(a, b)
    if not in_g:
      f = e_zero
    elif (
      (a != b and dim_edge_features == 0)
      or (a == b and dim_node_features == 0)):
      f = []
    else:
      f = g.get_edge_data(a, b).get("features", e_zero)

    nbs = (
      ([a] if a == b else [a, b])
      + list(nx.common_neighbors(g_p, a, b)))
    n_a = np.array([eid_lookup(e_ids, a, k) for k in nbs])
    n_b = np.array([eid_lookup(e_ids, b, k) for k in nbs])
    nbs_count = len(nbs)

    if nbs_count > max_ref_dim:
      max_ref_dim = nbs_count

    x.append(np.concatenate(
      ([1, 0, 0], f, e_zero) if a == b else
      ([0], [1, 0] if in_g else [0, 1], n_zero, f)))
    ref_a.append(n_a)
    ref_b.append(n_b)

  n = g.order()
  res = x, ref_a, ref_b, max_ref_dim, n

  if with_indices:
    res += (list(g_p.edges),)

  return res

def make_wl2_batch(encoded_graphs, with_indices=False):
  "Takes a sequence of graphs that were encoded via `wl2_encode`."
  max_ref_dim = np.max([g[3] for g in encoded_graphs])

  b_x = []
  b_ref_a = []
  b_ref_b = []
  b_e_map = []
  b_n = []
  b_idx = []
  e_offset = 0

  for i, e in enumerate(encoded_graphs):
    if with_indices:
      b_idx += e[5]
      e = e[:-1]
    x, ref_a, ref_b, _, n = e
    e_count = len(x)
    b_x += x
    b_ref_a += [np.pad(
      r + e_offset,
      (0, max_ref_dim - len(r)), "constant", constant_values=-1)
      for r in ref_a]
    b_ref_b += [np.pad(
      r + e_offset,
      (0, max_ref_dim - len(r)), "constant", constant_values=-1)
      for r in ref_b]
    b_e_map += [i] * e_count
    b_n.append(n)
    e_offset += e_count

  res = (
    np.array(b_x),
    np.array(b_ref_a), np.array(b_ref_b),
    np.array(b_e_map),
    np.array(b_n))

  if with_indices:
    res += (b_idx,)

  return res

def get_graph_feature_dims(g):
  dim_node_features = 0
  dim_edge_features = 0

  for _, data in g.nodes(data=True):
    f = data.get("features")
    if f is not None:
      dim_node_features = len(f)
    else:
      break

  for _, _, data in g.edges(data=True):
    f = data.get("features")
    if f is not None:
      dim_edge_features = len(f)
    else:
      break

  return dim_node_features, dim_edge_features

def wl2_batches_to_dataset(batches, dim_wl2, y_dim):
  if callable(batches):
    batch_gen = batches
  else:
    def batch_gen():
      for b in batches:
        yield b

  return tf.data.Dataset.from_generator(
    batch_gen,
    output_types=((
      tf.float32, tf.int32, tf.int32, tf.int32, tf.int32), tf.float32),
    output_shapes=((
      tf.TensorShape([None, dim_wl2]),
      tf.TensorShape([None, None]),
      tf.TensorShape([None, None]),
      tf.TensorShape([None]),
      tf.TensorShape([None])),
      tf.TensorShape([None, *y_dim])))

@cp.tolerant
def to_wl2_ds(
  graphs, ys,
  dim_node_features,
  dim_edge_features,
  fuzzy_batch_edge_count=100000,
  upper_batch_edge_count=120000,
  batch_graph_count=100,
  neighborhood=1,
  with_indices=False,
  lazy=False, preencoded=False,
  as_list=False, log=False):
  ds_size = len(graphs)
  y_shape = ys.shape
  y_dim = y_shape[1:] if len(y_shape) > 1 else []
  dim_wl2 = 3 + dim_node_features + dim_edge_features

  if log:
    print(
      "Batching", ds_size, "preencoded" if preencoded else "raw", "graphs.",
      f"b_gc={batch_graph_count}",
      f"dim_wl2={dim_wl2}",
      f"(node={dim_node_features}, edge={dim_edge_features})")

  if batch_graph_count > 1:
    def gen():
      b_gs = []
      b_ys = []
      e_count = 0

      if ds_size > 0:
        enc_next = graphs[0] if preencoded else wl2_encode(
          graphs[0],
          dim_node_features, dim_edge_features,
          neighborhood, with_indices)

      for i in range(ds_size):
        enc = enc_next
        y = ys[i]
        if i + 1 < ds_size:
          enc_next = graphs[i + 1] if preencoded else wl2_encode(
            graphs[i + 1],
            dim_node_features, dim_edge_features,
            neighborhood, with_indices)
        else:
          enc_next = None

        b_gs.append(enc)
        b_ys.append(y)

        e_count += len(enc[0])
        e_count_next = (
          e_count if enc_next is None
          else e_count + len(enc_next[0]))

        if (
          len(b_gs) >= batch_graph_count
          or e_count >= fuzzy_batch_edge_count
          or (len(b_gs) > 0 and e_count_next >= upper_batch_edge_count)):
          if log:
            print("Batch with", len(b_gs), "graphs and", e_count, "edges.")
          yield (make_wl2_batch(b_gs, with_indices), b_ys)
          e_count = 0
          b_gs = []
          b_ys = []

      if len(b_gs) > 0:
        if log:
          print("Batch with", len(b_gs), "graphs and", e_count, "edges with.")
        yield (make_wl2_batch(b_gs, with_indices), b_ys)
  else:
    def gen():
      for i in range(ds_size):
        yield (make_wl2_batch([graphs[i]], with_indices), [ys[i]])

  if lazy and not as_list:
    batches = gen
  else:
    batches_list = list(gen())
    batches = np.empty(len(batches_list), dtype='O')
    batches[:] = batches_list

    if as_list:
      return batches, dim_wl2, y_dim

  return wl2_batches_to_dataset(batches, dim_wl2, y_dim)


def draw_graph(g, y, with_features=False):
  plt.figure()
  plt.title('Label: {}'.format(y))

  g = g.copy()

  if with_features:
    nx.relabel_nodes(g, dict([
      (n, f"{n}: " + str(data.get("features")))
      for n, data in g.nodes(data=True)
    ]))

  nx.draw_spring(g, with_labels=True)
  plt.show()
