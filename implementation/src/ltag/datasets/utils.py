from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
import funcy as fy

import ltag.chaining.pipeline as cp

@cp.tolerant
def to_vert_ds(x, adjs, y, ragged=False, sparse=False):
  x_in = x
  x = tf.ragged.constant(x)
  adjs = tf.ragged.constant(adjs)

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
      tf.constant([s.shape[0] for s in x_in], dtype=tf.int32)
    ),
    tf.constant(y, dtype=tf.float32)
  ))

def eid_lookup(e_ids, g, i, j):
  if i > j:
    i, j = j, i

  return e_ids[(g, i, j)]

@cp.tolerant
def to_edge_ds(xs, adjs, ys, shuffle=False):
  def gen():
    il = np.arange(len(xs))

    if shuffle:
      np.random.shuffle(il)

    for i in il:
      x = xs[i]
      y = ys[i]
      (n, f) = x.shape

      if n == 0:
        continue

      adj = adjs[i]
      y = ys[i]
      g = nx.from_numpy_array(adj)
      e_zero = np.zeros(f)
      e_ids = {}
      x_e = []
      ref_v = []
      ref_u = []

      for node in nx.nodes(g):
        g.add_edge(node, node)

      e_count = 0
      for v, u in nx.edges(g):
        e_ids[(i, v, u)] = e_count
        e_count += 1

      for edge in nx.edges(g):
        v, u = edge
        n = list(nx.common_neighbors(g, v, u))
        n_v = [eid_lookup(e_ids, i, v, k) for k in n]
        n_u = [eid_lookup(e_ids, i, u, k) for k in n]

        x_e.append(x[v] if v == u else e_zero)
        ref_v.append(n_v)
        ref_u.append(n_u)

      max_ref = np.max([len(r) for r in ref_v])

      ref_v = [
        np.pad(r, (0, max_ref - len(r)), "constant", constant_values=-1)
        for r in ref_v]
      ref_u = [
        np.pad(r, (0, max_ref - len(r)), "constant", constant_values=-1)
        for r in ref_u]

      yield np.array(x_e), np.array(ref_v), np.array(ref_u), y

  return tf.data.Dataset.from_generator(
    gen,
    output_types=(tf.float32, tf.int32, tf.int32, tf.float32))

def tf_dataset_generator(f):
  @fy.wraps(f)
  def w(*args, edge_featured=False, **kwargs):
    r = cp.tolerant(f)(*args, **kwargs)

    return (
      to_edge_ds(*r, **kwargs) if edge_featured
      else to_vert_ds(*r, **kwargs))

  return w

def draw_graph(x, adj, y):
  plt.figure()
  plt.title('Label: {}'.format(y))

  g = nx.from_numpy_array(adj)

  nx.relabel_nodes(g, dict([
    (i, str(x[i]))
    for i in range(x.shape[0])
  ]))

  nx.draw_spring(g, with_labels=True)
  plt.show()

def draw_from_ds(ds, i):
  (x, adj, n), y = list(ds)[i]
  x = x.numpy()
  adj = adj.numpy()
  y = y.numpy()

  x = x[:n, :]
  adj = adj[:n, :n]

  draw_graph(x, adj, y)
