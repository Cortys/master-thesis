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
def to_edge_ds(
  xs, adjs, ys,
  shuffle=False,
  fuzzy_batch_edge_count=1000,
  batch_graph_count=10):
  def gen():
    il = np.arange(len(xs))

    if shuffle:
      np.random.shuffle(il)

    b_x_e = []
    b_ref_v = []
    b_ref_u = []
    b_ns = []
    b_ys = []
    e_ids = {}
    e_count = 0
    g_count = 0
    local_e_counts = []

    def make_batch(b_x_e, b_ref_v, b_ref_u, b_ns, b_ys, local_e_counts):
      max_ref = np.max([len(r) for r in b_ref_v])

      b_ref_v = [
        np.pad(r, (0, max_ref - len(r)), "constant", constant_values=-1)
        for r in b_ref_v]
      b_ref_u = [
        np.pad(r, (0, max_ref - len(r)), "constant", constant_values=-1)
        for r in b_ref_u]

      b_ns = np.array(b_ns)

      return ((
        np.array(b_x_e),
        np.array(b_ref_v), np.array(b_ref_u),
        np.array(local_e_counts),
        np.array(b_ns)),
        np.array(b_ys))

    for i in il:
      x = xs[i]
      y = ys[i]
      n, f = x.shape

      if n == 0:
        continue

      adj = adjs[i]
      y = ys[i]
      g = nx.from_numpy_array(adj)
      e_zero = np.zeros(f)
      local_e_count = 0

      for node in g.nodes:
        g.add_edge(node, node)

      for v, u in g.edges:
        e_ids[(i, v, u)] = e_count
        e_count += 1
        local_e_count += 1

      local_e_counts.append(local_e_count)

      for edge in g.edges(data=True):
        v, u, d = edge
        w = d.get("weight", 0)
        ne = list(nx.common_neighbors(g, v, u))
        n_v = [eid_lookup(e_ids, i, v, k) for k in ne]
        n_u = [eid_lookup(e_ids, i, u, k) for k in ne]

        b_x_e.append(np.concatenate(
          (x[v], [1, 0]) if v == u
          else (e_zero, [0, w])))
        b_ref_v.append(n_v)
        b_ref_u.append(n_u)

      b_ns.append(n)
      b_ys.append(y)
      g_count += 1

      if g_count >= batch_graph_count or e_count >= fuzzy_batch_edge_count:
        yield make_batch(b_x_e, b_ref_v, b_ref_u, b_ns, b_ys, local_e_counts)
        b_x_e = []
        b_ref_v = []
        b_ref_u = []
        b_ns = []
        b_ys = []
        e_ids = {}
        e_count = 0
        g_count = 0
        local_e_counts = []

    if g_count > 0:
      yield make_batch(b_x_e, b_ref_v, b_ref_u, b_ns, b_ys, local_e_counts)

  return tf.data.Dataset.from_generator(
    gen,
    output_types=((
      tf.float32, tf.int32, tf.int32, tf.int32, tf.int32), tf.float32))

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
