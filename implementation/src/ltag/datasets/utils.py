from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
import funcy as fy

import ltag.chaining.pipeline as cp

@cp.tolerant
def to_vert_ds(x, adjs, n_s, y, ragged=False, sparse=False):
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
      tf.constant(
        [s.shape[0] for s in x_in] if n_s is None else n_s, dtype=tf.int32)
    ),
    tf.constant(y, dtype=tf.float32)
  ))

def eid_lookup(e_ids, g, i, j):
  if i > j:
    i, j = j, i

  return e_ids[(g, i, j)]

@cp.tolerant
def to_edge2_ds(
  xs, adjs, n_s, ys,
  shuffle=False,
  fuzzy_batch_edge_count=100000,
  batch_graph_count=100):
  f = xs[0].shape[1] if len(xs) > 0 else 0

  il = np.arange(len(xs))

  y_shape = ys.shape
  y_dim = y_shape[1:] if len(y_shape) > 1 else []

  if shuffle:
    np.random.shuffle(il)

  def gen():
    b_x_e = []
    b_ref_a = []
    b_ref_b = []
    b_ns = []
    b_ys = []
    e_ids = {}
    e_count = 0
    g_count = 0
    e_map = []

    def make_batch(b_x_e, b_ref_a, b_ref_b, e_map, b_ns, b_ys):
      max_ref = np.max([len(r) for r in b_ref_a])

      b_ref_a = [
        np.pad(r, (0, max_ref - len(r)), "constant", constant_values=-1)
        for r in b_ref_a]
      b_ref_b = [
        np.pad(r, (0, max_ref - len(r)), "constant", constant_values=-1)
        for r in b_ref_b]

      b_ns = np.array(b_ns)

      return ((
        np.array(b_x_e),
        np.array(b_ref_a), np.array(b_ref_b),
        np.array(e_map),
        np.array(b_ns)),
        np.array(b_ys))

    for i in il:
      x = xs[i]
      y = ys[i]
      n = x.shape[0] if n_s is None else n_s[i]

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

      e_map += [g_count] * local_e_count

      for edge in g.edges(data=True):
        a, b, d = edge
        w = d.get("weight", 0)
        ne = (
          ([a] if a == b else [a, b])
          + list(nx.common_neighbors(g, a, b)))
        n_a = [eid_lookup(e_ids, i, a, k) for k in ne]
        n_b = [eid_lookup(e_ids, i, b, k) for k in ne]

        b_x_e.append(np.concatenate(
          (x[a], [1, 0]) if a == b
          else (e_zero, [0, w])))
        b_ref_a.append(n_a)
        b_ref_b.append(n_b)

      b_ns.append(n)
      b_ys.append(y)
      g_count += 1

      if g_count >= batch_graph_count or e_count >= fuzzy_batch_edge_count:
        yield make_batch(b_x_e, b_ref_a, b_ref_b, e_map, b_ns, b_ys)
        b_x_e = []
        b_ref_a = []
        b_ref_b = []
        b_ns = []
        b_ys = []
        e_ids = {}
        e_count = 0
        g_count = 0
        e_map = []

    if g_count > 0:
      yield make_batch(b_x_e, b_ref_a, b_ref_b, e_map, b_ns, b_ys)

  return tf.data.Dataset.from_generator(
    gen,
    output_types=((
      tf.float32, tf.int32, tf.int32, tf.int32, tf.int32), tf.float32),
    output_shapes=((
      tf.TensorShape([None, f + 2]),
      tf.TensorShape([None, None]),
      tf.TensorShape([None, None]),
      tf.TensorShape([None]),
      tf.TensorShape([None])),
      tf.TensorShape([None, *y_dim])))


output_types = {
  "vert": to_vert_ds,
  "edge2": to_edge2_ds
}

def tf_dataset_generator(f):
  @fy.wraps(f)
  def w(*args, output_type="vert", **kwargs):
    r = cp.tolerant(f)(*args, **kwargs)

    if len(r) == 2:
      name, r = r
    else:
      name = None

    if len(r) == 3:
      X, A, y = r
      n = None
    else:
      X, A, n, y = r

    ds = output_types[output_type](X, A, n, y, **kwargs)
    ds.name = f.__name__ if name is None else name

    return ds

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
