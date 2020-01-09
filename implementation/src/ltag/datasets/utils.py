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

def make_wl2_batch(b_x_e, b_ref_a, b_ref_b, e_map, b_ns, b_ys):
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

@cp.tolerant
def to_wl2_ds(
  xs, adjs, n_s, ys,
  shuffle=False,
  fuzzy_batch_edge_count=100000,
  upper_batch_edge_count=120000,
  batch_graph_count=100,
  neighborhood=1,
  lazy=False,
  as_list=False, log=False):
  ds_size = len(xs)

  f = xs[0].shape[1] if ds_size > 0 else 0

  il = np.arange(ds_size)

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

    if ds_size > 0:
      g_next = nx.from_numpy_array(adjs[il[0]])
      g_next_p = nx.power(g_next, neighborhood)

      for node in g_next.nodes:
        g_next.add_edge(node, node)
        g_next_p.add_edge(node, node)

    for i_pos, i in enumerate(il):
      x = xs[i]
      y = ys[i]
      n = x.shape[0] if n_s is None else n_s[i]

      y = ys[i]
      e_zero = np.zeros(f)
      local_e_count = 0
      g = g_next
      g_p = g_next_p

      if i_pos + 1 < ds_size:
        i_next = il[i_pos + 1]
        adj = adjs[i_next]
        g_next = nx.from_numpy_array(adj)
        g_next_p = nx.power(g_next, neighborhood)

        for node in g_next.nodes:
          g_next.add_edge(node, node)
          g_next_p.add_edge(node, node)
      else:
        g_next_p = None

      if n == 0:
        continue

      for v, u in g_p.edges:
        e_ids[(i, v, u)] = e_count
        e_count += 1
        local_e_count += 1

      e_map += [g_count] * local_e_count

      for edge in g_p.edges:
        a, b = edge
        in_g = g.has_edge(a, b)
        w = g.get_edge_data(a, b).get("weight", 0) if in_g else 0
        ne = (
          ([a] if a == b else [a, b])
          + list(nx.common_neighbors(g_p, a, b)))
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
      next_e_count = (
        e_count if g_next_p is None
        else e_count + nx.number_of_edges(g_next_p))

      if (
        g_count >= batch_graph_count
        or e_count >= fuzzy_batch_edge_count
        or (g_count > 0 and next_e_count >= upper_batch_edge_count)):
        if log:
          print(
            "Batch with", g_count, "graphs and",
            e_count, "edges with", f, "features.")
        yield make_wl2_batch(b_x_e, b_ref_a, b_ref_b, e_map, b_ns, b_ys)
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
      if log:
        print(
          "Batch with", g_count, "graphs and",
          e_count, "edges with", f, "features.")
      yield make_wl2_batch(b_x_e, b_ref_a, b_ref_b, e_map, b_ns, b_ys)

  if lazy and not as_list:
    gen_out = gen
  else:
    batches = list(gen())

    if as_list:
      return batches

    def gen_out():
      for b in batches:
        yield b

  return tf.data.Dataset.from_generator(
    gen_out,
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
  "wl2": to_wl2_ds
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
