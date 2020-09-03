from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
import funcy as fy

import ltag.chaining.pipeline as cp

@cp.tolerant
def to_dense_ds(
  graphs, ys,
  dim_node_features, num_node_labels,
  ragged=False, sparse=False):

  if dim_node_features == 0 and num_node_labels == 0:
    x = [np.ones((g.order(), 1)) for g in graphs]
  elif num_node_labels == 0:
    x = [
      [f for _, f in g.nodes(data="features")]
      for g in graphs]
  else:
    I_n = np.eye(num_node_labels)
    n_zero = np.zeros(dim_node_features)
    x = [[
      np.concatenate((
        I_n[data["label"] - 1],
        data.get("features", n_zero)))
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

@cp.tolerant
def wl1_encode(g, dim_node_features=None, num_node_labels=None):
  if dim_node_features is None:
    dim_node_features = 0

  if num_node_labels is None:
    num_node_labels = 0

  dim_node_wl1 = dim_node_features + num_node_labels

  n_zero_f = np.zeros(dim_node_features)
  I_n = np.eye(num_node_labels)

  x = []
  ref_a = []
  ref_b = []

  n_ids = {}
  i = 0

  for node, data in g.nodes(data=True):
    if dim_node_wl1 == 0:
      f = [1]
      lab = []
    else:
      f = data.get("features", n_zero_f)
      if num_node_labels > 0:
        lab = I_n[data["label"] - 1]
      else:
        lab = []

    n_ids[node] = i
    i += 1
    x.append(np.concatenate((lab, f)))

  for (a, b) in g.edges():
    ref_a.append(n_ids[a])
    ref_b.append(n_ids[b])

  n = g.order()
  return (
    np.array(x),
    np.array(ref_a),
    None,
    np.array(ref_b),
    n)

@cp.tolerant
def edge_neighborhood_encode(
  g,
  dim_node_features=None, dim_edge_features=None,
  num_node_labels=None, num_edge_labels=None,
  with_indices=False):
  """
    Takes a graph with node and edge features and converts it
    into an sparse edge neighborhood representation for 2-GNNs.
  """

  if dim_node_features is None:
    dim_node_features = 0

  if dim_edge_features is None:
    dim_edge_features = 0

  if num_node_labels is None:
    num_node_labels = 0

  if num_edge_labels is None:
    num_edge_labels = 0

  dim_node_wl2 = dim_node_features + num_node_labels
  dim_edge_wl2 = dim_edge_features + num_edge_labels
  n_zero_f = np.zeros(dim_node_features)
  n_zero_l = np.zeros(num_node_labels)
  e_zero_f = np.zeros(dim_edge_features)
  e_zero_l = np.zeros(num_edge_labels)
  I_n = np.eye(num_node_labels)
  I_e = np.eye(num_edge_labels)

  for node, data in g.nodes(data=True):
    g.add_edge(
      node, node,
      features=data.get("features", n_zero_f),
      label=data.get("label"))

  x = []
  ref_a = []
  backref = []
  e_ids = {}

  for i, edge in enumerate(g.edges):
    e_ids[edge] = i

  for i, edge in enumerate(g.edges):
    a, b = edge
    if (
      (a != b and dim_edge_wl2 == 0)
      or (a == b and dim_node_wl2 == 0)):
      f = []
      lab = []
    else:
      d = g.get_edge_data(a, b)
      f = d.get("features", e_zero_f)
      if a == b and num_node_labels > 0:
        lab = I_n[d["label"] - 1]
      elif a != b and num_edge_labels > 0:
        lab = I_e[d["label"] - 1]
      else:
        lab = []

    n_ab = {eid_lookup(e_ids, a, k) for k in nx.neighbors(g, a) if k != b}
    if a != b:
      n_ab |= {
        eid_lookup(e_ids, b, k)
        for k in nx.neighbors(g, b) if k != a}

    n_ab = list(n_ab)
    nbs_count = len(n_ab)

    x.append(np.concatenate(
      ([1, 0, 0], lab, f, e_zero_l, e_zero_f) if a == b else
      ([0, 1, 0], n_zero_l, n_zero_f, lab, f)))

    ref_a += n_ab
    backref += [e_ids[edge]] * nbs_count

  n = g.order()
  res = (
    np.array(x),
    np.array(ref_a),
    None,
    np.array(backref),
    n)

  if with_indices:
    res += (list(g.edges),)

  return res

def wl2_encode(
  g,
  dim_node_features=None, dim_edge_features=None,
  num_node_labels=None, num_edge_labels=None,
  neighborhood=1, with_indices=False, compact=False):
  """
    Takes a graph with node and edge features and converts it
    into a edge + row/col ref list for sparse WL2 implementations.
  """
  g_p = nx.power(g, neighborhood) if neighborhood > 1 else g

  if dim_node_features is None:
    dim_node_features = 0

  if dim_edge_features is None:
    dim_edge_features = 0

  if num_node_labels is None:
    dim_edge_features = 0

  if num_edge_labels is None:
    num_edge_labels = 0

  dim_node_wl2 = dim_node_features + num_node_labels
  dim_edge_wl2 = dim_edge_features + num_edge_labels
  n_zero_f = np.zeros(dim_node_features)
  n_zero_l = np.zeros(num_node_labels)
  e_zero_f = np.zeros(dim_edge_features)
  e_zero_l = np.zeros(num_edge_labels)
  I_n = np.eye(num_node_labels)
  I_e = np.eye(num_edge_labels)

  for node, data in g.nodes(data=True):
    g_p.add_edge(node, node)
    g.add_edge(
      node, node,
      features=data.get("features", n_zero_f),
      label=data.get("label"))

  x = []
  ref_a = []
  ref_b = []
  backref = []
  max_ref_dim = 0
  e_ids = {}

  for i, edge in enumerate(g_p.edges):
    e_ids[edge] = i

  for i, edge in enumerate(g_p.edges):
    a, b = edge
    in_g = g.has_edge(a, b)
    if not in_g:
      f = e_zero_f
      lab = e_zero_l
    elif (
      (a != b and dim_edge_wl2 == 0)
      or (a == b and dim_node_wl2 == 0)):
      f = []
      lab = []
    else:
      d = g.get_edge_data(a, b)
      f = d.get("features", e_zero_f)
      if a == b and num_node_labels > 0:
        lab = I_n[d["label"] - 1]
      elif a != b and num_edge_labels > 0:
        lab = I_e[d["label"] - 1]
      else:
        lab = []

    nbs = (
      ([a] if a == b else [a, b])
      + list(nx.common_neighbors(g_p, a, b)))
    n_a = [eid_lookup(e_ids, a, k) for k in nbs]
    n_b = [eid_lookup(e_ids, b, k) for k in nbs]
    nbs_count = len(nbs)

    x.append(np.concatenate(
      ([1, 0, 0], lab, f, e_zero_l, e_zero_f) if a == b else
      ([0], [1, 0] if in_g else [0, 1], n_zero_l, n_zero_f, lab, f)))

    if compact:
      ref_a += n_a
      ref_b += n_b
      backref += [e_ids[edge]] * nbs_count
    else:
      if nbs_count > max_ref_dim:
        max_ref_dim = nbs_count

      ref_a.append(np.array(n_a))
      ref_b.append(np.array(n_b))

  n = g.order()
  res = (
    np.array(x),
    np.array(ref_a), np.array(ref_b),
    np.array(backref if compact else max_ref_dim),
    n)

  if with_indices:
    res += (list(g_p.edges),)

  return res

def make_wl_batch(
  encoded_graphs, dim_wl, with_indices=False,
  compact=False, en_encode=False, wl1_encode=False):
  "Takes a sequence of graphs that were encoded via `wl2_encode` or\
   `edge_neighborhood_encode` (if en_encode=True)."
  if not compact and not en_encode and not wl1_encode:
    max_ref_dim = np.max([g[3] for g in encoded_graphs])

  x_len = 0
  r_len = 0
  for e in encoded_graphs:
    x = e[0]
    r = e[1]
    x_len += len(x)
    r_len += len(r)

  b_x = np.empty((x_len, dim_wl), dtype=float)
  if compact:
    b_ref_a = np.empty((r_len,), dtype=int)
    b_ref_b = np.empty((r_len,), dtype=int)
    b_backref = np.empty((r_len,), dtype=int)
  elif en_encode or wl1_encode:
    b_ref_a = np.empty((r_len,), dtype=int)
    b_backref = np.empty((r_len,), dtype=int)
  else:
    b_ref_a = np.empty((r_len, max_ref_dim), dtype=int)
    b_ref_b = np.empty((r_len, max_ref_dim), dtype=int)

  b_e_map = np.empty((x_len,), dtype=float)
  b_n = np.empty(len(encoded_graphs))
  b_idx = np.empty((x_len, 2), dtype=int)
  e_offset = 0
  r_offset = 0

  for i, e in enumerate(encoded_graphs):
    if with_indices:
      x, ref_a, ref_b, backref, n, idx = e
      e_count = len(x)
      next_e_offset = e_offset + e_count
      b_idx[e_offset:next_e_offset] = idx
    else:
      x, ref_a, ref_b, backref, n = e
      e_count = len(x)
      next_e_offset = e_offset + e_count

    if e_count == 0:  # discard empty graphs
      continue

    r_count = len(ref_a)
    next_r_offset = r_offset + r_count
    b_x[e_offset:next_e_offset] = x
    b_e_map[e_offset:next_e_offset] = [i] * e_count
    b_n[i] = n

    if compact:
      b_ref_a[r_offset:next_r_offset] = ref_a + e_offset
      b_ref_b[r_offset:next_r_offset] = ref_b + e_offset
      b_backref[r_offset:next_r_offset] = backref + e_offset
    elif en_encode or wl1_encode:
      b_ref_a[r_offset:next_r_offset] = ref_a + e_offset
      b_backref[r_offset:next_r_offset] = backref + e_offset
    else:
      for i_r in range(r_count):
        r_a = ref_a[i_r]
        r_b = ref_b[i_r]
        b_ref_a[r_offset + i_r, :] = np.pad(
          r_a + e_offset,
          (0, max_ref_dim - len(r_a)), "constant", constant_values=-1)
        b_ref_b[r_offset + i_r, :] = np.pad(
          r_b + e_offset,
          (0, max_ref_dim - len(r_b)), "constant", constant_values=-1)

    e_offset = next_e_offset
    r_offset = next_r_offset

  if compact:
    res = (b_x, b_ref_a, b_ref_b, b_backref, b_e_map, b_n)
  elif en_encode or wl1_encode:
    res = (b_x, b_ref_a, b_backref, b_e_map, b_n)
  else:
    res = (b_x, b_ref_a, b_ref_b, b_e_map, b_n)

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

def wl_batches_to_dataset(
  batches, dim_wl, y_dim, compact=False, en_encode=False, wl1_encode=False):
  if callable(batches):
    batch_gen = batches
  else:
    def batch_gen():
      for b in batches:
        yield b

  if compact:
    return tf.data.Dataset.from_generator(
      batch_gen,
      output_types=((
        tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
        tf.float32),
      output_shapes=((
        tf.TensorShape([None, dim_wl]),   # X
        tf.TensorShape([None]),           # ref_a
        tf.TensorShape([None]),           # ref_b
        tf.TensorShape([None]),           # backref
        tf.TensorShape([None]),           # e_map
        tf.TensorShape([None])),          # n
        tf.TensorShape([None, *y_dim])))  # y
  if en_encode or wl1_encode:
    return tf.data.Dataset.from_generator(
      batch_gen,
      output_types=((
        tf.float32, tf.int32, tf.int32, tf.int32, tf.int32),
        tf.float32),
      output_shapes=((
        tf.TensorShape([None, dim_wl]),   # X
        tf.TensorShape([None]),           # ref_a
        tf.TensorShape([None]),           # backref/ref_b
        tf.TensorShape([None]),           # e_map
        tf.TensorShape([None])),          # n
        tf.TensorShape([None, *y_dim])))  # y
  else:
    return tf.data.Dataset.from_generator(
      batch_gen,
      output_types=((
        tf.float32, tf.int32, tf.int32, tf.int32, tf.int32), tf.float32),
      output_shapes=((
        tf.TensorShape([None, dim_wl]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None]),
        tf.TensorShape([None])),
        tf.TensorShape([None, *y_dim])))


@cp.tolerant
def to_wl_ds(
  graphs, ys,
  dim_node_features=0,
  dim_edge_features=0,
  num_node_labels=0,
  num_edge_labels=0,
  fuzzy_batch_edge_count=100000,
  upper_batch_edge_count=120000,
  batch_graph_count=100,
  neighborhood=1,
  with_indices=False,
  compact=False, en_encode=False,
  wl1_encode=False,
  lazy=False, preencoded=False,
  as_list=False):
  ds_size = len(graphs)
  y_shape = ys.shape
  y_dim = y_shape[1:] if len(y_shape) > 1 else []

  if wl1_encode:
    dim_wl = max(1, dim_node_features + num_node_labels)
  else:
    dim_wl = 3 + dim_node_features + dim_edge_features\
        + num_node_labels + num_edge_labels

  print(
    "Batching", ds_size,
    "preencoded" if preencoded else "raw",
    "compact" if compact else "non-compact",
    "graphs.",
    f"b_gc={batch_graph_count}",
    f"dim_wl={dim_wl}",
    f"(node={dim_node_features}+{num_node_labels},",
    f"edge={dim_edge_features}+{num_edge_labels})")

  if compact:
    encoder = fy.partial(wl2_encode, compact=True)
    batcher = fy.partial(
      make_wl_batch, compact=True,
      dim_wl=dim_wl, with_indices=with_indices)
    ds_conv = fy.partial(wl_batches_to_dataset, compact=True)
  elif en_encode:
    encoder = edge_neighborhood_encode
    batcher = fy.partial(
      make_wl_batch, dim_wl=dim_wl,
      with_indices=with_indices, en_encode=True)
    ds_conv = fy.partial(wl_batches_to_dataset, en_encode=True)
  elif wl1_encode:
    encoder = wl1_encode
    batcher = fy.partial(
      make_wl_batch, dim_wl=dim_wl,
      with_indices=with_indices, wl1_encode=True)
    ds_conv = fy.partial(wl_batches_to_dataset, wl1_encode=True)
  else:
    encoder = wl2_encode
    batcher = fy.partial(
      make_wl_batch, dim_wl=dim_wl, with_indices=with_indices)
    ds_conv = wl_batches_to_dataset

  if batch_graph_count > 1:
    def gen():
      b_gs = []
      b_ys = []
      e_count = 0

      if ds_size > 0:
        enc_next = graphs[0] if preencoded else encoder(
          graphs[0],
          dim_node_features, dim_edge_features,
          num_node_labels, num_edge_labels,
          neighborhood, with_indices)

      for i in range(ds_size):
        enc = enc_next
        y = ys[i]
        if i + 1 < ds_size:
          enc_next = graphs[i + 1] if preencoded else encoder(
            graphs[i + 1],
            dim_node_features=dim_node_features,
            dim_edge_features=dim_edge_features,
            num_node_labels=num_node_labels,
            num_edge_labels=num_edge_labels,
            neighborhood=neighborhood, with_indices=with_indices)
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
          yield (batcher(b_gs), b_ys)
          e_count = 0
          b_gs = []
          b_ys = []

      if len(b_gs) > 0:
        yield (batcher(b_gs), b_ys)
  else:
    def gen():
      for i in range(ds_size):
        yield (batcher([graphs[i]]), [ys[i]])

  if lazy:
    batches = gen
  else:
    batches_list = list(gen())
    batches = np.empty(len(batches_list), dtype='O')
    batches[:] = batches_list

    if as_list:
      return batches, dim_wl, y_dim

  return ds_conv(batches, dim_wl, y_dim)


def vec_to_unit(feat):
  u = 0
  for i, s in enumerate(np.clip(feat, 0, 1), 1):
    u += (2 ** -i) * s

  return u

def draw_graph(
  g, y, with_features=False, with_colors=True, label_colors=False):
  plt.figure()
  plt.title('Label: {}'.format(y))

  cmap = plt.get_cmap("hsv")
  node_color = [
    vec_to_unit([d.get("label", 0)] if label_colors else d.get("features", []))
    for n, d in g.nodes(data=True)] if with_colors else "#1f78b4"

  if with_features:
    labels = {
      n: f"{n}:" + str(data.get("features"))
      for n, data in g.nodes(data=True)
    }
    nx.draw_spring(
      g, labels=labels,
      node_color=node_color, vmin=0, vmax=1, cmap=cmap)
  else:
    nx.draw_spring(
      g, with_labels=True,
      node_color=node_color, vmin=0, vmax=1, cmap=cmap)

  plt.show()
