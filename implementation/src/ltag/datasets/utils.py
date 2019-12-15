from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

def to_tf(x, adjs, y, ragged=False):
  x_in = x
  x = tf.ragged.constant(x)
  adjs = tf.ragged.constant(adjs)

  if not ragged:
    x = x.to_tensor()
    adjs = adjs.to_tensor()

  return tf.data.Dataset.from_tensor_slices((
    (
      tf.cast(x, tf.float32),
      tf.cast(adjs, tf.float32),
      tf.constant([s.shape[0] for s in x_in], dtype=tf.int32)
    ),
    tf.constant(y, dtype=tf.float32)
  ))

def tf_dataset_generator(f):
  def w(*args, ragged=False):
    return to_tf(*f(*args), ragged=ragged)

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
