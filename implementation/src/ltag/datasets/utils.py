from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

def to_tf(x, adjs, y):
  return tf.data.Dataset.from_tensor_slices((
    (
      tf.cast(tf.ragged.constant(x).to_tensor(), tf.float32),
      tf.cast(tf.ragged.constant(adjs).to_tensor(), tf.float32)
    ),
    tf.cast(tf.constant(y), tf.float32)
  ))

def tf_dataset_generator(f):
  def w(*args):
    return to_tf(*f(*args))

  return w

def draw_graph(x, adj, y):
  plt.figure()
  plt.title('Label: {}'.format(y))
  nx.draw_spring(nx.from_numpy_array(adj))
  plt.show()
