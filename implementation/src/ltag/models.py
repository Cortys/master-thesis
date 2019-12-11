from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
import funcy as fy
from tensorflow import keras

from ltag.layers import EFGCNLayer

def gcn(layer, layer_dims=[], **kwargs):
  X = keras.Input(shape=(None, layer_dims[0]), name="X")
  A = keras.Input(shape=(None, None), name="A")
  n = keras.Input(shape=(), dtype=tf.int32, name="n")
  inputs = (X, A, n)

  h = inputs
  for f_dim in layer_dims[1:]:
    h = layer(f_dim, **kwargs)(h)

  Y = h[0]

  return inputs, Y


def as_model(name, io_fn, *args, **kwargs):
  def Model(*args2, **kwargs2):
    inputs, output = io_fn(*args, *args2, **kwargs, **kwargs2)

    return keras.Model(inputs=inputs, outputs=output, name=name)

  return Model


def avg_agg(io):
  (X, A, n), Y = io

  Y_shape = tf.shape(Y)
  max_n = Y_shape[-2]
  n_mask = tf.broadcast_to(
    tf.expand_dims(tf.sequence_mask(n, maxlen=max_n), 2), Y_shape)

  Y = tf.where(n_mask, Y, tf.zeros_like(Y))

  y = tf.transpose(
    tf.math.divide_no_nan(
      tf.transpose(tf.reduce_sum(Y, 1)),
      tf.cast(n, tf.float32)))

  return (X, A, n), y


EFGCN = as_model("EFGCN", gcn, EFGCNLayer)
AVG_EFGCN = as_model("AVG_EFGCN", fy.rcompose(gcn, avg_agg), EFGCNLayer)
