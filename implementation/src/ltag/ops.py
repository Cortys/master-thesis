from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
import networkx as nx
import functools as ft

@tf.function
def normalize_mat(M):
  diags = tf.reduce_sum(M, axis=-1)
  diags_norm = tf.pow(diags, -0.5)
  diags_norm_fixed = tf.where(
    tf.math.is_finite(diags_norm),
    diags_norm, tf.zeros(tf.shape(diags_norm)))
  D_norm = tf.linalg.diag(diags_norm_fixed)
  M_norm = tf.linalg.matmul(D_norm, tf.linalg.matmul(M, D_norm))

  return M_norm
