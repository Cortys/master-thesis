from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf

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


@tf.function
def edge_feature_aggregation(X, agg):
  X_shape = tf.shape(X)
  n = X_shape[-2]

  X_b_shape = tf.concat([[n], X_shape], axis=0)

  X_b = tf.broadcast_to(X, X_b_shape)

  X_1 = tf.transpose(X_b, perm=(1, 2, 0, 3, 4))
  X_2 = tf.transpose(X_b, perm=(1, 0, 3, 2, 4))
  X_prod = agg(X_1, X_2)

  return tf.reduce_sum(X_prod, axis=-2)
