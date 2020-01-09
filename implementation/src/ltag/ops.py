from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf

import sys
sys.argv = sys.argv[:1]

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
def vec_mask(n, max_n):
  vec_mask = tf.cast(tf.sequence_mask(n, maxlen=max_n), tf.float32)

  return tf.expand_dims(vec_mask, -1)

@tf.function
def matrix_mask(n, max_n, sparse=False):
  vec_mask = tf.cast(tf.sequence_mask(n, maxlen=max_n), tf.int32)
  m_1 = tf.expand_dims(vec_mask, 2)
  m_2 = tf.expand_dims(vec_mask, 1)
  m = tf.cast(tf.expand_dims(tf.linalg.matmul(m_1, m_2), -1), tf.float32)

  return m

@tf.function
def neighborhood_mask(AX_e, degree=1):
  mask = tf.where(
    tf.reduce_sum(AX_e, axis=-1) == 0,
    0, 1)

  mask_p = mask

  for _ in tf.range(degree - 1):
    mask_p = tf.linalg.matmul(mask_p, mask)

  return tf.expand_dims(tf.where(mask_p == 0, 0.0, 1.0), -1)

@tf.function
def aggregate_edge_features(X, agg):
  X_shape = tf.shape(X)
  n = X_shape[-2]

  X_b_shape = tf.concat([[n], X_shape], axis=0)

  X_b = tf.broadcast_to(X, X_b_shape)

  X_1 = tf.transpose(X_b, perm=(1, 2, 0, 3, 4))
  X_2 = tf.transpose(X_b, perm=(1, 0, 3, 2, 4))

  X_prod = agg(X_1, X_2)

  return tf.reduce_sum(X_prod, axis=-2)

@tf.function
def aggregate_edge_features_using_refs(X, ref_a, ref_b, agg):
  X_a = tf.gather(X, ref_a, axis=0, batch_dims=1)
  X_b = tf.gather(X, ref_b, axis=0, batch_dims=1)
  X_ab = agg(X_a, X_b)
  X_agg = tf.reduce_sum(X_ab, axis=-2)

  return X_agg

# import ltag.datasets.synthetic as synthetic
#
# ds = synthetic.triangle_dataset(output_type="wl2", batch_graph_count=1)
#
# (X, ref_a, ref_b, e_map, v_count), y = list(ds)[0]
#
# print(aggregate_edge_features_using_refs(X, ref_a, ref_b, tf.multiply))
