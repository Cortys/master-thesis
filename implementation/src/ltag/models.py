from __future__ import absolute_import, division, print_function,\
  unicode_literals

import tensorflow as tf
from tensorflow import keras

from ltag.layers import EFGCNLayer

def EFGCN(in_dim=1, hidden_dim=10, out_dim=1):
  X = keras.Input(shape=(None, in_dim), name="X")
  A = keras.Input(shape=(None, None), name="A")
  inputs = (X, A)

  h_1 = EFGCNLayer(hidden_dim)(inputs)
  h_2 = EFGCNLayer(out_dim)(h_1)
  y = h_2[0]

  return keras.Model(inputs=inputs, outputs=y, name="EFGCN")
