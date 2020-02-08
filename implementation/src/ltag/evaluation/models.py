from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.models.gnn as gnn_models
from ltag.utils import cart, cart_merge
from ltag.evaluation.model_factories import binary_classifier
from ltag.evaluation.kernel_models import (
  WL_st, WL_sp, LWL2, GWL2)

kernel_models = [WL_st, WL_sp, LWL2, GWL2]

@binary_classifier(gnn_models.AvgWL2GCN)
def AvgWL2GCN_Binary(dsm):
  "Small hyperparam space for averaging WL2GCNs + binary classification."
  in_dim = dsm.dim_wl2_features()

  hidden = [
    [8], [8, 8, 8],
    [32], [32, 32]
  ]

  return cart(
    layer_dims=[[in_dim, *h, 1] for h in hidden],
    bias=[True],
    learning_rate=[0.0001]
  )

@binary_classifier(gnn_models.AvgWL2GCN)
def AvgWL2GCN_Binary_3x32(dsm):
  "Like AvgWL2GCN_Binary but only one hyperparam config."
  in_dim = dsm.dim_wl2_features()

  hidden = [
    [32, 32, 32]
  ]

  return cart(
    layer_dims=[[in_dim, *h, 1] for h in hidden],
    bias=[True],
    learning_rate=[0.0001]
  )

@binary_classifier(gnn_models.with_fc(gnn_models.AvgWL2GCN))
def AvgWL2GCN_FC_Binary(dsm):
  "Small hyperparam space for averaging FC WL2GCNs + binary classification."
  in_dim = dsm.dim_wl2_features()

  base = cart(
    squeeze_output=[False],
    bias=[True],
    learning_rate=[0.0001])

  hidden = [
    ([8, 8, 8], [8, 8]),
    ([32, 32], [32, 32])
  ]

  hidden_hp = [dict(
    conv_layer_dims=[in_dim, *ch],
    fc_layer_dims=[*fh, 1]
  ) for ch, fh in hidden]

  return cart_merge(base, hidden_hp)
