from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.models.gnn as gnn_models
import ltag.models.kernel as kernel_models
from ltag.evaluation.model_factories import (
  binary_classifier, kernel_classifier)
from ltag.utils import cart, cart_merge

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


kernel_hps = lambda dsm: cart(C=[1, 0.1, 0.01, 0.001, 0.0001])
WL_st = kernel_classifier(kernel_models.WL_st)(kernel_hps)
WL_sp = kernel_classifier(kernel_models.WL_sp)(kernel_hps)
LWL2 = kernel_classifier(kernel_models.LWL2)(kernel_hps)
GWL2 = kernel_classifier(kernel_models.GWL2)(kernel_hps)
