from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy

import ltag.models.gnn as gnn_models
from ltag.utils import cart, cart_merge, entry_duplicator
from ltag.evaluation.model_factories import binary_classifier
from ltag.evaluation.kernel_models import (
  WL_st, WL_st_1, WL_st_2, WL_st_3, WL_st_4, WL_sp, WL_sp_3, LWL2, LWL2_1, GWL2)

kernel_models = [
  WL_st, WL_st_1, WL_st_2, WL_st_3, WL_st_4,
  WL_sp, WL_sp_3,
  LWL2, LWL2_1, GWL2]

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
def AvgWL2GCN_FC_Binary_Small(dsm):
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


# CWL2GCN:

@binary_classifier(gnn_models.AvgCWL2GCN)
def AvgCWL2GCN_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[32, 64],
  cwl2_layer_depths=[1, 3],
  cwl2_stack_tfs=[None, "keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    [b] * l
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  return cart(
    conv_layer_dims=[[in_dim, *h, 1] for h in hidden],
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    learning_rate=[0.01, 0.001, 0.0001],
    squeeze_output=[True]
  )

@binary_classifier(gnn_models.with_fc(gnn_models.AvgCWL2GCN))
def AvgCWL2GCN_FC_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[32, 64],
  cwl2_layer_depths=[1, 3],
  cwl2_stack_tfs=[None, "keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    ([b] * l, [b, b])
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  hidden_hp = [dict(
    conv_layer_dims=[in_dim, *ch],
    fc_layer_dims=[*fh, 1]
  ) for ch, fh in hidden]

  return cart_merge(cart(
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    fc_bias=[True],
    learning_rate=[0.01, 0.001, 0.0001],
    squeeze_output=[True]
  ), hidden_hp)

@binary_classifier(gnn_models.SagCWL2GCN)
def SagCWL2GCN_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[32, 64],
  cwl2_layer_depths=[1, 3],
  cwl2_stack_tfs=[None, "keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    [b] * l
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  hps = cart(
    conv_layer_dims=[[in_dim, *h, 1] for h in hidden],
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    learning_rate=[0.01, 0.001, 0.0001],
    squeeze_output=[True]
  )

  duplicate_settings = {
    "conv_layer_dims": ["att_conv_layer_dims"],
    "conv_act": ["att_conv_act"],
    "conv_local_act": ["att_conv_local_act"],
    "conv_stack_tf": ["att_conv_stack_tf"],
    "conv_bias": ["att_conv_bias"],
  }

  return fy.map(entry_duplicator(duplicate_settings), hps)

@binary_classifier(gnn_models.with_fc(gnn_models.SagCWL2GCN))
def SagCWL2GCN_FC_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[32, 64],
  cwl2_layer_depths=[2, 4],
  cwl2_stack_tfs=[None, "keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    ([b] * l, [b, b])
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  hidden_hp = [dict(
    conv_layer_dims=[in_dim, *ch],
    fc_layer_dims=[*fh, 1]
  ) for ch, fh in hidden]

  base_hp = cart(
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    fc_bias=[True],
    learning_rate=[0.01, 0.001, 0.0001],
    squeeze_output=[True]
  )

  hps = cart_merge(base_hp, hidden_hp)

  duplicate_settings = {
    "conv_layer_dims": ["att_conv_layer_dims"],
    "conv_act": ["att_conv_act"],
    "conv_local_act": ["att_conv_local_act"],
    "conv_stack_tf": ["att_conv_stack_tf"],
    "conv_bias": ["att_conv_bias"],
  }

  return fy.map(entry_duplicator(duplicate_settings), hps)

@binary_classifier(gnn_models.SagCWL2GCN)
def SagCWL2GCN_Binary_quick_max(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[64],
  cwl2_layer_depths=[3],
  cwl2_stack_tfs=["keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    [b] * l
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  hps = cart(
    conv_layer_dims=[[in_dim, *h, 1] for h in hidden],
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    learning_rate=[0.001],
    squeeze_output=[True]
  )

  duplicate_settings = {
    "conv_layer_dims": ["att_conv_layer_dims"],
    "conv_act": ["att_conv_act"],
    "conv_local_act": ["att_conv_local_act"],
    "conv_stack_tf": ["att_conv_stack_tf"],
    "conv_bias": ["att_conv_bias"],
  }

  return fy.map(entry_duplicator(duplicate_settings), hps)


# K2GNN:

@binary_classifier(gnn_models.with_fc(gnn_models.AvgK2GNN))
def AvgK2GNN_FC_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[32, 64],
  cwl2_layer_depths=[1, 3],
  cwl2_stack_tfs=[None, "keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    ([b] * l, [b, b])
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  hidden_hp = [dict(
    conv_layer_dims=[in_dim, *ch],
    fc_layer_dims=[*fh, 1]
  ) for ch, fh in hidden]

  return cart_merge(cart(
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    fc_bias=[True],
    learning_rate=[0.01, 0.001, 0.0001],
    squeeze_output=[True]
  ), hidden_hp)

@binary_classifier(gnn_models.with_fc(gnn_models.SagK2GNN))
def SagK2GNN_FC_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_layer_widths=[32, 64],
  cwl2_layer_depths=[2, 4],
  cwl2_stack_tfs=[None, "keep_input"]):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    ([b] * l, [b, b])
    for b, l in cart(cwl2_layer_widths, cwl2_layer_depths)]

  hidden_hp = [dict(
    conv_layer_dims=[in_dim, *ch],
    fc_layer_dims=[*fh, 1]
  ) for ch, fh in hidden]

  base_hp = cart(
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_bias=[True],
    fc_bias=[True],
    learning_rate=[0.01, 0.001, 0.0001],
    squeeze_output=[True]
  )

  hps = cart_merge(base_hp, hidden_hp)

  duplicate_settings = {
    "conv_layer_dims": ["att_conv_layer_dims"],
    "conv_act": ["att_conv_act"],
    "conv_local_act": ["att_conv_local_act"],
    "conv_stack_tf": ["att_conv_stack_tf"],
    "conv_bias": ["att_conv_bias"],
  }

  return fy.map(entry_duplicator(duplicate_settings), hps)


# Timing:

@binary_classifier(gnn_models.AvgCWL2GCN)
def AvgCWL2GCN_Timing_Binary(
  dsm,
  cwl2_local_act="sigmoid",
  cwl2_stack_tfs=[None]):
  in_dim = dsm.dim_wl2_features()

  return cart(
    conv_act=["sigmoid"],
    conv_local_act=[cwl2_local_act],
    conv_stack_tf=cwl2_stack_tfs,
    conv_layer_dims=[[in_dim, 24, 24, 1]],
    conv_bias=[True],
    fc_bias=[True],
    learning_rate=[0.01],
    squeeze_output=[True]
  )

@binary_classifier(gnn_models.AvgGIN)
def AvgGIN_Timing_Binary(
  dsm,
  wl1_local_act="sigmoid",
  wl1_stack_tfs=[None]):
  in_dim = dsm.dim_wl1_features()

  return cart(
    conv_act=["sigmoid"],
    conv_stack_tf=wl1_stack_tfs,
    conv_layer_dims=[[in_dim, 32, 1]],
    conv_bias=[True],
    fc_bias=[True],
    learning_rate=[0.01],
    squeeze_output=[True]
  )
