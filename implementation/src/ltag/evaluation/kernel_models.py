from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.models.kernel as kernel_models
from ltag.evaluation.model_factories import kernel_classifier
from ltag.utils import cart

kernel_hps = lambda dsm: cart(C=[1, 0.1, 0.01, 0.001, 0.0001])
WL_st = kernel_classifier(kernel_models.WL_st)(kernel_hps)
WL_st_1 = kernel_classifier(kernel_models.WL_st_1)(kernel_hps)
WL_st_2 = kernel_classifier(kernel_models.WL_st_2)(kernel_hps)
WL_sp = kernel_classifier(kernel_models.WL_sp)(kernel_hps)
LWL2 = kernel_classifier(kernel_models.LWL2)(kernel_hps)
GWL2 = kernel_classifier(kernel_models.GWL2)(kernel_hps)
