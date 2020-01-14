from __future__ import absolute_import, division, print_function,\
  unicode_literals

from tensorflow import keras
import itertools
import funcy as fy

import ltag.models as models

def model_factory(mf):
  def model_factory_instance(model_class):
    def model_factory_with_class(hyperparam_gen):
      @fy.wraps(hyperparam_gen)
      def wrapper(ds_manager, **params):
        hps = list(hyperparam_gen(ds_manager, **params))
        return fy.partial(mf, model_class), hps

      wrapper.input_type = model_class.input_type
      wrapper.name = model_class.name

      return wrapper

    return model_factory_with_class

  return model_factory_instance

@model_factory
def binary_classifier(model_class, learning_rate=0.001, **params):
  model = model_class(act="sigmoid", squeeze_output=True, **params)

  opt = keras.optimizers.Adam(learning_rate)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  return model

@model_factory
def logistic_regression(
  model_class, act="tanh", learning_rate=0.001, **params):
  model = model_class(act=act, squeeze_output=True, **params)

  opt = keras.optimizers.Adam(learning_rate)

  model.compile(
    optimizer=opt,
    loss="mse",
    metrics=["mae"])

  return model

def cart(*pos_params, **params):
  if len(pos_params) > 0:
    return itertools.product(*pos_params)

  return (dict(zip(params, x)) for x in itertools.product(*params.values()))


@binary_classifier(models.AvgWL2GCN)
def AvgWL2GCN_Binary(dsm):
  in_dim = dsm.dim_wl2_features()

  hidden = [
    [b] * l
    for b, l in cart([8, 32], [1, 3])]

  hidden = [
    [8], [8, 8, 8],
    [32], [32, 32]
  ]

  return cart(
    layer_dims=[[in_dim, *h, 1] for h in hidden],
    bias=[True],
    learning_rate=[0.0001]
  )
