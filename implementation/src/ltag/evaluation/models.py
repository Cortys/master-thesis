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
      def wrapper(ds, **params):
        hps = hyperparam_gen(ds, **params)

        def mf_caller(hp):
          model_ctr = fy.partial(mf, model_class, **hp)
          model_ctr.hyperparams = hp
          return model_ctr

        return fy.map(mf_caller, hps)

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

def wl2_in_dim(ds):
  return ds.element_spec[0][0].shape[-1]


@binary_classifier(models.AvgWL2GCN)
def AvgWL2GCN_Binary(ds):
  in_dim = wl2_in_dim(ds)

  hidden = [
    [b] * l
    for b, l in cart([8, 32], [1, 3])]

  return cart(
    layer_dims=[[in_dim, *h, 1] for h in hidden],
    bias=[True, False],
    learning_rate=[0.01, 0.001, 0.0001]
  )
