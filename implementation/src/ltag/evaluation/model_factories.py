from __future__ import absolute_import, division, print_function,\
  unicode_literals

from tensorflow import keras
import itertools
import funcy as fy

def model_factory(mf):
  """
    Takes a fn `mf` that creates a trainable model from hyperparams.
    Returns a function that takes a fn `hyperparam_gen`
    which generates a set of hyperparams for a given dataset maanger.
    Finally returns a function that takes a dataset manager and
    returns a model factory and a list of hyperparams to supply to the factory.
  """
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
def binary_classifier(
  model_class, act="sigmoid", learning_rate=0.001,
  squeeze_output=True, **params):
  model = model_class(
    act=act, squeeze_output=squeeze_output, **params)

  opt = keras.optimizers.Adam(learning_rate)

  model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"])

  return model

@model_factory
def logistic_regression(
  model_class, act="tanh", learning_rate=0.001,
  squeeze_output=True, **params):
  model = model_class(act=act, squeeze_output=squeeze_output, **params)

  opt = keras.optimizers.Adam(learning_rate)

  model.compile(
    optimizer=opt,
    loss="mse",
    metrics=["mae"])

  return model

def cart(*pos_params, **params):
  "Lazyly computes the cartesian product of the given lists or dicts."
  if len(pos_params) > 0:
    return itertools.product(*pos_params)

  return (dict(zip(params, x)) for x in itertools.product(*params.values()))

def cart_merge(*dicts):
  "Lazyly computes all possible merge combinations of the given dicts."
  return (fy.merge(*c) for c in itertools.product(*dicts))