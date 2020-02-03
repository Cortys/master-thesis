from __future__ import absolute_import, division, print_function,\
  unicode_literals

from tensorflow import keras
import funcy as fy

class DenseLayer(keras.layers.Dense):
  def __init__(
    self, in_dim, out_dim, act="linear",
    W_regularizer=None, b_regularizer=None,
    bias=False, wrapped_input=False, **kwargs):
    kwargs = fy.project(kwargs, [
      "units", "input_shape", "activation", "use_bias",
      "kernel_initializer", "bias_initializer",
      "kernel_regularizer", "bias_regularizer",
      "kernel_constraint", "bias_constraint"])

    # keras.layer.Dense arg names have precedence over their aliases:
    kwargs = fy.merge(dict(
      units=out_dim, input_shape=(in_dim,),
      activation=act, use_bias=bias,
      kernel_regularizer=W_regularizer,
      bias_regularizer=b_regularizer
    ), kwargs)

    super().__init__(**kwargs)

    self.wrapped_input = wrapped_input

  def get_config(self):
    base_config = super().get_config()
    base_config["wrapped_input"] = self.wrapped_input

    return base_config

  def build(self, input_shape):
    super().build(
      input_shape[0] if self.wrapped_input
      else input_shape)

  def call(self, input):
    if self.wrapped_input:
      X = input[0]
      X_out = super().call(X)

      return (X_out, *input[1:])

    return super().call(input)
