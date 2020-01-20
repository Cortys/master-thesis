from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy
from tensorflow import keras

import ltag.chaining.pipeline as pipeline

@pipeline.pipeline_step
def as_model(io, name):
  inputs, outputs = io

  return keras.Model(inputs=inputs, outputs=outputs, name=name)

def model_inputs(f):
  f = pipeline.tolerant(f)

  @fy.wraps(f)
  def wrapper(*args, **kwargs):
    inputs = f(*args, **kwargs)

    return inputs, inputs

  wrapper.__tolerant__ = True

  return pipeline.pipeline_start(wrapper)

def model_step(f):
  f = pipeline.tolerant(f)

  @fy.wraps(f)
  def wrapper(io, *args, **kwargs):
    inputs, outputs = io
    new_outputs = f(outputs, *args, **kwargs)

    return inputs, new_outputs

  wrapper.__tolerant__ = True

  return pipeline.pipeline_step(wrapper)

def create_model(name, steps, extend_at=None, **kwargs):
  modelFactory = pipeline.create_pipeline([*steps, as_model(name)], **kwargs)

  def extend(name, additional_steps, at=None, **additional_kwargs):
    ext_kwargs = fy.merge(kwargs, additional_kwargs)
    at = at or extend_at
    if at is not None and at != "end":
      before = steps[:at]
      after = steps[at:]
    else:
      before = steps
      after = []

    return create_model(
      name, before + additional_steps + after,
      extend_at=extend_at, **ext_kwargs)

  modelFactory.extend = extend
  modelFactory.name = name

  return modelFactory

@model_step
def with_layers(inputs, layer, layer_dims, layer_args=None, **kwargs):
  layer = pipeline.tolerant(layer)
  h = inputs

  for i in range(1, len(layer_dims)):
    if layer_args is None or layer_args[i - 1] is None:
      args = kwargs
    else:
      args = fy.merge(kwargs, layer_args[i - 1])

    h = layer(in_dim=layer_dims[i - 1], out_dim=layer_dims[i], **args)(h)

  return h

@pipeline.pipeline_step
def with_layer(io, layer, with_inputs=False, **kwargs):
  input, output = io

  p = io if with_inputs else output

  return input, pipeline.tolerant(layer)(**kwargs)(p)
