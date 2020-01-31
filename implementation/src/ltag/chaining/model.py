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

@pipeline.pipeline_step
def with_layers(
  io, layer, layer_dims=[], layer_args=None,
  stack_tf=None, stack_tf_lookup=None,
  **kwargs):
  input, output = io
  layer = pipeline.tolerant(layer)
  hs = [output]

  if stack_tf is not None:
    if stack_tf_lookup is not None:
      stack_tf = stack_tf_lookup[stack_tf]

    if not callable(stack_tf):
      raise TypeError(
        "Stack transformers need to be callable or resolve to a callable.")

    stack_tf = pipeline.tolerant(stack_tf)

  for i in range(1, len(layer_dims)):
    if layer_args is None or layer_args[i - 1] is None:
      args = kwargs
    else:
      args = fy.merge(kwargs, layer_args[i - 1])

    in_dim = layer_dims[i - 1]
    out_dim = layer_dims[i]
    h = hs[i - 1]

    if stack_tf is not None:
      h, in_dim, out_dim = stack_tf(
        h, in_dim, out_dim,
        input=input, hs=hs, layer_dims=layer_dims, i=i)

    hs.append(
      layer(in_dim=in_dim, out_dim=out_dim, **args)(h))

  return input, hs[-1]

@pipeline.pipeline_step
def with_layer(io, layer, with_inputs=False, **kwargs):
  input, output = io

  p = io if with_inputs else output

  return input, pipeline.tolerant(layer)(**kwargs)(p)

@pipeline.pipeline_step
def merge_ios(ios):
  input = ios[0][0]

  return input, (io[1] for io in ios)

# Layer Stack Transformers:

def stack_tf_seq(*transformers):
  transformers = [pipeline.tolerant(t) for t in transformers]

  def tf_seq(h, in_dim, out_dim, **kwargs):
    for t in transformers:
      h, in_dim, out_dim = t(h, in_dim, out_dim, *kwargs)

    return h, in_dim, out_dim

  return tf_seq

def add_input_tf(h, in_dim, out_dim, input):
  return (input, h), in_dim, out_dim
