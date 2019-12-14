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

def create_model(name, steps, **kwargs):
  modelFactory = pipeline.create_pipeline([*steps, as_model(name)], **kwargs)

  def extend(name, additional_steps, **additional_kwargs):
    ext_kwargs = fy.merge(kwargs, additional_kwargs)

    return create_model(name, steps + additional_steps, **ext_kwargs)

  modelFactory.extend = extend

  return modelFactory
