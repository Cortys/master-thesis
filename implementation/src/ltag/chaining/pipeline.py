from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy
import inspect
import collections as coll

def tolerant(f):
  if hasattr(f, "__tolerant__"):
    return f

  spec = inspect.getfullargspec(f.__init__ if inspect.isclass(f) else f)
  f_varkws = spec.varkw is not None

  if f_varkws:
    return f

  f_args = spec.args

  @fy.wraps(f)
  def wrapper(*args, **kwargs):
    applicable_kwargs = fy.project(kwargs, f_args)

    return f(*args, **applicable_kwargs)

  wrapper.__tolerant__ = True

  return wrapper

def pipeline_step(f):
  if hasattr(f, "__pipeline_step__"):
    return f

  f = tolerant(f)

  @fy.wraps(f)
  def step(*args, prefix=None, **kwargs1):
    @fy.wraps(f)
    def execute(input, **kwargs2):
      kwargs = fy.merge(kwargs2, kwargs1)

      if prefix is not None:
        p = prefix + "_"
        for k, v in kwargs2.items():
          if k.startswith(p):
            kwargs[k[len(p):]] = v

      return f(input, *args, **kwargs)

    execute.__tolerant__ = True

    return execute

  step.__pipeline_step__ = True

  return step

def pipeline_start(f):
  f = tolerant(f)

  @fy.wraps(f)
  def wrapper(_, **kwargs):
    return f(**kwargs)

  wrapper.__tolerant__ = True

  return pipeline_step(wrapper)


def to_executable_step(f):
  if hasattr(f, "__pipeline_step__"):
    return f()

  if isinstance(f, coll.Iterable):
    pipelines = [(
      create_pipeline(s) if isinstance(s, coll.Iterable)
      else to_executable_step(s))
      for s in f]

    def split_step(input, **kwargs):
      return [p(input, **kwargs) for p in pipelines]

    return split_step

  return tolerant(f)

def create_pipeline(steps, **kwargs1):
  executable_steps = [to_executable_step(step) for step in steps]

  def pipeline(input=None, **kwargs2):
    a = input
    kwargs = fy.merge(kwargs1, kwargs2)

    for executable_step in executable_steps:
      a = executable_step(a, **kwargs)

    return a

  return pipeline
