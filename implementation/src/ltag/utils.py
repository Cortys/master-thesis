from __future__ import absolute_import, division, print_function,\
  unicode_literals

import json
import itertools
import contextlib
import numpy as np
import funcy as fy
from collections import Sized

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if (
      isinstance(obj, np.float32)
      or isinstance(obj, np.float64)
      or isinstance(obj, np.int32)
      or isinstance(obj, np.int64)):
      return np.asscalar(obj)
    return json.JSONEncoder.default(self, obj)

def statistics(vals, mask_invalid=False):
  if mask_invalid:
    vals_masked = np.ma.masked_invalid(vals)
    return {
      "mean": np.mean(vals_masked),
      "std": np.std(vals_masked),
      "min": np.min(vals),
      "max": np.max(vals),
      "max_masked": np.max(vals_masked),
      "min_masked": np.min(vals_masked),
      "count": len(vals),
      "count_masked": vals_masked.count()
    }
  else:
    return {
      "mean": np.mean(vals),
      "std": np.std(vals),
      "min": np.min(vals),
      "max": np.max(vals),
      "count": len(vals) if isinstance(vals, Sized) else 1
    }

@contextlib.contextmanager
def local_seed(seed):
  state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  finally:
    np.random.set_state(state)

def cart(*pos_params, **params):
  "Lazily computes the cartesian product of the given lists or dicts."
  if len(pos_params) > 0:
    return itertools.product(*pos_params)

  return (dict(zip(params, x)) for x in itertools.product(*params.values()))

def cart_merge(*dicts):
  "Lazily computes all possible merge combinations of the given dicts."
  return (fy.merge(*c) for c in itertools.product(*dicts))

def entry_duplicator(duplicates):
  def f(d):
    for source, targets in duplicates.items():
      d_source = d[source]
      for target in targets:
        d[target] = d_source

    return d

  return f
