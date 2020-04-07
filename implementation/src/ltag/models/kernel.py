from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import sklearn as skl
import grakel as gk

class History:
  def __init__(self, history):
    self.history = history

class KernelModel:
  def __init__(self, classification=True, **hp):
    if classification:
      self.svm = skl.svm.SVC(kernel="precomputed", **hp)
      self.metric_name = "accuracy"
    else:
      self.svm = skl.svm.SVR(kernel="precomputed", **hp)
      self.metric_name = "r2"

  @property
  def metrics_names(self):
    return [self.metric_name]

  def fit(
    self, training_data, validation_data=None,
    **kwargs):
    train = self.svm.fit(*training_data).score(*training_data)

    history = {
      self.metric_name: [train]
    }

    if validation_data is not None:
      val = self.svm.score(*validation_data)
      history["val_" + self.metric_name] = [val]

    return History(history)

  def evaluate(self, test_data):
    return [self.svm.score(*test_data)]

def kernel_model(name, kernel):
  n = name

  class Model(KernelModel):
    name = n
    input_type = kernel

  return Model

def grakel_kernel(name, *args, **kwargs):
  def kernel_fn(dsm):
    kernel = gk.GraphKernel(*args, **kwargs)
    graphs = dsm.get_all(output_type="grakel")

    return kernel.fit_transform(*graphs)

  kernel_fn.__name__ = name

  return kernel_fn

def fs_kernel(name):
  def kernel_fn(dsm):
    ds_dir = dsm.root_dir
    gram_file = ds_dir / "gram" / name / "gram.csv"

    if not gram_file.exists():
      raise Exception("No gram matrix found.")

    return np.loadtxt(
      gram_file, dtype=np.float32,
      delimiter=", ", encoding="utf8", ndmin=2)

  kernel_fn.__name__ = name

  return kernel_fn


# Models:
WL_st = kernel_model("WL_st", grakel_kernel(
  "WL_st", kernel=[{"name": "WL"}, "VH"]))
WL_st_1 = kernel_model("WL_st_1", grakel_kernel(
  "WL_st_1", kernel=[{"name": "WL", "n_iter": 1}, "VH"]))
WL_st_2 = kernel_model("WL_st_2", grakel_kernel(
  "WL_st_2", kernel=[{"name": "WL", "n_iter": 2}, "VH"]))
WL_st_3 = kernel_model("WL_st_3", grakel_kernel(
  "WL_st_3", kernel=[{"name": "WL", "n_iter": 3}, "VH"]))
WL_st_4 = kernel_model("WL_st_4", grakel_kernel(
  "WL_st_4", kernel=[{"name": "WL", "n_iter": 4}, "VH"]))

WL_sp = kernel_model("WL_sp", grakel_kernel(
  "WL_sp", kernel=[{"name": "WL"}, "SP"]))
WL_sp_3 = kernel_model("WL_sp_3", grakel_kernel(
  "WL_sp_3", kernel=[{"name": "WL", "n_iter": 3}, "SP"]))

LWL2 = kernel_model("LWL2", fs_kernel("LWL2"))
GWL2 = kernel_model("GWL2", fs_kernel("GWL2"))
