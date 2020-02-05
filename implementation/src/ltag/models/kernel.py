from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy
import sklearn as skl
import grakel as gk

import ltag.chaining.model as cm

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

    return dict(history=history)

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
    pass

  kernel_fn.__name__ = name

  return kernel_fn


# Models:
WL_st = kernel_model("WL_st", grakel_kernel(
  "WL_st", kernel=[{"name": "WL"}, "VH"]))
WL_sp = kernel_model("WL_sp", grakel_kernel(
  "WL_sp", kernel=[{"name": "WL"}, "SP"]))
