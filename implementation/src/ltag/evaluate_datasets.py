from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.evaluation.evaluate as evaluate
import ltag.evaluation.models as models
import ltag.evaluation.datasets as ds

epochs = 500
repeat = 5  # fewer repeats for now.

mf = models.AvgWL2GCN_Binary
dsm = ds.Proteins

def run(**kwargs):
  evaluate.evaluate(
    mf, dsm(), epochs=epochs, repeat=repeat,
    **kwargs)

def resume(eval_dir_name, **kwargs):
  return evaluate.resume_evaluation(
    mf, dsm(), evaluate.eval_dir_base / eval_dir_name,
    **kwargs)


if __name__ == "__main__":
  run()
