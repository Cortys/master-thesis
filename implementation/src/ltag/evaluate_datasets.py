from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.evaluation.evaluate as evaluate
import ltag.evaluation.summary as summary
import ltag.evaluation.models as models
import ltag.evaluation.datasets as ds

epochs = 1000
repeat = 3  # fewer repeats for now.

mf = models.AvgWL2GCN_Binary_2
dsm = ds.Mutag_8

def quick_run(**kwargs):
  evaluate.quick_evaluate(mf, dsm(), **kwargs)

def run(**kwargs):
  evaluate.evaluate(
    mf, dsm(), epochs=epochs, repeat=repeat,
    **kwargs)

def resume(eval_dir_name, **kwargs):
  return evaluate.resume_evaluation(
    mf, dsm(), evaluate.eval_dir_base / eval_dir_name,
    **kwargs)


def summarize(eval_dir_name):
  return summary.summarize_evaluation(evaluate.eval_dir_base / eval_dir_name)

if __name__ == "__main__":
  run()
  # resume("2020-01-15_14-31-53_NCI1_AvgWL2GCN")
