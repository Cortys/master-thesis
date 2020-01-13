from __future__ import absolute_import, division, print_function,\
  unicode_literals

import ltag.evaluation.evaluate as evaluate
import ltag.evaluation.models as models
import ltag.evaluation.datasets as ds

def run():
  evaluate.evaluate(
    models.AvgWL2GCN_Binary, ds.Proteins())


if __name__ == "__main__":
  run()
