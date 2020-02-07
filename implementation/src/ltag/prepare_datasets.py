from __future__ import absolute_import, division, print_function,\
  unicode_literals

import multiprocessing as mp
import argparse

import warnings
# Ignore future warnings caused by grakel:
warnings.simplefilter(action='ignore', category=FutureWarning)

import ltag.evaluation.datasets as datasets

def prepare_ds(d, all=False):
  manager = getattr(datasets, d)(no_wl2_load=True)
  print(f"Preparing {manager.name}...")
  manager.prepare_wl2_batches()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Prebatch WL2 encoded datasets.')
  parser.add_argument(
    "-p", "--parallel", action="store_true",
    help="Preprocess datasets in parallel.")
  parser.add_argument(
    "-d", "--dataset", action="append",
    help="Prepare this dataset.")
  args = parser.parse_args()
  p = mp.cpu_count() if args.parallel else 1

  if args.dataset is None or len(args.dataset) == 0:
    ds = datasets.stored
  else:
    ds = args.dataset

  dsl = len(ds)

  print(f"Will prepare {dsl} stored datasets with parallelism {p}:")
  for d in ds:
    print(f"- {d}")

  print("Starting...")

  with mp.Pool(p) as p:
    p.starmap(
      prepare_ds,
      zip(ds))

  print("Prepared all stored datasets.")
