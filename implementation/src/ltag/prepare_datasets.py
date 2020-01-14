from __future__ import absolute_import, division, print_function,\
  unicode_literals

import multiprocessing as mp
import argparse

import ltag.evaluation.datasets as datasets

def prepare_ds(i):
  manager = datasets.stored[i](no_wl2_load=True)
  print(f"Preparing {manager.name}...")
  manager.prepare_wl2_batches()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Prebatch WL2 encoded datasets.')
  parser.add_argument(
    "-p", "--parallel", action="store_true",
    help="Preprocess datasets in parallel.")
  args = parser.parse_args()
  p = mp.cpu_count() if args.parallel else 1

  print(f"Preparing all stored datasets with parallelism {p}...")

  with mp.Pool(p) as p:
    p.map(prepare_ds, range(len(datasets.stored)))

  print("Prepared all stored datasets.")
