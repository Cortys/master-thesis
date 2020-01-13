from __future__ import absolute_import, division, print_function,\
  unicode_literals

import multiprocessing as mp

import ltag.evaluation.datasets as datasets

def prepare_ds(i):
  manager = datasets.all_stored[i](no_wl2_load=True)
  print(f"Preparing {manager.name}...")
  manager.prepare_wl2_batches()


if __name__ == "__main__":
  print("Preparing all stored datasets...")

  with mp.Pool(mp.cpu_count()) as p:
    p.map(prepare_ds, range(len(datasets.all_stored)))

  print("Prepared all stored datasets.")
