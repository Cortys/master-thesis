from __future__ import absolute_import, division, print_function,\
  unicode_literals

import multiprocessing as mp
import argparse
import funcy as fy

import warnings
# Ignore future warnings caused by grakel:
warnings.simplefilter(action='ignore', category=FutureWarning)

import ltag.evaluation.datasets as datasets
import ltag.evaluation.kernel_models as kernel

def prepare_ds(d, wl2=True, gram=True):
  manager = getattr(datasets, d)(no_wl2_load=True)
  if wl2:
    print(f"Preparing {manager.name} WL2 encoding...")
    manager.prepare_wl2_batches()
  if gram:
    print(f"Preparing {manager.name} gram matrices...")
    manager.prepare_gram_matrices([
      kernel.WL_st.input_type,
      kernel.WL_sp.input_type])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Prepare dataset WL2 encodedings and gram matrices.')
  parser.add_argument(
    "-p", "--parallel", action="store_true",
    help="Preprocess datasets in parallel.")
  parser.add_argument(
    "--no-gram", action="store_true",
    help="Do not compute gram matrices.")
  parser.add_argument(
    "--no-wl2", action="store_true",
    help="Do not compute WL2 encodings.")
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

  print(args)

  print(f"Will prepare {dsl} stored datasets with parallelism {p}:")
  for d in ds:
    print(f"- {d}")

  print("Starting...")

  with mp.Pool(p) as p:
    p.starmap(
      prepare_ds,
      zip(ds, fy.repeat(not args.no_wl2), fy.repeat(not args.no_gram)))

  print("Prepared all stored datasets.")
