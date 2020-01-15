from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import json
from timeit import default_timer as timer
import warnings
from pathlib import Path
from datetime import datetime
from tensorflow import keras
import funcy as fy

from ltag.utils import NumpyEncoder
import ltag.evaluation.summary as summary

# Sparse gradient updates don't work for some reason. Disable the warning:
warnings.filterwarnings(
  "ignore",
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape.*",
  UserWarning)

eval_dir_base = Path("../evaluations")
log_dir_base = Path("../logs")

def time_str():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def train(
  model, train_ds, val_ds=None, label=None,
  log_dir_base=log_dir_base,
  epochs=500, patience=50, restore_best=False,
  stopping_min_delta=0.0001,
  verbose=2):
  label = "_" + label if label is not None else ""

  t = time_str()
  tb = keras.callbacks.TensorBoard(
    log_dir=log_dir_base / f"{t}{label}/",
    histogram_freq=1,
    write_images=False)
  es = keras.callbacks.EarlyStopping(
    monitor="loss" if val_ds is None else "val_loss",
    patience=patience,
    min_delta=stopping_min_delta,
    restore_best_weights=restore_best)

  return model.fit(
    train_ds, validation_data=val_ds,
    epochs=epochs, callbacks=[tb, es],
    verbose=verbose)

def evaluate(
  model_factory, ds_manager,
  outer_k=None, repeat=10, epochs=500,
  patience=50, stopping_min_delta=0.0001,
  restore_best=False, label=None,
  eval_dir=None, verbose=2):
  label = "_" + label if label is not None else ""
  outer_k = outer_k or ds_manager.outer_k
  inner_k = None

  mf_name = model_factory.name
  ds_type = model_factory.input_type
  ds_name = ds_manager.name
  t = time_str()
  if eval_dir is None:
    eval_dir = eval_dir_base / f"{t}_{ds_name}_{mf_name}{label}"
    if not eval_dir.exists():
      os.makedirs(eval_dir)

    with open(eval_dir / "config.json", "w") as f:
      config = {
        "outer_k": outer_k,
        "inner_k": inner_k,
        "repeat": repeat,
        "epochs": epochs,
        "patience": patience,
        "stopping_min_delta": stopping_min_delta,
        "restore_best": restore_best,
        "ds_type": ds_type,
        "ds_name": ds_name,
        "mf_name": mf_name,
        "start_time": t,
        "end_time": None,
        "duration": 0
      }
      json.dump(config, f, indent="\t", sort_keys=True, cls=NumpyEncoder)
    resume = False
  else:
    assert eval_dir.exists(), "Invalid resume directory."
    with open(eval_dir / "config.json", "r") as f:
      config = json.load(f)
      assert (
        config["outer_k"] == outer_k
        and config["inner_k"] == inner_k
        and config["repeat"] == repeat
        and config["epochs"] == epochs
        and config["patience"] == patience
        and config["stopping_min_delta"] == stopping_min_delta
        and config["restore_best"] == restore_best
        and config["ds_type"] == ds_type
        and config["ds_name"] == ds_name
        and config["mf_name"] == mf_name), "Incompatible config."
    resume = True

  log_dir_base = eval_dir / "logs"
  res_dir = eval_dir / "results"
  pos_file = eval_dir / "state.txt"

  if not res_dir.exists():
    os.makedirs(res_dir)

  if not log_dir_base.exists():
    os.makedirs(log_dir_base)

  k_start = 0
  hp_start = 0
  i_start = -1
  print(t, f"- Evaluating {ds_name} using {mf_name}...")

  if resume and pos_file.exists():
    pos = pos_file.read_text().split(",")
    if len(pos) == 3:
      k_start, hp_start, i_start = fy.map(int, pos)
      print(f"Continuing at {k_start}, {hp_start}, {i_start}.")

  model_ctr, hps = model_factory(ds_manager)
  hpc = len(hps)

  with open(eval_dir / f"hyperparams.json", "w") as f:
    json.dump(hps, f, indent="\t", sort_keys=True, cls=NumpyEncoder)

  t_start_eval = timer()
  try:
    for k in range(k_start, outer_k):
      print(time_str(), f"- Evaluating fold {k+1}/{outer_k}...")
      t_start_fold = timer()
      test_ds = ds_manager.get_test_fold(k, output_type=ds_type)
      train_ds, val_ds = ds_manager.get_train_fold(k, output_type=ds_type)
      fold_str = f"{k+1}/{outer_k}"

      for hp_i, hp in enumerate(hps):
        hp_str = f"{hp_i+1}/{hpc}"
        curr_i_start = 0
        if k == k_start:
          if hp_i < hp_start:
            print(f"Already evaluated {fold_str} with hyperparams {hp_str}.")
            continue
          elif hp_i == hp_start and i_start >= 0:
            print(
              f"Already evaluated {fold_str} with hyperparams {hp_str}",
              f"{i_start + 1}/{repeat} times.")
            curr_i_start = i_start + 1

        print(f"\nFold {fold_str} with hyperparams {hp_str}.")

        for i in range(curr_i_start, repeat):
          rep_str = f"{i+1}/{repeat}"
          print(
            time_str(),
            f"- Iteration {rep_str}, fold {fold_str}, hps {hp_str}...")
          t_start = timer()
          label = f"{k}-{hp_i}-{i}"
          model = model_ctr(**hp)
          h = train(
            model, train_ds, val_ds, label,
            epochs=epochs, patience=patience,
            stopping_min_delta=stopping_min_delta,
            restore_best=restore_best,
            log_dir_base=log_dir_base,
            verbose=verbose)
          train_res = h.history
          t_end = timer()
          train_dur = t_end - t_start
          t_start = t_end
          test_res = model.evaluate(test_ds)
          t_end = timer()
          test_dur = t_end - t_start
          test_res = dict(zip(model.metrics_names, test_res))

          with open(res_dir / f"{label}.json", "w") as f:
            json.dump({
              "train": train_res,
              "test": test_res,
              "train_duration": train_dur,
              "test_duration": test_dur
            }, f, cls=NumpyEncoder)

          pos_file.write_text(f"{k},{hp_i},{i}")
          print(
            f"\nTest results in {train_dur}s/{test_dur}s for",
            f"it {rep_str}, fold {fold_str}, hps {hp_str}:",
            test_res)

      t_end_fold = timer()
      dur_fold = t_end_fold - t_start_fold
      print(time_str(), f"- Evaluated fold {fold_str} in {dur_fold}s.")
  finally:
    t_end_eval = timer()
    dur_eval = t_end_eval - t_start_eval

    with open(eval_dir / "config.json", "w") as f:
      config["duration"] += dur_eval
      config["end_time"] = time_str()
      json.dump(config, f, indent="\t", sort_keys=True, cls=NumpyEncoder)

  summary.summarize_evaluation(eval_dir)
  print(
    time_str(),
    f"- Evaluation of {ds_name} using {mf_name} completed in {dur_eval}s.")

def resume_evaluation(model_factory, ds_manager, eval_dir, **kwargs):
  with open(eval_dir / "config.json", "r") as f:
    config = json.load(f)

  return evaluate(
    model_factory, ds_manager,
    outer_k=config["outer_k"],
    repeat=config["repeat"],
    epochs=config["epochs"],
    patience=config["patience"],
    stopping_min_delta=config["stopping_min_delta"],
    restore_best=config["restore_best"],
    eval_dir=eval_dir,
    **kwargs)

def quick_evaluate(model_factory, ds_manager, **kwargs):
  return evaluate(
    model_factory, ds_manager,
    epochs=1, repeat=1, label="quick",
    **kwargs)
