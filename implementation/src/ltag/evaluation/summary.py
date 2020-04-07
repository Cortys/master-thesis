from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import json
from collections import defaultdict
import funcy as fy
import numpy as np

from ltag.utils import NumpyEncoder, statistics

selection_metrics = {
  "accuracy": "max",
  "loss": "min",
  "mse": "min",
  "val_accuracy": "max",
  "val_loss": "min",
  "val_mse": "min"
}

def dict_map(f, d):
  return {k: f(v) for k, v in d.items()}

def summarize_evaluation(
  eval_dir, selection_metric="val_accuracy", ignore_worst=0):
  assert eval_dir.exists(), f"No evalutation '{eval_dir}' found."

  with open(eval_dir / "config.json") as f:
    config = json.load(f)

  with open(eval_dir / "hyperparams.json") as f:
    hps = json.load(f)

  results_dir = eval_dir / "results"
  assert results_dir.exists(), f"No results found for '{eval_dir}'."
  summary_dir = eval_dir / "summary"

  if not summary_dir.exists():
    os.makedirs(summary_dir)

  result_files = [
    (list(fy.map(int, f[:-5].split("-"))), results_dir / f)
    for f in os.listdir(results_dir)]

  fold_files = fy.group_by(lambda f: f[0][0], result_files)
  fold_param_files = {
    fold: fy.group_by(lambda f: f[0][1], files)
    for fold, files in fold_files.items()}
  folds = list(fold_param_files.items())
  folds.sort(key=fy.first)

  best_goal = selection_metrics[selection_metric]

  results = []

  for fold_i, param_files in folds:
    best_res = None

    for hp_i, files in param_files.items():
      hp_train_results = defaultdict(list)
      hp_test_results = defaultdict(list)
      selection_vals = []
      all_selection_vals = []
      for (_, _, i), file in files:
        with open(file, "r") as f:
          result = json.load(f)

        selection_val = result["train"][selection_metric][-1]
        all_selection_vals.append(selection_val)
        if i < config["repeat"]:
          selection_vals.append(selection_val)

        for metric, val in result["train"].items():
          hp_train_results[metric].append(val[-1])
        for metric, val in result["test"].items():
          hp_test_results[metric].append(val)

      top_idxs = np.argsort(np.array(all_selection_vals))

      if len(all_selection_vals) > ignore_worst:
        if best_goal == "max":
          top_idxs = top_idxs[ignore_worst:]
        elif best_goal == "min":
          top_idxs = top_idxs[:-ignore_worst]

      top_statistics = fy.compose(
        statistics,
        lambda l: np.array(l)[top_idxs])

      hp_res = dict(
        fold_idx=fold_i,
        train=dict_map(top_statistics, hp_train_results),
        test=dict_map(top_statistics, hp_test_results),
        select=np.mean(selection_vals),
        hp_i=hp_i,
        hp=hps[hp_i],
        select_repeats=len(selection_vals),
        eval_repeats=len(files))

      if (
        best_res is None
        or (best_goal == "max" and best_res["select"] < hp_res["select"])
        or (best_goal == "min" and best_res["select"] > hp_res["select"])
        or (
          best_res["select"] == hp_res["select"]
          and best_res["eval_repeats"] < hp_res["eval_repeats"])):
        best_res = hp_res

    if best_res is not None:
      results.append(best_res)
    else:
      print(f"No results for {fold_i}.")

  combined_train = dict_map(
    statistics,
    fy.merge_with(np.array, *map(
      lambda res: dict_map(lambda t: t["mean"], res["train"]), results)))
  combined_test = dict_map(
    statistics,
    fy.merge_with(np.array, *map(
      lambda res: dict_map(lambda t: t["mean"], res["test"]), results)))

  results_summary = {
    "folds": results,
    "combined_train": combined_train,
    "combined_test": combined_test,
    "args": {
      "ignore_worst": ignore_worst
    }
  }

  with open(summary_dir / "results.json", "w") as f:
    json.dump(
      results_summary, f,
      cls=NumpyEncoder, indent="\t")

  return results_summary
