from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import json
from collections import defaultdict
import funcy as fy
import numpy as np

from ltag.utils import NumpyEncoder

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

def statistics(vals):
  return {
    "mean": np.mean(vals),
    "std": np.std(vals),
    "min": np.min(vals),
    "max": np.max(vals)
  }

def summarize_evaluation(
  eval_dir, selection_metric="val_accuracy"):
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

  best_goal = selection_metrics[selection_metric]

  results = []

  for fold_i, param_files in fold_param_files.items():
    best_result = None

    for hp_i, files in param_files.items():
      hp_results = defaultdict(list)
      selection_vals = []
      for _, file in files:
        with open(file, "r") as f:
          result = json.load(f)

        selection_vals.append(result["train"][selection_metric][-1])

        for metric, val in result["test"].items():
          hp_results[metric].append(val)

      hp_result = (
        dict_map(np.mean, hp_results),
        hp_i, np.mean(selection_vals))

      if (
        best_result is None
        or (best_goal == "max" and best_result[2] < hp_result[2])
        or (best_goal == "min" and best_result[2] > hp_result[2])):
        best_result = hp_result

    if best_result is not None:
      results.append(best_result)
    else:
      print(f"No results for {fold_i}.")

  combined_result = dict_map(
    statistics,
    fy.merge_with(np.array, *map(fy.first, results)))

  results_summary = {
    "folds": results,
    "combined": combined_result
  }

  with open(summary_dir / "results.json", "w") as f:
    json.dump(
      results_summary, f,
      cls=NumpyEncoder, indent="\t")

  return results_summary
