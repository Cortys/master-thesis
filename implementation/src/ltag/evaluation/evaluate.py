from __future__ import absolute_import, division, print_function,\
  unicode_literals

import warnings
from pathlib import Path
from datetime import datetime
from tensorflow import keras

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
  epochs=500, patience=50):
  label = "_" + label if label is not None else ""

  t = time_str()
  tb = keras.callbacks.TensorBoard(
    log_dir=log_dir_base / f"{t}{label}/",
    histogram_freq=1,
    write_images=False)
  es = keras.callbacks.EarlyStopping(
    monitor="loss" if val_ds is None else "val_loss",
    patience=patience,
    min_delta=0.0001,
    restore_best_weights=True)

  return model.fit(
    train_ds, validation_data=val_ds,
    epochs=epochs, callbacks=[tb, es])

def evaluate(
  model_factory, ds_manager,
  outer_k=None, repeat=10, epochs=500):
  outer_k = outer_k or ds_manager.outer_k

  ds_type = model_factory.input_type
  ds_name = ds_manager.name
  t = time_str()
  eval_dir = eval_dir_base / f"{t}_{ds_name}_{model_factory.name}"
  log_dir_base = eval_dir / "logs"

  for k in range(outer_k):
    test_ds = ds_manager.get_test_fold(k, output_type=ds_type)
    train_ds, val_ds = ds_manager.get_train_fold(k, output_type=ds_type)

    for hp_i, model_ctr in enumerate(model_factory(train_ds)):
      hp = model_ctr.hyperparams
      print("\nSplit", k, "with hyperparams:", hp)

      for i in range(repeat):
        print("\nIteration", i, "of split", k, "...")

        model = model_ctr()
        label = f"{k}-{hp_i}-{i}"
        h = train(
          model, train_ds, val_ds, label,
          epochs=epochs, log_dir_base=log_dir_base)
        history = h.history
        results = model.evaluate(test_ds)
        print("\nEvaluation", k, i, "|", model.metrics_names, "=", results)
        print("hist", history)
