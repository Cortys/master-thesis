from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy

import ltag.datasets.disk.datasets as tu
import ltag.datasets.synthetic.datasets as syn

# Chemical:

Mutag_8_mini = fy.partial(
  tu.Mutag,
  wl2_neighborhood=8,
  wl2_batch_size={
    "batch_graph_count": 1
  })

Mutag_8 = fy.partial(
  tu.Mutag,
  wl2_neighborhood=8)

NCI1_8 = fy.partial(
  tu.NCI1,
  wl2_neighborhood=8)

Proteins_5 = fy.partial(
  tu.Proteins,
  wl2_neighborhood=5,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 20000,
    "upper_batch_edge_count": 50000,
    "batch_graph_count": 20
  })

Proteins_6 = fy.partial(
  tu.Proteins,
  wl2_neighborhood=6,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 20000,
    "upper_batch_edge_count": 50000,
    "batch_graph_count": 20
  })

DD_2 = fy.partial(
  tu.DD,
  wl2_neighborhood=2,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 30000,
    "upper_batch_edge_count": 60000,
    "batch_graph_count": 10
  })

chemical_binary = [
  "NCI1_8",
  "Proteins_5",
  "DD_2"
]
chemical = chemical_binary

# Social:

RedditBinary_1 = fy.partial(
  tu.RedditBinary,
  wl2_neighborhood=1,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 100000,
    "upper_batch_edge_count": 200000,
    "batch_graph_count": 10
  }
)

social_binary = [
  "RedditBinary_1"
]
social = social_binary

# Synthetic:

noisy_triangle_classification_2 = fy.partial(
  syn.noisy_triangle_classification_dataset(stored=True),
  wl2_neighborhood=2,
  wl2_batch_size={
    "batch_graph_count": 208
  },
  evaluation_args={
    "epochs": 3000,
    "patience": 600,
    "hp_args": {
      # other widths ignored based on previous experiments:
      "cwl2_layer_widths": [32],
      "cwl2_layer_depths": [3],
      "cwl2_local_act": "relu"
    }
  })

balanced_triangle_classification_2 = fy.partial(
  syn.balanced_triangle_classification_dataset(stored=True),
  wl2_neighborhood=2,
  wl2_batch_size={
    "batch_graph_count": 228
  },
  evaluation_args={
    "epochs": 5000,
    "patience": 1000,
    "winner_repeat": 5,
    "hp_args": {
      # other widths ignored based on previous experiments:
      "cwl2_layer_widths": [32],
      "cwl2_layer_depths": [3],
      "cwl2_local_act": "relu"
    },
    "ignore_worst": 2
  })

synthetic_binary = [
  "noisy_triangle_classification_2",
  "balanced_triangle_classification_2"
]
synthetic = synthetic_binary
stored_synthetic = synthetic
dynamic_synthetic = []

# Other categories:

stored = stored_synthetic + chemical + social
binary = synthetic_binary + chemical_binary + social_binary
all = dynamic_synthetic + stored
