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

Mutag_1 = fy.partial(
  tu.Mutag,
  name_suffix="_n1",
  wl2_neighborhood=1)
Mutag_8 = fy.partial(
  tu.Mutag,
  wl2_neighborhood=8)

NCI1_1 = fy.partial(
  tu.NCI1,
  name_suffix="_n1",
  wl2_neighborhood=1)
NCI1_8 = fy.partial(
  tu.NCI1,
  wl2_neighborhood=8)

Proteins_1 = fy.partial(
  tu.Proteins,
  name_suffix="_n1",
  wl2_neighborhood=1,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 20000,
    "upper_batch_edge_count": 50000,
    "batch_graph_count": 20
  })
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
  name_suffix="_n6",
  wl2_batch_size={
    "fuzzy_batch_edge_count": 20000,
    "upper_batch_edge_count": 50000,
    "batch_graph_count": 20
  })

DD_1 = fy.partial(
  tu.DD,
  name_suffix="_n1",
  wl2_neighborhood=1,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 30000,
    "upper_batch_edge_count": 60000,
    "batch_graph_count": 10
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

IMDB_1 = fy.partial(
  tu.IMDBBinary,
  name_suffix="_n1",
  wl2_neighborhood=1)
IMDB_2 = fy.partial(
  tu.IMDBBinary,
  name_suffix="_n2",
  wl2_neighborhood=2)
IMDB_4 = fy.partial(
  tu.IMDBBinary,
  name_suffix="_n4",
  wl2_neighborhood=4)
IMDB_6 = fy.partial(
  tu.IMDBBinary,
  name_suffix="_n6",
  wl2_neighborhood=6)
IMDB_8 = fy.partial(
  tu.IMDBBinary,
  wl2_neighborhood=8)

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
  "IMDB_8",
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

balanced_eval_args = {
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
}

balanced_triangle_classification_1 = fy.partial(
  syn.balanced_triangle_classification_dataset(stored=True),
  name_suffix="_n1",
  wl2_neighborhood=1,
  wl2_batch_size={
    "batch_graph_count": 228
  },
  evaluation_args=balanced_eval_args)

balanced_triangle_classification_2 = fy.partial(
  syn.balanced_triangle_classification_dataset(stored=True),
  wl2_neighborhood=2,
  wl2_batch_size={
    "batch_graph_count": 228
  },
  evaluation_args=balanced_eval_args)

synthetic_binary = [
  "balanced_triangle_classification_2"
]
synthetic = synthetic_binary
stored_synthetic = synthetic
dynamic_synthetic = []

# Other categories:

stored = stored_synthetic + chemical + social
binary = synthetic_binary + chemical_binary + social_binary
all = dynamic_synthetic + stored
