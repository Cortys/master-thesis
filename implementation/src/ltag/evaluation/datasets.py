from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy

import ltag.datasets.disk.tu.datasets as tu

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
  "Mutag_8",
  "NCI1_8",
  "Proteins_6",
  "DD_2"
]
chemical = chemical_binary

# Social:

RedditBinary_1 = fy.partial(
  tu.RedditBinary,
  wl2_neighborhood=1,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 50000,
    "upper_batch_edge_count": 100000,
    "batch_graph_count": 5
  }
)

social_binary = [
  RedditBinary_1
]
social = social_binary

# Other categories:

stored = chemical + social

binary = chemical_binary + social_binary

all = stored
