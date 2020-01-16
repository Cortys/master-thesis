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
    "fuzzy_batch_edge_count": 10000,
    "upper_batch_edge_count": 30000,
    "batch_graph_count": 20
  })

DD_2 = fy.partial(
  tu.DD,
  wl2_neighborhood=2,
  wl2_batch_size={
    "batch_graph_count": 1
  })

chemical = [
  "Mutag_8",
  "NCI1_8",
  "Proteins_6",
  "DD_2"
]

# Other categories:

stored = chemical

binary = [
  "Mutag_8",
  "NCI1_8",
  "Proteins_6",
  "DD_2"
]

all = stored
