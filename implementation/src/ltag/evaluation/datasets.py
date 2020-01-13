from __future__ import absolute_import, division, print_function,\
  unicode_literals

import funcy as fy

import ltag.datasets.disk.tu.datasets as tu

# Chemical:

Mutag = fy.partial(
  tu.Mutag,
  wl2_neighborhood=8)

NCI1 = fy.partial(
  tu.NCI1,
  wl2_neighborhood=8)

Proteins = fy.partial(
  tu.Proteins,
  wl2_neighborhood=6,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 10000,
    "upper_batch_edge_count": 30000,
    "batch_graph_count": 20
  })

DD = fy.partial(
  tu.DD,
  wl2_neighborhood=2,
  wl2_batch_size={
    "fuzzy_batch_edge_count": 5000,
    "upper_batch_edge_count": 15000,
    "batch_graph_count": 4
  })

chemical = [
  Mutag,
  NCI1,
  Proteins,
  DD
]

# All:

all_stored = chemical

all = all_stored
