import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.gsheet_api import GsheetApi
from functions.helpers import split_by_rarity_groups, get_crystal_system_obj, prepare_data, get_symmetry_indexes

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for analysing the crystal systems of minerals in the light of their rarity
"""


GsheetApi = GsheetApi()
GsheetApi.run_main()

ns = GsheetApi.nickel_strunz.copy()
crystal = GsheetApi.crystal.copy()

mr_data = prepare_data(ns, crystal)

# create Crystal System helper object
crystal_system = get_crystal_system_obj()

r, re_rr_tr, re_true, re, rr, t, tr, tu, u, tu_u = split_by_rarity_groups(mr_data)

# Pie chart: Crystal classes for RE

pie_ = u.groupby('Crystal System').size()
pie_ = pd.DataFrame(pie_/pie_.sum() * 100)
pie_ = pie_.join(crystal_system).sort_values('order')

fig1, ax1 = plt.subplots()
ax1.pie(pie_[0], colors=pie_['color'], labels=pie_.index, autopct='%1.1f%%',startangle=90)

ax1.axis('equal')

plt.savefig(f"figures/crystal/u_pie_chart.jpeg", dpi=300, format='jpeg')

plt.close()


# symmetry index
(pie_.loc['isometric', 0] + pie_.loc['hexagonal', 0] + pie_.loc['trigonal', 0] + pie_.loc['tetragonal', 0]) \
/ (pie_.loc['triclinic', 0] + pie_.loc['monoclinic', 0] + pie_.loc['orthorhombic', 0])

# calculate symmetry index yearly

discovery_rate = mr_data.loc[mr_data['discovery_year'].notna()]
discovery_rate = get_symmetry_indexes(discovery_rate)
