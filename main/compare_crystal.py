import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.gsheet_api import GsheetApi
from functions.helpers import parse_rruff, parse_mindat, split_by_rarity_groups, get_crystal_system_obj

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for analysing the crystal systems of minerals in the light of their rarity
"""


GsheetApi = GsheetApi()
GsheetApi.run_main()

ns = GsheetApi.nickel_strunz.copy()
crystal = GsheetApi.crystal.copy()

locs_md = pd.read_csv('data/mindat_locs.csv', sep=',')
rruff_data = pd.read_csv('data/RRUFF_Export.csv', sep=',')

rruff_data = parse_rruff(rruff_data)
locs_md = parse_mindat(locs_md)

# Clean and transform MR data
ns.set_index('Mineral_Name', inplace=True)
crystal.set_index('Mineral_Name', inplace=True)

mindat_rruff = locs_md.join(rruff_data, how='outer')
mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

# create final subset for the analysis
mr_data = ns.join(mindat_rruff, how='inner')
mr_data = mr_data.join(crystal[['Crystal System']], how='inner')
mr_data.loc[mr_data['Crystal System'].isin(['icosahedral', 'amorphous']), 'Crystal System'] = np.nan

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

mr_data.groupby('discovery_year').agg(
    isometric=pd.NamedAgg(column="isometric", aggfunc="sum"),
    triclinic=pd.NamedAgg(column="triclinic", aggfunc="sum"),
)

# calculate "triclinic" index yearly

discovery_rate = mr_data.loc[mr_data['discovery_year'].notna()]

discovery_rate.loc[:,'triclinic'] = 0
discovery_rate.loc[:,'isometric'] = 0
discovery_rate.loc[discovery_rate['Crystal System'] == 'triclinic', 'triclinic'] = 1
discovery_rate.loc[discovery_rate['Crystal System'] == 'isometric', 'isometric'] = 1

discovery_rate = discovery_rate.sort_values('discovery_year', ascending=True)

discovery_rate = discovery_rate.groupby('discovery_year').agg(
    isometric=pd.NamedAgg(column="isometric", aggfunc="sum"),
    triclinic=pd.NamedAgg(column="triclinic", aggfunc="sum"),
)

discovery_rate['triclinic_index'] = discovery_rate['triclinic'] / discovery_rate['isometric']

