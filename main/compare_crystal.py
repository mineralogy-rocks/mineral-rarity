import pandas as pd

import matplotlib
matplotlib.use('Agg')

from modules.gsheet_api import GsheetApi
from functions.helpers import parse_rruff, parse_mindat, split_by_rarity_groups

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

r, re_true, re, rr, t, tr, tu, u = split_by_rarity_groups(mr_data)


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

