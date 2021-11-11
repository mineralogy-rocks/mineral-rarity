import pandas as pd
import numpy as np

import re

from modules.gsheet_api import GsheetApi
# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for assessing the evolution of mineral endemicity with time and
make some assumptions
"""

GsheetApi = GsheetApi()
GsheetApi.run_main()
locs_md = pd.read_csv('data/locs_year.csv', sep=',')

locs_mr = GsheetApi.locs.copy()
discovery_year = GsheetApi.names.copy()

# Clean and transform md data

locs_md.replace('\\N', np.nan, inplace=True)

## Don't have any clue why mindat keeps `0` in imayear column and mixes dtypes in others (?!)

locs_md['imayear'].replace('0', np.nan, inplace=True)
locs_md = locs_md[['name', 'imayear', 'yeardiscovery', 'yearrruff', 'loccount']]

locs_md['imayear'] = pd.to_numeric(locs_md['imayear'])
locs_md['yearrruff'] = pd.to_numeric(locs_md['yearrruff'])
locs_md['loccount'] = pd.to_numeric(locs_md['loccount'])

locs_md.loc[~locs_md['yeardiscovery'].str.match(r'[0-9]{4}', na=False), 'yeardiscovery'] = np.nan

locs_md['yeardiscovery'] = pd.to_numeric(locs_md['yeardiscovery'])

locs_md_2020 = locs_md.loc[locs_md['yearrruff'] == 2020]
locs_md_2021 = locs_md.loc[locs_md['yearrruff'] == 2021]

# Clean and transform MR/RRUFF data

locs_mr.set_index('mineral_name', inplace=True)
discovery_year.set_index('Mineral_Name', inplace=True)

locs_year_mr = locs_mr.join(discovery_year, how='inner')[['locality_counts', 'discovery_year_min']]

locs_year_mr['locality_counts'] = pd.to_numeric(locs_year_mr['locality_counts'])

locs_year = locs_year_mr.loc[(locs_year_mr['locality_counts'] == 1) & (locs_year_mr['discovery_year_min'].notna())]

locs_year.sort_values('discovery_year_min', inplace=True)