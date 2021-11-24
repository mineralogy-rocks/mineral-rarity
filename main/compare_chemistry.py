import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.gsheet_api import GsheetApi
from functions.helpers import get_discovery_rate_all, get_discovery_rate_endemic, get_endemic_proportion, parse_rruff, \
    parse_mindat

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for analysing the chemistry of endemic minerals
"""

GsheetApi = GsheetApi()
GsheetApi.run_main()

status = GsheetApi.status_data.copy()
ns = GsheetApi.nickel_strunz.copy()

locs_md = pd.read_csv('data/mindat_locs.csv', sep=',')
rruff_data = pd.read_csv('data/RRUFF_Export.csv', sep=',')

rruff_data = parse_rruff(rruff_data)
locs_md = parse_mindat(locs_md)


# Clean and transform MR data
status.set_index('Mineral_Name', inplace=True)
ns.set_index('Mineral_Name', inplace=True)


mindat_rruff = locs_md.join(rruff_data, how='outer')
mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

# create final subset for the analysis
mr_data = ns.join(mindat_rruff, how='inner')


##### RE MINERALS  #####

re = mr_data.loc[(mr_data['locality_counts'] == 1)]
re.sort_values(by='discovery_year', inplace=True)

re.loc[re['discovery_year'] < 1950]

# during SD period
re.loc[re['discovery_year'] < 1950].groupby('CLASS').size()

# during MPRD period
re.loc[re['discovery_year'] >= 1950].groupby('CLASS').size()

# during all periods
re.groupby('CLASS').size()


##### RR MINERALS  #####

rr = mr_data.loc[(mr_data['locality_counts'] <= 4) & (mr_data['locality_counts'] >=2)]
rr.sort_values(by='discovery_year', inplace=True)

# during SD period
rr.loc[rr['discovery_year'] < 1950].groupby('CLASS').size()

# during MPRD period
rr.loc[rr['discovery_year'] >= 1950].groupby('CLASS').size()

# during all periods
rr.groupby('CLASS').size()


