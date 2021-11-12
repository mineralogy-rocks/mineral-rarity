import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.gsheet_api import GsheetApi
from functions.helpers import get_discovery_rate_all, get_discovery_rate_endemic

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for assessing the evolution of mineral endemicity with time and
make some assumptions
"""

GsheetApi = GsheetApi()
GsheetApi.run_main()

locs_mr = GsheetApi.locs.copy()
discovery_year = GsheetApi.names.copy()
status = GsheetApi.status_data.copy()


locs_md = pd.read_csv('data/mindat_locs.csv', sep=',')
rruff_data = pd.read_csv('data/RRUFF_Export.csv', sep=',')


# Clean and transform RRUFF data

rruff_data.loc[rruff_data['Year First Published'] == 0, 'Year First Published'] = np.nan
rruff_data.rename(columns={ 'Mineral Name': 'mineral_name', 'Year First Published': 'discovery_year' }, inplace=True)

rruff_data = rruff_data[['mineral_name', 'discovery_year']]
rruff_data.set_index('mineral_name', inplace=True)


# Clean and transform md data

locs_md.replace('\\N', np.nan, inplace=True)

locs_md.rename(columns={ 'loccount': 'locality_counts' }, inplace=True)

## Don't have any clue why mindat keeps `0` in imayear column and mixes dtypes in others (?!)

locs_md['imayear'].replace('0', np.nan, inplace=True)

locs_md = locs_md[['name', 'imayear', 'yeardiscovery', 'yearrruff', 'locality_counts']]
locs_md['imayear'] = pd.to_numeric(locs_md['imayear'])
locs_md['yearrruff'] = pd.to_numeric(locs_md['yearrruff'])

locs_md['locality_counts'] = pd.to_numeric(locs_md['locality_counts'])
locs_md.loc[~locs_md['yeardiscovery'].str.match(r'[0-9]{4}', na=False), 'yeardiscovery'] = np.nan
locs_md['yeardiscovery'] = pd.to_numeric(locs_md['yeardiscovery'])

locs_md.set_index('name', inplace=True)


# Clean and transform MR data

locs_mr.set_index('mineral_name', inplace=True)
discovery_year.set_index('Mineral_Name', inplace=True)
discovery_year['discovery_year_min'] = pd.to_numeric(discovery_year['discovery_year_min'])

locs_mr = locs_mr.join(discovery_year, how='inner')[['locality_counts', 'discovery_year_min']]

locs_mr['locality_counts'] = pd.to_numeric(locs_mr['locality_counts'])

discovery_year.rename(columns={ 'discovery_year_min': 'discovery_year' }, inplace=True)


## Get all minerals counts grouped by the discovery year (all from MR)

discovery_rate_all = get_discovery_rate_all(discovery_year)
discovery_rate_endemic = get_discovery_rate_endemic(locs_mr)

endemic_all_prop = discovery_rate_endemic.join(discovery_rate_all, how='inner', lsuffix='_endemic', rsuffix='_all')

endemic_all_prop['proportion'] = endemic_all_prop['count_endemic'] / endemic_all_prop['count_all'] * 100



# Check MR discovery year with RRUFF first published year

status.set_index('Mineral_Name', inplace=True)
status = status[['all_indexes']]

status_mr = status.join(discovery_year, how='inner')
status_mr_ima = status_mr.loc[status_mr.all_indexes.str.contains('0.0')]

rruff_mr_all = rruff_data.join(status_mr_ima, how='inner')

diff = rruff_mr_all.loc[rruff_mr_all['discovery_year'] != rruff_mr_all['discovery_year_min']] # 1922 diff


# Use mindat localities and RRUFF discovery year

mindat_rruff = locs_md.join(rruff_data, how='outer')

mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

mindat_rruff.loc[mindat_rruff['discovery_year'] == 2021]


## Get discovery rates (localities from mindat, discovery year from RRUFF)

discovery_rate_all = get_discovery_rate_all(mindat_rruff)
discovery_rate_endemic = get_discovery_rate_endemic(mindat_rruff)


## Get a propostion of endemic / all minerals during discovery year

endemic_all_prop = discovery_rate_endemic.join(discovery_rate_all, how='inner', lsuffix='_endemic', rsuffix='_all')

endemic_all_prop['proportion'] = endemic_all_prop['count_endemic'] / endemic_all_prop['count_all'] * 100


# bar chart of discovery rate of all minerals

y_pos = np.arange(len(discovery_rate_all.index))
x_ticks = np.arange(len(discovery_rate_all.index), step=15)
x_labels = discovery_rate_all.index[x_ticks]

plt.bar(y_pos, discovery_rate_all['count'])

plt.xticks(x_ticks, np.array(x_labels, dtype=int), rotation=45)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/all_minerals/discovery_rate.jpeg", dpi=300, format='jpeg')

plt.close()


# scatter plot of endemic mineral counts / discovery year

plt.scatter(discovery_rate_endemic.index, discovery_rate_endemic['count'], color='#5FD6D1', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

plt.xlabel('Discovery Year')
plt.ylabel('Endemic minerals count')
plt.title('Discovery rate of endemic minerals')

plt.savefig(f"figures/endemic_minerals/discovery_rate_scatter.jpeg", dpi=300, format='jpeg')

plt.close()


# bar chart of discovery rate of endemic minerals only

y_pos = np.arange(len(discovery_rate_endemic.index))
x_ticks = np.arange(len(discovery_rate_endemic.index), step=15)
x_labels = discovery_rate_endemic.index[x_ticks]

plt.bar(y_pos, discovery_rate_endemic['count'])

plt.xticks(x_ticks, np.array(x_labels, dtype=int), rotation=45)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/endemic_minerals/discovery_rate_bar.jpeg", dpi=300, format='jpeg')

plt.close()
