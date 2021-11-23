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

rruff_data = parse_rruff(rruff_data)
locs_md = parse_mindat(locs_md)


# Clean and transform MR data
locs_mr.set_index('mineral_name', inplace=True)
status.set_index('Mineral_Name', inplace=True)
discovery_year.set_index('Mineral_Name', inplace=True)
discovery_year['discovery_year_min'] = pd.to_numeric(discovery_year['discovery_year_min'])

locs_mr = locs_mr.join(discovery_year, how='inner')[['locality_counts', 'discovery_year_min']]

locs_mr['locality_counts'] = pd.to_numeric(locs_mr['locality_counts'])

discovery_year.rename(columns={ 'discovery_year_min': 'discovery_year' }, inplace=True)
locs_mr.rename(columns={ 'discovery_year_min': 'discovery_year' }, inplace=True)

## Get all minerals counts grouped by the discovery year (all from MR)

discovery_rate_all = get_discovery_rate_all(discovery_year)
discovery_rate_endemic = get_discovery_rate_endemic(locs_mr)
endemic_proportion = get_endemic_proportion(discovery_rate_endemic, discovery_rate_all)

# Use mindat localities and RRUFF discovery year

mindat_rruff = locs_md.join(rruff_data, how='outer')

mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

# get proportions of rare and endemic minerals
mindat_rruff.loc[(mindat_rruff['locality_counts'] == 1)]

len(mindat_rruff.loc[(mindat_rruff['locality_counts'] <= 18) & (mindat_rruff['locality_counts'] > 5)]) / 5762 * 100

## Get discovery rates (localities from mindat, discovery year from RRUFF)

discovery_rate_all = get_discovery_rate_all(mindat_rruff)
discovery_rate_endemic = get_discovery_rate_endemic(mindat_rruff)
endemic_proportion = get_endemic_proportion(discovery_rate_endemic, discovery_rate_all)


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
