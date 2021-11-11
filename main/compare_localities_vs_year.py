import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.gsheet_api import GsheetApi

GsheetApi = GsheetApi()

GsheetApi.run_main()

locs_mr = GsheetApi.locs.copy()
names = GsheetApi.names.copy()


locs_mr.set_index('mineral_name', inplace=True)
names.set_index('Mineral_Name', inplace=True)
names['discovery_year_min'] = pd.to_numeric(names['discovery_year_min'])

locs_mr = locs_mr.join(names, how='inner')[['locality_counts', 'discovery_year_min']]

locs_mr['locality_counts'] = pd.to_numeric(locs_mr['locality_counts'])


# Get all minerals counts grouped by the discovery year (all from MR)

discovery_rate_all = names.loc[(names['discovery_year_min'].notna()) & (names['discovery_year_min'] > 1500)]

discovery_rate_all = discovery_rate_all.sort_values('discovery_year_min', ascending=True)[['discovery_year_min']]

discovery_rate_all = discovery_rate_all.groupby('discovery_year_min').agg({ 'discovery_year_min': 'count' })

discovery_rate_all.rename(columns={ 'discovery_year_min': 'count' }, inplace=True)


## bar chart of discovery rate of all minerals

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


# Get endemic minerals counts grouped by the discovery year (all from MR)

discovery_rate_endemic = locs_mr.loc[(locs_mr['locality_counts'] == 1) & (locs_mr['discovery_year_min'].notna())]

discovery_rate_endemic = discovery_rate_endemic.sort_values('discovery_year_min', ascending=True)

discovery_rate_endemic = discovery_rate_endemic.groupby('discovery_year_min').agg({ 'locality_counts': 'sum' })

discovery_rate_endemic.rename(columns={ 'locality_counts': 'count' }, inplace=True)

## scatter plot of endemic mineral counts / discovery year

plt.scatter(discovery_rate_endemic.index, discovery_rate_endemic['count'], color='#5FD6D1', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

plt.xlabel('Discovery Year')
plt.ylabel('Endemic minerals count')
plt.title('Discovery rate of endemic minerals')

plt.savefig(f"figures/endemic_minerals/discovery_rate_scatter.jpeg", dpi=300, format='jpeg')

plt.close()


## bar chart of discovery rate of endemic minerals only

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


# Get a propostion of endemic / all minerals during discovery year

endemic_all_prop = discovery_rate_endemic.join(discovery_rate_all, how='inner', lsuffix='_endemic', rsuffix='_all')

endemic_all_prop['proportion'] = endemic_all_prop['count_endemic'] / endemic_all_prop['count_all'] * 100

