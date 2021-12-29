import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["font.family"] = "Arial"
import seaborn as sns

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

locs_mr.join(status)

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
endemic = mindat_rruff.loc[(mindat_rruff['locality_counts'] == 1)]
endemic.sort_values(by='discovery_year', inplace=True)

mindat_rruff.loc[(mindat_rruff['locality_counts'] >= 2) & (mindat_rruff['locality_counts'] <= 4)]

mindat_rruff.loc[mindat_rruff['locality_counts'].isna(), 'locality_counts'] = 1.0
len(mindat_rruff.loc[(mindat_rruff['locality_counts'] > 18)]) / len(mindat_rruff) * 100
len(mindat_rruff.loc[(mindat_rruff['locality_counts'] <= 18) & (mindat_rruff['locality_counts'] >= 5)]) / len(mindat_rruff) * 100
len(mindat_rruff.loc[(mindat_rruff['locality_counts'] <= 4) ]) / len(mindat_rruff) * 100

## Get discovery rates (localities from mindat, discovery year from RRUFF)

discovery_rate_all = get_discovery_rate_all(mindat_rruff)
discovery_rate_endemic = get_discovery_rate_endemic(mindat_rruff)
endemic_proportion = get_endemic_proportion(discovery_rate_endemic, discovery_rate_all)

discovery_rate_all['count_cumulative'] = discovery_rate_all.cumsum()
discovery_rate_all = discovery_rate_all.loc[discovery_rate_all.index >= 1800]


### split discovery rate into Rarity groups
mindat_rruff.loc[(mindat_rruff['locality_counts'] > 1) & (mindat_rruff['locality_counts'] <= 16), 'rarity'] = 'RR + TR'
mindat_rruff.loc[mindat_rruff['locality_counts'] > 16, 'rarity'] = 'TU + U'
mindat_rruff.loc[mindat_rruff['locality_counts'] == 1, 'rarity'] = 'RE'

discovery_rate_classes = mindat_rruff.groupby(['discovery_year','rarity']).count().unstack()
discovery_rate_classes.columns = ['RE', 'RR+TR', 'TU+U']

# stacked bar chart of discovery rate vs rarity groups
sns.set(style="darkgrid")

plt.figure(figsize=(8, 8), dpi=300)

bar0 = sns.barplot(x="discovery_year", y="RR+TR", data=discovery_rate_classes.reset_index(), color='darkblue', dodge=False)
bar1 = sns.barplot(x="discovery_year", y="TU+U", data=discovery_rate_classes.reset_index(), color='lightseagreen', dodge=False)
bar2 = sns.barplot(x="discovery_year", y="RE", data=discovery_rate_classes.reset_index(), color='tomato', dodge=False)

# add legend
x_ticks = np.arange(len(discovery_rate_classes.index), step=15)
x_labels = discovery_rate_classes.index[x_ticks]

top_bar0 = mpatches.Patch(color='tomato', label='RE')
top_bar = mpatches.Patch(color='darkblue', label='RR+TR')
bottom_bar = mpatches.Patch(color='lightseagreen', label='TU+U')
plt.legend(handles=[top_bar0, top_bar, bottom_bar])

plt.xticks(x_ticks, np.array(x_labels, dtype=int), rotation=45)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')

plt.savefig(f"figures/discovery_rate/stacked_all_rarity_groups.jpeg", dpi=300, format='jpeg')

plt.close()

# same but show sum of minerals
sns.set_context("paper")
ax1 = sns.set(style="white")

fig, ax1 = plt.subplots(figsize=(8, 8), dpi=300)

temp = sns.histplot(
    mindat_rruff,
    x='discovery_year',
    hue='rarity',
    multiple='stack',
    palette=['darkblue', 'lightseagreen', 'tomato'],
    # Add white borders to the bars.
    edgecolor='white',
    # kde=True,
    # Shrink the bars a bit so they don't touch.
    shrink=0.5,
    binwidth=1,
    ax=ax1,
    legend=False
)

ax1.set(xlim=(1750, 2021))

ax2 = ax1.twinx()

sns.kdeplot(data=mindat_rruff, x="discovery_year", hue="rarity", palette=['darkblue', 'lightseagreen', 'tomato'], ax=ax2,
            legend=False)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')

top_bar0 = mpatches.Patch(color='tomato', label='RE')
top_bar = mpatches.Patch(color='darkblue', label='RR+TR')
bottom_bar = mpatches.Patch(color='lightseagreen', label='TU+U')
plt.legend(handles=[top_bar0, top_bar, bottom_bar])

plt.savefig(f"figures/discovery_rate/stacked_all_rarity_groups.jpeg", dpi=300, format='jpeg')

plt.close()


# bar chart of discovery rate of all minerals between 1800 and 2021

y_pos = np.arange(len(discovery_rate_all.index))
x_ticks = np.arange(len(discovery_rate_all.index), step=15)
x_labels = discovery_rate_all.index[x_ticks]

plt.bar(y_pos, discovery_rate_all['count'])

plt.xticks(x_ticks, np.array(x_labels, dtype=int), rotation=45)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/discovery_rate/all.jpeg", dpi=300, format='jpeg')

plt.close()


# cumulative curve of discovery rate for all minerals

plt.plot(discovery_rate_all.index, discovery_rate_all[['count_cumulative']], color='red', linestyle='dotted', linewidth=1)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/all_minerals/discovery_rate_cumulative.jpeg", dpi=300, format='jpeg')

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


