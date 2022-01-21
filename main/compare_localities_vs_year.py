import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
import seaborn as sns

from modules.gsheet_api import GsheetApi
from functions.helpers import get_discovery_rate_all, get_discovery_rate_endemic, get_endemic_proportion, parse_rruff, \
    parse_mindat, prepare_data, get_symmetry_indexes

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for assessing the evolution of mineral endemicity with time and the discovery rate and make some assumptions
"""

GsheetApi = GsheetApi()
GsheetApi.run_main()

ns = GsheetApi.nickel_strunz.copy()
crystal = GsheetApi.crystal.copy()

mr_data = prepare_data(ns, crystal)

locs_md = pd.read_csv('data/mindat_locs.csv', sep=',')
rruff_data = pd.read_csv('data/RRUFF_Export.csv', sep=',')
rruff_data = parse_rruff(rruff_data)
locs_md = parse_mindat(locs_md)

# Use mindat localities and RRUFF discovery year
mindat_rruff = locs_md.join(rruff_data, how='outer')
mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]
mindat_rruff = mindat_rruff.join(mr_data['Crystal System'], how='left')
mindat_rruff.loc[mindat_rruff['locality_counts'].isna(), 'locality_counts'] = 1.0

symmetry_rate = get_symmetry_indexes(mindat_rruff)

# get basic stats
len(mindat_rruff.loc[(mindat_rruff['locality_counts'] > 18)]) / len(mindat_rruff) * 100
len(mindat_rruff.loc[(mindat_rruff['locality_counts'] <= 18) & (mindat_rruff['locality_counts'] >= 5)]) / len(mindat_rruff) * 100
len(mindat_rruff.loc[(mindat_rruff['locality_counts'] <= 4) ]) / len(mindat_rruff) * 100

## Get discovery rates (localities from mindat, discovery year from RRUFF)
discovery_rate_all = get_discovery_rate_all(mindat_rruff)
discovery_rate_endemic = get_discovery_rate_endemic(mindat_rruff)
endemic_proportion = get_endemic_proportion(discovery_rate_endemic, discovery_rate_all)

discovery_rate_all['count_cumulative'] = discovery_rate_all.cumsum()
discovery_rate_all = discovery_rate_all.loc[discovery_rate_all.index >= 1800]

discovery_rate = discovery_rate_all.join(endemic_proportion, how='left')
discovery_rate = discovery_rate.join(symmetry_rate[['triclinic_index', 'symmetry_index']], how='left')


### split discovery rate into Rarity groups
mindat_rruff.loc[(mindat_rruff['locality_counts'] > 1) & (mindat_rruff['locality_counts'] <= 16), 'rarity'] = 'RR + TR'
mindat_rruff.loc[mindat_rruff['locality_counts'] > 16, 'rarity'] = 'TU + U'
mindat_rruff.loc[mindat_rruff['locality_counts'] == 1, 'rarity'] = 'RE'

discovery_rate_classes = mindat_rruff.groupby(['discovery_year','rarity'])[['rarity']].count().unstack()
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


# stacked bar chart of rarity groups discovery rate and with KDEs
ax1 = sns.set_theme(palette=None, style={ 'figure.facecolor': 'white', 'xtick.bottom': True, 'ytick.left': True })

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
plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')

ax2 = ax1.twinx()
ax3 = ax1.twinx()

sns.kdeplot(data=mindat_rruff, x="discovery_year", hue="rarity", palette=['darkblue', 'lightseagreen', 'tomato'], ax=ax2,
            legend=False)

temp_ = discovery_rate.reset_index().dropna(subset=['symmetry_index'])
temp_.loc[temp_['symmetry_index'] == np.inf, 'symmetry_index'] = 0
sns.kdeplot(data=temp_, x="discovery_year", weights='symmetry_index', ax=ax2, fill=False, color="red", alpha=.3,
            linewidth=1, legend=False)


top_bar0 = mpatches.Patch(color='tomato', label='RE')
top_bar = mpatches.Patch(color='darkblue', label='RR+TR')
bottom_bar = mpatches.Patch(color='lightseagreen', label='TU+U')
bottom_bar1 = mpatches.Patch(color='red', label='symmetry index')

plt.legend(handles=[top_bar0, top_bar, bottom_bar, bottom_bar1], loc='upper left')
plt.axis('off')
plt.savefig(f"figures/discovery_rate/stacked_all_rarity_groups_kdes.eps", dpi=300, format='eps')
plt.close()


# stacked bar chart of rarity groups discovery rate
sns.set_theme(palette=None, style={ 'figure.facecolor': 'white', 'xtick.bottom': True, 'ytick.left': True })
fig, ax1 = plt.subplots(figsize=(8, 8), dpi=300)

temp = sns.histplot(
    mindat_rruff,
    x='discovery_year',
    hue='rarity',
    multiple='stack',
    palette=['darkblue', 'lightseagreen', 'tomato'],
    edgecolor='white',
    shrink=0.5,
    binwidth=1,
    ax=ax1,
    legend=False
)

ax1.set(xlim=(1750, 2021))

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')

top_bar0 = mpatches.Patch(color='tomato', label='RE')
top_bar = mpatches.Patch(color='darkblue', label='RR+TR')
bottom_bar = mpatches.Patch(color='lightseagreen', label='TU+U')

plt.legend(handles=[top_bar0, top_bar, bottom_bar])

plt.savefig(f"figures/discovery_rate/stacked_all_rarity_groups.jpeg", dpi=300, format='jpeg')
plt.close()



# symmetry index with endemicity index
sns.set(style="darkgrid")

plt.figure(figsize=(8, 8), dpi=300)

sns.lineplot(data=discovery_rate, x=discovery_rate.index, y=discovery_rate.symmetry_index, palette="tab10", linewidth=1)
sns.lineplot(data=discovery_rate, x=discovery_rate.index, y=discovery_rate.proportion, palette="tab10", linewidth=1)

plt.savefig(f"figures/discovery_rate/symmetry_index.jpeg", dpi=300, format='jpeg')
plt.close()


# bar chart of discovery rate of all minerals between 1800 and 2021
y_pos = np.arange(len(discovery_rate.index))
x_ticks = np.arange(len(discovery_rate.index), step=15)
x_labels = discovery_rate.index[x_ticks]

plt.bar(y_pos, discovery_rate['count'])

plt.xticks(x_ticks, np.array(x_labels, dtype=int), rotation=45)
plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/discovery_rate/all.jpeg", dpi=300, format='jpeg')
plt.close()


# cumulative curve of discovery rate for all minerals
plt.plot(discovery_rate.index, discovery_rate[['count_cumulative']], color='red', linestyle='dotted', linewidth=1)

plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/discovery_rate/all_cumulative.jpeg", dpi=300, format='jpeg')

plt.close()


# scatter plot of endemic mineral counts / discovery year
sns.set_theme(palette=None, style={ 'figure.facecolor': 'white', 'xtick.bottom': True, 'ytick.left': True })

plt.scatter(discovery_rate.index, discovery_rate['count_endemic'], color='green', marker='o', s=60,
            edgecolor='black', linewidth=1, alpha=0.5)

plt.xlabel('Discovery year')
plt.ylabel('Minerals count')
plt.xlim([1800, 2022])

plt.savefig(f"figures/discovery_rate/endemic_scatter.jpeg", dpi=300, format='jpeg')

plt.close()


# bar chart of discovery rate of endemic minerals only

y_pos = np.arange(len(discovery_rate.index))
x_ticks = np.arange(len(discovery_rate.index), step=15)
x_labels = discovery_rate.index[x_ticks]

plt.bar(y_pos, discovery_rate['count_endemic'])

plt.xticks(x_ticks, np.array(x_labels, dtype=int), rotation=45)
plt.xlabel('Discovery Year')
plt.ylabel('Minerals count')
plt.title('Discovery rate of minerals')

plt.savefig(f"figures/discovery_rate/endemic_bar.jpeg", dpi=300, format='jpeg')
plt.close()


