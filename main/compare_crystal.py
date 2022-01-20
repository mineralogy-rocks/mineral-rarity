import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

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

crystal_system = get_crystal_system_obj()

r, re_rr_tr, re_true, re, rr, t, tr, tu, u, tu_u = split_by_rarity_groups(mr_data)


# build Crystal Systems pie charts
pie_ = pd.DataFrame(columns=['re', 'rr', 'tr', 'tu', 'u', ])
for key, item in { 're': re, 'rr': rr, 'tr': tr, 'tu': tu, 'u': u}.items():
    item['Crystal System'] = item['Crystal System'].str.capitalize()
    pie_[key] = item.groupby('Crystal System').size() / item.groupby('Crystal System').size().sum() * 100

pie_ = pie_.join(crystal_system).sort_values('order')


# RE, RR and TR species
fig, ax = plt.subplots(nrows=1, ncols=3)
plt.rcParams['axes.titlepad'] = 16

ax[0].pie(pie_['re'], colors=pie_['color'], autopct='%1.1f%%', startangle=90, pctdistance=0.8,
          wedgeprops = {'linewidth': 0.4, 'edgecolor': 'black'}, textprops={ 'fontsize': 7 }, radius=1.3)
ax[0].set_title('a', fontsize = 11)

ax[1].pie(pie_['rr'], colors=pie_['color'], autopct='%1.1f%%',startangle=90, pctdistance=0.8,
          wedgeprops = {'linewidth': 0.4, 'edgecolor': 'black'}, textprops={ 'fontsize': 7 }, radius=1.3)
ax[1].set_title('b', fontsize = 11)

ax[2].pie(pie_['tr'], colors=pie_['color'], autopct='%1.1f%%',startangle=90, pctdistance=0.8,
          wedgeprops = {'linewidth': 0.4, 'edgecolor': 'black'}, textprops={ 'fontsize': 7 }, radius=1.3)
ax[2].set_title('c', fontsize = 11)

# plt.tight_layout()
fig.subplots_adjust(right=0.8, left=0.05)
fig.legend(pie_.index, fontsize=7, loc='right', labelspacing=.3) # , bbox_to_anchor=(1.8, 0.5),

plt.savefig(f"figures/crystal/re_rr_tr.jpeg", dpi=300, format='jpeg')
plt.close()


# TU and U species
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.rcParams['axes.titlepad'] = 16

ax[0].pie(pie_['tu'], colors=pie_['color'], autopct='%1.1f%%', startangle=90, pctdistance=0.8,
          wedgeprops = {'linewidth': 0.4, 'edgecolor': 'black'}, textprops={ 'fontsize': 7 }, radius=1.3)
ax[0].set_title('a', fontsize = 11)

ax[1].pie(pie_['u'], colors=pie_['color'], autopct='%1.1f%%',startangle=90, pctdistance=0.8,
          wedgeprops = {'linewidth': 0.4, 'edgecolor': 'black'}, textprops={ 'fontsize': 7 }, radius=1.3)
ax[1].set_title('b', fontsize = 11)

# plt.tight_layout()
fig.subplots_adjust(right=0.8, left=0.05)
fig.legend(pie_.index, fontsize=7, loc='right', labelspacing=.3) # , bbox_to_anchor=(1.8, 0.5),

plt.savefig(f"figures/crystal/tu_u.jpeg", dpi=300, format='jpeg')
plt.close()



# symmetry index
(pie_.loc['isometric', 0] + pie_.loc['hexagonal', 0] + pie_.loc['trigonal', 0] + pie_.loc['tetragonal', 0]) \
/ (pie_.loc['triclinic', 0] + pie_.loc['monoclinic', 0] + pie_.loc['orthorhombic', 0])

# calculate symmetry index yearly

discovery_rate = mr_data.loc[mr_data['discovery_year'].notna()]
discovery_rate = get_symmetry_indexes(discovery_rate)
