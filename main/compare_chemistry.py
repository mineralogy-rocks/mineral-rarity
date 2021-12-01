import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from modules.gsheet_api import GsheetApi
from functions.helpers import parse_rruff, parse_mindat, calculate_cooccurrence_matrix

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

elements = pd.read_csv('data/elements_data.csv', sep=',')
elements.set_index('element', inplace=True)

# Petrological classifications

elements.loc[elements.index.isin(['Pb', 'Sr', 'Ba', 'K', 'Rb', 'Cs']), 'petrology'] = 'LILE'
elements.loc[elements.index.isin(['P', 'Nb', 'Ta', 'W', 'Zr', 'Hf', 'Th', 'U', 'Ti', 'Y', 'Lu', 'La', 'Eu']), 'petrology'] = 'HFSE'
elements.loc[elements.index.isin(['Si', 'Al', 'Cr', 'Fe', 'Mg', 'Cu', 'Zn', 'Ni', 'Co', 'Ca']), 'petrology'] = 'MRFE'

# Clean and transform MR data
status.set_index('Mineral_Name', inplace=True)
ns.set_index('Mineral_Name', inplace=True)


mindat_rruff = locs_md.join(rruff_data, how='outer')
mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

# create final subset for the analysis
mr_data = ns.join(mindat_rruff, how='inner')

##### RE MINERALS  #####

re = mr_data.loc[(mr_data['locality_counts'] == 1)]
re_true = mr_data.loc[~((mr_data['discovery_year'] > 2000) & (mr_data['locality_counts'] == 1)) & (mr_data['locality_counts'] == 1)]

re.sort_values(by='discovery_year', inplace=True)
re_true.sort_values(by='discovery_year', inplace=True)

re.loc[re['discovery_year'] < 1950]
re_true.loc[re['discovery_year'] < 1950]

# during SD period
re.loc[re['discovery_year'] < 1950].groupby('CLASS').size()
re_true.loc[re_true['discovery_year'] < 1950].groupby('CLASS').size()

# during MPRD period
re.loc[re['discovery_year'] >= 1950].groupby('CLASS').size()
re_true.loc[re_true['discovery_year'] >= 1950].groupby('CLASS').size()

# further analytics
re.loc[(re['CLASS'] == 'Elements') & (re['locality_counts'] == 1)]
re_true.loc[(re_true['CLASS'] == 'Elements') & (re_true['locality_counts'] == 1)]

# Analyse carbides
carbon_minerals = mr_data[mr_data['Formula'].str.contains(r'C(?![a-z])') == True]
carbon_minerals_re = carbon_minerals.loc[carbon_minerals['locality_counts'] == 1]
carbon_minerals_re.loc[carbon_minerals_re['CLASS'] == 'Elements']
carbon_minerals_re.groupby('CLASS').size()

# during all periods
RE = pd.DataFrame(re.groupby('CLASS').size(), columns=['RE'])
tRE = pd.DataFrame(re_true.groupby('CLASS').size(), columns=['tRE'])
RE.join(tRE, how='inner')

pie_ = re.groupby('CLASS').size()
pie_ = pie_/pie_.sum() * 100


# Pie chart: Nickel-Strunz classes for RE

fig1, ax1 = plt.subplots()
ax1.pie(pie_, labels=pie_.index, autopct='%1.1f%%',startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig(f"figures/endemic_minerals/pie_chart_nickel_strunz.jpeg", dpi=300, format='jpeg')

plt.close()


##### RR MINERALS  #####

rr = mr_data.loc[(mr_data['locality_counts'] <= 4) & (mr_data['locality_counts'] >=2)]
rr.sort_values(by='discovery_year', inplace=True)
rr.loc[rr['discovery_year'] < 1950]

# during SD period
rr.loc[rr['discovery_year'] < 1950].groupby('CLASS').size()

# during MPRD period
rr.loc[rr['discovery_year'] >= 1950].groupby('CLASS').size()

# during all periods
rr.groupby('CLASS').size()

# during all periods
pie_ = rr.groupby('CLASS').size()
pie_ = pie_/pie_.sum() * 100


# Pie chart: Nickel-Strunz classes

fig1, ax1 = plt.subplots()
ax1.pie(pie_, labels=pie_.index, autopct='%1.1f%%',startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig(f"figures/rare_minerals/pie_chart_nickel_strunz.jpeg", dpi=300, format='jpeg')

plt.close()

r = mr_data.loc[(mr_data['locality_counts'] <= 4)]
r.sort_values(by='discovery_year', inplace=True)

##### T MINERALS  #####

t = mr_data.loc[(mr_data['locality_counts'] > 4) & (mr_data['locality_counts'] <= 70)]
t.sort_values(by='discovery_year', inplace=True)

tr = mr_data.loc[(mr_data['locality_counts'] > 4) & (mr_data['locality_counts'] <= 16)]
tr.sort_values(by='discovery_year', inplace=True)

tu = mr_data.loc[(mr_data['locality_counts'] > 16) & (mr_data['locality_counts'] <= 70)]
tu.sort_values(by='discovery_year', inplace=True)

##### T MINERALS  #####

u = mr_data.loc[(mr_data['locality_counts'] > 70)]
u.sort_values(by='discovery_year', inplace=True)


##### Chemistry analytics #####

# Get a share by elements for RE
re_el = pd.DataFrame(re.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
re_el.rename(columns={0 : 'Elements'}, inplace=True)
re_el = re_el.explode('Elements')
re_el_spread = pd.DataFrame(re_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for tRE
re_true_el = pd.DataFrame(re_true.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
re_true_el.rename(columns={0 : 'Elements'}, inplace=True)
re_true_el = re_true_el.explode('Elements')
re_true_el_spread = pd.DataFrame(re_true_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for RR
rr_el = pd.DataFrame(rr.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
rr_el.rename(columns={0 : 'Elements'}, inplace=True)
rr_el = rr_el.explode('Elements')
rr_el_spread = pd.DataFrame(rr_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for all R
r_el = pd.DataFrame(r.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
r_el.rename(columns={0 : 'Elements'}, inplace=True)
r_el = r_el.explode('Elements')
r_el_spread = pd.DataFrame(r_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for TR
tr_el = pd.DataFrame(tr.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
tr_el.rename(columns={0 : 'Elements'}, inplace=True)
tr_el = tr_el.explode('Elements')
tr_el_spread = pd.DataFrame(tr_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for TU
tu_el = pd.DataFrame(tu.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
tu_el.rename(columns={0 : 'Elements'}, inplace=True)
tu_el = tu_el.explode('Elements')
tu_el_spread = pd.DataFrame(tu_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for T
t_el = pd.DataFrame(t.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
t_el.rename(columns={0 : 'Elements'}, inplace=True)
t_el = t_el.explode('Elements')
t_el_spread = pd.DataFrame(t_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for U
u_el = pd.DataFrame(u.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
u_el.rename(columns={0 : 'Elements'}, inplace=True)
u_el = u_el.explode('Elements')
u_el_spread = pd.DataFrame(u_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# Get a share by elements for All
mr_el = pd.DataFrame(mr_data.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
mr_el.rename(columns={0 : 'Elements'}, inplace=True)
mr_el = mr_el.explode('Elements')
mr_el_spread = pd.DataFrame(mr_el.groupby('Elements').size().sort_values(), columns=['abundance'])

# concat all and RE
abundance = mr_el_spread.join(re_el_spread, how='outer', lsuffix='_all', rsuffix='_re')
abundance = abundance.join(re_true_el_spread.rename(columns={'abundance': 'abundance_re_true'}), how='outer')
abundance = abundance.join(rr_el_spread.rename(columns={'abundance': 'abundance_rr'}), how='outer')
abundance = abundance.join(r_el_spread.rename(columns={'abundance': 'abundance_r'}), how='outer')
abundance = abundance.join(tr_el_spread.rename(columns={'abundance': 'abundance_tr'}), how='outer')
abundance = abundance.join(tu_el_spread.rename(columns={'abundance': 'abundance_tu'}), how='outer')
abundance = abundance.join(t_el_spread.rename(columns={'abundance': 'abundance_t'}), how='outer')
abundance = abundance.join(u_el_spread.rename(columns={'abundance': 'abundance_u'}), how='outer')
abundance.replace(np.nan, 0, inplace=True)
abundance['re/all'] = abundance['abundance_re'] / abundance['abundance_all'] * 100
abundance['re_true/all'] = abundance['abundance_re_true'] / abundance['abundance_all'] * 100
abundance['rr/all'] = abundance['abundance_rr'] / abundance['abundance_all'] * 100
abundance['re + rr/all'] = (abundance['abundance_re'] + abundance['abundance_rr']) / abundance['abundance_all'] * 100
abundance['re + rr + tr/all'] = (abundance['abundance_re'] + abundance['abundance_rr'] + abundance['abundance_tr']) / abundance['abundance_all'] * 100
abundance['tr/all'] = abundance['abundance_tr'] / abundance['abundance_all'] * 100
abundance['tu/all'] = abundance['abundance_tu'] / abundance['abundance_all'] * 100
abundance['t/all'] = abundance['abundance_t'] / abundance['abundance_all'] * 100
abundance['u/all'] = abundance['abundance_u'] / abundance['abundance_all'] * 100
abundance['tu + u/all'] = (abundance['abundance_tu'] + abundance['abundance_u']) / abundance['abundance_all'] * 100


# join elements data
abundance = abundance.join(elements, how='left')
abundance['ion_radius'].replace(',','.', regex=True, inplace=True)
abundance['electronegativity'].replace(',','.', regex=True, inplace=True)
abundance['crust_crc_handbook'].replace(',','.', regex=True, inplace=True)

abundance['Elements'] = abundance.index

# Dot plot of elements, arranged by abundance in crust grouped by rarity groups
sns.set_theme(style="whitegrid")

# Make the PairGrid
g = sns.PairGrid(data=abundance.sort_values('crust_crc_handbook', ascending=False),
                 x_vars=['re_true/all', 're + rr/all', 're + rr + tr/all', 't/all', 'u/all', 'tu + u/all'], y_vars=["Elements"],
                 hue="goldschmidt_classification", hue_order=None, height=10, aspect=.25)


g.map(sns.scatterplot, size=abundance['ion_radius'].to_numpy(dtype=float), legend='brief', linewidth=0.5, marker='o', edgecolor='black')

g.add_legend(adjust_subtitles=True)

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 100), xlabel="% of minerals", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['tRE/All', 'RE + RR', 'RE + RR + TR', 'T', 'U', 'TU + U']

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

g.savefig(f"figures/chemistry/dot_plot_proportion_from_all.jpeg", dpi=300, format='jpeg')


# Chalcophile and Siderophile elements, rare in Earth's Crust
chalc_sidero_el = ['Sn', 'As', 'Ge', 'Mo', 'Tl', 'In', 'Sb', 'Cd', 'Hg', 'Ag', 'Se', 'Pd', 'Bi', 'Pt', 'Au', 'Os', 'Te',
                   'Rh', 'Ru', 'Ir', 'Re']

abundance.loc[abundance.index.isin(chalc_sidero_el)]['re + rr + tr/all'].median()


# calculate elements co-occurrence matrixes
# All
mr_el_vector = pd.DataFrame(mr_data.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1))
mr_el_vector[1] = pd.DataFrame(mr_data.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
mr_el_vector = mr_el_vector.explode(1)

# tRE
re_true_el_vector = pd.DataFrame(re_true.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1))
re_true_el_vector[1] = pd.DataFrame(re_true.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
re_true_el_vector = re_true_el_vector.explode(1)

# RE
re_el_vector = pd.DataFrame(re.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1))
re_el_vector[1] = pd.DataFrame(re.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
re_el_vector = re_el_vector.explode(1)

# RR
rr_el_vector = pd.DataFrame(rr.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1))
rr_el_vector[1] = pd.DataFrame(rr.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
rr_el_vector = rr_el_vector.explode(1)

# T
t_el_vector = pd.DataFrame(t.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1))
t_el_vector[1] = pd.DataFrame(t.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
t_el_vector = t_el_vector.explode(1)

# U
u_el_vector = pd.DataFrame(u.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1))
u_el_vector[1] = pd.DataFrame(u.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
u_el_vector = u_el_vector.explode(1)

cooccurrence_all = calculate_cooccurrence_matrix(mr_el_vector[0], mr_el_vector[1])
cooccurrence_t_re = calculate_cooccurrence_matrix(re_true_el_vector[0], re_true_el_vector[1])
cooccurrence_re = calculate_cooccurrence_matrix(re_el_vector[0], re_el_vector[1])
cooccurrence_rr = calculate_cooccurrence_matrix(rr_el_vector[0], rr_el_vector[1])
cooccurrence_t = calculate_cooccurrence_matrix(t_el_vector[0], t_el_vector[1])
cooccurrence_u = calculate_cooccurrence_matrix(u_el_vector[0], u_el_vector[1])

cooccurrence_all_norm = calculate_cooccurrence_matrix(mr_el_vector[0], mr_el_vector[1], norm='index')
cooccurrence_t_re_norm = calculate_cooccurrence_matrix(re_true_el_vector[0], re_true_el_vector[1], norm='index')
cooccurrence_re_norm = calculate_cooccurrence_matrix(re_el_vector[0], re_el_vector[1], norm='index')
cooccurrence_rr_norm = calculate_cooccurrence_matrix(rr_el_vector[0], rr_el_vector[1], norm='index')
cooccurrence_t_norm = calculate_cooccurrence_matrix(t_el_vector[0], t_el_vector[1], norm='index')
cooccurrence_u_norm = calculate_cooccurrence_matrix(u_el_vector[0], u_el_vector[1], norm='index')

# add heat maps
sns.set_theme(style="whitegrid")
temp = sns.heatmap(cooccurrence_t_re_norm, linewidths=0.1)
temp.set_xticks(range(len(cooccurrence_t_re_norm))) # <--- set the ticks first
temp.set_xticklabels(cooccurrence_t_re_norm.columns)
temp.tick_params(labelsize=2)
temp.set_xlabel(None)
temp.set_ylabel(None)
plt.savefig(f"figures/chemistry/cooccurrence_tre.jpeg", dpi=300, format='jpeg')
plt.close()