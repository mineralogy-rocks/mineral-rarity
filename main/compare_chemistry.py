import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.gsheet_api import GsheetApi
from functions.helpers import parse_rruff, parse_mindat

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
abundance = abundance.join(rr_el_spread.rename(columns={'abundance': 'abundance_rr'}), how='outer')
abundance = abundance.join(tr_el_spread.rename(columns={'abundance': 'abundance_tr'}), how='outer')
abundance = abundance.join(tu_el_spread.rename(columns={'abundance': 'abundance_tu'}), how='outer')
abundance = abundance.join(t_el_spread.rename(columns={'abundance': 'abundance_t'}), how='outer')
abundance = abundance.join(u_el_spread.rename(columns={'abundance': 'abundance_u'}), how='outer')
abundance['re/all'] = abundance['abundance_re'] / abundance['abundance_all'] * 100
abundance['rr/all'] = abundance['abundance_rr'] / abundance['abundance_all'] * 100
abundance['re + rr/all'] = (abundance['abundance_re'] + abundance['abundance_rr']) / abundance['abundance_all'] * 100
abundance['re + rr + tr/all'] = (abundance['abundance_re'] + abundance['abundance_rr'] + abundance['abundance_tr']) / abundance['abundance_all'] * 100
abundance['tr/all'] = abundance['abundance_tr'] / abundance['abundance_all'] * 100
abundance['tu/all'] = abundance['abundance_tu'] / abundance['abundance_all'] * 100
abundance['t/all'] = abundance['abundance_t'] / abundance['abundance_all'] * 100
abundance['u/all'] = abundance['abundance_u'] / abundance['abundance_all'] * 100
abundance['tu + u/all'] = (abundance['abundance_tu'] + abundance['abundance_u']) / abundance['abundance_all'] * 100


# Geochemical classifications
lile = abundance.loc[['Pb', 'Sr', 'Ba', 'K', 'Rb', 'Cs']]
hfse = abundance.loc[['P', 'Nb', 'Ta', 'W', 'Zr', 'Hf', 'Th', 'U', 'Ti', 'Y', 'La']]


# calculate elements co-occurrence matrix
elements = mr_el['Elements'].drop_duplicates().sort_values().reset_index(drop=True)
test = mr_el.copy()
test['number'] = 1
test['mineral_name'] = test.index
test = test.pivot(index='mineral_name', columns='Elements', values='number')
test = test.replace(0, np.nan)

import scipy.sparse as sp
X = sp.csr_matrix(test.values) # convert dataframe to sparse matrix
Xc = X.T * X # multiply sparse matrix #
Xc.setdiag(0) # reset diagonal
print(Xc.todense()) # to print co-occurence matrix in dense format

test1 = pd.DataFrame(data = Xc.toarray(), columns = test.columns)