import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from modules.gsheet_api import GsheetApi
from functions.helpers import parse_rruff, parse_mindat, calculate_cooccurrence_matrix, split_by_rarity_groups,\
    get_mineral_clarks

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for analysing the chemistry of minerals in the light of their rarity
"""

GsheetApi = GsheetApi()
GsheetApi.run_main()

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
ns.set_index('Mineral_Name', inplace=True)

mindat_rruff = locs_md.join(rruff_data, how='outer')
mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

# create final subset for the analysis
mr_data = ns.join(mindat_rruff, how='inner')

r, re_true, re, rr, t, tr, tu, u = split_by_rarity_groups(mr_data)

##### RE MINERALS  #####

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


##### Chemistry analytics #####

re_true_el, re_true_el_spread = get_mineral_clarks(re_true)
re_el, re_el_spread = get_mineral_clarks(re)
rr_el, rr_el_spread = get_mineral_clarks(rr)
r_el, r_el_spread = get_mineral_clarks(r)

tr_el, tr_el_spread = get_mineral_clarks(tr)
tu_el, tu_el_spread = get_mineral_clarks(tu)
t_el, t_el_spread = get_mineral_clarks(t)

u_el, u_el_spread = get_mineral_clarks(u)

mr_el,mr_el_spread = get_mineral_clarks(mr_data)

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
abundance['all'] = abundance['abundance_all'] / len(mr_data) * 100
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
abundance['atomic_mass'].replace(',','.', regex=True, inplace=True)
abundance['electronegativity'].replace(',','.', regex=True, inplace=True)
abundance['crust_crc_handbook'].replace(',','.', regex=True, inplace=True)
abundance['crust_kaye_laby'].replace(',','.', regex=True, inplace=True)
abundance['crust_greenwood'].replace(',','.', regex=True, inplace=True)
abundance['crust_ahrens_taylor'].replace(',','.', regex=True, inplace=True)
abundance['crust_ahrens_wanke'].replace(',','.', regex=True, inplace=True)
abundance['crust_ahrens_waver'].replace(',','.', regex=True, inplace=True)
abundance['upper_crust_ahrens_taylor'].replace(',','.', regex=True, inplace=True)
abundance['upper_crust_ahrens_shaw'].replace(',','.', regex=True, inplace=True)

abundance.replace(r'^\s*$', np.nan, regex=True, inplace=True)

abundance['electron_affinity'] = pd.to_numeric(abundance['electron_affinity'])
abundance['atomic_number'] = pd.to_numeric(abundance['atomic_number'])
abundance['crust_crc_handbook'] = pd.to_numeric(abundance['crust_crc_handbook'])
abundance['ion_radius'] = pd.to_numeric(abundance['ion_radius'])
abundance['Elements'] = abundance.index


# Dot plot of elements, arranged by abundance in crust grouped by rarity groups
sns.set_theme(style="whitegrid")

# Make the PairGrid
g = sns.PairGrid(data=abundance.sort_values('crust_crc_handbook', ascending=False),
                 x_vars=['all', 're_true/all', 're + rr/all', 're + rr + tr/all', 't/all', 'u/all', 'tu + u/all'], y_vars=["Elements"],
                 hue="goldschmidt_classification", hue_order=None, height=10, aspect=.25)


g.map(sns.scatterplot, size=abundance['ion_radius'].to_numpy(), legend='brief', linewidth=0.5, marker='o', edgecolor='black')

g.add_legend(adjust_subtitles=True)

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 100), xlabel="% of minerals", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['ALL', 'tRE/All', 'RE + RR', 'RE + RR + TR', 'T', 'U', 'TU + U']

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

plt.savefig(f"figures/chemistry/dot_plot_proportion_from_all.jpeg", dpi=300, format='jpeg')
plt.close()


# Dot plot of elements, sorted by goldschmidt groups (for Vitalii)

sns.set_theme(style="whitegrid")
initial_data = abundance.sort_values(['goldschmidt_classification', 'crust_crc_handbook'], ascending=False)

# Make the PairGrid
g = sns.PairGrid(data=initial_data,
                 x_vars=['all', 're_true/all', 're + rr/all', 're + rr + tr/all', 't/all', 'u/all', 'tu + u/all'], y_vars=["Elements"],
                 hue="goldschmidt_classification", hue_order=None, height=10, aspect=.25)


g.map(sns.scatterplot, size=initial_data['ion_radius'], linewidth=0.5, marker='o', edgecolor='black')

g.add_legend()


# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 100), xlabel="% of minerals", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['All IMA-approved minerals', 'tRE/All', '(RE + RR)/All', '(RE + RR + TR)/All', 'T/All', 'U/All', '(TU + U)/All']

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

plt.savefig(f"figures/chemistry/dot_plot_sorted_by_crustal_abundance_size_ion_radius.jpeg", dpi=300, format='jpeg')
plt.close()


# Dot plot with a Facet Grid

sns.set_theme(style="whitegrid")
initial_data = abundance.sort_values(['goldschmidt_classification', 'ion_radius'])

g = sns.FacetGrid(initial_data, col='goldschmidt_classification', height=10, aspect=.25, hue='crust_crc_handbook', palette='flare')


g.map_dataframe(sns.scatterplot, x='re_true/all', y='Elements', legend='brief', size=initial_data['ion_radius'], linewidth=0.5, marker='o', edgecolor='black')

g.add_legend()

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 100), xlabel="% of minerals", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['Atmophile', 'Chalcophile', 'Lithophile', 'Siderophile']

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

plt.savefig(f"figures/chemistry/dot_plot_sorted_by_ion_radius.jpeg", dpi=300, format='jpeg')
plt.close()


# Binary plot: mineral abundance vs crustal abundance of element

abundance_log = abundance.copy()[['abundance_all', 'crust_crc_handbook', 'goldschmidt_classification']]
abundance_log['crust_crc_handbook'] = pd.to_numeric(abundance_log['crust_crc_handbook'])
abundance_log['crust_crc_handbook'] = np.log(abundance_log['crust_crc_handbook'])
abundance_log['abundance_all'] = np.log(abundance_log['abundance_all'])
abundance_log['Elements'] = abundance_log.index

sns.set_theme(style="whitegrid")

sc = sns.scatterplot(x="abundance_all",
                y="crust_crc_handbook",
                hue="goldschmidt_classification",
                sizes=(1, 2),
                linewidth=0,
                data=abundance_log)

for line in range(0, abundance_log.shape[0]):
    sc.text(abundance_log['abundance_all'][line] + 0.01, abundance_log['crust_crc_handbook'][line],
            abundance_log['Elements'][line], horizontalalignment='right',
            color='black', weight='light', fontsize='xx-small')

plt.xlabel("log(Number of minerals)")
plt.ylabel("log(Crustal abundance)")
plt.title(None)

plt.savefig(f"figures/chemistry/element_abundance_vs_mineral_clark.jpeg", dpi=300, format='jpeg')
plt.close()


# calculate elements co-occurrence matrixes
# All
mr_el_vector = mr_data.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
mr_el_vector[1] = pd.DataFrame(mr_data.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
mr_el_vector = mr_el_vector.explode(1)
mr_el_vector = mr_el_vector.loc[mr_el_vector[0] != mr_el_vector[1]]

# tRE
re_true_el_vector = re_true.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
re_true_el_vector[1] = pd.DataFrame(re_true.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
re_true_el_vector = re_true_el_vector.explode(1)
re_true_el_vector = re_true_el_vector.loc[re_true_el_vector[0] != re_true_el_vector[1]]


# RE
re_el_vector = re.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
re_el_vector[1] = pd.DataFrame(re.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
re_el_vector = re_el_vector.explode(1)
re_el_vector = re_el_vector.loc[re_el_vector[0] != re_el_vector[1]]

# RR
rr_el_vector = rr.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
rr_el_vector[1] = pd.DataFrame(rr.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
rr_el_vector = rr_el_vector.explode(1)
rr_el_vector = rr_el_vector.loc[rr_el_vector[0] != rr_el_vector[1]]

# RR
r_el_vector = r.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
r_el_vector[1] = pd.DataFrame(r.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
r_el_vector = r_el_vector.explode(1)
r_el_vector = r_el_vector.loc[r_el_vector[0] != r_el_vector[1]]

# T
t_el_vector = t.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
t_el_vector[1] = pd.DataFrame(t.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
t_el_vector = t_el_vector.explode(1)
t_el_vector = t_el_vector.loc[t_el_vector[0] != t_el_vector[1]]

# U
u_el_vector = u.Formula.str.extractall('(REE|[A-Z][a-z]?)').droplevel(1).reset_index().drop_duplicates(subset=['index', 0]).set_index('index')
u_el_vector[1] = pd.DataFrame(u.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))[0]
u_el_vector = u_el_vector.explode(1)
u_el_vector = u_el_vector.loc[u_el_vector[0] != u_el_vector[1]]

cooccurrence_all = calculate_cooccurrence_matrix(mr_el_vector[0], mr_el_vector[1])
cooccurrence_re_true = calculate_cooccurrence_matrix(re_true_el_vector[0], re_true_el_vector[1])
cooccurrence_re = calculate_cooccurrence_matrix(re_el_vector[0], re_el_vector[1])
cooccurrence_rr = calculate_cooccurrence_matrix(rr_el_vector[0], rr_el_vector[1])
cooccurrence_r = calculate_cooccurrence_matrix(r_el_vector[0], r_el_vector[1])
cooccurrence_t = calculate_cooccurrence_matrix(t_el_vector[0], t_el_vector[1])
cooccurrence_u = calculate_cooccurrence_matrix(u_el_vector[0], u_el_vector[1])


# Group by each element and calculate sum of occurrences for each
cooccurrence_size = cooccurrence_r.sum()
cooccurrence_size.sort_values(0, inplace=True, ascending=False)

cooccurrence_size = cooccurrence_re.sum()
cooccurrence_size.sort_values(0, inplace=True, ascending=False)

# calculate unique cooccurrences

r_el_vector.drop_duplicates(ignore_index=True).groupby(0).count().sort_values(1, ascending=False)[:10]

# create network graphs

## Circle graph
circular_data = re_true_el_vector.drop_duplicates(ignore_index=True).join(abundance, on=0, how='left')[[0, 1, 'crust_crc_handbook',]].sort_values('crust_crc_handbook')
circular_data['crust_crc_handbook'] = pd.to_numeric(circular_data['crust_crc_handbook'])
circular_data['crust_crc_handbook'] = np.log(circular_data['crust_crc_handbook'])

G = nx.from_pandas_edgelist(circular_data, source=0, target=1)
node_size =  [(v * 10 if not np.isnan(v) else 100) for v in circular_data['crust_crc_handbook'].drop_duplicates().values]

nx.draw_circular(G, with_labels=True, node_size=node_size, width=0.1, font_size=5)
plt.savefig(f"figures/chemistry/circular_network/re_true_elements.jpeg", dpi=300, format='jpeg')
plt.close()


## Spring layout

G = nx.from_pandas_edgelist(r_el_vector, source=0, target=1)
degree_centrality = nx.centrality.degree_centrality(G)
betweenness_centrality = nx.centrality.betweenness_centrality(G)
closeness_centrality = nx.centrality.closeness_centrality(G)  # save results in a variable to use again
eigenvector_centrality = nx.centrality.eigenvector_centrality(G)
pos = nx.spring_layout(G, iterations=15, seed=1721)

node_size =  [v * 1000 for v in degree_centrality.values()]
node_color =  [v * 10 for v in eigenvector_centrality.values()]
plt.figure(figsize=(10, 9))
plt.axis('off')

nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=node_color, node_size=node_size, alpha=0.7, cmap=plt.cm.jet)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=G.edges(),
    width=0.1,
)
nx.draw_networkx_labels(G, pos=pos,  font_size=10)

plt.tight_layout()
plt.savefig(f"figures/chemistry/spring_network/r_elements_eigenvector.jpeg", dpi=300, format='jpeg')
plt.close()

## test the degree centrality
(sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True))[:10]
(sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True))[:10]
(sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True))[:10]
(sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True))[:10]

## analyse specific elements EXACT OR INEXACT MATCH
elements = ['Pb']
initial_data = re_true_el
exact = False

minerals = initial_data.loc[initial_data['Elements'].isin(elements)]
minerals['count'] = minerals.groupby(minerals.index).count()
if exact:
    minerals = minerals.loc[minerals['count'] == len(elements)]
minerals = minerals.index
test = mr_data.loc[minerals].drop_duplicates()
r_el.loc[minerals].groupby('Elements').size().sort_values()
test.groupby('CLASS').size().sort_values()
test.groupby('SUBCLASS').size().sort_values()
test.groupby('FAMILY').size().sort_values()

### further analyse anions of these elements
anions_to_check = ['SO_4_']

anions = pd.DataFrame(test['anions_theoretical'].str.split('; *?').explode(0))
anions = anions.loc[anions['anions_theoretical'].isin(anions_to_check)].index.drop_duplicates()
test = mr_data.loc[anions].drop_duplicates()
test.groupby('CLASS').size().sort_values()

### check unique anions within a subset
anions = pd.DataFrame(test['anions_theoretical'].str.split('; *?').explode(0))
anions_unique = anions.groupby('anions_theoretical').size().sort_values()


### further analyse cations of these elements
cations_to_check = ['Na', 'Pb', 'As']

cations = pd.DataFrame(test['cations_theoretical'].str.split('; *?').explode(0))
cations = cations.loc[cations['cations_theoretical'].isin(cations_to_check)].index.drop_duplicates()
test = mr_data.loc[cations].drop_duplicates()
test.groupby('CLASS').size().sort_values()

### check unique cations within a subset
cations = pd.DataFrame(test['cations_theoretical'].str.split('; *?').explode(0))
cations_unique = cations.groupby('cations_theoretical').size().sort_values()

# export to csv
cooccurrence_re_true.to_csv('supplementary_data/cooc_re_true.csv', sep=',')
cooccurrence_all.to_csv('supplementary_data/cooc_all.csv', sep=',')
cooccurrence_r.to_csv('supplementary_data/cooc_r.csv', sep=',')
cooccurrence_re.to_csv('supplementary_data/cooc_re.csv', sep=',')
cooccurrence_rr.to_csv('supplementary_data/cooc_rr.csv', sep=',')
cooccurrence_t.to_csv('supplementary_data/cooc_t.csv', sep=',')
cooccurrence_u.to_csv('supplementary_data/cooc_u.csv', sep=',')