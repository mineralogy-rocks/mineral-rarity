import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import StandardScaler

# -*- coding: utf-8 -*-

class Scaler():
    def __init__(self):
        self.scaler = StandardScaler()

    def scale_features(self, np_array):
        """Scale by StandardScale of the dataset"""

        self.scaler.fit(np_array)

        return self.scaler.transform(np_array)

    def descale_features(self, np_array):
        """ Descale by StandardScale of the dataset"""

        return self.scaler.inverse_transform(np_array)


def transform_data(df, ScalerClass):
    """ Transform data and return transformed, raw, scaled, normalized datasets """

    df = df.loc[df['locality_counts'].notna()]
    df['locality_counts'] = pd.to_numeric(df['locality_counts'])

    raw_locality_mineral_pairs = _construct_mineral_locality_pairs(df.copy())
    log_locality_mineral_pairs = _standardize_features(raw_locality_mineral_pairs.copy(), skewed_features=['mineral_count', 'locality_counts'])

    raw_locality_1d = _construct_locality_features(df.copy())
    log_locality_1d = _standardize_features(raw_locality_1d)
    scaled_locality_1d = ScalerClass.scale_features(log_locality_1d)

    return raw_locality_mineral_pairs, log_locality_mineral_pairs, scaled_locality_1d


def _construct_mineral_locality_pairs(df):
    """ Create dataframe grouped by locality counts
    Returns:
        df['locality_count', 'mineral_count']
    """

    df['mineral_count'] = 1

    df = df.groupby(by='locality_counts').agg(mineral_count=pd.NamedAgg(column="mineral_count", aggfunc="sum"))
    df = df.sort_values(by=['mineral_count', 'locality_counts'], ascending=(False, True), axis=0).reset_index(
        level=0)

    return df


def _construct_locality_features(df):
    """ Create numpy array with localities features in (-1,1) shape, eg [ [1], [2], [3] ] """

    df = df['locality_counts'].to_numpy(dtype=int).reshape(-1, 1)

    return df


def _standardize_features(data, skewed_features=[]):
    """ Log transform of the df or numpy array """

    if isinstance(data, pd.DataFrame):
        for feature in skewed_features:
            data[feature] = np.log(data[feature])
    else:
        data = np.log(data)

    return data


def parse_rruff(data):
    """ Clean and transform RRUFF data  """

    data.loc[data['Year First Published'] == 0, 'Year First Published'] = np.nan
    data.rename(columns={'Mineral Name': 'mineral_name', 'Year First Published': 'discovery_year'}, inplace=True)

    rruff_data = data[['mineral_name', 'discovery_year']]
    rruff_data.set_index('mineral_name', inplace=True)

    return rruff_data


def parse_mindat(data):
    """ Clean and transform mindat data  """

    data.replace('\\N', np.nan, inplace=True)

    data.rename(columns={'loccount': 'locality_counts'}, inplace=True)

    ## Don't have any clue why mindat keeps `0` in imayear column and mixes dtypes in others (?!)

    data['imayear'].replace('0', np.nan, inplace=True)

    locs_md = data[['name', 'imayear', 'yeardiscovery', 'yearrruff', 'locality_counts']]
    locs_md.loc[:, 'imayear'] = pd.to_numeric(locs_md['imayear'])
    locs_md.loc[:, 'yearrruff'] = pd.to_numeric(locs_md['yearrruff'])

    locs_md.loc[:, 'locality_counts'] = pd.to_numeric(locs_md['locality_counts'])
    locs_md.loc[~locs_md['yeardiscovery'].str.match(r'[0-9]{4}', na=False), 'yeardiscovery'] = np.nan
    locs_md.loc[:, 'yeardiscovery'] = pd.to_numeric(locs_md['yeardiscovery'])

    locs_md.set_index('name', inplace=True)

    return locs_md


def get_discovery_rate_all(data, min_year=1500):
    """ Get all minerals counts grouped by the discovery year (all from MR)"""

    discovery_rate_all = data.loc[
        (data['discovery_year'].notna()) & (data['discovery_year'] > min_year)]

    discovery_rate_all = discovery_rate_all.sort_values('discovery_year', ascending=True)[['discovery_year']]

    discovery_rate_all = discovery_rate_all.groupby('discovery_year').agg({'discovery_year': 'count'})

    discovery_rate_all.rename(columns={'discovery_year': 'count'}, inplace=True)

    return discovery_rate_all


def get_discovery_rate_endemic(data):
    """ Get endemic minerals counts grouped by the discovery year """

    discovery_rate_endemic = data.loc[
        (data['locality_counts'] == 1) & (data['discovery_year'].notna())]

    discovery_rate_endemic = discovery_rate_endemic.sort_values('discovery_year', ascending=True)

    discovery_rate_endemic = discovery_rate_endemic.groupby('discovery_year').agg({'locality_counts': 'sum'})

    discovery_rate_endemic.rename(columns={'locality_counts': 'count'}, inplace=True)

    return discovery_rate_endemic


def get_endemic_proportion(discovery_rate_endemic, discovery_rate_all):
    """ Calculate the proportion of endemic minerals from all """

    endemic_all_prop = discovery_rate_endemic.join(discovery_rate_all, how='inner', lsuffix='_endemic', rsuffix='_all')

    endemic_all_prop['proportion'] = endemic_all_prop['count_endemic'] / endemic_all_prop['count_all'] * 100

    return endemic_all_prop


def calculate_cooccurrence_matrix(data0, data1, norm=False):
    cooccurrence = pd.crosstab(data0, data1, normalize=norm)
    np.fill_diagonal(cooccurrence.values, 0)
    return cooccurrence


def split_by_rarity_groups(data):
    r = data.loc[(data['locality_counts'] <= 4)]
    re_rr_tr = data.loc[data['locality_counts'] <= 16]
    re_true = data.loc[
        ~((data['discovery_year'] > 2000) & (data['locality_counts'] == 1)) & (data['locality_counts'] == 1)]
    re = data.loc[(data['locality_counts'] == 1)]
    rr = data.loc[(data['locality_counts'] <= 4) & (data['locality_counts'] >= 2)]

    t = data.loc[(data['locality_counts'] > 4) & (data['locality_counts'] <= 70)]
    tr = data.loc[(data['locality_counts'] > 4) & (data['locality_counts'] <= 16)]
    tu = data.loc[(data['locality_counts'] > 16) & (data['locality_counts'] <= 70)]

    u = data.loc[(data['locality_counts'] > 70)]
    tu_u = data.loc[(data['locality_counts'] > 16)]

    r.sort_values(by='discovery_year', inplace=True)
    re_true.sort_values(by='discovery_year', inplace=True)
    re.sort_values(by='discovery_year', inplace=True)
    rr.sort_values(by='discovery_year', inplace=True)
    t.sort_values(by='discovery_year', inplace=True)
    tr.sort_values(by='discovery_year', inplace=True)
    tu.sort_values(by='discovery_year', inplace=True)
    u.sort_values(by='discovery_year', inplace=True)

    return r, re_rr_tr, re_true, re, rr, t, tr, tu, u, tu_u


def get_mineral_clarks(data):
    data_el = pd.DataFrame(
        data.Formula.str.extractall('(REE|[A-Z][a-z]?)').groupby(level=0)[0].apply(lambda x: list(set(x))))
    data_el.rename(columns={0: 'Elements'}, inplace=True)
    data_el = data_el.explode('Elements')
    data_el_spread = pd.DataFrame(data_el.groupby('Elements').size().sort_values(), columns=['abundance'])

    return data_el, data_el_spread


def get_ns_obj():
    return pd.DataFrame([
        {'class': 'Borates', 'order': 0, 'color': 'green' },
        {'class': 'Carbonates (Nitrates)', 'order': 1, 'color': 'teal'},
        {'class': 'Elements', 'order': 2, 'color': 'indigo'},
        {'class': 'Halides', 'order': 3, 'color': 'chartreuse'},
        {'class': 'Organic Compounds', 'order': 4, 'color': 'coral'},
        {'class': 'Oxides', 'order': 5, 'color': 'gold'},
        {'class': 'Phosphates, Arsenates, Vanadates', 'order': 6, 'color': 'crimson'},
        {'class': 'Silicates (Germanates)', 'order': 6, 'color': 'purple'},
        {'class': 'Sulfates (selenates, tellurates, chromates, molybdates, wolframates)', 'order': 6, 'color': 'darkslateblue'},
        {'class': 'Sulfides and Sulfosalts', 'order': 6, 'color': 'coral'},
        ]).set_index('class')


def get_crystal_system_obj():
    return pd.DataFrame([
        {'system': 'triclinic', 'order': 0, 'color': 'cadetblue' },
        {'system': 'monoclinic', 'order': 1, 'color': 'teal'},
        {'system': 'orthorhombic', 'order': 2, 'color': 'indigo'},
        {'system': 'trigonal', 'order': 3, 'color': 'chartreuse'},
        {'system': 'tetragonal', 'order': 4, 'color': 'coral'},
        {'system': 'hexagonal', 'order': 5, 'color': 'gold'},
        {'system': 'isometric', 'order': 6, 'color': 'crimson'},
        ]).set_index('system')


def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


def heaviest_edge(G):
    u, v, w = max(G.edges(data="size"), key=itemgetter(2))
    return (u, v)


def prepare_data(ns, crystal):
    locs_md = pd.read_csv('data/mindat_locs.csv', sep=',')
    rruff_data = pd.read_csv('data/RRUFF_Export.csv', sep=',')

    rruff_data = parse_rruff(rruff_data)
    locs_md = parse_mindat(locs_md)

    # Clean and transform MR data
    ns.set_index('Mineral_Name', inplace=True)
    crystal.set_index('Mineral_Name', inplace=True)

    mindat_rruff = locs_md.join(rruff_data, how='outer')
    mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

    # create final subset for the analysis
    mr_data = ns.join(mindat_rruff, how='inner')
    mr_data = mr_data.join(crystal[['Crystal System']], how='inner')
    mr_data.loc[mr_data['Crystal System'].isin(['icosahedral', 'amorphous']), 'Crystal System'] = np.nan

    return mr_data


def get_symmetry_indexes(data):

    data.loc[:, 'triclinic'] = 0
    data.loc[:, 'isometric'] = 0
    data.loc[:, 'hexagonal'] = 0
    data.loc[:, 'trigonal'] = 0
    data.loc[:, 'orthorhombic'] = 0
    data.loc[:, 'monoclinic'] = 0
    data.loc[:, 'tetragonal'] = 0
    data.loc[data['Crystal System'] == 'triclinic', 'triclinic'] = 1
    data.loc[data['Crystal System'] == 'isometric', 'isometric'] = 1
    data.loc[data['Crystal System'] == 'hexagonal', 'hexagonal'] = 1
    data.loc[data['Crystal System'] == 'trigonal', 'trigonal'] = 1
    data.loc[data['Crystal System'] == 'orthorhombic', 'orthorhombic'] = 1
    data.loc[data['Crystal System'] == 'monoclinic', 'monoclinic'] = 1
    data.loc[data['Crystal System'] == 'tetragonal', 'tetragonal'] = 1

    discovery_rate = data.sort_values('discovery_year', ascending=True)

    discovery_rate = discovery_rate.groupby('discovery_year').agg(
        isometric=pd.NamedAgg(column="isometric", aggfunc="sum"),
        triclinic=pd.NamedAgg(column="triclinic", aggfunc="sum"),
        hexagonal=pd.NamedAgg(column="hexagonal", aggfunc="sum"),
        trigonal=pd.NamedAgg(column="trigonal", aggfunc="sum"),
        orthorhombic=pd.NamedAgg(column="orthorhombic", aggfunc="sum"),
        monoclinic=pd.NamedAgg(column="monoclinic", aggfunc="sum"),
        tetragonal=pd.NamedAgg(column="tetragonal", aggfunc="sum"),
    )

    discovery_rate['triclinic_index'] = discovery_rate['triclinic'] / discovery_rate['isometric']
    discovery_rate['symmetry_index'] = (discovery_rate['isometric'] + discovery_rate['hexagonal'] + discovery_rate[
        'trigonal'] +
                                        discovery_rate['tetragonal']) \
                                       / (discovery_rate['triclinic'] + discovery_rate['monoclinic'] + discovery_rate[
        'orthorhombic'])

    return discovery_rate